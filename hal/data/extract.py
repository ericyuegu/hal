"""peppi-py-based per-frame extractor.

``extract_replay(path) -> dict[str, np.ndarray] | None`` returns one ndarray
per column in ``MDS_PER_FRAME_DTYPES`` ready to feed into ``MDSWriter.write``.
Returns ``None`` on any unrecoverable parse failure.

The output stream is rollback-deduplicated via ``wire.dedupe_keep_idx``: one
row per ``frame_id``, keeping the engine's committed (last) value. Pre-
countdown frames (``frame_id < wire.GAME_START_FRAME``) are dropped.

Values are stored game-causal: slp-logical sticks as-is, per-shoulder triggers
with sub-deadzone jitter zeroed (``wire.TRIGGER_DEADZONE``), buttons unpacked
from the physical bitmask. Slp-version-unavailable fields get dtype-specific
mask sentinels. Items (projectiles) are global, not per-player: the K lowest
spawn ids alive on a frame fill the ``item{k}_*`` slots, the rest are masked.
See ``hal.wire`` for the controller and replay wire contracts.
"""

from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any

import numpy as np
import peppi_py
from loguru import logger
from numpy.typing import DTypeLike
from peppi_py.frame import Data
from peppi_py.frame import Frame
from peppi_py.frame import Item
from peppi_py.frame import Post
from peppi_py.game import Game

from hal.data.policy_schema import FLOAT_STATE_SUFFIXES
from hal.data.policy_schema import PACKED_STATE_SUFFIXES
from hal.data.schema import MDS_PER_FRAME_DTYPES
from hal.data.schema import rank_from_player_name
from hal.wire import BUTTON_BITS
from hal.wire import GAME_START_FRAME
from hal.wire import ITEM_FIELD_PATHS
from hal.wire import ITEM_FIELD_SUFFIXES
from hal.wire import ITEM_SLOTS
from hal.wire import PLAYER_PREFIXES
from hal.wire import POST_FIELD_SUFFIXES
from hal.wire import TRIGGER_DEADZONE
from hal.wire import dedupe_keep_idx
from hal.wire import item_column
from hal.wire import item_owner_to_libmelee_port
from hal.wire import mask_value
from hal.wire import peppi_port_to_libmelee
from hal.wire import post_field_path
from hal.wire import slp_character_to_libmelee
from hal.wire import slp_stage_to_libmelee


def _arr_to_np(arr: Any, dtype: DTypeLike, length: int) -> np.ndarray:
    """Convert a pyarrow Array (or None) to numpy with mask substitution.

    None whole-column means peppi didn't parse this field for this slp
    version — fill with the dtype's mask sentinel. None scalars within a
    column (e.g. peppi.hitlag for one frame) get the same treatment.
    """
    mask = mask_value(dtype)
    if arr is None:
        return np.full(length, mask, dtype=dtype)
    return _list_to_np(arr.to_pylist(), dtype, length)


def _list_to_np(values: Sequence[Any] | None, dtype: DTypeLike, length: int) -> np.ndarray:
    mask = mask_value(dtype)
    if values is None:
        return np.full(length, mask, dtype=dtype)
    return np.array([v if v is not None else mask for v in values], dtype=dtype)


def _walk(obj: Any, path: Sequence[str | int]) -> Any:
    """Follow a ``wire`` accessor path into peppi's SoA (attribute for a named
    step, index for a positional one). ``None`` anywhere means this slp version
    never recorded the block, so the whole column is masked."""
    cur: Any = obj
    for step in path:
        cur = getattr(cur, step) if isinstance(step, str) else cur[step]
        if cur is None:
            return None
    return cur


def _gamestate_arrays(
    post: Post,
    key_prefix: str,
    keep_idx: np.ndarray,
    length: int,
    suffixes: Sequence[str] = POST_FIELD_SUFFIXES,
) -> dict[str, np.ndarray]:
    """Pull every gamestate column for one post block under ``key_prefix``.

    Shared by leader (``p1``/``p2``) and follower (``p1_nana``/``p2_nana``).
    """
    out: dict[str, np.ndarray] = {}
    for suffix in suffixes:
        col = f"{key_prefix}_{suffix}"
        dtype = MDS_PER_FRAME_DTYPES[col]
        out[col] = _arr_to_np(_walk(post, post_field_path(suffix)), dtype, length)[keep_idx]
    return out


def _item_dtype(suffix: str) -> DTypeLike:
    """Storage dtype for one item field — identical across the slots."""
    return MDS_PER_FRAME_DTYPES[item_column(0, suffix)]


def _item_arrays(frames: Frame, keep_idx: np.ndarray) -> dict[str, np.ndarray]:
    """Global ``item{k}_*`` columns, already indexed to the kept rows.

    peppi stores items as one flat SoA plus ``item_offset``, an Arrow-style
    offsets array where frame ``i``'s items are ``[offset[i], offset[i+1])``.
    Slot assignment is by ascending spawn id (``wire.ITEM_SPAWN_ID_FIELD``) so
    it matches ``wire.canonical_item_columns`` on the closed-loop side; items
    past ``ITEM_SLOTS`` are dropped and empty slots stay masked.
    """
    out_length = int(keep_idx.size)
    out = {
        item_column(slot, suffix): np.full(out_length, mask_value(_item_dtype(suffix)), dtype=_item_dtype(suffix))
        for slot in range(ITEM_SLOTS)
        for suffix in ITEM_FIELD_SUFFIXES
    }
    items = frames.items
    if items is None or frames.item_offset is None:
        return out  # slp < 3.0 predates item events

    offsets = np.asarray(frames.item_offset.to_pylist(), dtype=np.int64)
    n_items = int(offsets[-1])
    if n_items == 0:
        return out

    flat = {
        suffix: _arr_to_np(_walk(items, ITEM_FIELD_PATHS[suffix]), _item_dtype(suffix), n_items)
        for suffix in ITEM_FIELD_SUFFIXES
        if suffix != "owner"
    }
    flat["owner"] = _item_owner_array(items, n_items)
    spawn_id = _arr_to_np(items.id, np.int64, n_items)

    # Rank every flat item by (its frame, then spawn id), then scatter the first
    # ITEM_SLOTS of each frame into their slot columns — one vectorized pass
    # instead of a per-frame Python loop over the whole replay.
    raw_row_of_item = np.repeat(np.arange(offsets.size - 1), np.diff(offsets))
    order = np.lexsort((spawn_id, raw_row_of_item))
    ranked_raw_row = raw_row_of_item[order]
    slot_of_item = np.arange(n_items) - offsets[ranked_raw_row]
    out_row_of_raw = np.full(offsets.size - 1, -1, dtype=np.int64)
    out_row_of_raw[keep_idx] = np.arange(out_length)
    out_row_of_item = out_row_of_raw[ranked_raw_row]
    for slot in range(ITEM_SLOTS):
        take = (slot_of_item == slot) & (out_row_of_item >= 0)
        rows = out_row_of_item[take]
        src = order[take]
        for suffix in ITEM_FIELD_SUFFIXES:
            out[item_column(slot, suffix)][rows] = flat[suffix][src]
    return out


def _item_owner_array(items: Item, n_items: int) -> np.ndarray:
    """Item owners as libmelee ports (1..4), ``wire.ITEM_OWNER_NONE`` preserved.

    The slp value is a peppi 0..3 port; every other port in the MDS is libmelee
    1..4. Converted element-wise through the wire bridge (items are sparse —
    hundreds per replay, not one per frame) so the online reader can't drift.
    """
    dtype = _item_dtype("owner")
    if items.owner is None:
        return np.full(n_items, mask_value(dtype), dtype=dtype)
    owners = [None if v is None else item_owner_to_libmelee_port(v) for v in items.owner.to_pylist()]
    return _list_to_np(owners, dtype, n_items)


def _unpack_buttons(physical: Any, length: int) -> dict[str, np.ndarray]:
    if physical is None:
        return {b: np.zeros(length, dtype=np.int32) for b in BUTTON_BITS}
    bits = np.array([v if v is not None else 0 for v in physical.to_pylist()], dtype=np.int32)
    return {b: ((bits & mask) != 0).astype(np.int32) for b, mask in BUTTON_BITS.items()}


def netplay_name(start_player: Any, metadata_players: Mapping[str, Any]) -> str | None:
    """Netplay display name of one start-block player, or None if it has none.

    ``metadata.players`` (keyed by the 0-indexed port as a string) wins. Anonymized
    .slps ship empty metadata but still carry ``netplay.name`` on the start block,
    hence the fallback. Shared with ``index.extract_index_entry`` so the name that
    lands in ``p{1,2}_rank`` is the same one the index records.
    """
    names = (metadata_players.get(str(int(start_player.port))) or {}).get("names") or {}
    netplay = getattr(start_player, "netplay", None)
    return names.get("netplay") or (getattr(netplay, "name", "") or None)


def _metadata_players(g: Game) -> Mapping[str, Any]:
    return (g.metadata or {}).get("players") or {}


def _peppi_idx_by_libmelee_port(g: Game) -> dict[int, int]:
    """Map libmelee port (1..4) to peppi's port-array index (0..3)."""
    return {peppi_port_to_libmelee(pl.port): i for i, pl in enumerate(g.start.players)}


def _extract_player(
    leader: Data,
    prefix: str,
    keep_idx: np.ndarray,
    length: int,
    state_suffixes: Sequence[str] = POST_FIELD_SUFFIXES,
) -> dict[str, np.ndarray]:
    """Pull the gamestate + controller columns for one port's leader (main char)."""
    pre = leader.pre
    out = _gamestate_arrays(leader.post, prefix, keep_idx, length, state_suffixes)

    # Buttons
    for b, arr in _unpack_buttons(pre.buttons_physical, length).items():
        out[f"{prefix}_button_{b}"] = arr[keep_idx]

    # Sticks (slp-logical: post-deadzone, [-1, 1])
    out[f"{prefix}_main_stick_x"] = _arr_to_np(pre.joystick.x, np.float32, length)[keep_idx]
    out[f"{prefix}_main_stick_y"] = _arr_to_np(pre.joystick.y, np.float32, length)[keep_idx]
    out[f"{prefix}_c_stick_x"] = _arr_to_np(pre.cstick.x, np.float32, length)[keep_idx]
    out[f"{prefix}_c_stick_y"] = _arr_to_np(pre.cstick.y, np.float32, length)[keep_idx]

    # Per-shoulder triggers, zeroed below the game's deadzone so the stored
    # value is game-causal — the same post-deadzone convention the slp logical
    # stick already has. (slp has no per-shoulder logical channel; its fused
    # ``pre.triggers`` scalar loses which shoulder moved, so we derive ours.)
    tp = pre.triggers_physical
    for shoulder, arr in (("l", tp.l if tp is not None else None), ("r", tp.r if tp is not None else None)):
        trigger = _arr_to_np(arr, np.float32, length)[keep_idx]
        trigger[trigger < TRIGGER_DEADZONE] = 0.0
        out[f"{prefix}_trigger_{shoulder}"] = trigger

    return out


def _extract_nana(
    follower: Data | None,
    prefix: str,
    keep_idx: np.ndarray,
    length: int,
    state_suffixes: Sequence[str] = POST_FIELD_SUFFIXES,
) -> dict[str, np.ndarray]:
    """Nana columns: gamestate only (no controller). Mask if no follower."""
    nana_prefix = f"{prefix}_nana"
    if follower is None:
        out_length = int(keep_idx.size)
        return {
            f"{nana_prefix}_{suffix}": np.full(
                out_length,
                mask_value(MDS_PER_FRAME_DTYPES[f"{nana_prefix}_{suffix}"]),
                dtype=MDS_PER_FRAME_DTYPES[f"{nana_prefix}_{suffix}"],
            )
            for suffix in state_suffixes
        }
    return _gamestate_arrays(follower.post, nana_prefix, keep_idx, length, state_suffixes)


def _extract_replay(
    replay_path: str,
    state_suffixes: Sequence[str],
) -> dict[str, np.ndarray] | None:
    """Parse a slp file and return per-frame ndarrays keyed by MDS column name.

    Returns None if peppi can't parse the file or the start block is missing
    expected players. Caller logs.
    """
    try:
        g = peppi_py.read_slippi(str(replay_path), skip_frames=False)
    except Exception as e:
        logger.debug(f"peppi failed for {replay_path}: {e}")
        return None

    if g.frames is None or g.frames.id is None:
        logger.debug(f"empty frames for {replay_path}")
        return None

    ids = g.frames.id.to_pylist()
    raw_length = len(ids)
    # Rollback dedup (last occurrence per frame_id) AND drop pre-countdown
    # frames. Each frame_id falls in exactly one bucket, so the two filters
    # compose by intersection regardless of order.
    keep_idx = dedupe_keep_idx(ids)
    kept_frame_ids = np.fromiter((ids[i] for i in keep_idx), dtype=np.int32, count=len(keep_idx))
    in_game = kept_frame_ids >= GAME_START_FRAME
    keep_idx = keep_idx[in_game]
    out_length = int(keep_idx.size)

    if out_length <= 0:
        logger.debug(f"no in-game frames for {replay_path}")
        return None

    peppi_idx_by_libmelee_port = _peppi_idx_by_libmelee_port(g)
    # Map p1/p2 to the two lowest occupied libmelee ports (in ascending order).
    # Replays on ports (3, 4) — common in tournament setups — would otherwise
    # be silently dropped. We still require exactly two players (1v1).
    occupied_libmelee_ports = sorted(peppi_idx_by_libmelee_port)
    if len(occupied_libmelee_ports) != len(PLAYER_PREFIXES):
        logger.debug(f"{replay_path}: {len(occupied_libmelee_ports)} players; expected {len(PLAYER_PREFIXES)} (1v1)")
        return None

    sample: dict[str, np.ndarray] = {
        "frame": kept_frame_ids[in_game],
        # Per-replay constants broadcast across frames (SCHEMA_VERSION 5). Both stage and
        # character are stored as their libmelee enum values — the same id space the
        # closed-loop obs uses, so no second translation is needed downstream. Stage goes
        # through slp_stage_to_libmelee; character (below) through slp_character_to_libmelee.
        # The slp EXTERNAL / character-select id survives only here, at the peppi read.
        "stage": np.full(out_length, int(slp_stage_to_libmelee(int(g.start.stage)).value), dtype=np.int32),
        **_item_arrays(g.frames, keep_idx),
    }

    metadata_players = _metadata_players(g)
    for prefix, port in zip(PLAYER_PREFIXES, occupied_libmelee_ports, strict=True):
        peppi_idx = peppi_idx_by_libmelee_port[port]
        port_data = g.frames.ports[peppi_idx]
        start_player = g.start.players[peppi_idx]
        slp_character = int(start_player.character)
        try:
            character = slp_character_to_libmelee(slp_character).value
        except ValueError:
            logger.debug(f"{replay_path}: unknown external character id {slp_character}; dropping replay")
            return None
        sample[f"{prefix}_character"] = np.full(out_length, character, dtype=np.int32)
        # Ranked tier, broadcast per frame (SCHEMA_VERSION 7). Unrecognized or absent
        # netplay names store Rank.UNKNOWN — a non-ranked corpus is then visibly untiered
        # rather than silently labeled.
        rank = rank_from_player_name(netplay_name(start_player, metadata_players))
        sample[f"{prefix}_rank"] = np.full(out_length, int(rank), dtype=np.uint8)
        sample.update(_extract_player(port_data.leader, prefix, keep_idx, raw_length, state_suffixes))
        sample.update(_extract_nana(port_data.follower, prefix, keep_idx, raw_length, state_suffixes))

    # Sanity: every column has the expected length.
    bad = [(k, v.shape[0]) for k, v in sample.items() if v.shape[0] != out_length]
    if bad:
        logger.debug(f"{replay_path}: column length mismatch {bad[:3]}")
        return None

    return sample


def extract_replay(replay_path: str) -> dict[str, np.ndarray] | None:
    """Extract every canonical-v7 per-frame column."""
    return _extract_replay(replay_path, POST_FIELD_SUFFIXES)


_POLICY_STATE_SUFFIXES: tuple[str, ...] = tuple(dict.fromkeys((*FLOAT_STATE_SUFFIXES, *PACKED_STATE_SUFFIXES)))


def extract_policy_world_replay(replay_path: str) -> dict[str, np.ndarray] | None:
    """Extract only fields needed by ``mds-policy-world-v7``.

    Peppi still validates and parses the whole replay, but avoiding conversion
    of unused Arrow columns into NumPy substantially reduces scale-up CPU and
    inter-process memory without changing any retained values.
    """
    return _extract_replay(replay_path, _POLICY_STATE_SUFFIXES)
