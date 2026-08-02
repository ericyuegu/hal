"""Tests for hal.data.extract: pure-function helpers plus dev-archive .slp
extraction (schema coverage, id spaces, item slots, offline/online parity)."""

from pathlib import Path

import numpy as np
import pytest

from hal.data.extract import _list_to_np
from hal.data.extract import _unpack_buttons
from hal.data.extract import extract_replay
from hal.data.schema import MDS_PER_FRAME_DTYPES
from hal.paths import DEV_ARCHIVE_PATH
from hal.wire import BUTTON_BITS
from hal.wire import ITEM_FIELD_SUFFIXES
from hal.wire import ITEM_SLOTS
from hal.wire import MASK_INT32
from hal.wire import TRIGGER_DEADZONE
from hal.wire import VALID_LIBMELEE_PORTS
from hal.wire import item_column
from hal.wire import mask_value as _mask_value


class _ArrowLike:
    """Minimal stand-in for a pyarrow Array exposing only ``.to_pylist()``."""

    def __init__(self, values: list[object]) -> None:
        self._values = values

    def to_pylist(self) -> list[object]:
        return list(self._values)


@pytest.fixture(scope="module")
def dev_slp(tmp_path_factory: pytest.TempPathFactory) -> str:
    """First .slp member of the dev archive, extracted once per module."""
    import py7zr

    if not Path(DEV_ARCHIVE_PATH).exists():
        pytest.skip(f"dev archive missing at {DEV_ARCHIVE_PATH}; run `python -m hal.scripts.fetch --name dev.7z`")
    out = tmp_path_factory.mktemp("dev_slp")
    with py7zr.SevenZipFile(DEV_ARCHIVE_PATH, "r") as z:
        member = next(m for m in z.getnames() if m.endswith(".slp"))
        z.extract(path=out, targets=[member])
    return str(out / member)


@pytest.fixture(scope="module")
def dev_sample(dev_slp: str) -> dict[str, np.ndarray]:
    sample = extract_replay(dev_slp)
    assert sample is not None, f"extract_replay returned None for {dev_slp}"
    return sample


def test_mask_value_float_is_nan() -> None:
    assert np.isnan(_mask_value(np.float32))
    assert np.isnan(_mask_value(np.float64))


def test_mask_value_signed_int_is_dtype_min_for_narrow() -> None:
    assert _mask_value(np.int8) == np.iinfo(np.int8).min
    assert _mask_value(np.int16) == np.iinfo(np.int16).min


def test_mask_value_int32_is_np_mask_value() -> None:
    assert _mask_value(np.int32) == MASK_INT32
    assert _mask_value(np.int32) == (1 << 31) - 1


def test_mask_value_unsigned_is_dtype_max() -> None:
    assert _mask_value(np.uint8) == np.iinfo(np.uint8).max
    assert _mask_value(np.uint16) == np.iinfo(np.uint16).max


def test_mask_value_float_callers_must_use_isnan_not_eq() -> None:
    """Regression for the silent-mask-detection footgun: ``arr == mask`` is
    always False for the NaN sentinel because ``nan != nan``."""
    mask = _mask_value(np.float32)
    arr = np.full(5, mask, dtype=np.float32)
    assert not np.any(arr == mask)  # the trap
    assert np.all(np.isnan(arr))  # the correct check


def test_schema_has_no_action_frame_column() -> None:
    """action_frame is dropped: training reconstructed it as a run-length while
    inference reads the engine's state_age — they never matched (0/N frames).
    The column is gone from the schema entirely."""
    assert not any("action_frame" in col for col in MDS_PER_FRAME_DTYPES)


def test_schema_gamestate_fields_match_post_suffixes() -> None:
    """With action_frame gone, every per-player gamestate column is exactly one
    POST_FIELD_SUFFIXES entry — no schema field without a wire counterpart."""
    from hal.wire import POST_FIELD_SUFFIXES

    p1_gamestate = {
        col.removeprefix("p1_")
        for col in MDS_PER_FRAME_DTYPES
        if col.startswith("p1_")
        and not col.startswith("p1_nana_")
        and "_button_" not in col
        and "stick" not in col
        and not col.endswith(("_trigger_l", "_trigger_r"))
        and col != "p1_character"  # per-replay constant, not a post field
    }
    assert p1_gamestate == set(POST_FIELD_SUFFIXES)


def test_unpack_buttons_decodes_bitmask() -> None:
    a_bit = BUTTON_BITS["a"]
    b_bit = BUTTON_BITS["b"]
    # Frame 0: A only; frame 1: A+B; frame 2: none.
    physical = _ArrowLike([a_bit, a_bit | b_bit, 0])
    out = _unpack_buttons(physical, length=3)
    assert list(out["a"]) == [1, 1, 0]
    assert list(out["b"]) == [0, 1, 0]
    assert list(out["x"]) == [0, 0, 0]
    assert out["a"].dtype == np.int32


def test_unpack_buttons_none_yields_all_zeros() -> None:
    out = _unpack_buttons(None, length=4)
    assert set(out) == set(BUTTON_BITS)
    for arr in out.values():
        assert list(arr) == [0, 0, 0, 0]


def test_list_to_np_substitutes_none_with_mask() -> None:
    arr = _list_to_np([1.0, None, 3.0], np.float32, length=3)
    assert arr[0] == 1.0
    assert np.isnan(arr[1])
    assert arr[2] == 3.0


def test_list_to_np_none_input_returns_full_mask() -> None:
    arr = _list_to_np(None, np.int8, length=4)
    assert all(v == np.iinfo(np.int8).min for v in arr)


def test_schema_item_columns_cover_every_slot_and_field() -> None:
    """The global item block is exactly ITEM_SLOTS x ITEM_FIELD_SUFFIXES, with no
    player prefix — items are global state, not per-player."""
    item_cols = {col for col in MDS_PER_FRAME_DTYPES if col.startswith("item")}
    assert item_cols == {item_column(k, s) for k in range(ITEM_SLOTS) for s in ITEM_FIELD_SUFFIXES}
    assert not any(col.startswith(("p1_", "p2_")) for col in item_cols)


def test_extract_replay_produces_full_schema(dev_sample: dict[str, np.ndarray]) -> None:
    """Every column declared in MDS_PER_FRAME_DTYPES is present after extract,
    with the right dtype and a common frame length. Pins schema/extract drift."""
    missing = set(MDS_PER_FRAME_DTYPES) - set(dev_sample)
    extra = set(dev_sample) - set(MDS_PER_FRAME_DTYPES)
    assert not missing, f"missing columns: {sorted(missing)[:8]}"
    assert not extra, f"unexpected columns: {sorted(extra)[:8]}"

    frame_len = dev_sample["frame"].shape[0]
    assert frame_len > 0
    for col, dtype in MDS_PER_FRAME_DTYPES.items():
        assert dev_sample[col].shape == (frame_len,), f"length mismatch on {col}"
        assert dev_sample[col].dtype == np.dtype(dtype), f"dtype mismatch on {col}"

    # Triggers are stored game-causal: sub-deadzone hardware jitter zeroed.
    for col in ("p1_trigger_l", "p1_trigger_r", "p2_trigger_l", "p2_trigger_r"):
        t = dev_sample[col][~np.isnan(dev_sample[col])]
        assert np.all((t == 0.0) | (t >= TRIGGER_DEADZONE)), f"{col} has sub-deadzone values"


def test_extract_replay_stores_internal_character_id(dev_slp: str, dev_sample: dict[str, np.ndarray]) -> None:
    """SCHEMA_VERSION 5 normalization: ``p{1,2}_character`` is the libmelee
    INTERNAL Character value (the in-game / post-frame id), NOT the slp external
    character-select id from the start block. Pre-v5 this stored the external id;
    this test fails on that code (external != internal for these dev replays)."""
    import peppi_py

    stored = {int(dev_sample["p1_character"][0]), int(dev_sample["p2_character"][0])}

    # Ground truth straight from peppi: start.character is the EXTERNAL id, the
    # post-frame character is the INTERNAL id libmelee/the engine use.
    g = peppi_py.read_slippi(dev_slp, skip_frames=False)
    occupied = [
        (i, sp) for i, sp in enumerate(g.start.players) if str(getattr(sp.type, "name", sp.type)).upper() != "EMPTY"
    ]
    external = {int(sp.character) for _, sp in occupied}
    internal = {int(g.frames.ports[i].leader.post.character[0]) for i, _ in occupied}

    assert stored == internal, f"stored {stored} should be the internal/post-frame ids {internal}"
    # Guard the test's own teeth: if this dev replay used a coincide-id character
    # (external == internal) the assertion above couldn't distinguish the spaces.
    assert external != internal, f"fixture has external==internal ({external}); pick a replay where they differ"


def test_character_live_is_the_post_frame_internal_id(dev_slp: str, dev_sample: dict[str, np.ndarray]) -> None:
    """SCHEMA_VERSION 6: the per-frame ``character_live`` column comes from the post
    block, which is ALREADY the libmelee internal id space — no external→internal
    conversion applies to it (unlike the start block's select pick). It must equal
    peppi's post-frame character verbatim."""
    import peppi_py

    g = peppi_py.read_slippi(dev_slp, skip_frames=False)
    # This fixture has no Sheik<->Zelda transform, so the live id is constant and
    # equal to the (normalized) select pick — the id SPACE is what's under test.
    for prefix in ("p1", "p2"):
        live = dev_sample[f"{prefix}_character_live"]
        assert np.all(live == live[0])
        assert int(live[0]) in {int(g.frames.ports[i].leader.post.character[0]) for i in range(len(g.start.players))}
        assert int(live[0]) == int(dev_sample[f"{prefix}_character"][0])


def test_extract_items_are_slot_ordered_by_spawn_id(dev_slp: str, dev_sample: dict[str, np.ndarray]) -> None:
    """Slot 0 holds the frame's lowest live spawn id, slot 1 the next, and so on —
    the ordering contract ``wire.canonical_item_columns`` repeats online."""
    import peppi_py

    g = peppi_py.read_slippi(dev_slp, skip_frames=False)
    offsets = np.asarray(g.frames.item_offset.to_pylist(), dtype=np.int64)
    spawn_id = np.asarray(g.frames.items.id.to_pylist(), dtype=np.int64)
    item_type = np.asarray(g.frames.items.type.to_pylist(), dtype=np.int64)
    frame_ids = np.asarray(g.frames.id.to_pylist(), dtype=np.int64)

    checked = 0
    for raw in range(offsets.size - 1):
        lo, hi = int(offsets[raw]), int(offsets[raw + 1])
        if hi - lo < 2:
            continue
        rows = np.flatnonzero(dev_sample["frame"] == frame_ids[raw])
        if rows.size != 1:
            continue  # rollback-superseded row; the kept row is a later occurrence
        row = int(rows[0])
        want = item_type[lo + np.argsort(spawn_id[lo:hi], kind="stable")][:ITEM_SLOTS]
        got = [int(dev_sample[item_column(k, "type")][row]) for k in range(len(want))]
        assert got == list(want), f"frame {frame_ids[raw]}: item slots {got} != spawn-id order {list(want)}"
        checked += 1
    assert checked > 0, "fixture has no frame with 2+ simultaneous items; ordering is untested"


def test_extract_item_owner_is_a_libmelee_port(dev_sample: dict[str, np.ndarray]) -> None:
    """Owner is normalized out of peppi's 0..3 space into the libmelee 1..4 ports
    the rest of the schema speaks; unowned stays ITEM_OWNER_NONE."""
    owner = dev_sample[item_column(0, "owner")]
    live = owner[owner != MASK_INT32]
    assert live.size > 0, "fixture has no owned items"
    assert set(np.unique(live)) <= {-1, *VALID_LIBMELEE_PORTS}
    assert set(np.unique(live)) & set(VALID_LIBMELEE_PORTS), "owner never resolved to a real port"


def test_empty_item_slots_are_masked(dev_sample: dict[str, np.ndarray]) -> None:
    """A frame with no items carries the full item column set, all masked — the
    column set never varies with item count."""
    empty = np.isnan(dev_sample[item_column(0, "pos_x")])
    assert empty.any(), "fixture has items on every frame"
    for slot in range(ITEM_SLOTS):
        assert np.all(dev_sample[item_column(slot, "type")][empty] == MASK_INT32)
        assert np.all(np.isnan(dev_sample[item_column(slot, "vel_x")][empty]))


def test_extract_matches_canonical_flatten_column_for_column(dev_slp: str, dev_sample: dict[str, np.ndarray]) -> None:
    """The offline MDS row and the online observation must agree value-for-value.

    ``peppi_py.read_frame_dicts`` emits the canonical per-frame dict libmelee's
    ``GameState.to_canonical_dict`` reproduces byte-for-byte, so flattening it is
    a faithful stand-in for the closed-loop observation path — no Dolphin needed.
    Every column the two paths share (post block, nana block, item slots) is
    compared on every frame.
    """
    import peppi_py

    from hal.training.canonical import flatten_canonical_frame

    frames = peppi_py.read_frame_dicts(dev_slp)
    row_of_frame_id = {int(f): i for i, f in enumerate(dev_sample["frame"])}
    shared = sorted(set(flatten_canonical_frame(frames[0])) & set(dev_sample))
    assert len(shared) > len(MDS_PER_FRAME_DTYPES) // 2, f"only {len(shared)} shared columns; the paths have diverged"

    mismatches: list[str] = []
    for frame in frames:
        row = row_of_frame_id.get(int(frame["id"]))
        if row is None:
            continue
        flat = flatten_canonical_frame(frame)
        for col in shared:
            online = flat[col]
            offline = dev_sample[col][row]
            masked = np.isnan(offline) if dev_sample[col].dtype.kind == "f" else offline == MASK_INT32
            if np.isnan(online) and masked:
                continue
            if float(online) != float(offline):
                mismatches.append(f"frame {frame['id']} {col}: online {online} != offline {offline}")
    assert not mismatches, f"{len(mismatches)} online/offline mismatches:\n" + "\n".join(mismatches[:10])


def test_schema_controller_block_is_logical_only() -> None:
    """No raw byte columns and no fused trigger_logical — the pre-frame block
    is exactly the model action space."""
    assert not any("_raw_" in col or "trigger_logical" in col for col in MDS_PER_FRAME_DTYPES)
