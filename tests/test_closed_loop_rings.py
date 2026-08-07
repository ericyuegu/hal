"""The ring context builder must reproduce the window builder it replaced, exactly.

``RecedingHorizon`` used to rebuild the whole ``L_ctx`` window from a rolling list of
per-frame dicts at every replan, then run ``preprocess`` over it. It now preprocesses
each frame ONCE into per-feature ring buffers and reads a window as one contiguous
slice. The two must produce the SAME model input — same keys, same dtypes, same bits
— through the cold-start pad, the pad→real seam, an instant-restart boundary and a
saturated context, with and without the schema-v6 ``extra`` routing.

``_reference_*`` below is the replaced implementation, copied verbatim. It is the
oracle, so it must not be "improved": a change to it is a change to the contract.
"""

import melee
import numpy as np
import pytest
import torch

from hal.data.feature_stats import FeatureStats
from hal.sim.vec import Slot
from hal.training.canonical import flatten_canonical_frame
from hal.training.closed_loop import RecedingHorizon
from hal.training.dataloader import relabel_ego
from hal.training.features import A_DIM
from hal.training.features import ACTION_CHANNELS
from hal.training.features import NEUTRAL_ACTION
from hal.training.features import V6_PLAYER_COLUMNS
from hal.training.features import ExtraColumns
from hal.training.features import preprocess

STAGE = int(melee.Stage.FINAL_DESTINATION.value)
EGO_PORT, OPP_PORT = 1, 2
L_CTX = 64
L_CHUNK = 8


# --- the replaced implementation, verbatim -----------------------------------


def _reference_window(
    flat_history: list[dict],
    ego_inputs_hist: list[np.ndarray],
    ego_prefix: str,
    L_ctx: int,
) -> dict[str, np.ndarray]:
    """``[1, L_ctx]`` batch the model expects, built from one slot's rolling buffers."""
    pad_g = L_ctx - len(flat_history)
    out: dict[str, np.ndarray] = {}
    keys = flat_history[0].keys()
    for k in keys:
        sample = flat_history[0][k]
        dtype = np.int32 if isinstance(sample, int) else np.float32
        vals = [h[k] for h in flat_history]
        if pad_g > 0:
            vals = [0] * pad_g + vals
        out[k] = np.array(vals, dtype=dtype)
    ego_aligned = [NEUTRAL_ACTION] * (len(flat_history) - len(ego_inputs_hist)) + list(ego_inputs_hist)
    if pad_g > 0:
        ego_aligned = [NEUTRAL_ACTION] * pad_g + ego_aligned
    hist_arr = np.stack(ego_aligned)
    for i, ch in enumerate(ACTION_CHANNELS):
        col = hist_arr[:, i]
        if ch.startswith("button_"):
            out[f"{ego_prefix}_{ch}"] = (col > 0.5).astype(np.int32)
        else:
            out[f"{ego_prefix}_{ch}"] = col.astype(np.float32)
    out.pop("frame", None)
    relabeled = relabel_ego(out, ego_prefix)
    return {k: v[None, ...] for k, v in relabeled.items()}


def _reference_features(
    batch: list[_ReferenceSlot],
    L_ctx: int,
    stats: dict[str, FeatureStats],
    extra: ExtraColumns | None,
) -> dict[str, torch.Tensor]:
    """The replaced ``_build_stacked_batch`` + ``preprocess`` + dtype packing."""
    per_slot = [_reference_window(s.flat_hist, s.ego_hist, s.prefix, L_ctx) for s in batch]
    stacked = {k: np.concatenate([d[k] for d in per_slot], axis=0) for k in per_slot[0]}
    preprocessed = preprocess(stacked, stats, extra=extra)
    float_items = [(k, v) for k, v in preprocessed.items() if v.dtype.is_floating_point]
    int_items = [(k, v) for k, v in preprocessed.items() if not v.dtype.is_floating_point]
    feats: dict[str, torch.Tensor] = {}
    for items in (float_items, int_items):
        if items:
            packed = torch.stack([v for _, v in items], dim=0)
            feats.update({k: v for (k, _), v in zip(items, packed.unbind(0), strict=True)})
    return feats


class _ReferenceSlot:
    """One slot's rolling buffers, driven by the replaced ``__call__`` bookkeeping."""

    def __init__(self, prefix: str, L_ctx: int) -> None:
        self.prefix = prefix
        self.L_ctx = L_ctx
        self.flat_hist: list[dict] = []
        self.ego_hist: list[np.ndarray] = []
        self.last_id: int | None = None

    def observe(self, obs: dict) -> None:
        fid = obs["id"]
        if self.last_id is not None and fid < self.last_id:
            self.flat_hist.clear()
            self.ego_hist.clear()
        self.last_id = fid
        self.flat_hist.append(flatten_canonical_frame(obs))
        if len(self.flat_hist) > self.L_ctx:
            self.flat_hist.pop(0)

    def act(self, a: np.ndarray) -> None:
        self.ego_hist.append(a.astype(np.float32))
        if len(self.ego_hist) > self.L_ctx:
            self.ego_hist.pop(0)

    def newest(self) -> _ReferenceSlot:
        """The single-frame view an incremental decoder got."""
        view = _ReferenceSlot(self.prefix, 1)
        view.flat_hist = self.flat_hist[-1:]
        view.ego_hist = self.ego_hist[-1:]
        return view


# --- synthetic frame stream ---------------------------------------------------


def _post(t: int, side: int, *, v6: bool) -> dict:
    phase = 0.37 * t + 1.1 * side
    post = {
        "position": {"x": 110.0 * np.cos(phase), "y": 40.0 * np.sin(0.21 * t) - 12.0 * side},
        "direction": -1.0 if (t + side) % 3 == 0 else 1.0,
        "percent": float((3 * t + 7 * side) % 180),
        "shield": 60.0 - (t % 41),
        "stock": 4 - (t // 137),
        "action": 14 + (t % 23),
        "jumps_used": t % 3,
        "airborne": (t + side) % 2,
        "hurtbox_state": t % 3,
        "hitlag_left": float(t % 5),
    }
    if v6:
        post |= {
            "state_age": float(t % 29),
            "misc_as": float((t * 3) % 17),
            "l_cancel": t % 3,
            "ground": 65535 if t % 2 else t % 54,
            "character_live": 1 + (t % 2) * 21,
            "velocities": {
                "self_x_air": 0.5 * np.sin(0.3 * t),
                "self_y": 0.25 * np.cos(0.11 * t),
                "knockback_x": float(t % 7) - 3.0,
                "knockback_y": float(t % 11) - 5.0,
                "self_x_ground": 0.75 * np.sin(0.07 * t),
            },
            "state_flags": [t % 256, 0, 1, 2, 3],
        }
    return post


def _obs(t: int, frame_id: int, *, v6: bool, follower: bool) -> dict:
    """One canonical closed-loop frame with the matchup metadata ``drive_vec`` injects.

    Port 1 can carry a Nana follower while port 2 never does, so within one batch a
    given nana column is masked on one slot and real on the other — which is what
    decides whether ``preprocess`` emits that column's ``_mask`` sidecar.
    """
    ports = {
        EGO_PORT: {
            "leader": {"post": _post(t, 0, v6=v6)},
            "follower": {"post": _post(t + 3, 0, v6=v6)} if follower else None,
        },
        OPP_PORT: {"leader": {"post": _post(t, 1, v6=v6)}, "follower": None},
    }
    return {
        "id": frame_id,
        "ports": ports,
        "items": [],
        "stage": STAGE,
        "_matchup": {"stage": STAGE, "character": {EGO_PORT: 14, OPP_PORT: 22}},
    }


def _frame_ids(n_first: int, n_second: int) -> list[int]:
    """A rising run, then a drop to a new pre-game countdown: the instant-restart seam."""
    return list(range(400, 400 + n_first)) + list(range(-123, -123 + n_second))


def _stats(v6: bool, degenerate: bool) -> dict[str, FeatureStats]:
    """Deliberately asymmetric stats, so standardize and min-max are both non-trivial."""
    keys = ["position_x", "position_y", "percent", "shield", "direction", "hitlag_left"]
    if v6:
        keys += ["state_age", "misc_as"] + [
            f"velocities_{c}" for c in ("self_x_air", "self_y", "knockback_x", "knockback_y", "self_x_ground")
        ]
    rng = np.random.default_rng(11)
    out: dict[str, FeatureStats] = {}
    for key in keys + [f"nana_{k}" for k in keys]:
        mean, std = float(rng.normal(0, 30)), float(abs(rng.normal(0, 20)) + 0.5)
        low = float(rng.normal(-50, 10))
        out[key] = FeatureStats(mean=mean, std=std, min=low, max=low + float(abs(rng.normal(0, 60)) + 1.0))
    if degenerate:
        # Zero spread on both transforms: preprocess collapses each to a zero column.
        out["percent"] = FeatureStats(mean=3.0, std=0.0, min=-1.0, max=1.0)
        out["shield"] = FeatureStats(mean=0.0, std=1.0, min=7.0, max=7.0)
    return out


# --- the parity sweep ---------------------------------------------------------


def _drive(
    *,
    ports: tuple[int, ...],
    frame_ids: list[int],
    s: int,
    d: int,
    v6: bool,
    follower: bool,
    incremental: bool = False,
    degenerate: bool = False,
) -> tuple[list[tuple[list[int], dict[str, torch.Tensor]]], list[tuple[list[int], dict[str, torch.Tensor]]]]:
    """Run the ring policy over the stream, then replay the reference builder on the
    same frames and the same executed actions. Returns the two capture lists."""
    extra = V6_PLAYER_COLUMNS if v6 else None
    stats = _stats(v6, degenerate)
    rng = np.random.default_rng(5)
    slots = [Slot(0, p) for p in ports]
    frames = [_obs(t, fid, v6=v6, follower=follower) for t, fid in enumerate(frame_ids)]
    replans: list[tuple[int, list[int], dict[str, torch.Tensor]]] = []
    executed: list[tuple[int, int, np.ndarray]] = []
    at = 0

    def predict_chunk(ctx, committed):
        assert ctx.slot_ids is not None
        replans.append((at, [int(v) for v in ctx.slot_ids], {k: v.clone() for k, v in ctx.features.items()}))
        # Action chunks whose button channels straddle the 0.5 threshold.
        return rng.uniform(-1.0, 1.0, size=(ctx.batch, L_CHUNK, A_DIM)).astype(np.float32)

    policy = RecedingHorizon(
        predict_chunk=predict_chunk,
        stats=stats,
        L_ctx=L_CTX,
        L_chunk=L_CHUNK,
        s=s,
        d=d,
        device="cpu",
        predict_incremental=predict_chunk if incremental else None,
        extra=extra,
    )
    real_push = policy._push_ego

    def spy_push(slot, a):
        executed.append((at, slot.port, np.asarray(a, dtype=np.float32).copy()))
        real_push(slot, a)

    policy._push_ego = spy_push
    for at, obs in enumerate(frames):
        policy(at, {sl: obs for sl in slots})

    reference: list[tuple[list[int], dict[str, torch.Tensor]]] = []
    ref_slots = {sl.match * 8 + sl.port: _ReferenceSlot(f"p{sl.port}", L_CTX) for sl in slots}
    for at, obs in enumerate(frames):
        for ref in ref_slots.values():
            ref.observe(obs)
        for frame, slot_ids, _ in replans:
            if frame != at:
                continue
            batch = [ref_slots[i] for i in slot_ids]
            features = _reference_features(batch, L_CTX, stats, extra)
            if incremental:
                # Keep the newest row's finite differences from the full rolling context.
                features = {name: value[:, -1:] for name, value in features.items()}
            reference.append((slot_ids, features))
        for frame, port, a in executed:
            if frame == at:
                ref_slots[port].act(a)
    return [(ids, feats) for _, ids, feats in replans], reference


def _assert_identical(new: list, reference: list, *, min_replans: int) -> None:
    assert len(new) == len(reference) >= min_replans
    varying = 0
    for step, ((new_ids, new_feats), (ref_ids, ref_feats)) in enumerate(zip(new, reference, strict=True)):
        assert new_ids == ref_ids, f"replan {step} covered different slots"
        assert set(new_feats) == set(ref_feats), (
            f"replan {step} key sets differ: "
            f"+{sorted(set(new_feats) - set(ref_feats))} -{sorted(set(ref_feats) - set(new_feats))}"
        )
        for name, ref in ref_feats.items():
            got = new_feats[name]
            assert got.dtype == ref.dtype, f"{name} is {got.dtype}, reference is {ref.dtype} (replan {step})"
            assert got.shape == ref.shape, f"{name} is {got.shape}, reference is {ref.shape} (replan {step})"
            assert torch.equal(got, ref), f"{name} differs at replan {step}"
            varying += int(bool(torch.any(ref != ref.flatten()[0]).item()))
    # Non-vacuity: the agreement is on live signal, not on a batch of constant columns.
    assert varying > 10 * len(new)


@pytest.mark.parametrize(
    "v6, follower, s, d, ports, degenerate, min_replans",
    [
        (False, True, 1, 0, (EGO_PORT, OPP_PORT), False, 350),
        (True, True, 1, 0, (EGO_PORT, OPP_PORT), False, 350),
        (True, False, 1, 0, (EGO_PORT,), False, 350),
        (True, True, 4, 2, (EGO_PORT,), True, 80),
    ],
)
def test_ring_context_matches_the_window_builder(v6, follower, s, d, ports, degenerate, min_replans) -> None:
    """350 frames across an instant-restart seam: every feature tensor, every replan."""
    new, reference = _drive(
        ports=ports, frame_ids=_frame_ids(190, 160), s=s, d=d, v6=v6, follower=follower, degenerate=degenerate
    )
    _assert_identical(new, reference, min_replans=min_replans)


def test_ring_context_matches_the_window_builder_incrementally() -> None:
    new, reference = _drive(
        ports=(EGO_PORT, OPP_PORT), frame_ids=_frame_ids(190, 160), s=1, d=0, v6=True, follower=True, incremental=True
    )
    _assert_identical(new, reference, min_replans=350)


def test_encode_frame_takes_every_frame_while_a_chunk_executes() -> None:
    """An incremental decoder's cache must not skip the frames between replans, or a chunked
    execution horizon would decode from a state several frames stale. ``encode_frame`` runs every
    frame for every live slot; the replan callback still runs every ``s``."""
    s = 4
    encoded: list[tuple[int, list[int], list[bool]]] = []
    replanned: list[int] = []

    def encode_frame(ctx):
        assert ctx.slot_ids is not None and ctx.reset is not None
        assert next(iter(ctx.features.values())).shape[1] == 1  # one frame, not a window
        encoded.append((at, [int(v) for v in ctx.slot_ids], [bool(v) for v in ctx.reset]))

    def predict_incremental(ctx, committed):
        replanned.append(at)
        assert ctx.reset is not None and not any(bool(v) for v in ctx.reset), "encode_frame consumes the reset"
        return np.zeros((ctx.batch, s, A_DIM), dtype=np.float32)

    slots = [Slot(0, EGO_PORT), Slot(0, OPP_PORT)]
    policy = RecedingHorizon(
        predict_chunk=predict_incremental,
        predict_incremental=predict_incremental,
        encode_frame=encode_frame,
        stats=_stats(False, False),
        L_ctx=L_CTX,
        L_chunk=s,
        s=s,
        d=0,
        device="cpu",
    )
    ids = _frame_ids(40, 30)
    for at, fid in enumerate(ids):
        policy(at, {sl: _obs(at, fid, v6=False, follower=False) for sl in slots})

    assert [frame for frame, _, _ in encoded] == list(range(len(ids)))
    assert all(len(slot_ids) == len(slots) for _, slot_ids, _ in encoded)
    # Replans at the restart (frame 40) and every s frames from each fresh start.
    assert replanned == sorted(set(range(0, 40, s)) | set(range(40, len(ids), s)))
    assert [frame for frame, _, resets in encoded if any(resets)] == [0, 40]


def test_encode_frame_without_an_incremental_decoder_is_refused() -> None:
    with pytest.raises(ValueError, match="predict_incremental"):
        RecedingHorizon(
            predict_chunk=lambda ctx, committed: np.zeros((ctx.batch, 1, A_DIM), dtype=np.float32),
            encode_frame=lambda ctx: None,
            stats=_stats(False, False),
            L_ctx=L_CTX,
            L_chunk=1,
            s=1,
            d=0,
            device="cpu",
        )


def test_context_pad_tracks_the_refilling_context() -> None:
    """``ctx_pad`` counts the not-yet-observed prefix, and an instant restart re-opens it."""
    pads: list[int] = []

    def predict_chunk(ctx, committed):
        pads.append(int(ctx.ctx_pad[0]))
        return np.zeros((ctx.batch, 1, A_DIM), dtype=np.float32)

    policy = RecedingHorizon(
        predict_chunk=predict_chunk, stats=_stats(False, False), L_ctx=L_CTX, L_chunk=1, s=1, d=0, device="cpu"
    )
    slot = Slot(0, EGO_PORT)
    ids = _frame_ids(2 * L_CTX, 2 * L_CTX)
    for at, fid in enumerate(ids):
        policy(at, {slot: _obs(at, fid, v6=False, follower=False)})
    assert pads[: L_CTX + 1] == [L_CTX - 1 - i for i in range(L_CTX)] + [0]
    assert pads[2 * L_CTX - 1] == 0
    assert pads[2 * L_CTX : 3 * L_CTX] == [L_CTX - 1 - i for i in range(L_CTX)]
