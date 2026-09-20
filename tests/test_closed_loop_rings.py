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
from hal.sim.rollout import ObservationRow
from hal.sim.vec import Slot
from hal.training.canonical import flatten_canonical_frame
from hal.training.closed_loop import RecedingHorizon
from hal.training.dataloader import relabel_ego
from hal.training.features import ACTION_CHANNELS
from hal.training.features import BASE_ACTION_PROJECTION
from hal.training.features import BASE_ITEMS_PROJECTION
from hal.training.features import ITEM_COLUMNS
from hal.training.features import ITEM_INPUT_COLUMNS
from hal.training.features import NEUTRAL_ACTION
from hal.training.features import V6_PLAYER_COLUMNS
from hal.training.features import ExtraColumns
from hal.training.features import FeatureProjection
from hal.training.features import preprocess
from hal.wire import ACTION_DIM
from hal.wire import ITEM_SLOTS
from hal.wire import MASK_INT32
from hal.wire import item_column

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
    projection: FeatureProjection | None = None,
) -> dict[str, torch.Tensor]:
    """The replaced ``_build_stacked_batch`` + ``preprocess`` + dtype packing."""
    per_slot = [_reference_window(s.flat_hist, s.ego_hist, s.prefix, L_ctx) for s in batch]
    stacked = {k: np.concatenate([d[k] for d in per_slot], axis=0) for k in per_slot[0]}
    preprocessed = preprocess(stacked, stats, extra=extra, projection=projection)
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
        """Return the newest stored frame."""
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
    degenerate: bool = False,
    projection: FeatureProjection | None = None,
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
        return rng.uniform(-1.0, 1.0, size=(ctx.batch, L_CHUNK, ACTION_DIM)).astype(np.float32)

    policy = RecedingHorizon(
        predict_chunk=predict_chunk,
        stats=stats,
        L_ctx=L_CTX,
        L_chunk=L_CHUNK,
        s=s,
        d=d,
        device="cpu",
        extra=extra,
        projection=projection,
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
            features = _reference_features(batch, L_CTX, stats, extra, projection)
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


def test_projected_ring_context_matches_projected_window_builder() -> None:
    new, reference = _drive(
        ports=(EGO_PORT, OPP_PORT),
        frame_ids=_frame_ids(90, 80),
        s=1,
        d=0,
        v6=True,
        follower=True,
        projection=BASE_ACTION_PROJECTION,
    )
    _assert_identical(new, reference, min_replans=170)


def test_context_pad_tracks_the_refilling_context() -> None:
    """``ctx_pad`` counts the not-yet-observed prefix, and an instant restart re-opens it."""
    pads: list[int] = []

    def predict_chunk(ctx, committed):
        pads.append(int(ctx.ctx_pad[0]))
        return np.zeros((ctx.batch, 1, ACTION_DIM), dtype=np.float32)

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


# --- the global item block ----------------------------------------------------
#
# The two observation paths deliver the same projectile frame in DIFFERENT dtypes:
# the MDS stores the ids as int32 with ``MASK_INT32`` in an empty slot, while the
# closed loop hands ``preprocess`` all seven fields as float NaN-sentinel columns
# (``wire.canonical_item_columns``). Both must produce the same item tensors.

ITEM_L = 8
_ITEM_FLOATS: tuple[str, ...] = tuple(ITEM_COLUMNS.floats)
_ITEM_CATS: tuple[str, ...] = tuple(ITEM_COLUMNS.cats)


def _item_live(slot: int) -> np.ndarray:
    """Which of the ``ITEM_L`` frames carry an item in ``slot``. Every slot is empty on
    at least one frame, so every routed float column emits its ``_mask`` sidecar."""
    return np.array([(t + slot) % 3 != 0 for t in range(ITEM_L)])


def _item_value(slot: int, suffix: str) -> np.ndarray:
    """One item column's logical value over the ``ITEM_L`` frames, before masking."""
    t = np.arange(ITEM_L, dtype=np.float64)
    if suffix == "type":  # a Fox laser (6) up to a Peach turnip (210)
        return 6.0 + 51.0 * slot + (t % 3)
    if suffix == "state":
        return (t + slot) % 5
    if suffix == "owner":  # a libmelee port, 1..4
        return np.full(ITEM_L, 1.0 + slot % 2)
    offset = 0.0 if suffix.endswith("_x") else 1.7
    return (90.0 if suffix.startswith("pos") else 4.0) * np.sin(0.4 * t + 1.1 * slot + offset)


def _item_batch(*, online: bool) -> dict[str, np.ndarray]:
    """One ``[L]`` stream of every item column, in the offline MDS dtypes or in the
    online all-float sentinel form."""
    batch: dict[str, np.ndarray] = {}
    for slot in range(ITEM_SLOTS):
        live = _item_live(slot)
        for suffix in (*_ITEM_CATS, *_ITEM_FLOATS, "owner"):
            values = _item_value(slot, suffix)
            if online or suffix in _ITEM_FLOATS:
                column = np.where(live, values, np.nan).astype(np.float32)
            else:
                column = np.where(live, values.astype(np.int32), MASK_INT32).astype(np.int32)
            batch[item_column(slot, suffix)] = column
    return batch


def _item_stats() -> dict[str, FeatureStats]:
    """The four consolidated ``item_*`` keys: one shared scale over the four slots."""
    rng = np.random.default_rng(23)
    out: dict[str, FeatureStats] = {}
    for suffix in _ITEM_FLOATS:
        low = float(rng.normal(-50, 10))
        out[f"item_{suffix}"] = FeatureStats(
            mean=float(rng.normal(0, 20)),
            std=float(abs(rng.normal(0, 10)) + 0.5),
            min=low,
            max=low + float(abs(rng.normal(0, 60)) + 1.0),
        )
    return out


def test_preprocess_routes_the_item_block() -> None:
    """``ITEM_COLUMNS`` over offline dtypes: the ids come back as int64 with 0 in an
    empty slot, the floats standardize against the one consolidated scale and flag the
    empty frames, and ``owner`` stays dropped."""
    stats = _item_stats()
    out = preprocess(_item_batch(online=False), stats, extra=ITEM_COLUMNS)

    assert [name for name in out if name.endswith("_owner")] == []
    for slot in range(ITEM_SLOTS):
        live = _item_live(slot)
        for suffix in _ITEM_CATS:
            got = out[item_column(slot, suffix)]
            assert got.dtype == torch.int64
            expected = np.where(live, _item_value(slot, suffix).astype(np.int64), 0)
            assert torch.equal(got, torch.from_numpy(expected))
        for suffix in _ITEM_FLOATS:
            name = item_column(slot, suffix)
            s = stats[f"item_{suffix}"]
            got = out[name]
            assert got.dtype == torch.float32
            expected = np.where(live, (_item_value(slot, suffix) - s.mean) / s.std, 0.0)
            assert got.numpy() == pytest.approx(expected, rel=1e-5, abs=1e-6)
            assert torch.equal(out[f"{name}_mask"], torch.from_numpy((~live).astype(np.float32)))


def test_item_projection_extends_the_base_action_projection() -> None:
    """``BASE_ITEMS_PROJECTION`` adds exactly the routed item columns to the base
    projection: four slots, six suffixes, no ``owner``."""
    added = BASE_ITEMS_PROJECTION.columns - BASE_ACTION_PROJECTION.columns
    assert added == ITEM_INPUT_COLUMNS
    assert len(added) == ITEM_SLOTS * (len(_ITEM_CATS) + len(_ITEM_FLOATS)) == 24
    assert [name for name in added if name.endswith("_owner")] == []
    assert BASE_ITEMS_PROJECTION.derive_spatial is False


def test_preprocess_item_routing_is_dtype_agnostic() -> None:
    """The same logical item frame, offline int32/NaN and online all-float, produces
    the same tensors. Pins the dtype asymmetry the two observation paths carry."""
    stats = _item_stats()
    offline = preprocess(_item_batch(online=False), stats, extra=ITEM_COLUMNS)
    online = preprocess(_item_batch(online=True), stats, extra=ITEM_COLUMNS)

    assert len(offline) == ITEM_SLOTS * (len(_ITEM_CATS) + 2 * len(_ITEM_FLOATS))
    assert set(online) == set(offline)
    for name, expected in offline.items():
        got = online[name]
        assert got.dtype == expected.dtype, f"{name} is {got.dtype}, offline is {expected.dtype}"
        assert torch.equal(got, expected), f"{name} differs between the two dtype forms"


def test_shared_row_planning_uses_policy_schedule_and_resets_context() -> None:
    pads: list[int] = []

    def predict_chunk(ctx, committed):
        pads.append(int(ctx.ctx_pad[0]))
        return np.zeros((ctx.batch, L_CHUNK, ACTION_DIM), dtype=np.float32)

    policy = RecedingHorizon(
        predict_chunk=predict_chunk,
        stats=_stats(False, False),
        L_ctx=L_CTX,
        L_chunk=L_CHUNK,
        s=4,
        d=0,
        device="cpu",
    )
    slot = Slot(0, EGO_PORT)

    def row(t: int, frame_id: int, *, reset: bool = False) -> ObservationRow:
        return ObservationRow(
            frame_id=frame_id,
            flat=flatten_canonical_frame(_obs(t, frame_id, v6=False, follower=False)),
            action=NEUTRAL_ACTION,
            reset=reset,
        )

    first = policy.plan_rows({slot: [row(0, 400, reset=True)]})[slot]
    second = policy.plan_rows({slot: [row(t, 400 + t) for t in range(1, 5)]})[slot]
    reset_plan = policy.plan_rows({slot: [row(5, -123, reset=True)]})[slot]

    assert policy.runtime_spec.execution_stride == 4
    assert first.shape == second.shape == reset_plan.shape == (L_CHUNK, ACTION_DIM)
    assert pads == [L_CTX - 1, L_CTX - 5, L_CTX - 1]


@pytest.mark.parametrize(("stride", "delay"), [(2, 0), (2, 1), (2, 2), (2, 3), (3, 1), (1, 3)])
def test_neutral_commitment_frame_and_chunk_paths_match(stride: int, delay: int) -> None:
    from hal.eval.policy_sampling import SlotGroupRng
    from hal.sim.inputs import controller_to_action_vec

    class Predictor:
        def __init__(self) -> None:
            self.rng = SlotGroupRng(71, ("x", "y"))
            self.calls = []

        def __call__(self, ctx, committed):
            assert committed.shape == (ctx.batch, delay, ACTION_DIM)
            self.rng.begin(ctx)
            plans = np.zeros((ctx.batch, L_CHUNK, ACTION_DIM), dtype=np.float32)
            draws = []
            for depth in range(delay, L_CHUNK):
                values = torch.stack([self.rng.uniforms(name) for name in ("x", "y")], dim=-1)
                draws.append(values.clone())
                plans[:, depth, :2] = values.numpy() * 1.8 - 0.9
            # The scheduler must retain the supplied commitment, even if the
            # predictor changes it during quantization or uses a shared buffer.
            plans[:, :delay] = -0.37
            self.calls.append((ctx, committed.copy(), torch.stack(draws), self.rng.state()))
            return plans

    frame_predictor, row_predictor = Predictor(), Predictor()

    def policy(predictor):
        return RecedingHorizon(
            predictor,
            _stats(False, False),
            5,
            L_CHUNK,
            stride,
            delay,
            device="cpu",
            bootstrap_committed="neutral",
        )

    frame_policy, row_policy = policy(frame_predictor), policy(row_predictor)
    slots = [Slot(0, 1), Slot(0, 2), Slot(1, 1)]
    buffers = {slot: [] for slot in slots}
    plans = {}
    positions = {}
    previous = {slot: NEUTRAL_ACTION.copy() for slot in slots}
    ids = {slot: 100 for slot in slots}
    for tick in range(10 * stride + 7):
        live = slots[:2] if tick < 2 * stride else slots
        live = live if tick % 2 else list(reversed(live))
        observations, requests = {}, {}
        for slot in live:
            reset = (slot == slots[0] and tick == 3 * stride + 1) or (slot == slots[1] and tick == 4 * stride)
            ids[slot] = -123 if reset else ids[slot] + 1
            frame = _obs(tick, ids[slot], v6=False, follower=False)
            observations[slot] = frame
            buffers[slot].append(
                ObservationRow(
                    ids[slot],
                    flatten_canonical_frame(frame),
                    NEUTRAL_ACTION.copy() if reset else previous[slot].copy(),
                    reset=reset,
                )
            )
            if reset or slot not in plans or positions[slot] >= stride:
                requests[slot] = buffers[slot]
                buffers[slot] = []
        for slot, plan in row_policy.plan_rows(requests).items():
            plans[slot] = plan
            positions[slot] = 0
        actions = frame_policy(tick, observations)
        for slot in live:
            actual = controller_to_action_vec(actions[slot])
            np.testing.assert_array_equal(actual, plans[slot][positions[slot]])
            previous[slot] = actual
            positions[slot] += 1

    assert len(frame_predictor.calls) == len(row_predictor.calls)
    for frame_call, row_call in zip(frame_predictor.calls, row_predictor.calls, strict=True):
        frame_ctx, frame_committed, frame_draws, frame_rng = frame_call
        row_ctx, row_committed, row_draws, row_rng = row_call
        assert frame_ctx.features.keys() == row_ctx.features.keys()
        for name in frame_ctx.features:
            torch.testing.assert_close(frame_ctx.features[name], row_ctx.features[name], rtol=0, atol=0)
        for name in ("ctx_pad", "slot_ids", "reset"):
            assert torch.equal(getattr(frame_ctx, name), getattr(row_ctx, name))
        np.testing.assert_array_equal(frame_committed, row_committed)
        assert torch.equal(frame_draws, row_draws)
        assert frame_rng == row_rng
    np.testing.assert_array_equal(frame_predictor.calls[0][1], np.zeros((2, delay, ACTION_DIM)))
    assert any(call[0].reset.tolist().count(True) == 1 and call[0].batch > 1 for call in frame_predictor.calls)


@pytest.mark.parametrize("delay", [0, 2])
@pytest.mark.parametrize("entry", ["frames", "rows"])
def test_historical_bootstrap_keeps_its_grouping_and_predictions(delay: int, entry: str) -> None:
    calls = []
    output = np.full((2, L_CHUNK, ACTION_DIM), 0.25, dtype=np.float32)

    def predict(ctx, committed):
        calls.append((ctx.slot_ids.tolist(), committed))
        return output[: ctx.batch]

    policy = RecedingHorizon(predict, _stats(False, False), 5, L_CHUNK, 1, delay, device="cpu")
    first, second = Slot(0, 1), Slot(0, 2)

    def step(tick, slots):
        frame = _obs(tick, 100 + tick, v6=False, follower=False)
        if entry == "frames":
            policy(tick, {slot: frame for slot in slots})
        else:
            policy.plan_rows(
                {slot: [ObservationRow(100 + tick, flatten_canonical_frame(frame), NEUTRAL_ACTION)] for slot in slots}
            )

    step(0, [first])
    step(1, [first, second])
    assert calls[0] == ([1], None)
    if entry == "frames" and delay == 0:
        assert calls[1] == ([1, 2], None)
    else:
        assert calls[1] == ([2], None)
        assert calls[2][0] == [1]
        if delay:
            np.testing.assert_array_equal(calls[2][1], output[:1, 1 : 1 + delay])
        else:
            assert calls[2][1] is None
    np.testing.assert_array_equal(policy._slots[first].pending, output[0])
    output.fill(0.75)
    assert np.all(policy._slots[first].pending == 0.25)


def test_fault_snapshot_retains_values_before_reset_flags_change() -> None:
    policy = RecedingHorizon(
        lambda ctx, committed: np.zeros((ctx.batch, L_CHUNK, ACTION_DIM), dtype=np.float32),
        _stats(False, False),
        5,
        L_CHUNK,
        3,
        2,
        device="cpu",
        bootstrap_committed="neutral",
        fault_metadata=lambda: {"experiment": "test"},
    )
    assert policy.fault_snapshot() == ({"experiment": "test"}, {})
    slots = [Slot(0, 1), Slot(0, 2)]
    policy(0, {slot: _obs(0, 100, v6=False, follower=False) for slot in slots})
    metadata, arrays = policy.fault_snapshot()
    assert all(not policy._slots[slot].reset_pending for slot in slots)
    assert metadata["reset"] == [True, True]
    assert metadata["ctx_pad"] == [4, 4]
    assert metadata["slots"] == [{"match": 0, "port": 1}, {"match": 0, "port": 2}]
    assert set(metadata) == {
        "slots",
        "ctx_pad",
        "reset",
        "value_names",
        "mask_names",
        "cat_names",
        "emitted_masks",
        "experiment",
    }
    expected_arrays = {name: value.copy() for name, value in arrays.items()}
    policy(1, {slot: _obs(1, 101, v6=False, follower=False) for slot in slots})
    later_metadata, later_arrays = policy.fault_snapshot()
    assert later_metadata == metadata
    for name, expected in expected_arrays.items():
        np.testing.assert_array_equal(later_arrays[name], expected)
