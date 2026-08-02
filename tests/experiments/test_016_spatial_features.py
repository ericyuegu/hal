"""The derived spatial block must be the SAME numbers at train and at closed-loop eval.

``hal.training.features.derive_spatial`` runs inside ``preprocess``, which is the
one function ``dataloader.collate_train_batch`` and ``closed_loop.RecedingHorizon``
share. That is only worth anything if the two paths actually agree frame for frame
— including at the two places where they differ structurally: the train window's
left-padded cold start, and the rolling buffer filling from empty (also after an
instant-restart boundary clears it). A finite difference taken across either of
those pad→real seams would read a jump from the origin, so both must land on the
same masked-out delta.

Also pins the libmelee-sourced stage geometry, the fail-loud on an unmapped stage
id, and that ``spatial_features=False`` leaves experiment 016 identical to 013.
"""

import importlib.util
import math
from pathlib import Path

import melee
import numpy as np
import pytest
import torch

from hal.data.feature_stats import FeatureStats
from hal.sim.vec import Slot
from hal.training.canonical import flatten_canonical_frame
from hal.training.closed_loop import RecedingHorizon
from hal.training.dataloader import WindowDataset
from hal.training.dataloader import collate_train_batch
from hal.training.dataloader import relabel_ego
from hal.training.features import A_DIM
from hal.training.features import ACTION_CHANNELS
from hal.training.features import CAT_FEATURES
from hal.training.features import FLOAT_FEATURES
from hal.training.features import SPATIAL_COLUMNS
from hal.training.features import SPATIAL_FEATURES
from hal.training.features import derive_spatial
from hal.training.features import preprocess

_EXP_DIR = Path(__file__).resolve().parent.parent.parent / "experiments"

STAGE = melee.Stage.FINAL_DESTINATION
EGO_PORT, OPP_PORT = 1, 2
L_CTX = 6


def _load_experiment(filename: str):
    spec = importlib.util.spec_from_file_location(filename.split(".")[0], _EXP_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _stats() -> dict[str, FeatureStats]:
    """Identity-ish stats for every float column, incl. the masked nana block."""
    keys = (*FLOAT_FEATURES, *(f"nana_{k}" for k in FLOAT_FEATURES))
    return {k: FeatureStats(mean=0.0, std=1.0, min=-1.0, max=1.0) for k in keys}


# --- one trajectory, two shapes ----------------------------------------------
# Asymmetric per-player kinematics that walk the ego from deep offstage across the
# ledge and back, so ledge/offstage/blastzone/velocity channels all move.


def _kinematics(t: int) -> dict[str, float]:
    return {
        "ego_position_x": 130.0 * math.cos(0.55 * t),
        "ego_position_y": 4.0 * t - 26.0,
        "ego_direction": -1.0 if t % 3 == 0 else 1.0,
        "opp_position_x": 120.0 * math.sin(0.4 * t + 0.3),
        "opp_position_y": 1.5 * t + 2.0,
        "opp_direction": 1.0 if t % 2 == 0 else -1.0,
    }


def _post(t: int, side: str) -> dict:
    k = _kinematics(t)
    return {
        "position": {"x": k[f"{side}_position_x"], "y": k[f"{side}_position_y"]},
        "direction": k[f"{side}_direction"],
        "percent": float(t),
        "shield": 60.0,
        "stock": 4,
        "action": 14,
        "jumps_used": 0,
        "airborne": 1,
        "hurtbox_state": 0,
        "hitlag_left": 0.0,
    }


def _obs(t: int, *, frame_id: int | None = None) -> dict:
    """One canonical closed-loop frame, with the matchup metadata ``drive_vec`` injects."""
    return {
        "id": t if frame_id is None else frame_id,
        "ports": {
            EGO_PORT: {"leader": {"post": _post(t, "ego")}, "follower": None},
            OPP_PORT: {"leader": {"post": _post(t, "opp")}, "follower": None},
        },
        "_matchup": {"stage": STAGE.value, "character": {EGO_PORT: melee.Character.FOX.value, OPP_PORT: 1}},
    }


def _mds_sample(n_frames: int) -> dict[str, np.ndarray]:
    """The same trajectory as an MDS replay row.

    Columns come from the closed-loop observation bridge itself, stacked over time and
    typed the way ``_live_batch_from_rolling`` types them. That is what makes the two
    paths comparable key-for-key on the SAME frames without this test pinning a column
    list that the extraction schema keeps growing."""
    frames = [flatten_canonical_frame(_obs(t)) for t in range(n_frames)]
    sample: dict[str, np.ndarray] = {"frame": np.arange(n_frames, dtype=np.int32)}
    for key, first in frames[0].items():
        dtype = np.int32 if isinstance(first, int) else np.float32
        sample[key] = np.array([frame[key] for frame in frames], dtype=dtype)
    for channel in ACTION_CHANNELS:
        dtype = np.int32 if channel.startswith("button_") else np.float32
        sample[f"p{EGO_PORT}_{channel}"] = np.zeros(n_frames, dtype=dtype)
    return sample


def _train_context(sample: dict[str, np.ndarray], last_frame: int) -> tuple[dict[str, torch.Tensor], int]:
    """The train-path context whose newest real frame is ``last_frame``: the production
    ``WindowDataset`` left-pad + ``collate_train_batch``. Returns ``(features, ctx_pad)``."""
    sampler = WindowDataset([], L_CTX, 1, seed=0)
    start = last_frame + 1 - L_CTX
    pad = max(0, -start)
    window = sampler._padded_window(sample, start, pad)
    window["ctx_pad"] = np.int64(min(pad, L_CTX))
    batch = collate_train_batch([relabel_ego(window, f"p{EGO_PORT}")], stats=_stats(), L_ctx=L_CTX)
    return batch.context.features, int(batch.context.ctx_pad[0])


def _closed_loop_contexts(frame_ids: list[int]) -> list[tuple[dict[str, torch.Tensor], int]]:
    """Drive ``RecedingHorizon`` over the trajectory (one replan per frame) and capture the
    Context it hands the model. ``frame_ids`` supplies the canonical ids, so a drop triggers
    the instant-restart buffer clear."""
    captured: list[tuple[dict[str, torch.Tensor], int]] = []

    def predict_chunk(ctx, committed):
        assert committed is None
        captured.append(({k: v.clone() for k, v in ctx.features.items()}, int(ctx.ctx_pad[0])))
        return np.zeros((ctx.batch, 1, A_DIM), dtype=np.float32)

    policy = RecedingHorizon(
        predict_chunk=predict_chunk, stats=_stats(), L_ctx=L_CTX, L_chunk=1, s=1, d=0, device="cpu"
    )
    slot = Slot(0, EGO_PORT)
    for t, frame_id in enumerate(frame_ids):
        policy(t, {slot: _obs(t, frame_id=frame_id)})
    return captured


# --- parity ------------------------------------------------------------------


def test_train_and_closed_loop_derive_identical_spatial_features() -> None:
    """Every derived column, at every position of every cold-start-through-saturated
    context, must be bit-equal across the two paths — including the leading pad frames
    and the pad→real seam where the finite difference has no predecessor."""
    n_frames = 2 * L_CTX
    sample = _mds_sample(n_frames)
    live = _closed_loop_contexts(list(range(n_frames)))

    for t in range(n_frames - 1):  # -1: the train window also needs one chunk frame
        train_features, train_pad = _train_context(sample, t)
        loop_features, loop_pad = live[t]
        assert train_pad == loop_pad == max(0, L_CTX - (t + 1))
        assert set(train_features) == set(loop_features)
        for name in SPATIAL_COLUMNS:
            torch.testing.assert_close(
                train_features[name], loop_features[name], rtol=0, atol=0, msg=f"{name} differs at frame {t}"
            )
    # Non-vacuity: the agreement above is on live signal, not on two all-zero blocks.
    saturated, pad = live[n_frames - 2]
    assert pad == 0
    for name in SPATIAL_FEATURES:
        assert torch.any(saturated[name] != 0.0), f"{name} is identically zero over the whole trajectory"


def test_pad_boundary_delta_is_masked_on_both_paths() -> None:
    """The first real frame has no predecessor in context, so its velocity must be masked
    (not a jump from the zero-filled pad). Every later real frame must be unmasked."""
    n_frames = 2 * L_CTX
    sample = _mds_sample(n_frames)
    live = _closed_loop_contexts(list(range(n_frames)))

    for t in range(n_frames - 1):
        pad = max(0, L_CTX - (t + 1))
        expected_frame_mask = torch.tensor([[1.0] * pad + [0.0] * (L_CTX - pad)])
        expected_delta_mask = torch.tensor([[1.0] * min(pad + 1, L_CTX) + [0.0] * max(0, L_CTX - pad - 1)])
        for features, _ in (_train_context(sample, t), live[t]):
            torch.testing.assert_close(features["spatial_mask"], expected_frame_mask, rtol=0, atol=0)
            torch.testing.assert_close(features["spatial_dpos_mask"], expected_delta_mask, rtol=0, atol=0)
            # Masked positions carry no signal at all, not a stale or wrapped value.
            for name in SPATIAL_FEATURES:
                invalid = expected_delta_mask if name.endswith(("_dpos_x", "_dpos_y")) else expected_frame_mask
                assert torch.all(features[name][invalid > 0] == 0.0), f"{name} nonzero where masked (frame {t})"


def test_instant_restart_remasks_the_velocity_seam() -> None:
    """An instant restart clears the rolling buffer mid-boot. The re-warming context must
    re-pad and re-mask exactly like a cold start, so no delta spans two matches."""
    per_match = 2 * L_CTX
    ids = list(range(400, 400 + per_match)) + list(range(-123, -123 + per_match))
    live = _closed_loop_contexts(ids)

    assert live[per_match - 1][1] == 0, "buffer never saturated before the restart"
    for offset in range(per_match):
        features, pad = live[per_match + offset]
        assert pad == max(0, L_CTX - (offset + 1))
        assert features["spatial_dpos_mask"][0, pad] == 1.0, "delta crossed the restart boundary"
        assert torch.all(features["ego_dpos_x"][0, : pad + 1] == 0.0)


def test_spatial_mask_reproduces_ctx_pad() -> None:
    """The stage-id presence rule the derivation uses IS the window's ``ctx_pad``; if it ever
    drifted, the masks would silently stop lining up with the attention mask."""
    sample = _mds_sample(2 * L_CTX)
    for t in range(2 * L_CTX - 1):
        features, pad = _train_context(sample, t)
        positions = torch.arange(L_CTX)
        torch.testing.assert_close(features["spatial_mask"][0], (positions < pad).float(), rtol=0, atol=0)


# --- geometry + fail-loud ----------------------------------------------------


def _single_frame(stage_id: int, **kinematics: float) -> dict[str, np.ndarray]:
    batch = {"stage": np.array([stage_id], dtype=np.int32)}
    for name in (
        "ego_position_x",
        "ego_position_y",
        "ego_direction",
        "opp_position_x",
        "opp_position_y",
        "opp_direction",
    ):
        batch[name] = np.array([kinematics.get(name, 0.0)], dtype=np.float32)
    return batch


@pytest.mark.parametrize("stage", melee.stages.EDGE_POSITION)
def test_stage_geometry_matches_libmelee_tables(stage: melee.Stage) -> None:
    """Ledge and blastzone channels are libmelee's own tables, not a second copy."""
    x, y = 12.5, -7.5
    out = derive_spatial(_single_frame(stage.value, ego_position_x=x, ego_position_y=y))
    left, right, top, bottom = melee.stages.BLASTZONES[stage]
    edge = melee.stages.EDGE_POSITION[stage]
    assert out["ego_ledge_dx"][0] == pytest.approx((abs(x) - edge) / 100.0, abs=1e-6)
    assert out["ego_ledge_dy"][0] == pytest.approx(y / 100.0, abs=1e-6)
    assert out["ego_offstage"][0] == float(abs(x) > edge)
    assert out["ego_blast_left"][0] == pytest.approx((x - left) / 100.0, abs=1e-6)
    assert out["ego_blast_right"][0] == pytest.approx((right - x) / 100.0, abs=1e-6)
    assert out["ego_blast_top"][0] == pytest.approx((top - y) / 100.0, abs=1e-6)
    assert out["ego_blast_bottom"][0] == pytest.approx((y - bottom) / 100.0, abs=1e-6)


def test_facing_relative_dx_is_signed_by_each_players_facing() -> None:
    """``rel_dx_ego_facing`` > 0 iff the opponent is in FRONT of the ego, and its mirror
    answers the same question from the opponent's side."""
    right_of_ego = dict(ego_position_x=0.0, opp_position_x=40.0)
    facing_in = derive_spatial(_single_frame(STAGE.value, **right_of_ego, ego_direction=1.0, opp_direction=-1.0))
    facing_away = derive_spatial(_single_frame(STAGE.value, **right_of_ego, ego_direction=-1.0, opp_direction=1.0))
    assert facing_in["rel_dx_ego_facing"][0] > 0 and facing_in["rel_dx_opp_facing"][0] > 0
    assert facing_away["rel_dx_ego_facing"][0] < 0 and facing_away["rel_dx_opp_facing"][0] < 0
    assert facing_in["rel_dist"][0] == pytest.approx(0.4, abs=1e-6)


def test_unmapped_stage_id_fails_loud() -> None:
    """No fallback geometry: a stage we have no ledge/blastzone data for is a hard error."""
    with pytest.raises(ValueError, match="no libmelee ledge/blastzone geometry"):
        derive_spatial(_single_frame(melee.Stage.NO_STAGE.value + 1))


def test_no_stage_is_the_pad_sentinel_not_an_error() -> None:
    """``Stage.NO_STAGE`` is what a zero-filled cold-start row reads, so it masks rather
    than raising — that is the whole mechanism the two paths share."""
    out = derive_spatial(_single_frame(melee.Stage.NO_STAGE.value))
    assert out["spatial_mask"][0] == 1.0 and out["spatial_dpos_mask"][0] == 1.0


def test_derived_column_arriving_as_input_is_rejected() -> None:
    """If the MDS ever materializes the block, deriving it again would double-write it."""
    batch = _mds_sample(4)
    batch["rel_dx"] = np.zeros(4, dtype=np.float32)
    with pytest.raises(ValueError, match="derived on the fly"):
        preprocess(relabel_ego(batch, f"p{EGO_PORT}"), _stats())


def test_batch_without_stage_gets_no_spatial_block() -> None:
    """Pre-conditioning batches carry no stage, hence no derivable geometry — inert, not fatal."""
    batch = {k: v for k, v in _mds_sample(4).items() if k != "stage"}
    out = preprocess(relabel_ego(batch, f"p{EGO_PORT}"), _stats())
    assert not set(out) & set(SPATIAL_COLUMNS)


# --- 013 parity when the flag is off -----------------------------------------

exp013 = _load_experiment("013_multi_token.py")
exp016 = _load_experiment("016_spatial_features.py")


def _tiny_cfg(exp, **kwargs):
    defaults = dict(d_model=64, n_layers=1, n_heads=2, L_ctx=4, head_offsets=(1, 2), batch_size=2, max_steps=8)
    return exp.TrainConfig(**{**defaults, **kwargs})


def _zeros_features(exp, batch_size: int, length: int) -> dict[str, torch.Tensor]:
    features: dict[str, torch.Tensor] = {}
    for prefix in exp._PLAYER_PREFIXES:
        for feat in FLOAT_FEATURES:
            features[f"{prefix}_{feat}"] = torch.zeros(batch_size, length)
        for name in CAT_FEATURES:
            features[f"{prefix}_{name}"] = torch.zeros(batch_size, length, dtype=torch.long)
    for channel in ACTION_CHANNELS:
        features[f"ego_{channel}"] = torch.zeros(batch_size, length)
    for key in ("ego_character", "opp_character", "stage"):
        features[key] = torch.zeros(batch_size, length, dtype=torch.long)
    return features


def test_spatial_features_off_is_identical_to_013() -> None:
    """The baseline arm must be the SAME model, not merely a similar one: same input width,
    same parameters from the same seed, same hidden state — otherwise the A/B has two axes."""
    cfg013, cfg016 = _tiny_cfg(exp013), _tiny_cfg(exp016, spatial_features=False)
    torch.manual_seed(7)
    model013 = exp013.GPT(cfg013).eval()
    torch.manual_seed(7)
    model016 = exp016.GPT(cfg016).eval()

    assert model016.ctx_proj.in_features == model013.ctx_proj.in_features
    assert {k: tuple(v.shape) for k, v in model016.state_dict().items()} == {
        k: tuple(v.shape) for k, v in model013.state_dict().items()
    }
    features = _zeros_features(exp016, 2, cfg016.L_ctx)
    ctx_pad = torch.tensor([0, 1], dtype=torch.long)
    with torch.no_grad():
        torch.testing.assert_close(model013(features, ctx_pad), model016(features, ctx_pad), rtol=0, atol=0)


def test_spatial_features_on_widens_the_token_and_needs_the_block() -> None:
    """The enabled arm consumes exactly the derived columns, and says so when they are absent."""
    cfg = _tiny_cfg(exp016, spatial_features=True)
    model = exp016.GPT(cfg).eval()
    assert model.ctx_proj.in_features == exp013.GPT(_tiny_cfg(exp013)).ctx_proj.in_features + len(SPATIAL_COLUMNS)
    features = _zeros_features(exp016, 2, cfg.L_ctx)
    with pytest.raises(ValueError, match="missing"):
        model(features, torch.tensor([0, 0], dtype=torch.long))
    for name in SPATIAL_COLUMNS:
        features[name] = torch.zeros(2, cfg.L_ctx)
    with torch.no_grad():
        assert model(features, torch.tensor([0, 0], dtype=torch.long)).shape == (2, cfg.L_ctx, cfg.d_model)


def test_incremental_decode_is_rejected_with_spatial_features() -> None:
    """A one-frame token cannot carry a finite difference, so the combination would train and
    deploy on different feature distributions."""
    with pytest.raises(ValueError, match="finite-difference velocity"):
        exp016.validate_config(
            _tiny_cfg(exp016, spatial_features=True, eval_incremental_kv=True), has_button_combo_counts=False
        )
