"""021 is the 016 recipe on a new observation bundle: the schema-v6 post block plus a
lean spatial subset. These tests pin the three things that makes that a clean baseline.

1. Nothing else moved. ``hal.training.features`` routes the v6 columns only for a
   consumer that asks; with no routing its output is what it always was, so every
   earlier experiment and checkpoint keeps its exact input width.
2. The token is exactly the declared bundle. 486 dims with a documented breakdown,
   the lean spatial columns in their documented order, and the per-replay
   character-SELECT pick deliberately unread.
3. Train and closed-loop eval see the SAME v6 values, frame for frame. Both paths
   run one ``preprocess`` with one routing object; a column that libmelee never
   fills online would silently disable a feature at eval only.

Plus the frozen 016-base recipe at schema v6, the config rejections, the deploy
contract (closed-loop play samples, never argmax) and an overfit smoke.
"""

import dataclasses
import importlib.util
import inspect
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
from hal.training.features import SPATIAL_COLUMNS_LEAN
from hal.training.features import V6_PLAYER_COLUMNS
from hal.training.features import Context
from hal.training.features import ExtraColumns
from hal.training.features import TrainBatch
from hal.training.features import _classify
from hal.training.features import preprocess

_EXP_DIR = Path(__file__).resolve().parent.parent.parent / "experiments"

STAGE = melee.Stage.FINAL_DESTINATION
EGO_PORT, OPP_PORT = 1, 2
L_CTX = 6
# One hitstun action state and one that is definitely not (WAIT).
HITSTUN_ACTION = melee.Action.DAMAGE_HIGH_1.value
IDLE_ACTION = melee.Action.STANDING.value


def _load_experiment(filename: str):
    spec = importlib.util.spec_from_file_location(filename.split(".")[0], _EXP_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


exp021 = _load_experiment("021_v6_features.py")

_V6_FLOATS = tuple(V6_PLAYER_COLUMNS.floats)
_V6_CATS = tuple(V6_PLAYER_COLUMNS.cats)
_V6_SUFFIXES = _V6_FLOATS + _V6_CATS


def _stats() -> dict[str, FeatureStats]:
    """Every float column's stats. The v6 entries are deliberately NOT unit-Gaussian so a
    standardize-vs-min-max mix-up shows up as a wrong number, not just a wrong shape."""
    stats = {
        key: FeatureStats(mean=0.0, std=1.0, min=-1.0, max=1.0)
        for key in (*FLOAT_FEATURES, *(f"nana_{k}" for k in FLOAT_FEATURES))
    }
    stats.update(
        {
            key: FeatureStats(mean=2.0, std=4.0, min=-1.0, max=1.0)
            for key in (*_V6_FLOATS, *(f"nana_{k}" for k in _V6_FLOATS))
        }
    )
    return stats


def _is_v6(column: str) -> bool:
    return any(column.endswith(f"_{suffix}") for suffix in _V6_SUFFIXES)


# --- one trajectory, two shapes ----------------------------------------------
# The closed-loop bridge builds the column set, so these tests never pin a column list
# that the extraction schema keeps growing.


def _post(t: int, side: str) -> dict:
    """One canonical post block, shaped like libmelee's ``to_canonical_dict``."""
    sign = 1.0 if side == "ego" else -1.0
    return {
        "position": {"x": sign * (40.0 + 9.0 * t), "y": 3.0 * t - 12.0},
        "direction": sign,
        "percent": float(t),
        "shield": 60.0,
        "stock": 4,
        "action": HITSTUN_ACTION if t % 2 == 0 else IDLE_ACTION,
        "jumps_used": 0,
        "airborne": 1,
        "hurtbox_state": 0,
        "hitlag_left": 0.0,
        "character": melee.Character.FOX.value if side == "ego" else melee.Character.MARTH.value,
        "state_age": float(t),
        "misc_as": 7.0 + t,
        "l_cancel": t % 3,
        "ground": 65535 if t % 2 else 3,
        "velocities": {
            "self_x_air": 0.5 * sign,
            "self_x_ground": 0.25 * t,
            "self_y": -1.5 + 0.1 * t,
            "knockback_x": 0.75 * sign,
            "knockback_y": 0.2 * t,
        },
        "state_flags": (0, 0, 0, 0, 0),
    }


def _obs(t: int, *, frame_id: int | None = None) -> dict:
    """One canonical closed-loop frame, with the matchup metadata ``drive_vec`` injects."""
    return {
        "id": t if frame_id is None else frame_id,
        "ports": {
            EGO_PORT: {"leader": {"post": _post(t, "ego")}, "follower": None},
            OPP_PORT: {"leader": {"post": _post(t, "opp")}, "follower": None},
        },
        "_matchup": {
            "stage": STAGE.value,
            "character": {EGO_PORT: melee.Character.FOX.value, OPP_PORT: melee.Character.MARTH.value},
        },
    }


def _mds_sample(n_frames: int) -> dict[str, np.ndarray]:
    """The same trajectory as an MDS replay row."""
    frames = [flatten_canonical_frame(_obs(t)) for t in range(n_frames)]
    sample: dict[str, np.ndarray] = {"frame": np.arange(n_frames, dtype=np.int32)}
    for key, first in frames[0].items():
        dtype = np.int32 if isinstance(first, int) else np.float32
        sample[key] = np.array([frame[key] for frame in frames], dtype=dtype)
    for channel in ACTION_CHANNELS:
        dtype = np.int32 if channel.startswith("button_") else np.float32
        sample[f"p{EGO_PORT}_{channel}"] = np.zeros(n_frames, dtype=dtype)
    return sample


def _mds_batch(n_frames: int = 4) -> dict[str, np.ndarray]:
    return relabel_ego(_mds_sample(n_frames), f"p{EGO_PORT}")


def _train_context(sample: dict[str, np.ndarray], last_frame: int) -> tuple[dict[str, torch.Tensor], int]:
    """The train-path context whose newest real frame is ``last_frame``."""
    sampler = WindowDataset([], L_CTX, 1, seed=0)
    start = last_frame + 1 - L_CTX
    pad = max(0, -start)
    window = sampler._padded_window(sample, start, pad)
    window["ctx_pad"] = np.int64(min(pad, L_CTX))
    batch = collate_train_batch(
        [relabel_ego(window, f"p{EGO_PORT}")], stats=_stats(), L_ctx=L_CTX, extra=V6_PLAYER_COLUMNS
    )
    return batch.context.features, int(batch.context.ctx_pad[0])


def _closed_loop_contexts(n_frames: int) -> list[tuple[dict[str, torch.Tensor], int]]:
    """Drive ``RecedingHorizon`` over the trajectory and capture the Context it hands the model."""
    captured: list[tuple[dict[str, torch.Tensor], int]] = []

    def predict_chunk(ctx, committed):
        assert committed is None
        captured.append(({k: v.clone() for k, v in ctx.features.items()}, int(ctx.ctx_pad[0])))
        return np.zeros((ctx.batch, 1, A_DIM), dtype=np.float32)

    policy = RecedingHorizon(
        predict_chunk=predict_chunk,
        stats=_stats(),
        L_ctx=L_CTX,
        L_chunk=1,
        s=1,
        d=0,
        device="cpu",
        extra=V6_PLAYER_COLUMNS,
    )
    slot = Slot(0, EGO_PORT)
    for t in range(n_frames):
        policy(t, {slot: _obs(t)})
    return captured


# --- nothing else moved ------------------------------------------------------


def test_default_routing_drops_the_v6_columns() -> None:
    """The regression that keeps every earlier experiment and checkpoint valid: with no
    routing, a batch that CARRIES the v6 columns preprocesses to exactly what the same
    batch without them ever did — same keys, same numbers."""
    with_v6 = _mds_batch()
    without_v6 = {k: v for k, v in with_v6.items() if not _is_v6(k)}
    routed_off = preprocess(with_v6, _stats())
    v5_only = preprocess(without_v6, _stats())
    assert set(routed_off) == set(v5_only)
    for name, value in routed_off.items():
        torch.testing.assert_close(value, v5_only[name], rtol=0, atol=0, msg=f"{name} moved")


def test_routing_adds_exactly_the_v6_player_columns() -> None:
    batch = _mds_batch()
    gained = set(preprocess(batch, _stats(), extra=V6_PLAYER_COLUMNS)) - set(preprocess(batch, _stats()))
    expected = {name for name in batch if _is_v6(name)}
    # A float column gets its mask sidecar only where the batch actually masks something —
    # here the nana block, which this two-Fox trajectory leaves empty.
    expected |= {
        f"{name}_mask"
        for name in expected
        if any(name.endswith(f"_{feat}") for feat in _V6_FLOATS) and np.isnan(batch[name]).any()
    }
    assert gained == expected
    assert {name for name in gained if name.endswith("_mask")} == {
        f"{prefix}_nana_{feat}_mask" for prefix in ("ego", "opp") for feat in _V6_FLOATS
    }


def test_velocities_and_state_age_standardize_with_the_dataset_stats() -> None:
    """Standardize, not min-max: with mean 2 / std 4 the two disagree on every value."""
    batch = _mds_batch()
    out = preprocess(batch, _stats(), extra=V6_PLAYER_COLUMNS)
    for name in ("ego_velocities_self_y", "opp_velocities_knockback_x", "ego_state_age"):
        expected = (batch[name] - 2.0) / 4.0
        torch.testing.assert_close(out[name], torch.from_numpy(expected.astype(np.float32)), rtol=0, atol=1e-6)


def test_velocity_suffix_does_not_collide_with_the_ground_categorical() -> None:
    """``velocities_self_x_ground`` also ends with the ``ground`` suffix, so the float
    table must resolve first — otherwise a float column would be read as class ids."""
    assert _classify("ego_velocities_self_x_ground", V6_PLAYER_COLUMNS) == "float"
    assert _classify("ego_ground", V6_PLAYER_COLUMNS) == "cat"
    assert _classify("ego_character_live", V6_PLAYER_COLUMNS) == "cat"
    assert _classify("ego_character", V6_PLAYER_COLUMNS) == "cat"
    out = preprocess(_mds_batch(), _stats(), extra=V6_PLAYER_COLUMNS)
    assert out["ego_velocities_self_x_ground"].dtype == torch.float32
    assert out["ego_ground"].dtype == torch.int64


@pytest.mark.parametrize(
    ("floats", "cats", "match"),
    [
        ({"state_age": "zscore"}, {}, "unknown float transform"),
        ({"ground": "standardize"}, {"ground": (4, 2)}, "both a float and a categorical"),
    ],
)
def test_extra_columns_rejects_an_invalid_declaration(floats: dict, cats: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ExtraColumns(floats=floats, cats=cats)


# --- the token is the declared bundle ----------------------------------------


def test_lean_spatial_columns_are_the_documented_subset() -> None:
    assert SPATIAL_COLUMNS_LEAN == (
        "ego_offstage",
        "opp_offstage",
        "ego_ledge_dx",
        "opp_ledge_dx",
        "ego_blast_bottom",
        "rel_dx_ego_facing",
        "rel_dy",
        "spatial_mask",
    )
    assert set(SPATIAL_COLUMNS_LEAN) < set(SPATIAL_COLUMNS)
    # The finite-difference proxy and its stricter mask are gone: v6 stores engine velocities.
    assert not [name for name in SPATIAL_COLUMNS_LEAN if "dpos" in name]


def _tiny_cfg(**kwargs):
    defaults = dict(
        d_model=64,
        n_layers=2,
        n_heads=2,
        L_ctx=16,
        head_offsets=(1, 2),
        batch_size=2,
        max_steps=8,
        warmup_steps=0,
    )
    return exp021.TrainConfig(**{**defaults, **kwargs})


def _features(batch: int, length: int, gen: torch.Generator | None = None) -> dict[str, torch.Tensor]:
    """A synthetic observation batch: zeros, or plausible random values when ``gen`` is given."""

    def randn(*shape: int) -> torch.Tensor:
        return torch.zeros(*shape) if gen is None else torch.randn(*shape, generator=gen)

    features: dict[str, torch.Tensor] = {}
    for prefix in exp021._PLAYER_PREFIXES:
        for feat in exp021._PLAYER_FLOATS:
            features[f"{prefix}_{feat}"] = randn(batch, length)
        for name, (vocab, _) in {**CAT_FEATURES, **exp021._V6_CATS}.items():
            hi = 1 if gen is None else vocab
            features[f"{prefix}_{name}"] = torch.randint(0, hi, (batch, length), generator=gen)
        features[f"{prefix}_{exp021._CHARACTER_LIVE}"] = torch.randint(
            0, 1 if gen is None else 26, (batch, length), generator=gen
        )
    for channel in ACTION_CHANNELS:
        features[f"ego_{channel}"] = randn(batch, length)
    for key in ("ego_character", "opp_character", "stage"):
        features[key] = torch.randint(0, 1 if gen is None else 26, (batch, length), generator=gen)
    for name in SPATIAL_COLUMNS_LEAN:
        features[name] = randn(batch, length)
    return features


def _context(cfg, batch: int = 2, seed: int | None = 0) -> Context:
    gen = None if seed is None else torch.Generator().manual_seed(seed)
    return Context(features=_features(batch, cfg.L_ctx, gen), ctx_pad=torch.zeros(batch, dtype=torch.long))


def _train_batch(cfg, batch: int = 2, seed: int = 0) -> TrainBatch:
    gen = torch.Generator().manual_seed(seed)
    ctx = Context(features=_features(batch, cfg.L_ctx, gen), ctx_pad=torch.tensor([0, 1] * (batch // 2)))
    target = torch.rand(batch, max(cfg.head_offsets), A_DIM, generator=gen) * 2 - 1
    return TrainBatch(context=ctx, target=target)


def _model(cfg, seed: int = 7):
    torch.manual_seed(seed)
    return exp021.GPT(cfg).eval()


def test_token_width_is_the_documented_breakdown() -> None:
    """486 = 4 players x (13 floats x 2 + 71 + 6 + 12) + 14 ego history + 4 stage + 8 spatial."""
    cfg = _tiny_cfg()
    model = _model(cfg)
    per_player = len(exp021._PLAYER_FLOATS) * 2 + 71 + 6 + cfg.char_dim
    assert per_player == 115
    assert model.ctx_proj.in_features == 4 * per_player + A_DIM + cfg.stage_dim + len(SPATIAL_COLUMNS_LEAN) == 486
    assert len(exp021._PLAYER_FLOATS) == len(FLOAT_FEATURES) + len(_V6_FLOATS) == 13


def test_model_reads_the_lean_block_only() -> None:
    """It consumes exactly the lean columns — absent one it says so, and it never touches
    the 17 derived columns the bundle dropped."""
    cfg = _tiny_cfg()
    model = _model(cfg)
    features = _features(2, cfg.L_ctx)
    ctx_pad = torch.zeros(2, dtype=torch.long)
    with torch.no_grad():
        assert model(features, ctx_pad).shape == (2, cfg.L_ctx, cfg.d_model)
    dropped = features.pop("ego_ledge_dx")
    with pytest.raises(ValueError, match="missing the spatial columns"):
        model(features, ctx_pad)
    features["ego_ledge_dx"] = dropped
    for name in set(SPATIAL_COLUMNS) - set(SPATIAL_COLUMNS_LEAN):
        features.pop(name, None)
    with torch.no_grad():
        assert model(features, ctx_pad).shape == (2, cfg.L_ctx, cfg.d_model)


def test_character_conditioning_is_per_frame_and_ignores_the_select_pick() -> None:
    """The live character enters every frame's token, so a mid-match transform moves the
    observation; the per-replay character-SELECT pick is deliberately unread."""
    cfg = _tiny_cfg()
    model = _model(cfg)
    features = _features(2, cfg.L_ctx, torch.Generator().manual_seed(1))
    ctx_pad = torch.zeros(2, dtype=torch.long)
    with torch.no_grad():
        base = model(features, ctx_pad)
        select_pick = dict(features)
        for key in ("ego_character", "opp_character"):
            select_pick[key] = torch.full_like(select_pick[key], 3)
        torch.testing.assert_close(model(select_pick, ctx_pad), base, rtol=0, atol=0)
        # Sheik -> Zelda inside the window: only the live column moves.
        transformed = dict(features)
        live = transformed[f"ego_{exp021._CHARACTER_LIVE}"].clone()
        live[:, cfg.L_ctx // 2 :] = melee.Character.ZELDA.value
        transformed[f"ego_{exp021._CHARACTER_LIVE}"] = live
        moved = model(transformed, ctx_pad)
    assert not torch.equal(moved, base), "live character is not conditioned on"


def _misc_as_channels(model, features: dict[str, torch.Tensor], prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
    """The (value, validity flag) pair the token carries for ``misc_as``."""
    block = model._per_player_features(features, prefix)
    index = exp021._PLAYER_FLOATS.index(exp021._MISC_AS)
    return block[..., index], block[..., len(exp021._PLAYER_FLOATS) + index]


def test_misc_as_is_carried_in_hitstun_and_flagged_out_of_it() -> None:
    """The column is action-state-multiplexed, so outside a hitstun state its value is not
    a hitstun counter at all: the token must zero it and raise its flag."""
    cfg = _tiny_cfg()
    model = _model(cfg)
    features = _features(2, cfg.L_ctx)
    features["ego_misc_as"] = torch.full((2, cfg.L_ctx), 0.6)
    action = torch.full((2, cfg.L_ctx), IDLE_ACTION, dtype=torch.long)
    action[:, ::2] = HITSTUN_ACTION
    features["ego_action"] = action
    with torch.no_grad():
        value, flag = _misc_as_channels(model, features, "ego")
    hitstun = action == HITSTUN_ACTION
    assert torch.all(value[hitstun] == 0.6) and torch.all(flag[hitstun] == 0.0)
    assert torch.all(value[~hitstun] == 0.0) and torch.all(flag[~hitstun] == 1.0)


def test_a_missing_misc_as_column_stays_flagged_inside_hitstun() -> None:
    """The two invalidity reasons compose: an slp too old to record the column is invalid
    even on a hitstun frame."""
    cfg = _tiny_cfg()
    model = _model(cfg)
    features = _features(2, cfg.L_ctx)
    features["ego_misc_as"] = torch.full((2, cfg.L_ctx), 0.6)
    features["ego_misc_as_mask"] = torch.ones(2, cfg.L_ctx)
    features["ego_action"] = torch.full((2, cfg.L_ctx), HITSTUN_ACTION, dtype=torch.long)
    with torch.no_grad():
        value, flag = _misc_as_channels(model, features, "ego")
    assert torch.all(value == 0.0) and torch.all(flag == 1.0)


def test_every_player_block_carries_the_v6_suffixes() -> None:
    """The nana follower block is suffix-driven, so it gains the same columns as its leader."""
    cfg = _tiny_cfg()
    model = _model(cfg)
    features = _features(2, cfg.L_ctx)
    for prefix in exp021._PLAYER_PREFIXES:
        with torch.no_grad():
            assert model._per_player_features(features, prefix).shape[-1] == 115
    features.pop("ego_nana_velocities_self_y")
    with pytest.raises(KeyError):
        model._per_player_features(features, "ego_nana")


# --- train and closed-loop eval agree ----------------------------------------


def test_train_and_closed_loop_build_the_same_v6_token() -> None:
    """Both paths run one preprocess with one routing object. Every v6 column and every
    lean spatial column must be bit-equal at every context position, cold start included."""
    n_frames = 2 * L_CTX
    sample = _mds_sample(n_frames)
    live = _closed_loop_contexts(n_frames)
    columns = [name for name in live[0][0] if _is_v6(name)] + list(SPATIAL_COLUMNS_LEAN)
    assert len(columns) > len(SPATIAL_COLUMNS_LEAN)
    for t in range(n_frames - 1):  # -1: the train window also needs one chunk frame
        train_features, train_pad = _train_context(sample, t)
        loop_features, loop_pad = live[t]
        assert train_pad == loop_pad == max(0, L_CTX - (t + 1))
        assert set(train_features) == set(loop_features)
        for name in columns:
            torch.testing.assert_close(
                train_features[name], loop_features[name], rtol=0, atol=0, msg=f"{name} differs at frame {t}"
            )


def test_closed_loop_reads_real_v6_values_not_mask_sentinels() -> None:
    """libmelee must actually fill the new post fields online. A field it left empty would
    reach the model as a masked zero — the feature would exist at train time only."""
    live = _closed_loop_contexts(2 * L_CTX)
    features, pad = live[-1]
    assert pad == 0
    for name in ("ego_velocities_self_y", "ego_state_age", "opp_velocities_knockback_y"):
        assert torch.any(features[name] != 0.0), f"{name} is identically zero online"
        assert f"{name}_mask" not in features or torch.all(features[f"{name}_mask"] == 0.0)
    # The live character is the ego's, per frame, not a masked zero.
    assert torch.all(features[f"ego_{exp021._CHARACTER_LIVE}"] == melee.Character.FOX.value)
    assert torch.all(features[f"opp_{exp021._CHARACTER_LIVE}"] == melee.Character.MARTH.value)
    assert set(torch.unique(features["ego_ground"]).tolist()) == {3, 65535}


# --- the frozen recipe -------------------------------------------------------


def test_defaults_are_the_deployed_016_base_recipe_on_v6_data() -> None:
    cfg = exp021.TrainConfig()
    assert (cfg.batch_size, cfg.max_steps) == (512, 16384)
    assert (cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.L_ctx) == (256, 8, 4, 256)
    assert (cfg.muon_lr, cfg.adam_lr) == (0.02, 8.5e-4)
    assert cfg.head_offsets == (1, 5, 9, 13)
    assert cfg.warmup_steps == 500 and cfg.val_every == 1024 and cfg.ckpt_every == 2048
    assert cfg.windows_per_replay == 4 and cfg.final_eval_n_matchups == 96
    assert (cfg.eval_every, cfg.eval_n_matchups, cfg.eval_timeout_seconds) == (4096, 32, 2700.0)
    assert cfg.mds_schema_version == 6
    assert cfg.data_root == "data/processed/ranked-anonymized-1/mds-v6"
    assert cfg.final_h2h_self_label == "021-v6feat" and cfg.final_h2h_reference_label == "016-base"
    assert cfg.final_h2h_n_configs == 64


def test_both_observation_paths_get_the_same_routing() -> None:
    cfg = _tiny_cfg()
    kwargs = exp021._loader_kwargs(cfg, _stats())
    assert kwargs["schema_version"] == 6
    assert kwargs["extra"] is V6_PLAYER_COLUMNS
    policy = exp021.make_policy(_model(cfg), _stats(), cfg, device="cpu")
    assert policy.extra is V6_PLAYER_COLUMNS


def test_validate_config_rejects_a_pre_v6_dataset() -> None:
    with pytest.raises(ValueError, match="mds_schema_version must be >= 6"):
        exp021.validate_config(_tiny_cfg(mds_schema_version=5), has_button_combo_counts=False)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (dict(final_h2h_reference_run="", final_h2h_reference_label="x"), "must be a run name or None"),
        (dict(final_h2h_reference_run="run", final_h2h_n_configs=0), "final_h2h_n_configs"),
        (dict(final_h2h_reference_run="run", final_h2h_self_label="016-base"), "labels must differ"),
        (
            dict(final_h2h_reference_run="run", final_h2h_reference_experiment="experiments/nope.py"),
            "does not exist",
        ),
    ],
)
def test_validate_config_rejects_a_bad_final_h2h(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        exp021.validate_config(_tiny_cfg(**kwargs), has_button_combo_counts=False)


# --- deploy contract ---------------------------------------------------------


def test_closed_loop_decode_never_uses_argmax() -> None:
    """Greedy decode collapses the closed-loop policy to doing nothing, so the deployed path
    has no argmax knob at all and the policy's draws must move with its seed."""
    assert "argmax" not in {f.name for f in dataclasses.fields(exp021.DecodeSettings)}
    for fn in (exp021.decode, exp021.decode_chunk):
        assert inspect.signature(fn).parameters["argmax"].default is False
    cfg = _tiny_cfg()
    model = _model(cfg)
    ctx = _context(cfg, batch=8)
    chunks = [
        exp021.make_policy(model, _stats(), cfg, device="cpu", decode_seed=seed).predict_chunk(ctx, None)
        for seed in (0, 1)
    ]
    assert not (chunks[0] == chunks[1]).all(), "policy decoded greedily"


# --- training smoke ----------------------------------------------------------


def test_objective_decreases_on_a_fixed_batch() -> None:
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    model = exp021.GPT(cfg)
    model.train()
    opt = exp021.make_optimizer(model, cfg)
    batch = _train_batch(cfg, batch=4)
    losses = []
    for _ in range(20):
        opt.zero_grad()
        nll, trans = exp021.action_loss(model, batch)
        obj = exp021.objective(nll, trans, cfg.aux_loss_weight, cfg.transition_loss_weight)
        obj.backward()
        opt.step()
        losses.append(obj.item())
    assert losses[-1] < losses[0], losses


def test_val_metrics_keep_the_016_surface_with_a_lean_spatial_probe() -> None:
    cfg = _tiny_cfg()
    model = _model(cfg)
    out = exp021.val_metrics(model, [_train_batch(cfg, batch=4)], cfg)
    for key in ("loss", "nll_off1", "nll_off2", "btn_logloss", "ablate_hist_kl", "ablate_hist_dnll"):
        assert key in out
    for name in exp021._GROUP_NAMES:
        assert f"nll_{name}" in out and f"brier_{name}" in out and f"changeF1_{name}" in out
        assert f"ablate_spatial_dnll_{name}" in out
    assert "ablate_spatial_kl" in out and "ablate_spatial_dnll" in out
    assert out["loss"] == pytest.approx(sum(out[f"nll_{name}"] for name in exp021._GROUP_NAMES), rel=1e-6)
