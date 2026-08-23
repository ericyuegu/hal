"""Focused contracts for the scaled light-AWR experiment."""

import importlib.util
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from streaming import MDSWriter

from hal.data.policy_schema import POLICY_SCHEMA_VERSION
from hal.data.policy_world_schema import POLICY_WORLD_MDS_COLUMNS
from hal.data.policy_world_schema import POLICY_WORLD_SCHEMA_VERSION
from hal.streams import StreamSource
from hal.training import returns as returns_lib
from hal.training.dataloader import _make_streaming_dataset
from hal.training.features import A_DIM
from hal.training.features import ACTION_CHANNELS
from hal.training.features import Context
from hal.training.features import FeatureProjection
from hal.training.features import TrainBatch
from hal.training.replay_reservoir import PolicyReplayPackDataset

_PATH = Path(__file__).resolve().parents[2] / "experiments" / "040_scaled_awr_bc.py"
_SPEC = importlib.util.spec_from_file_location("test_exp040", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
exp = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = exp
_SPEC.loader.exec_module(exp)


def _cfg(**overrides) -> exp.TrainConfig:
    values = {
        "d_model": 32,
        "n_layers": 1,
        "n_heads": 4,
        "L_ctx": 4,
        "temporal_d_model": 32,
        "temporal_layers": 1,
        "temporal_heads": 4,
        "temporal_ff_dim": 64,
        "group_head_dim": 64,
        "batch_size": 2,
        "target_loss_positions": 16,
        "reservoir_capacity": 4,
        "warmup_fraction": 0.5,
        "stable_fraction": 0.75,
        "compile_trunk": False,
        "compile_temporal": False,
        "num_workers": 0,
        "push_to_r2": False,
        "inference_mode": "eager",
        "awr_return_baseline": 10.0,
        "awr_weight_norm": 2.0,
    }
    return exp.TrainConfig(**{**values, **overrides})


def _actions(batch: int, length: int, generator: torch.Generator) -> torch.Tensor:
    actions = torch.empty(batch, length, A_DIM)
    actions[..., :4] = torch.rand(batch, length, 4, generator=generator) * 2 - 1
    actions[..., 4:6] = torch.rand(batch, length, 2, generator=generator)
    actions[..., 6:] = torch.randint(0, 2, (batch, length, 8), generator=generator).float()
    return actions


def _batch(cfg: exp.TrainConfig, pads: list[int] | None = None, seed: int = 0) -> TrainBatch:
    pads = [0, 1] if pads is None else pads
    generator = torch.Generator().manual_seed(seed)
    synthetic = exp.synthetic_context(cfg, len(pads), torch.device("cpu"))
    features = dict(synthetic.features)
    native = _actions(len(pads), cfg.L_ctx, generator)
    for channel, values in zip(ACTION_CHANNELS, native.unbind(-1), strict=True):
        features[f"ego_{channel}"] = values
    return TrainBatch(
        context=Context(features=features, ctx_pad=torch.tensor(pads, dtype=torch.int64)),
        target=_actions(len(pads), cfg.sample_chunk_length, generator),
    )


def _awr_batch(cfg: exp.TrainConfig) -> exp.AWRBatch:
    batch = _batch(cfg)
    batch.target[..., 6:] = 0.0
    returns = torch.full((batch.target.shape[0], cfg.L_ctx), cfg.awr_return_baseline)
    eligible = torch.ones_like(returns, dtype=torch.bool)
    return exp.AWRBatch(batch=batch, returns=returns, eligible=eligible)


def _write_policy_world(root: Path, replay_id: str, *, source_schema_version: int = 7) -> None:
    frames = 40
    sample: dict[str, object] = {
        "policy_schema_version": POLICY_SCHEMA_VERSION,
        "policy_world_schema_version": POLICY_WORLD_SCHEMA_VERSION,
        "source_schema_version": source_schema_version,
        "replay_id": replay_id,
        "num_frames": frames,
        "stage": 2,
        "p1_character": 1,
        "p2_character": 22,
        "p1_rank": 1,
        "p2_rank": 4,
        "p1_nana_present": 0,
        "p2_nana_present": 0,
    }
    for name, encoding in POLICY_WORLD_MDS_COLUMNS.items():
        if name in sample:
            continue
        dtype = np.dtype(encoding.removeprefix("ndarray:"))
        length = 1 if "nana" in name else frames
        values = np.zeros(length, dtype=dtype)
        if name.startswith("item") and dtype.kind == "f":
            values.fill(np.nan)
        sample[name] = values
    with MDSWriter(out=str(root / "train"), columns=POLICY_WORLD_MDS_COLUMNS, compression="zstd") as writer:
        writer.write(sample)


def test_production_geometry_and_schedule_endpoints() -> None:
    cfg = exp.TrainConfig()
    schedule = exp.lr_schedule(cfg)

    assert cfg.max_steps == 262_144
    assert cfg.warmup_steps == 7_864
    assert cfg.stable_steps == 209_715
    assert cfg.batch_size * cfg.L_ctx * cfg.max_steps == 2**35
    assert schedule(0) == 0.0
    assert schedule(7_864) == 1.0
    assert schedule(100_000) == 1.0
    assert schedule(209_715) == 1.0
    assert schedule(262_143) == pytest.approx(1 / 170)


def test_fixed_global_weights_match_hand_computation_and_warmup() -> None:
    returns = torch.tensor([12.0, 10.0, float("nan")])
    eligible = torch.tensor([True, True, False])
    weight, stats = exp.advantage_weights(
        returns,
        eligible,
        baseline=10.0,
        beta=2.0,
        weight_max=3.5,
        weight_norm=2.0,
    )

    torch.testing.assert_close(weight, torch.tensor([math.e / 2, 0.5, 1.0]))
    assert float(stats["weight_mean"]) == pytest.approx((math.e + 1) / 4)
    assert float(stats["weight_clip_frac"]) == 0.0
    warmup, warmup_stats = exp.advantage_weights(
        returns,
        eligible,
        baseline=10.0,
        beta=2.0,
        weight_max=3.5,
        weight_norm=2.0,
        active=False,
    )
    torch.testing.assert_close(warmup, torch.ones(3))
    assert float(warmup_stats["weight_ess"]) == 1.0


def test_weights_cap_before_fixed_normalization_and_reject_bad_eligible_rows() -> None:
    weight, stats = exp.advantage_weights(
        torch.tensor([100.0, float("nan")]),
        torch.tensor([True, False]),
        baseline=0.0,
        beta=1.0,
        weight_max=3.5,
        weight_norm=2.0,
    )
    torch.testing.assert_close(weight, torch.tensor([1.75, 1.0]))
    assert float(stats["weight_clip_frac"]) == 1.0
    with pytest.raises(FloatingPointError, match="non-finite"):
        exp.advantage_weights(
            torch.tensor([float("inf")]),
            torch.tensor([True]),
            baseline=0.0,
            beta=1.0,
            weight_max=3.5,
            weight_norm=1.0,
        )
    with pytest.raises(ValueError, match="detached"):
        exp.advantage_weights(
            torch.ones(1, requires_grad=True),
            torch.tensor([True]),
            baseline=0.0,
            beta=1.0,
            weight_max=3.5,
            weight_norm=1.0,
        )


def test_near_far_objective_matches_hand_computation() -> None:
    nll = torch.zeros(2, 15, exp.N_GROUPS)
    nll[0, :6] = 1.0
    nll[1, :6] = 2.0
    nll[0, 6:] = 3.0
    nll[1, 6:] = 5.0
    weight = torch.tensor([2.0, 0.5])

    near, far, total = exp.temporal_objective_parts(
        nll,
        weight,
        valid_prefixes=2,
        aux_loss_weight=0.5,
        n_near=6,
    )

    # Four controller groups: near=(4*1*2 + 4*2*.5)/2=6,
    # far=(4*3 + 4*5)/2=16, total=(6 + .5*16)/1.5.
    torch.testing.assert_close(near, torch.tensor(6.0))
    torch.testing.assert_close(far, torch.tensor(16.0))
    torch.testing.assert_close(total, torch.tensor(14 / 1.5))
    torch.testing.assert_close(
        exp.advantage_weighted_objective(
            nll,
            weight,
            valid_prefixes=2,
            aux_loss_weight=0.5,
            n_near=6,
        ),
        total,
    )


def test_unweighted_metric_uses_the_same_near_far_normalization() -> None:
    mean_nll = torch.ones(15, exp.N_GROUPS)
    mean_nll[6:] *= 3
    metrics = exp.nll_mean_metrics(mean_nll, tuple(range(1, 16)), n_near=6, aux_loss_weight=0.5)
    expected = (4 / exp._LN2 + 0.5 * 12 / exp._LN2) / 1.5
    assert metrics["loss_unweighted"] == pytest.approx(expected)
    assert metrics["temporal_loss_total_unweighted"] == pytest.approx(expected)


def test_rollout_button_mismatch_is_finite_and_uses_compatible_rows() -> None:
    cfg = _cfg()
    torch.manual_seed(0)
    model = exp.GPT(cfg)
    batch = _batch(cfg, pads=[0, 0, cfg.L_ctx])
    button_l = ACTION_CHANNELS.index("button_l")
    batch.target[..., 6:] = 0.0
    batch.target[0, :, button_l] = 1.0

    def incompatible_rollout(hidden: torch.Tensor, observed: torch.Tensor):
        del observed
        rows = hidden.shape[0]
        sampled = torch.zeros(rows, len(cfg.head_offsets), exp.N_GROUPS, dtype=torch.long)
        logits = []
        for _ in cfg.head_offsets:
            frame = {
                name: torch.zeros(rows, vocab) for name, vocab in zip(exp.GROUP_NAMES, exp.GROUP_VOCABS, strict=True)
            }
            frame["buttons"] = frame["buttons"].masked_fill(
                model.codec.button_mask(sampled[:, 0, exp.TRIG_G]), -torch.inf
            )
            logits.append(frame)
        return logits, sampled

    model.temporal.rollout_conditioned_logits = incompatible_rollout  # type: ignore[method-assign]
    metrics = exp.val_metrics(model, [batch], cfg)

    assert all(math.isfinite(value) for value in metrics.values())
    for offset in cfg.head_offsets:
        assert metrics[f"rollout_button_target_masked_rate_o{offset:02d}"] == 0.5
        assert math.isfinite(metrics[f"exposure_gap_o{offset:02d}_buttons"])


def test_parameter_partition_and_checkpoint_config_are_complete() -> None:
    cfg = _cfg()
    model = exp.GPT(cfg)
    counts = exp.subsystem_parameter_counts(model)
    assert sum(value for name, value in counts.items() if name != "total") == counts["total"]
    assert not hasattr(model, "value_head")
    owned = [parameter for group in exp.make_optimizer(model, cfg).param_groups for parameter in group["params"]]
    assert len(owned) == len({id(parameter) for parameter in owned}) == sum(1 for _ in model.parameters())

    checkpoint = exp._checkpoint_config(cfg)
    assert all(isinstance(name, str) for name in checkpoint["source_names"])
    assert exp.config_from_state(checkpoint) == cfg
    with pytest.raises(ValueError, match="experiment_id"):
        exp.config_from_state({**checkpoint, "experiment_id": "036_advantage_weighted_bc_v1"})


def test_two_policy_world_streams_label_returns_and_keep_schema_checks(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_policy_world(first, "first")
    _write_policy_world(second, "second")
    sources = (
        StreamSource("first", None, first),  # type: ignore[arg-type]
        StreamSource("second", None, second),  # type: ignore[arg-type]
    )
    dataset, names = _make_streaming_dataset(
        None,
        "train",
        sources=sources,
        remote=None,
        shuffle=False,
        shuffle_seed=0,
        cache_limit=None,
        shuffle_block_size=16,
        predownload=None,
    )
    projection = FeatureProjection(frozenset({exp.EGO_RETURN, exp.EGO_RETURN_VALID}), derive_spatial=False)
    packs = PolicyReplayPackDataset(
        dataset,
        L_ctx=4,
        L_chunk=2,
        seed=0,
        windows_per_replay=1,
        schema_version=7,
        projection=projection,
        replay_format="policy-world",
        replay_transform=lambda replay: returns_lib.label_replay(
            replay,
            gamma=0.9,
            damage_shaping=1.0,
            win_reward=50.0,
            stock_value=120.0,
            suffix=exp._RETURN_SUFFIX,
        ),
    )

    emitted = list(packs)
    assert names == ("first", "second")
    assert {pack.replay_id for pack in emitted} == {"first", "second"}
    assert all({exp.EGO_RETURN, exp.EGO_RETURN_VALID, "ctx_pad"} == pack.windows[0].keys() for pack in emitted)

    invalid = tmp_path / "invalid"
    _write_policy_world(invalid, "invalid", source_schema_version=6)
    bad_dataset, _ = _make_streaming_dataset(
        None,
        "train",
        sources=(StreamSource("invalid", None, invalid),),  # type: ignore[arg-type]
        remote=None,
        shuffle=False,
        shuffle_seed=0,
        cache_limit=None,
        shuffle_block_size=16,
        predownload=None,
    )
    bad_packs = PolicyReplayPackDataset(
        bad_dataset,
        L_ctx=4,
        L_chunk=2,
        seed=0,
        windows_per_replay=1,
        schema_version=7,
        projection=None,
        replay_format="policy-world",
    )
    with pytest.raises(ValueError, match="schema_version"):
        list(bad_packs)


def test_config_rejects_bad_chunk_dense_prefix_and_dose() -> None:
    with pytest.raises(ValueError, match="beyond sample_chunk_length"):
        exp.validate_config(_cfg(sample_chunk_length=20))
    with pytest.raises(ValueError, match="dense offset prefix"):
        exp.validate_config(_cfg(head_offsets=(1, 2, 3, 4, 5, 7, 8)))
    with pytest.raises(ValueError, match="divisible"):
        exp.validate_config(_cfg(target_loss_positions=17))
    assert asdict(exp.TrainConfig())["target_loss_positions"] == 2**35


def test_tiny_smoke_trains_checkpoints_and_resumes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = _cfg(
        wandb_hist_every=0,
        val_every=0,
        ckpt_every=0,
        eval_every=0,
        wandb_log_code=False,
    )
    batch = _awr_batch(cfg)

    class Loader:
        source_sample_counts = {name: exp.streams.POLICY_WORLD_V7_TRAIN_REPLAYS[name] for name in cfg.source_names}

        def __iter__(self):
            return iter((batch,))

    monkeypatch.setattr(exp, "_make_loaders", lambda _cfg, _stats: (Loader(), [batch.batch]))
    monkeypatch.setattr(exp, "make_run_name", lambda *args: "tiny-040")
    monkeypatch.setattr(exp, "setup_run_dir", lambda _name: (tmp_path, tmp_path / "replays"))

    class Run:
        id = "test"
        summary: dict[str, object] = {}

    monkeypatch.setattr(exp.wandb, "run", Run())
    monkeypatch.setattr(exp.wandb, "init", lambda **kwargs: None)
    monkeypatch.setattr(exp.wandb, "define_metric", lambda *args, **kwargs: None)
    monkeypatch.setattr(exp.wandb, "log", lambda values: None)
    monkeypatch.setattr(exp.wandb, "finish", lambda: None)

    exp.train(cfg, {}, smoke=True, stop_after_update=1, smoke_eval_matchups=0)
    first = torch.load(tmp_path / "latest.pt", map_location="cpu", weights_only=False)
    assert first["step"] == 0
    assert (tmp_path / "boundary-step-0000001.pt").is_file()

    exp.train(
        cfg,
        {},
        resume_run="tiny-040",
        resume_state=first,
        smoke=True,
        stop_after_update=2,
        smoke_eval_matchups=0,
    )
    second = torch.load(tmp_path / "latest.pt", map_location="cpu", weights_only=False)
    assert second["step"] == 1
    assert (tmp_path / "boundary-step-0000002.pt").is_file()
    assert (tmp_path / "smoke-final.pt").is_file()
