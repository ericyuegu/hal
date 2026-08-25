"""Contracts for scaled light AWR with projectile inputs.

The pooled set encoder must ignore which slots the live items occupy, the two
observation paths must deliver the same item tensors, and the configured sources
must be ones that carry the projectile block at all.
"""

import importlib.util
import json
import math
import sys
from collections.abc import Mapping
from dataclasses import asdict
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import melee
import numpy as np
import pytest
import torch
from streaming import MDSWriter
from streaming import StreamingDataset
from torch import Tensor

from hal.data.feature_stats import FeatureStats
from hal.data.policy_schema import POLICY_SCHEMA_VERSION
from hal.data.policy_world_schema import POLICY_WORLD_MDS_COLUMNS
from hal.data.policy_world_schema import POLICY_WORLD_SCHEMA_VERSION
from hal.data.policy_world_schema import encode_policy_world_replay
from hal.streams import StreamSource
from hal.training import returns as returns_lib
from hal.training.canonical import flatten_canonical_frame
from hal.training.closed_loop import _build_layout
from hal.training.closed_loop import _Rings
from hal.training.dataloader import _make_streaming_dataset
from hal.training.dataloader import make_loader
from hal.training.dataloader import relabel_ego
from hal.training.features import A_DIM
from hal.training.features import ACTION_CHANNELS
from hal.training.features import BASE_ITEMS_PROJECTION
from hal.training.features import ITEM_COLUMNS
from hal.training.features import ITEM_INPUT_COLUMNS
from hal.training.features import Context
from hal.training.features import FeatureProjection
from hal.training.features import TrainBatch
from hal.training.features import preprocess
from hal.training.replay_reservoir import PolicyReplayPackDataset
from hal.wire import ITEM_SLOTS
from hal.wire import MASK_INT32
from hal.wire import item_column


def _load(name: str, filename: str) -> ModuleType:
    """Experiments load by path: their filenames start with a digit."""
    path = Path(__file__).resolve().parents[2] / "experiments" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


exp = _load("test_exp040", "040_scaled_awr_bc.py")

# Every item width differs from every other, so a mixed-up concatenation cannot pass by
# coincidence.
_TINY_ITEMS: dict[str, object] = {
    "item_type_dim": 6,
    "item_state_dim": 3,
    "item_hidden_dim": 8,
    "item_dim": 5,
}

_ITEM_CATS: tuple[str, ...] = tuple(ITEM_COLUMNS.cats)
_ITEM_FLOATS: tuple[str, ...] = tuple(ITEM_COLUMNS.floats)
_ITEM_CAT_COLUMNS = {item_column(slot, name) for slot in range(ITEM_SLOTS) for name in _ITEM_CATS}

EGO_PORT, OPP_PORT = 1, 2
EGO_PREFIX = "p1"
STAGE = int(melee.Stage.FINAL_DESTINATION.value)

# The local v7 subset the schema tests also read. Absent on a fresh checkout.
_V7_TRAIN = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "ranked-anonymized-1" / "mds-v7-sub4" / "train"
)


def test_production_awr_constants_match_calibration_artifact() -> None:
    artifact_path = Path(__file__).resolve().parents[2] / "notebooks" / "040_awr_constants.json"
    artifact = json.loads(artifact_path.read_text())
    cfg = exp.TrainConfig()

    assert cfg.awr.return_baseline == artifact["return_baseline"]
    assert cfg.awr.weight_norm == artifact["weight_norm"]


def _cfg(**overrides) -> exp.TrainConfig:
    arch_values = {
        **asdict(exp.ARCHITECTURE),
        "d_model": 32,
        "n_layers": 1,
        "n_heads": 4,
        "L_ctx": 4,
        "temporal_d_model": 32,
        "temporal_layers": 1,
        "temporal_heads": 4,
        "temporal_ff_dim": 64,
        "group_head_dim": 64,
        **_TINY_ITEMS,
    }
    awr_values = {
        **asdict(exp.AWR_CALIBRATION),
        "return_baseline": 10.0,
        "weight_norm": 2.0,
    }
    values = {
        "batch_size": 2,
        "target_loss_positions": 16,
        "reservoir_capacity": 4,
        "warmup_fraction": 0.5,
        "stable_fraction": 0.75,
        "compile_trunk": False,
        "compile_temporal": False,
        "num_workers": 0,
        "prefetch_batches": 8,
        "push_to_r2": False,
        "inference_mode": "eager",
    }
    for name, value in overrides.items():
        if name in arch_values:
            arch_values[name] = value
        else:
            values[name] = value
    return exp.TrainConfig(arch=exp.Architecture(**arch_values), awr=exp.AWRCalibration(**awr_values), **values)


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
    native = _actions(len(pads), cfg.arch.L_ctx, generator)
    for channel, values in zip(ACTION_CHANNELS, native.unbind(-1), strict=True):
        features[f"ego_{channel}"] = values
    return TrainBatch(
        context=Context(features=features, ctx_pad=torch.tensor(pads, dtype=torch.int64)),
        target=_actions(len(pads), cfg.arch.sample_chunk_length, generator),
    )


def _awr_batch(cfg: exp.TrainConfig) -> exp.AWRBatch:
    batch = _batch(cfg)
    batch.target[..., 6:] = 0.0
    returns = torch.full((batch.target.shape[0], cfg.arch.L_ctx), cfg.awr.return_baseline)
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

    assert cfg.max_steps == 524_288
    assert cfg.ckpt_every == 2048
    assert cfg.warmup_steps == 15_728
    assert cfg.stable_steps == 419_430
    assert cfg.batch_size * cfg.arch.L_ctx * cfg.max_steps == 2**35
    assert schedule(0) == 0.0
    assert schedule(15_728) == 1.0
    assert schedule(100_000) == 1.0
    assert schedule(419_430) == 1.0
    assert schedule(524_287) == pytest.approx(1 / 170)

    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([parameter], lr=2.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    assert optimizer.param_groups[0]["lr"] == 0.0
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2.0 * schedule(1))


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


def test_extreme_negative_advantages_preserve_formula_and_keep_ess_finite() -> None:
    weights, stats = exp.advantage_weights(
        torch.tensor([-1e20, -1e30]),
        torch.ones(2, dtype=torch.bool),
        baseline=0.0,
        beta=1.0,
        weight_max=3.5,
        weight_norm=2.0,
    )

    torch.testing.assert_close(weights, torch.zeros(2))
    assert all(torch.isfinite(value) for value in stats.values())
    assert float(stats["weight_ess"]) == 0.0


def test_dense_awr_mask_matches_selecting_valid_prefixes() -> None:
    returns = torch.tensor([[12.0, float("nan"), 10.0], [8.0, 11.0, float("nan")]])
    eligible = torch.tensor([[True, False, True], [True, True, False]])
    valid = torch.tensor([[True, False, True], [False, True, False]])

    dense_weights, dense_stats = exp.advantage_weights(
        returns,
        eligible,
        baseline=10.0,
        beta=2.0,
        weight_max=3.5,
        weight_norm=2.0,
        valid=valid,
    )
    selected_weights, selected_stats = exp.advantage_weights(
        returns[valid],
        eligible[valid],
        baseline=10.0,
        beta=2.0,
        weight_max=3.5,
        weight_norm=2.0,
    )

    torch.testing.assert_close(dense_weights[valid], selected_weights)
    torch.testing.assert_close(dense_weights[~valid], torch.ones_like(dense_weights[~valid]))
    for name in dense_stats:
        torch.testing.assert_close(dense_stats[name], selected_stats[name])


def test_dense_temporal_objective_matches_selected_prefixes() -> None:
    generator = torch.Generator().manual_seed(17)
    nll = torch.rand(2, 3, 15, exp.N_GROUPS, generator=generator)
    weight = torch.rand(2, 3, generator=generator)
    valid = torch.tensor([[True, False, True], [False, True, True]])

    dense_parts = exp.temporal_objective_parts(
        nll,
        weight,
        valid_prefixes=int(valid.sum()),
        aux_loss_weight=0.5,
        valid=valid,
    )
    selected_parts = exp.temporal_objective_parts(
        nll[valid],
        weight[valid],
        valid_prefixes=int(valid.sum()),
        aux_loss_weight=0.5,
    )

    for dense, selected in zip(dense_parts, selected_parts, strict=True):
        torch.testing.assert_close(dense, selected)

    invalid_nan = nll.clone()
    invalid_nan[~valid] = torch.nan
    nan_masked_parts = exp.temporal_objective_parts(
        invalid_nan,
        weight,
        valid_prefixes=int(valid.sum()),
        aux_loss_weight=0.5,
        valid=valid,
    )
    for masked, selected in zip(nan_masked_parts, selected_parts, strict=True):
        torch.testing.assert_close(masked, selected)

    dense_nll = nll.clone().requires_grad_()
    selected_nll = nll[valid].clone().requires_grad_()
    exp.temporal_objective_parts(
        dense_nll,
        weight,
        valid_prefixes=int(valid.sum()),
        aux_loss_weight=0.5,
        valid=valid,
    )[2].backward()
    exp.temporal_objective_parts(
        selected_nll,
        weight[valid],
        valid_prefixes=int(valid.sum()),
        aux_loss_weight=0.5,
    )[2].backward()

    torch.testing.assert_close(dense_nll.grad[valid], selected_nll.grad)
    torch.testing.assert_close(dense_nll.grad[~valid], torch.zeros_like(dense_nll.grad[~valid]))


def test_device_batch_prefetcher_preserves_cpu_batches() -> None:
    cfg = _cfg()
    batches = [_awr_batch(cfg), _awr_batch(cfg)]
    batches[0].returns.fill_(1.0)
    batches[1].returns.fill_(2.0)
    prefetcher = exp.DeviceBatchPrefetcher(batches, cfg, "cpu")

    first, first_prefixes, _ = prefetcher.next()
    prefetcher.preload()
    second, second_prefixes, _ = prefetcher.next()

    assert first_prefixes == second_prefixes == 7
    torch.testing.assert_close(first.returns, torch.ones_like(first.returns))
    torch.testing.assert_close(second.returns, torch.full_like(second.returns, 2.0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for packed transfer")
def test_packed_awr_batch_copies_one_storage_per_dtype() -> None:
    cfg = _cfg()
    batch = _awr_batch(cfg)
    pinned = batch.pin_memory()
    tensors = pinned._tensors()

    assert all(tensor.is_pinned() for tensor in tensors)
    assert len({tensor.untyped_storage().data_ptr() for tensor in tensors}) == len(
        {tensor.dtype for tensor in tensors}
    )

    moved = pinned.to("cuda")
    torch.cuda.synchronize()
    for actual, expected in zip(moved._tensors(), batch._tensors(), strict=True):
        torch.testing.assert_close(actual.cpu(), expected)


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
    )

    # Four controller groups: near=(4*1*2 + 4*2*.5)/2=6,
    # far=(4*3 + 4*5)/2=16, total=(6 + .5*16)/1.5.
    torch.testing.assert_close(near, torch.tensor(6.0))
    torch.testing.assert_close(far, torch.tensor(16.0))
    torch.testing.assert_close(total, torch.tensor(14 / 1.5))


def test_unweighted_metric_uses_the_same_near_far_normalization() -> None:
    mean_nll = torch.ones(15, exp.N_GROUPS)
    mean_nll[6:] *= 3
    metrics = exp.nll_mean_metrics(mean_nll, tuple(range(1, 16)), aux_loss_weight=0.5)
    expected = (4 / exp._LN2 + 0.5 * 12 / exp._LN2) / 1.5
    assert metrics["loss_unweighted"] == pytest.approx(expected)


def test_rollout_button_mismatch_is_finite_and_uses_compatible_rows() -> None:
    cfg = _cfg()
    torch.manual_seed(0)
    model = exp.GPT(cfg)
    batch = _batch(cfg, pads=[0, 0, cfg.arch.L_ctx])
    button_l = ACTION_CHANNELS.index("button_l")
    batch.target[..., 6:] = 0.0
    batch.target[0, :, button_l] = 1.0

    def incompatible_rollout(hidden: torch.Tensor, observed: torch.Tensor):
        del observed
        rows = hidden.shape[0]
        sampled = torch.zeros(rows, len(cfg.arch.head_offsets), exp.N_GROUPS, dtype=torch.long)
        logits = []
        for _ in cfg.arch.head_offsets:
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
    for offset in cfg.arch.head_offsets:
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
        exp.config_from_state({**checkpoint, "experiment_id": "040_scaled_awr_bc_v1"})


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
        replay_labels=lambda replay: returns_lib.compact_policy_returns(
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
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        exp.validate_config(_cfg(batch_size=0))
    with pytest.raises(ValueError, match="beyond sample_chunk_length"):
        exp.validate_config(_cfg(sample_chunk_length=20))
    with pytest.raises(ValueError, match="dense offset prefix"):
        exp.validate_config(_cfg(head_offsets=(1, 2, 3, 4, 5, 7, 8)))
    with pytest.raises(ValueError, match="divisible"):
        exp.validate_config(_cfg(target_loss_positions=17))
    with pytest.raises(ValueError, match="layer_rms_every"):
        exp.validate_config(_cfg(layer_rms_every=-1))
    with pytest.raises(ValueError, match=r"num_workers must be an integer in \[0, 32\]"):
        exp.validate_config(_cfg(num_workers=33))
    assert asdict(exp.TrainConfig())["target_loss_positions"] == 2**35
    with pytest.raises(ValueError, match="frozen treatment"):
        exp.validate_production_config(exp.TrainConfig(exec_horizon=6))
    exp.validate_production_config(
        exp.TrainConfig(
            num_workers=8,
            cache_limit_gb=512,
            push_to_r2=False,
            gradient_hist_every=128,
            weight_hist_every=256,
            layer_rms_every=0,
            layer_rms_batch_size=4,
            prefetch_batches=4,
        )
    )


def test_production_loader_and_eval_defaults() -> None:
    cfg = exp.TrainConfig()

    assert cfg.windows_per_replay == 2
    assert cfg.num_workers == 32
    assert cfg.prefetch_batches == 8
    worker_prefetch, batch_prefetch = exp._loader_prefetch_depths(cfg)
    assert (worker_prefetch, batch_prefetch) == (8, 7)
    queued_worker_samples = cfg.num_workers * worker_prefetch * cfg.windows_per_replay
    queued_batch_samples = batch_prefetch * cfg.batch_size
    assert queued_worker_samples + queued_batch_samples == cfg.prefetch_batches * cfg.batch_size
    assert cfg.eval_every == 2**14
    assert cfg.eval_max_parallel == 32
    assert exp._eval_parallelism(cfg, 96) == 32
    assert exp._eval_inference_bucket(cfg, 96) == 32
    assert len(cfg.source_names) == len(cfg.source_weights) == 44
    assert dict(zip(cfg.source_names, cfg.source_weights, strict=True)) == {
        name: 2.0 if name == "professional-zain-policy-world-v7" else 1.0 for name in cfg.source_names
    }


def test_histogram_cadence_does_not_restart_on_resume() -> None:
    cfg = exp.TrainConfig()
    assert cfg.gradient_hist_every == 2**12
    assert cfg.weight_hist_every == 2**11
    assert exp.histogram_due(1, 4096)
    assert exp.histogram_due(4096, 4096)
    assert not exp.histogram_due(1001, 4096)


def test_wandb_watch_logs_gradient_histograms(monkeypatch: pytest.MonkeyPatch) -> None:
    model = torch.nn.Linear(2, 2)
    calls: list[tuple[torch.nn.Module, dict[str, object]]] = []
    monkeypatch.setattr(exp.wandb, "run", object())
    monkeypatch.setattr(exp.wandb, "watch", lambda watched, **kwargs: calls.append((watched, kwargs)))

    exp._watch_gradients(model, _cfg(gradient_hist_every=17))
    exp._watch_gradients(model, _cfg(gradient_hist_every=0))

    assert calls == [(model, {"log": "gradients", "log_freq": 17, "log_graph": False})]


def test_weight_histograms_use_their_own_cadence(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(weight_hist_every=2**11, layer_rms_every=0)
    model = exp.GPT(cfg)
    batch = _awr_batch(cfg)
    monkeypatch.setattr(exp, "wandb_weight_log", lambda _model: {"weights/test": "histogram"})

    assert exp._training_diagnostics(model, batch, cfg, 2**11) == {"weights/test": "histogram"}
    assert exp._training_diagnostics(model, batch, cfg, 2**11 + 1) == {}


def test_layer_rms_diagnostics_cover_every_residual_block() -> None:
    cfg = _cfg(n_layers=2, temporal_layers=2)
    model = exp.GPT(cfg)
    batch = _awr_batch(cfg)

    activations = exp.layer_activation_rms_log(model, batch, cfg, max_rows=1)
    layer_names = {
        "trunk_block_00",
        "trunk_block_01",
        "temporal_block_00",
        "temporal_block_01",
    }
    assert set(activations) == {
        f"{metric}/{name}"
        for metric in ("activation_rms", "residual_branch_rms", "residual_ratio")
        for name in layer_names
    }
    assert all(math.isfinite(value) and value >= 0 for value in activations.values())
    for name in layer_names:
        expected_ratio = activations[f"residual_branch_rms/{name}"] / activations[f"activation_rms/{name}"]
        assert activations[f"residual_ratio/{name}"] == pytest.approx(expected_ratio)


def test_temporal_nll_from_existing_logits_matches_direct_path() -> None:
    cfg = _cfg()
    model = exp.GPT(cfg)
    batch = _awr_batch(cfg)
    history, targets, _ = exp.prepared_targets(model, batch)
    hidden = model.forward_dense(batch.context.features, batch.context.ctx_pad, history)

    logits = model.temporal.teacher_forced_logits_by_group(hidden, history, targets)
    reused = model.temporal.nll_from_logits(logits, targets)
    direct = model.temporal.teacher_forced_nll(hidden, history, targets)

    torch.testing.assert_close(reused, direct)


def test_eval_prewarms_before_starting_dolphin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = _cfg(inference_mode="eager", eval_max_parallel=4)
    model = exp.GPT(cfg)
    engine = exp.BF16Inference(model, cfg, compiled=False)
    events: list[str] = []

    def prewarm(rows: int, horizon: int) -> float:
        events.append(f"prewarm:{rows}:{horizon}")
        return 1.25

    def sweep(*args, **kwargs):
        events.append("sweep")
        assert not model.training
        return [], []

    monkeypatch.setattr(engine, "prewarm", prewarm)
    monkeypatch.setattr(exp, "sweep_vs_cpu_prior_with_rows", sweep)
    monkeypatch.setattr(exp, "vs_cpu_metrics", lambda *args, **kwargs: {})
    metrics = exp.eval_vs_cpu(
        model,
        {},
        cfg,
        n_matchups=4,
        replay_dir=tmp_path,
        inference=engine,
    )

    assert events == ["prewarm:4:4", "sweep"]
    assert metrics["inference_compile_seconds"] == pytest.approx(1.25)
    assert model.training


def test_prewarm_uses_real_rows_before_bucket_padding(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(inference_mode="compiled")
    engine = exp.BF16Inference(exp.GPT(cfg), cfg, compiled=False, compiled_buckets=(8,))
    engine.compiled = True
    seen_rows: list[int] = []

    def decode(context: Context, horizon: int) -> None:
        del horizon
        seen_rows.append(context.ctx_pad.shape[0])

    monkeypatch.setattr(engine, "decode", decode)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)

    engine.prewarm(3, 4)

    assert seen_rows == [3, 3]
    assert engine._warmed == {(8, 4)}


def test_compiled_inference_routes_the_trunk_through_dense_sdpa(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(inference_mode="compiled")
    model = exp.GPT(cfg).eval()
    engine = exp.BF16Inference(model, cfg, compiled=False, compiled_buckets=(2,))
    engine.compiled = True
    calls: list[str] = []
    dense_forward = model.forward_dense

    def record_dense(features, pad, actions):
        calls.append("dense")
        return dense_forward(features, pad, actions)

    def forbidden(*args, **kwargs):
        raise AssertionError("compiled inference reached the model-default attention path")

    monkeypatch.setattr(model, "forward_dense", record_dense)
    monkeypatch.setattr(model, "forward", forbidden)
    compile_calls: list[dict[str, object]] = []

    def compile_once(fn, **kwargs):
        compile_calls.append(kwargs)
        return fn

    monkeypatch.setattr(torch, "compile", compile_once)
    context = exp.synthetic_context(cfg, 2, torch.device("cpu"))
    observed = model.codec.quantize(exp.stack_actions(context.features))

    output = engine._trunk(2)(context.features, context.ctx_pad, observed)

    assert output.shape == (2, cfg.arch.L_ctx, cfg.arch.d_model)
    assert calls == ["dense"]
    assert compile_calls == [{"dynamic": False, "fullgraph": True, "mode": "default"}]


def test_eager_inference_also_routes_the_trunk_through_dense_sdpa(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(inference_mode="eager")
    model = exp.GPT(cfg).eval()
    engine = exp.BF16Inference(model, cfg, compiled=False, compiled_buckets=(2,))
    calls: list[str] = []
    dense_forward = model.forward_dense

    def record_dense(features, pad, actions):
        calls.append("dense")
        return dense_forward(features, pad, actions)

    def forbidden(*args, **kwargs):
        raise AssertionError("eager inference reached the training attention path")

    monkeypatch.setattr(model, "forward_dense", record_dense)
    monkeypatch.setattr(model, "forward", forbidden)
    context = exp.synthetic_context(cfg, 2, torch.device("cpu"))
    observed = model.codec.quantize(exp.stack_actions(context.features))

    output = engine._trunk(2)(context.features, context.ctx_pad, observed)

    assert output.shape == (2, cfg.arch.L_ctx, cfg.arch.d_model)
    assert calls == ["dense"]


def test_tiny_smoke_trains_checkpoints_and_resumes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = _cfg(
        gradient_hist_every=0,
        weight_hist_every=0,
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
    assert first["actual_loss_positions"] == 7
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
    assert second["actual_loss_positions"] == 14
    assert (tmp_path / "boundary-step-0000002.pt").is_file()
    assert (tmp_path / "smoke-final.pt").is_file()


# --- projectile encoder ------------------------------------------------------


def test_model_includes_the_projectile_modules() -> None:
    cfg = _cfg()
    model = exp.GPT(cfg)

    item_keys = {name for name in model.state_dict() if name.startswith("item_")}
    assert item_keys == {"item_type_emb.weight", "item_state_emb.weight", "item_up.weight", "item_down.weight"}
    assert model.item_type_emb.weight.shape == (256, cfg.arch.item_type_dim)
    assert model.item_state_emb.weight.shape == (256, cfg.arch.item_state_dim)
    # type + state embeddings, four floats, four sidecars, one presence flag.
    slot_width = cfg.arch.item_type_dim + cfg.arch.item_state_dim + 2 * len(_ITEM_FLOATS) + 1
    assert model.item_up.weight.shape == (cfg.arch.item_hidden_dim, slot_width)
    assert model.item_down.weight.shape == (cfg.arch.item_dim, cfg.arch.item_hidden_dim)


def test_optimizer_and_counts_place_the_item_modules() -> None:
    cfg = _cfg()
    model = exp.GPT(cfg)
    optimizer = exp.make_optimizer(model, cfg)

    groups = {id(parameter): group for group in optimizer.param_groups for parameter in group["params"]}
    tables = (model.item_type_emb.weight, model.item_state_emb.weight)
    linears = (model.item_up.weight, model.item_down.weight)
    assert all(groups[id(w)]["weight_decay"] == 0.0 for w in tables)
    assert all(groups[id(w)]["weight_decay"] == cfg.adam_weight_decay for w in linears)
    assert all(not groups[id(w)]["use_muon"] for w in tables + linears)

    # The item encoder sits outside the trunk, the temporal chain, and the group heads.
    item_parameters = sum(w.numel() for w in tables + linears)
    assert exp.subsystem_parameter_counts(model)["other"] > item_parameters


# --- the pooled set encoder ---------------------------------------------------


def _item_columns(items: Mapping[int, Mapping[str, float]], *, masks: bool = True) -> dict[str, Tensor]:
    """Model-ready item columns for a ``[2, 3]`` batch: ``items`` maps a slot to its
    live projectile, and every other slot is the preprocessed empty form (id 0, zeroed
    floats, sidecar 1.0)."""
    shape = (2, 3)
    features: dict[str, Tensor] = {}
    for slot in range(ITEM_SLOTS):
        item = items.get(slot)
        for name in _ITEM_CATS:
            value = 0 if item is None else int(item[name])
            features[item_column(slot, name)] = torch.full(shape, value, dtype=torch.long)
        for name in _ITEM_FLOATS:
            column = item_column(slot, name)
            features[column] = torch.full(shape, 0.0 if item is None else float(item[name]))
            if masks:
                features[f"{column}_mask"] = torch.full(shape, 1.0 if item is None else 0.0)
    return features


_LASER = {"type": 6, "state": 2, "pos_x": 12.5, "pos_y": -3.25, "vel_x": 1.5, "vel_y": 0.0}
_TURNIP = {"type": 210, "state": 4, "pos_x": -40.0, "pos_y": 18.75, "vel_x": -0.5, "vel_y": 2.25}


def _pooled(model: torch.nn.Module, items: Mapping[int, Mapping[str, float]], *, masks: bool = True) -> Tensor:
    with torch.no_grad():
        return model._item_features(_item_columns(items, masks=masks))


def test_pooling_is_permutation_invariant_over_the_slots() -> None:
    """A slot holds its item until an OLDER item despawns, so live items shift slots
    mid-match. The pooled value must not move with them."""
    torch.manual_seed(0)
    model = exp.GPT(_cfg()).eval()

    first = _pooled(model, {0: _LASER, 1: _TURNIP})
    permuted = _pooled(model, {3: _TURNIP, 1: _LASER})
    one_item = _pooled(model, {2: _LASER})

    assert first.shape == (2, 3, model.cfg.arch.item_dim)
    assert torch.allclose(first, permuted, atol=1e-6)
    # Non-vacuity: the pooled value does depend on the set, just not on the slots.
    assert not torch.allclose(first, one_item, atol=1e-4)


def test_an_empty_frame_pools_to_exactly_zero() -> None:
    torch.manual_seed(0)
    model = exp.GPT(_cfg()).eval()

    pooled = _pooled(model, {})

    assert torch.equal(pooled, torch.zeros_like(pooled))


def test_a_missing_sidecar_reads_as_a_live_slot() -> None:
    """``preprocess`` emits ``{name}_mask`` only where a mask fires, so an absent
    sidecar means zero — a slot that holds an item."""
    torch.manual_seed(0)
    model = exp.GPT(_cfg()).eval()

    full = _pooled(model, {0: _LASER, 1: _TURNIP, 2: _LASER, 3: _TURNIP})
    without_sidecars = _pooled(model, {0: _LASER, 1: _TURNIP, 2: _LASER, 3: _TURNIP}, masks=False)

    assert torch.equal(full, without_sidecars)


def test_the_type_clamp_lands_unknown_ids_on_the_last_row() -> None:
    """The stored type is peppi's raw u16 id, which the routed table does not cover."""
    torch.manual_seed(0)
    model = exp.GPT(_cfg()).eval()

    unknown = _pooled(model, {0: {**_LASER, "type": 70_000}})
    last_row = _pooled(model, {0: {**_LASER, "type": model.item_type_emb.num_embeddings - 1}})

    assert torch.equal(unknown, last_row)


def test_context_tokens_accept_the_synthetic_item_context() -> None:
    cfg = _cfg()
    model = exp.GPT(cfg).eval()
    ctx = exp.synthetic_context(cfg, 2, torch.device("cpu"))

    sidecars = {f"{item_column(slot, name)}_mask" for slot in range(ITEM_SLOTS) for name in ITEM_COLUMNS.floats}
    assert set(ctx.features) >= ITEM_INPUT_COLUMNS | sidecars
    with torch.no_grad():
        tokens = model.context_tokens(ctx.features)
    assert tokens.shape == (2, cfg.arch.L_ctx, cfg.arch.d_model)


def test_decode_asks_for_the_item_canonical_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """``canonical_context`` does not self-gate: a decode that forgets ``items`` would
    compile a second program on a different key set."""
    cfg = _cfg()
    model = exp.GPT(cfg).eval()
    seen: list[bool] = []
    real = exp.canonical_context

    def spy(ctx: Context, observation_bundle: str, *, items: bool = False) -> Context:
        seen.append(items)
        return real(ctx, observation_bundle, items=items)

    monkeypatch.setattr(exp, "canonical_context", spy)
    with torch.no_grad():
        chunk = exp.BF16Inference(model, cfg).decode(exp.synthetic_context(cfg, 1, torch.device("cpu")), 4)

    assert seen == [True]
    assert chunk.shape[1:] == (4, A_DIM)


# --- routing ------------------------------------------------------------------


def test_projectile_routing_feeds_both_observation_paths() -> None:
    cfg = _cfg()
    model = exp.GPT(cfg).eval()

    loader = exp.loader_kwargs(cfg, {})
    policy = exp.make_policy(model, {}, cfg, device="cpu")
    assert (loader["extra"], loader["projection"]) == (ITEM_COLUMNS, BASE_ITEMS_PROJECTION)
    assert (policy.extra, policy.projection) == (ITEM_COLUMNS, BASE_ITEMS_PROJECTION)


def test_the_shipped_config_reads_only_sources_that_store_items() -> None:
    """No other decoder emits the item block, so the shipped source set must be the
    policy-world one, and the train loader must read it with the policy-world format."""
    cfg = exp.TrainConfig()

    assert set(cfg.source_names) == exp._POLICY_WORLD_NAMES
    assert exp.loader_kwargs(cfg, {})["projection"] is BASE_ITEMS_PROJECTION


def test_projectile_model_rejects_a_source_that_drops_items() -> None:
    compact = "ranked-anonymized-1-policy-v7"
    assert compact in exp.streams.BY_NAME and compact not in exp._POLICY_WORLD_NAMES

    with pytest.raises(ValueError, match="policy-world sources"):
        exp.validate_config(_cfg(source_names=(compact,), source_weights=(1.0,)))


def test_a_batch_without_item_columns_fails_loud() -> None:
    """A model handed an item-less observation must name the requirement rather
    than raise a bare KeyError deep inside the encoder."""
    cfg = _cfg()
    model = exp.GPT(cfg).eval()
    features = exp.synthetic_context(cfg, 2, torch.device("cpu")).features
    item_less = {name: value for name, value in features.items() if not name.startswith("item")}

    assert not any(name.startswith("item") for name in item_less)
    with pytest.raises(ValueError, match="policy-world"):
        model.context_tokens(item_less)


def _sliced_replay(source: Mapping[str, object], start: int, length: int) -> dict[str, object]:
    """One shorter replay: every per-frame column is cut, constants pass through."""
    frames = len(np.asarray(source["frame"]))
    out: dict[str, object] = {}
    for name, value in source.items():
        array = np.asarray(value)
        out[name] = array[start : start + length] if array.ndim == 1 and array.shape[0] == frames else value
    return out


def test_a_real_policy_world_window_reaches_context_tokens(tmp_path: Path) -> None:
    """End to end on real data: a v7 replay with live projectiles, encoded to the
    policy-world format, read back through ``loader_kwargs``'s routing, and forwarded."""
    if not _V7_TRAIN.is_dir():
        pytest.skip("local v7 subset is not available")
    cfg = _cfg()
    source = dict(StreamingDataset(local=str(_V7_TRAIN), batch_size=1, shuffle=False)[0])
    live = ~np.isnan(np.asarray(source[item_column(0, "pos_x")], dtype=np.float64))
    # The record is real data, so it decides the fixture: this asserts nothing without a
    # live projectile, and the window sampler needs a replay at least one window long.
    length = min(64, len(live))
    if not live.any() or length < cfg.arch.L_ctx + cfg.arch.sample_chunk_length:
        pytest.skip("the sampled replay is too short or carries no slot-0 projectile")
    start = max(0, min(int(np.flatnonzero(live)[0]), len(live) - length))
    replay = _sliced_replay(source, start, length)
    if np.count_nonzero(~np.isnan(np.asarray(replay[item_column(0, "pos_x")], dtype=np.float64))) < 8:
        pytest.skip("the sliced window carries too few live projectile frames")

    encoded = encode_policy_world_replay(replay, "items-0")
    with MDSWriter(out=str(tmp_path / "train"), columns=POLICY_WORLD_MDS_COLUMNS, compression="zstd") as writer:
        writer.write(encoded)

    kwargs = exp.loader_kwargs(cfg, _parity_stats())
    kwargs |= {
        "data_root": str(tmp_path),
        "sources": None,
        "source_weights": None,
        "remote": None,
        "batch_size": 1,
    }
    batch = next(
        iter(make_loader(split="train", num_workers=0, windows_per_replay=4, replay_format="policy-world", **kwargs))
    )
    features = batch.context.features

    assert set(features) >= ITEM_INPUT_COLUMNS
    model = exp.GPT(cfg).eval()
    with torch.no_grad():
        tokens = model.context_tokens(features)
    assert tokens.shape == (1, cfg.arch.L_ctx, cfg.arch.d_model)
    assert torch.isfinite(tokens).all()


# --- offline / online parity for one projectile frame -------------------------


def _post(side: int) -> dict[str, object]:
    return {
        "position": {"x": 42.0 - 11.0 * side, "y": -7.5 * side},
        "direction": 1.0 - 2.0 * side,
        "percent": 31.0 + side,
        "shield": 47.5,
        "stock": 3,
        "action": 14 + side,
        "jumps_used": side,
        "airborne": side,
        "hurtbox_state": 1,
        "hitlag_left": 2.0,
    }


def _obs_with_two_items() -> dict[str, object]:
    """One canonical closed-loop frame carrying two live projectiles. Spawn ids are
    ascending, so the laser takes slot 0 and the turnip slot 1."""
    items = [
        {
            "id": 3,
            "type": _LASER["type"],
            "state": _LASER["state"],
            "position": {"x": _LASER["pos_x"], "y": _LASER["pos_y"]},
            "velocity": {"x": _LASER["vel_x"], "y": _LASER["vel_y"]},
            "owner": 0,
        },
        {
            "id": 7,
            "type": _TURNIP["type"],
            "state": _TURNIP["state"],
            "position": {"x": _TURNIP["pos_x"], "y": _TURNIP["pos_y"]},
            "velocity": {"x": _TURNIP["vel_x"], "y": _TURNIP["vel_y"]},
            "owner": 1,
        },
    ]
    return {
        "id": 400,
        "ports": {
            EGO_PORT: {"leader": {"post": _post(0)}, "follower": None},
            OPP_PORT: {"leader": {"post": _post(1)}, "follower": None},
        },
        "items": items,
        "stage": STAGE,
        "_matchup": {"stage": STAGE, "character": {EGO_PORT: 1, OPP_PORT: 22}},
    }


def _parity_stats() -> dict[str, FeatureStats]:
    """Asymmetric stats so standardize and min-max both do real work."""
    rng = np.random.default_rng(41)
    names = ["position_x", "position_y", "percent", "shield", "direction", "hitlag_left"]
    keys = names + [f"nana_{name}" for name in names] + [f"item_{name}" for name in _ITEM_FLOATS]
    out: dict[str, FeatureStats] = {}
    for key in keys:
        low = float(rng.normal(-50, 10))
        out[key] = FeatureStats(
            mean=float(rng.normal(0, 20)),
            std=float(abs(rng.normal(0, 10)) + 0.5),
            min=low,
            max=low + float(abs(rng.normal(0, 60)) + 1.0),
        )
    return out


def _offline_batch(flat: Mapping[str, float], action: np.ndarray) -> dict[str, np.ndarray]:
    """The same frame in the OFFLINE MDS dtypes: item ids are int32 with ``MASK_INT32``
    in an empty slot, while the closed loop hands every item field over as a float NaN
    sentinel column."""
    out: dict[str, np.ndarray] = {}
    for key, value in flat.items():
        if key == "frame":
            continue
        if key in _ITEM_CAT_COLUMNS:
            out[key] = np.array([MASK_INT32 if math.isnan(value) else int(value)], dtype=np.int32)
        elif isinstance(value, int):
            out[key] = np.array([value], dtype=np.int32)
        else:
            out[key] = np.array([value], dtype=np.float32)
    for index, channel in enumerate(ACTION_CHANNELS):
        name = f"{EGO_PREFIX}_{channel}"
        if channel.startswith("button_"):
            out[name] = np.array([action[index] > 0.5], dtype=np.int32)
        else:
            out[name] = np.array([action[index]], dtype=np.float32)
    return relabel_ego(out, EGO_PREFIX)


def test_ring_item_rows_match_preprocess_on_the_offline_dtypes() -> None:
    """One two-projectile frame through the closed-loop ring builder reproduces
    ``preprocess`` on the equivalent MDS-dtype arrays, tensor for tensor."""
    cfg = _cfg()
    extra, projection = ITEM_COLUMNS, BASE_ITEMS_PROJECTION
    stats = _parity_stats()
    flat = flatten_canonical_frame(_obs_with_two_items())
    action = np.linspace(-1.0, 1.0, A_DIM, dtype=np.float32)

    layout = _build_layout(flat, EGO_PREFIX, stats, extra, projection)
    rings = _Rings(layout, cfg.arch.L_ctx)
    rings.gather(flat, action)
    rings.push(None)
    newest = rings.window(1)
    values = dict(zip(layout.value_names, rings.values[:, newest][:, 0], strict=True))
    cats = dict(zip(layout.cat_names, rings.cats[:, newest][:, 0], strict=True))
    masks = dict(zip(layout.mask_names, rings.masks[:, newest][:, 0], strict=True))

    reference = preprocess(_offline_batch(flat, action), stats, extra=extra, projection=projection)

    assert set(values) | set(cats) == {name for name in reference if not name.endswith("_mask")}
    assert set(values) | set(cats) >= ITEM_INPUT_COLUMNS
    for name, value in values.items():
        assert float(value) == pytest.approx(float(reference[name][0]), rel=1e-6, abs=1e-6), name
    for name, value in cats.items():
        assert int(value) == int(reference[name][0]), name
    for name, value in masks.items():
        expected = reference.get(name)
        assert float(value) == (0.0 if expected is None else float(expected[0])), name

    # The frame is not degenerate: two live slots, two empty ones.
    assert int(cats[item_column(0, "type")]) == _LASER["type"]
    assert int(cats[item_column(1, "type")]) == _TURNIP["type"]
    assert int(cats[item_column(2, "type")]) == 0
    for slot in range(ITEM_SLOTS):
        empty = 1.0 if slot >= 2 else 0.0
        assert masks[f"{item_column(slot, 'pos_x')}_mask"] == empty
    assert values[item_column(0, "pos_x")] != 0.0
    assert all(values[item_column(slot, name)] == 0.0 for slot in (2, 3) for name in _ITEM_FLOATS)
    assert [name for name in values if name.endswith("_owner")] == []


# --- configuration ------------------------------------------------------------


def test_config_round_trips_and_rejects_other_checkpoint_shapes() -> None:
    cfg = _cfg()
    values = exp._checkpoint_config(cfg)
    item_fields = {"item_type_dim", "item_state_dim", "item_hidden_dim", "item_dim"}

    assert item_fields <= values["architecture"].keys()
    assert exp.config_from_state(values) == cfg

    incomplete = dict(values)
    incomplete["architecture"] = {
        name: value for name, value in values["architecture"].items() if name not in item_fields
    }
    with pytest.raises(ValueError, match="architecture"):
        exp.config_from_state(incomplete)
    with pytest.raises(ValueError, match="unexpected"):
        exp.config_from_state({**values, "wandb_hist_every": 17})


def test_validate_config_rejects_non_positive_item_dims() -> None:
    exp.validate_config(_cfg())
    for name in ("item_type_dim", "item_state_dim", "item_hidden_dim", "item_dim"):
        with pytest.raises(ValueError, match=name):
            exp.validate_config(_cfg(**{name: 0}))


def test_model_tag_records_projectiles() -> None:
    assert exp.model_tag(_cfg()).startswith("scaled040-")
    assert "-projectiles-awr-near-" in exp.model_tag(_cfg())


def test_the_item_dims_are_frozen() -> None:
    with pytest.raises(ValueError, match="item_dim"):
        exp.validate_production_config(exp.TrainConfig(arch=replace(exp.ARCHITECTURE, item_dim=8)))


def test_gradients_are_clipped_to_the_frozen_norm() -> None:
    """A gradient spike must not reach the optimizer unscaled.

    Muon orthogonalizes its own updates, but the aux AdamW groups are not
    scale-bounded, so an unclipped spike can destroy the model in one step.
    """
    cfg = exp.TrainConfig()
    assert cfg.grad_clip == 1.0

    with pytest.raises(ValueError, match="grad_clip must be finite and positive"):
        exp.validate_config(_cfg(grad_clip=0.0))
    with pytest.raises(ValueError, match="grad_clip must be finite and positive"):
        exp.validate_config(_cfg(grad_clip=math.inf))
    with pytest.raises(ValueError, match="frozen treatment"):
        exp.validate_production_config(exp.TrainConfig(grad_clip=5.0))

    parameter = torch.nn.Parameter(torch.zeros(4))
    parameter.grad = torch.full((4,), 25.0)
    pre_clip = torch.nn.utils.clip_grad_norm_([parameter], cfg.grad_clip)

    assert pre_clip.item() == pytest.approx(50.0)
    assert parameter.grad.norm().item() == pytest.approx(cfg.grad_clip)
