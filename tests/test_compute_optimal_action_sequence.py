"""Frozen scientific and distributed contracts for experiment 060."""

import copy
import importlib.util
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from hal.data.feature_stats import FeatureStats
from hal.data.feature_stats import consolidate_key

_SPEC = importlib.util.spec_from_file_location(
    "hal_experiment_060",
    Path(__file__).parents[1] / "experiments" / "060_compute_optimal_action_sequence.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

_COMPARE_SPEC = importlib.util.spec_from_file_location(
    "hal_compare_060_resume", Path(__file__).parents[1] / "scripts" / "compare_060_resume.py"
)
assert _COMPARE_SPEC is not None and _COMPARE_SPEC.loader is not None
_COMPARE = importlib.util.module_from_spec(_COMPARE_SPEC)
sys.modules[_COMPARE_SPEC.name] = _COMPARE
_COMPARE_SPEC.loader.exec_module(_COMPARE)


def test_production_architecture_and_parameter_contract() -> None:
    cfg = _MODULE.TrainConfig()
    assert cfg.arch.d_model == 1_024
    assert cfg.arch.n_heads == 16
    assert cfg.arch.d_model // cfg.arch.n_heads == 64
    assert cfg.arch.d_model.bit_count() == 1
    assert cfg.arch.n_layers == 6
    assert cfg.arch.temporal_d_model == 1_024
    assert cfg.arch.temporal_layers == 8
    assert cfg.arch.temporal_heads == 16
    assert cfg.arch.return_embed_dim == 0
    assert not cfg.return_conditioning
    assert cfg.arch.head_offsets == tuple(range(1, 31))
    assert cfg.arch.sample_chunk_length == 30
    assert cfg.arch.main_stick_layout == "legacy65"
    with torch.device("meta"):
        model = _MODULE.make_model(cfg)
    counts = _MODULE.subsystem_parameter_counts(model)
    assert counts["return_conditioner"] == 0
    assert counts["total"] == 194_554_777
    assert _MODULE.compute_equivalent_parameter_count(cfg, counts) == 3_989_158_041


def test_proxy_arms_differ_only_by_the_state_head() -> None:
    state = _MODULE.proxy_config_for_arm("state")
    action_only = _MODULE.proxy_config_for_arm("action_only")
    assert replace(action_only, state_prediction=True) == state
    assert state.state_prediction and not action_only.state_prediction
    assert (state.arch.d_model, state.arch.n_layers, state.arch.temporal_layers) == (512, 6, 8)
    assert (state.world_size, state.local_batch_size, state.microbatch_size, state.batch_size) == (1, 512, 512, 512)
    assert state.source_names == ("ranked-anonymized-1-policy-world-v8",)
    assert state.val_n_samples == 1_024
    assert state.target_positions == 2**28
    assert state.max_steps == 16_384
    assert state.minimum_replay_gap_batches == 104
    assert _MODULE.TrainConfig().minimum_replay_gap_batches == 232
    for cfg in (state, action_only):
        _MODULE.validate_config(cfg)
    with torch.device("meta"):
        state_counts = _MODULE.subsystem_parameter_counts(_MODULE.make_model(state))
        action_counts = _MODULE.subsystem_parameter_counts(_MODULE.make_model(action_only))
    assert state_counts["total"] == 50_009_753
    assert action_counts["total"] == 50_009_753 - 664_316
    assert action_counts["state_head"] == 0
    assert {name: value for name, value in state_counts.items() if name not in ("state_head", "total")} == {
        name: value for name, value in action_counts.items() if name not in ("state_head", "total")
    }


def test_proxy_schedule_decays_over_the_last_quarter() -> None:
    cfg = _MODULE.proxy_config()
    assert (cfg.decay_start_update, cfg.decay_duration) == (12_288, 4_096)
    assert cfg.decay_start_update + cfg.decay_duration == cfg.max_steps
    schedule = _MODULE.lr_schedule(cfg)
    assert schedule(cfg.decay_start_update - 1) == pytest.approx(1.0)
    assert schedule(cfg.max_steps - 1) == pytest.approx(cfg.lr_floor_ratio)
    with pytest.raises(ValueError, match="proxy training boundaries"):
        _MODULE.validate_config(replace(cfg, decay_start_update=None, decay_duration=None))
    with pytest.raises(ValueError, match="one rank, batch 512"):
        _MODULE.validate_config(replace(cfg, microbatch_size=256))
    with pytest.raises(ValueError, match="1,024 validation windows"):
        _MODULE.validate_config(replace(cfg, val_n_samples=2_048))
    with pytest.raises(ValueError, match="one rank, batch 512"):
        _MODULE.validate_config(replace(cfg, source_names=_MODULE.TrainConfig().source_names))


def test_proxy_arms_share_the_action_path() -> None:
    cfg = _small_config()
    torch.manual_seed(3)
    state_model = _MODULE.make_model(cfg)
    torch.manual_seed(3)
    action_model = _MODULE.make_model(replace(cfg, state_prediction=False))
    assert action_model.state_head is None
    action_parameters = dict(action_model.named_parameters())
    for name, parameter in state_model.named_parameters():
        if name.startswith("state_head."):
            continue
        torch.testing.assert_close(parameter, action_parameters.pop(name), rtol=0, atol=0)
    assert not action_parameters
    batch = _MODULE.synthetic_awr_batch(cfg, torch.device("cpu"))
    positions = torch.tensor([[4, 7], [5, 6]])
    results = []
    for model, model_cfg in ((state_model, cfg), (action_model, replace(cfg, state_prediction=False))):
        trunk, temporal = _MODULE._training_functions(model, model_cfg)
        results.append(
            _MODULE.microbatch_loss(
                model,
                batch,
                model_cfg,
                step=0,
                valid_prefixes=4,
                trunk_fn=trunk,
                temporal_fn=temporal,
                prefix_positions=positions,
            )
        )
    (state_loss, state_nll, state_metrics), (action_loss, action_nll, action_metrics) = results
    torch.testing.assert_close(state_nll, action_nll, rtol=0, atol=0)
    torch.testing.assert_close(state_loss - state_metrics["state/loss"], action_loss)
    assert not any(name.startswith("state/") for name in action_metrics)
    assert "state_action_acc" not in _MODULE.val_metrics(action_model, [batch], replace(cfg, state_prediction=False))


def test_compute_curve_weights_schedule_and_batch_arithmetic() -> None:
    cfg = _MODULE.TrainConfig()
    _MODULE.validate_config(cfg)
    assert _MODULE.OFFSET_LOSS_WEIGHTS == (1 / 30,) * 30
    assert _MODULE.compute_flops_per_supervised_position(3_989_158_041) == 23_934_948_246
    assert cfg.batch_size == cfg.world_size * cfg.local_batch_size == 512
    assert cfg.microbatch_size == cfg.local_batch_size == 256
    assert cfg.supervised_positions_per_update == cfg.policy_prefixes_per_update == 16_384
    assert cfg.value_prefixes_per_update == 65_536
    assert cfg.max_steps == 714_752
    assert cfg.max_steps * cfg.supervised_positions_per_update == 11_710_496_768
    assert cfg.warmup_steps == 4_096
    assert cfg.decay_start_update == 536_576
    assert cfg.decay_duration == 178_176
    assert cfg.decay_start_update + cfg.decay_duration == cfg.max_steps
    assert (
        _MODULE.checkpoint_rounded_updates(
            _MODULE.FITTED_OPTIMAL_POSITIONS,
            positions_per_update=cfg.supervised_positions_per_update,
            checkpoint_interval=cfg.ckpt_every,
        )
        == cfg.max_steps
    )
    schedule = _MODULE.lr_schedule(cfg)
    assert schedule(cfg.warmup_steps - 1) == pytest.approx(1.0)
    assert schedule(cfg.decay_start_update - 1) == pytest.approx(1.0)
    assert schedule(cfg.max_steps - 1) == pytest.approx(cfg.lr_floor_ratio)


def test_microbatch_fallback_preserves_the_global_optimizer_batch() -> None:
    cfg = replace(_MODULE.TrainConfig(), microbatch_size=128)
    _MODULE.validate_config(cfg)
    assert cfg.local_batch_size // cfg.microbatch_size == 2
    assert cfg.batch_size == 512
    assert cfg.max_steps == 714_752


def test_rank_partitions_are_disjoint_and_cover_every_source() -> None:
    cfg = _MODULE.TrainConfig()
    full = _MODULE.data_selection(cfg)
    rank_zero = _MODULE.data_selection(cfg, rank=0, world_size=2)
    rank_one = _MODULE.data_selection(cfg, rank=1, world_size=2)
    assert rank_zero.sha256 != rank_one.sha256
    assert _MODULE.data_partition_hashes(cfg) == (rank_zero.sha256, rank_one.sha256)
    assert rank_zero.row_count + rank_one.row_count == full.row_count
    for complete, left, right in zip(full.sources, rank_zero.sources, rank_one.sources, strict=True):
        assert left.source == right.source == complete.source
        assert left.start == complete.start
        assert left.stop == right.start
        assert right.stop == complete.stop


def _rank_record(cfg: Any, rank: int) -> dict[str, object]:
    return {
        "rank": rank,
        "seed": _MODULE.rank_seed(cfg.seed, rank),
        "partition_sha256": _MODULE.data_partition_hashes(cfg)[rank],
        "loader": {"next_batch": rank},
        "identity_masker": {"state": rank},
        "return_masker": {"state": rank},
        "prefix_sampler": {"state": rank},
        "return_calibration": {"state": rank},
        "rng": {"state": rank},
    }


def test_resume_rejects_changed_distributed_geometry() -> None:
    cfg = _MODULE.TrainConfig()
    context = _MODULE.DistributedContext(0, 0, 2, torch.device("cpu"))
    ranks = [_rank_record(cfg, rank) for rank in range(cfg.world_size)]
    state = {
        "distributed": {
            "contract": _MODULE.distributed_training_contract(cfg),
            "ranks": ranks,
        }
    }
    assert _MODULE.rank_resume_state(state, cfg, context)["loader"] == {"next_batch": 0}
    changed = replace(cfg, microbatch_size=128)
    with pytest.raises(ValueError, match="world, batch, partition, architecture, offsets, or weights"):
        _MODULE.rank_resume_state(state, changed, context)


def _small_config() -> Any:
    return replace(
        _MODULE.TrainConfig(),
        arch=replace(
            _MODULE.Architecture(),
            d_model=32,
            n_layers=1,
            n_heads=4,
            L_ctx=8,
            temporal_d_model=32,
            temporal_layers=1,
            temporal_heads=4,
            temporal_ff_dim=64,
            group_head_dim=16,
            value_hidden_dim=16,
        ),
        local_batch_size=2,
        microbatch_size=2,
        prediction_frames=3,
        player_vocab_size=2,
        compile_trunk=False,
        compile_temporal=False,
    )


def _state_columns(frames: int) -> dict[str, np.ndarray]:
    values = {
        "position_x": (np.arange(frames) * 2.0, np.float32),
        "position_y": (np.arange(frames) * 4.0, np.float32),
        "percent": (np.arange(frames) * 10.0, np.float32),
        "shield": (np.full(frames, 60.0), np.float32),
        "direction": (np.where(np.arange(frames) % 2, 1.0, -1.0), np.float32),
        "airborne": (np.arange(frames) % 2, np.uint8),
        "action": (np.arange(frames) + 14, np.int32),
        "stock": (np.full(frames, 4), np.int32),
    }
    return {
        f"{player}_{name}": np.array([column], dtype=dtype)
        for player in ("ego", "opp")
        for name, (column, dtype) in values.items()
    }


_STATE_STATS = {
    "position_x": FeatureStats(mean=10, std=2, min=-100, max=100),
    "position_y": FeatureStats(mean=20, std=4, min=-100, max=100),
    "percent": FeatureStats(mean=50, std=10, min=0, max=500),
    "shield": FeatureStats(mean=50, std=5, min=0, max=60),
}


def test_state_labels_scale_changes_encode_binary_fields_and_mask_missing_values() -> None:
    columns = _state_columns(3)
    columns["ego_position_x"][0, 2] = np.nan
    columns["ego_action"][0, 2] = 2**31 - 1
    targets, valid = _MODULE.future_state_labels(columns, _STATE_STATS)
    assert targets.shape == (1, 3, 2, len(_MODULE.STATE_COLUMNS))
    assert targets[0, :, 0, :4].tolist() == [[0, 0, 0, 12], [1, 1, 1, 12], [0, 2, 2, 12]]
    assert targets[0, :, 0, 4].tolist() == [0, 1, 0]
    assert targets[0, :, 0, 5].tolist() == [0, 1, 0]
    assert targets[0, :2, 0, 6].tolist() == [14, 15]
    assert not valid[0, 2, 0, 0] and not valid[0, 2, 0, 6]
    assert valid[0, :, 1].all()
    assert torch.isfinite(targets).all()
    rare = {name: values.copy() for name, values in columns.items()}
    rare["ego_action"][0, 0] = 65_535
    rare["opp_action"][0, 1] = 511
    rare["ego_direction"][0, 1] = 0.0
    targets, valid = _MODULE.future_state_labels(rare, _STATE_STATS)
    # Engine action ids at or above 511 share the last class, and facing 0 is unlabeled.
    assert targets[0, 0, 0, 6] == targets[0, 1, 1, 6] == 511
    assert valid[0, 0, 0, 6] and valid[0, 1, 1, 6]
    assert not valid[0, 1, 0, 4] and valid[0, 0, 0, 4]
    for column, value, message in (
        ("ego_action", 65_536, r"unsupported action values \[65536\]"),
        ("ego_direction", 0.5, r"unsupported facing values \[0.5\]"),
        ("ego_airborne", 2, r"unsupported airborne values \[2\]"),
    ):
        changed = {name: values.copy() for name, values in columns.items()}
        changed[column][0, 0] = value
        with pytest.raises(ValueError, match=message):
            _MODULE.future_state_labels(changed, _STATE_STATS)
    columns["ego_action"] = columns["ego_action"].astype(np.float32)
    with pytest.raises(ValueError, match="integer ids"):
        _MODULE.future_state_labels(columns, _STATE_STATS)


def test_state_targets_are_changes_from_the_prefix_frame_within_one_stock() -> None:
    cfg = _small_config()
    batch = _MODULE.synthetic_awr_batch(cfg, torch.device("cpu"))
    frames = torch.arange(38).view(1, 38, 1, 1).expand_as(batch.state_targets).clone().float()
    frames[..., 7] = 4
    frames[1, 8:, 0, 7] = 3
    batch = replace(batch, state_targets=frames)
    targets, valid = _MODULE.aligned_state_targets(batch, torch.tensor([[4, 7], [5, 6]]), (1, 2, 3))
    assert targets.shape == (2, 2, 3, 2, _MODULE.STATE_FIELD_COUNT)
    assert targets[..., 0, 0].tolist() == [[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]]
    assert targets[..., 0, 6].tolist() == [[[5, 6, 7], [8, 9, 10]], [[6, 7, 8], [7, 8, 9]]]
    # Ego loses a stock at frame 8 in row 1, so changes that cross it are excluded.
    assert valid[1, 1, :, 0, :4].tolist() == [[True] * 4, [False] * 4, [False] * 4]
    assert valid[1, 1, :, 1].all() and valid[1, 1, :, 0, 4:].all() and valid[0].all()
    restored = _MODULE._return_batch_from_state(_MODULE._return_batch_state(batch))
    assert _MODULE._return_batch_sha256(restored) == _MODULE._return_batch_sha256(batch)
    assert torch.equal(restored.row_slice(1, 2).state_targets, frames[1:])


def test_training_and_validation_collation_use_the_same_future_states() -> None:
    cfg = _small_config()
    full_context = _MODULE.synthetic_context(replace(cfg, arch=replace(cfg.arch, L_ctx=38)), 2, torch.device("cpu"))
    columns = {name: value.numpy() for name, value in full_context.features.items() if not name.endswith("_mask")}
    columns["ctx_pad"] = np.zeros(2, dtype=np.int64)
    for name in ("ego_awr_return", "ego_return60"):
        columns[name] = np.zeros((2, 38), dtype=np.float32)
        columns[f"{name}_valid"] = np.ones((2, 38), dtype=bool)
    for player in ("ego", "opp"):
        columns[f"{player}_direction"][:] = 1
    columns["opp_position_x"][0, -1] = np.nan
    columns["ego_action"][1, -2] = 2**31 - 1
    stats = {consolidate_key(name): FeatureStats(mean=0, std=1, min=-1, max=1) for name in columns}
    training = _MODULE._collate_o60_batch(
        ("left", "right"),
        columns,
        stats=stats,
        projection=_MODULE.ITEM_PLAYER_PROJECTION,
        context_length=8,
        return_column="ego_awr_return",
        return_valid_column="ego_awr_return_valid",
    )
    windows = [{name: value[row] for name, value in columns.items()} for row in range(2)]
    validation = _MODULE.collate_awr_batch(windows, training.batch, L_ctx=8, stats=stats)
    for field in _MODULE.RETURN_BATCH_FIELDS:
        torch.testing.assert_close(getattr(training, field), getattr(validation, field), rtol=0, atol=0)
    _MODULE.validate_batch_geometry(training, cfg, 2)
    assert not training.state_valid[0, -1, 1, 0]
    assert not training.state_valid[1, -2, 0, _MODULE.STATE_COLUMNS.index("action")]


def test_state_objective_masks_missing_labels_and_accumulates_microbatches() -> None:
    torch.manual_seed(8)
    prediction = torch.randn(2, 2, 3, 2, _MODULE.STATE_OUTPUTS_PER_PLAYER, requires_grad=True)
    targets = torch.zeros(2, 2, 3, 2, _MODULE.STATE_FIELD_COUNT)
    targets[..., 6] = 7
    valid = torch.ones_like(targets, dtype=torch.bool)
    valid[0, 0, 0, 0] = False
    targets[0, 0, 0, 0] = float("nan")
    log_scale = torch.randn(3, 2, 4, requires_grad=True)
    whole, metrics = _MODULE.state_objective(prediction, targets, valid, log_scale, 4)
    pieces = [
        _MODULE.state_objective(prediction[i : i + 1], targets[i : i + 1], valid[i : i + 1], log_scale, 4)[0]
        for i in range(2)
    ]
    torch.testing.assert_close(whole, pieces[0] + pieces[1])
    assert set(metrics) == {
        "state/loss",
        "state/position_nll",
        "state/percent_nll",
        "state/shield_nll",
        "state/facing_nll",
        "state/airborne_nll",
        "state/action_nll",
    }
    whole.backward()
    assert torch.equal(prediction.grad[0, 0, 0, 0], torch.zeros(_MODULE.STATE_OUTPUTS_PER_PLAYER))
    assert prediction.grad[1].abs().sum() > 0
    assert log_scale.grad.abs().sum() > 0


def test_state_change_nll_is_gaussian_in_its_learned_scale() -> None:
    prediction = torch.zeros(1, 1, 1, 2, _MODULE.STATE_OUTPUTS_PER_PLAYER)
    targets = torch.zeros(1, 1, 1, 2, _MODULE.STATE_FIELD_COUNT)
    targets[..., 2] = 3.0
    valid = torch.zeros_like(targets, dtype=torch.bool)
    valid[..., 2] = True
    log_scale = torch.full((1, 2, 4), math.log(2.0))
    _, metrics = _MODULE.state_objective(prediction, targets, valid, log_scale, 1)
    expected = 0.5 * (3.0 / 2.0) ** 2 + math.log(2.0)
    torch.testing.assert_close(metrics["state/percent_nll"], torch.tensor(expected))
    assert metrics["state/position_nll"] == 0


def test_joint_training_gradients_and_action_only_decoder_parity() -> None:
    cfg = _small_config()
    model = _MODULE.make_model(cfg)
    batch = _MODULE.synthetic_awr_batch(cfg, torch.device("cpu"))
    positions = torch.tensor([[4, 7], [5, 6]])
    trunk, temporal = _MODULE._training_functions(model, cfg)
    loss, _, metrics = _MODULE.microbatch_loss(
        model,
        batch,
        cfg,
        step=0,
        valid_prefixes=4,
        trunk_fn=trunk,
        temporal_fn=temporal,
        prefix_positions=positions,
    )
    assert torch.isfinite(loss)
    assert metrics["state/loss"] > 0
    loss.backward()
    assert model.state_head.linear.weight.grad.abs().sum() > 0
    assert model.state_head.log_scale.grad.abs().sum() > 0
    assert model.temporal.blocks[0].qkv.weight.grad.abs().sum() > 0
    roles = _MODULE.optimizer_roles(model, cfg)
    assert roles["state_head.linear.weight"].lr_kind == "output"
    assert roles["state_head.linear.bias"].lr_kind == "vector"
    assert roles["state_head.log_scale"].lr_kind == "vector"

    history, targets, _ = _MODULE.prepared_targets(model, batch)
    local = positions - 4
    rows = torch.arange(2)[:, None]
    hidden = trunk(batch.context.features, batch.context.ctx_pad, None)
    args = (
        hidden,
        batch.context.ctx_pad,
        positions,
        history[rows, local],
        targets[rows, local],
        batch.future_return[rows, positions],
        batch.condition_present[rows, positions],
    )
    nll, diagnostics, states = model.temporal.teacher_forced_nll_with_states(*args)
    old_nll, old_diagnostics = model.temporal.teacher_forced_nll_with_diagnostics(*args)
    torch.testing.assert_close(nll, old_nll, rtol=0, atol=0)
    for name in diagnostics:
        torch.testing.assert_close(diagnostics[name], old_diagnostics[name], rtol=0, atol=0)
    torch.testing.assert_close(states, model.temporal.teacher_forced_states(*args), rtol=0, atol=0)
    # Future labels affect the auxiliary loss, never the decoder inputs or action NLL.
    changed = replace(batch, state_targets=batch.state_targets + 1)
    _, old_nll_sum, _ = _MODULE.microbatch_loss(
        model, batch, cfg, step=0, valid_prefixes=4, trunk_fn=trunk, temporal_fn=temporal, prefix_positions=positions
    )
    _, new_nll_sum, _ = _MODULE.microbatch_loss(
        model, changed, cfg, step=0, valid_prefixes=4, trunk_fn=trunk, temporal_fn=temporal, prefix_positions=positions
    )
    torch.testing.assert_close(old_nll_sum, new_nll_sum, rtol=0, atol=0)


def test_prediction_from_controller_chunk_and_exact_next_update_after_resume() -> None:
    cfg = _small_config()
    torch.manual_seed(18)
    model = _MODULE.make_model(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = _MODULE.synthetic_awr_batch(cfg, torch.device("cpu"))

    def update(model: Any, optimizer: Any, batch: Any) -> torch.Tensor:
        optimizer.zero_grad()
        positions = torch.tensor([[4, 7], [5, 6]])
        loss, _, _ = _MODULE.microbatch_loss(
            model,
            batch,
            cfg,
            step=0,
            valid_prefixes=4,
            trunk_fn=model.forward,
            temporal_fn=model.joint_teacher_forced_nll,
            prefix_positions=positions,
        )
        loss.backward()
        optimizer.step()
        return loss.detach()

    update(model, optimizer, batch)
    state = copy.deepcopy(model.state_dict())
    opt_state = copy.deepcopy(optimizer.state_dict())
    batch_state = _MODULE._return_batch_state(batch)
    expected_loss = update(model, optimizer, batch)
    resumed = _MODULE.make_model(cfg)
    resumed.load_state_dict(state)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    resumed_optimizer.load_state_dict(opt_state)
    restored_batch = _MODULE._return_batch_from_state(batch_state)
    torch.testing.assert_close(expected_loss, update(resumed, resumed_optimizer, restored_batch), rtol=0, atol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)
    actions = model.codec.quantize(batch.target)
    prediction = model.predict_chunk_states(batch.context, actions)
    assert prediction.shape == (2, 30, 2, _MODULE.STATE_FIELD_COUNT)
    assert torch.isfinite(prediction).all()
    assert torch.isin(prediction[..., 4], torch.tensor([-1.0, 1.0])).all()
    assert torch.isin(prediction[..., 5], torch.tensor([0.0, 1.0])).all()
    assert ((prediction[..., 6] >= 0) & (prediction[..., 6] < _MODULE.STATE_ACTION_CLASSES)).all()


def test_checkpoint_config_roundtrip_and_rejection_of_action_only_identity() -> None:
    cfg = _MODULE.TrainConfig()
    state = _MODULE._checkpoint_config(cfg)
    assert _MODULE.config_from_state(state) == cfg
    state["experiment_id"] = "060_compute_optimal_action_sequence_v2"
    with pytest.raises(ValueError, match="experiment_id"):
        _MODULE.config_from_state(state)
    with pytest.raises(ValueError, match="state_loss_weight"):
        _MODULE.validate_config(replace(cfg, state_loss_weight=float("nan")))


def test_validation_reports_state_errors_from_the_last_prefix() -> None:
    cfg = _small_config()
    model = _MODULE.make_model(cfg)
    batch = _MODULE.synthetic_awr_batch(cfg, torch.device("cpu"))
    _MODULE.validate_batch_geometry(batch, cfg, 2)
    metrics = _MODULE.val_metrics(model, [batch], cfg)
    assert model.training
    for name in _MODULE.STATE_CHANGE_FIELDS:
        assert metrics[f"state_{name}_change_rmse"] >= 0
    for name in ("state_facing_acc", "state_airborne_acc", "state_action_acc"):
        assert 0 <= metrics[name] <= 1
    assert "state_action_acc" in _MODULE._validation_wandb_metrics(metrics, cfg)
    with torch.no_grad():
        predictions = model.predict_chunk_states(batch.context, model.codec.quantize(batch.target))
    assert metrics["state_position_x_change_rmse"] == pytest.approx(float(predictions[..., 0].square().mean().sqrt()))
    valid = batch.state_valid.clone()
    valid[..., 2] = False
    with pytest.raises(ValueError, match="no observed labels"):
        _MODULE.val_metrics(model, [replace(batch, state_valid=valid)], cfg)


@pytest.mark.parametrize("version", ("v2", "v3"))
def test_resume_comparison_records_the_checkpoint_experiment_identity(tmp_path: Path, version: str) -> None:
    cfg = _MODULE.TrainConfig()
    identity = f"060_compute_optimal_action_sequence_{version}"
    state = {
        "step": 0,
        "cfg": {"experiment_id": identity},
        "provenance": {"git_sha": "a" * 40},
        "distributed": {"contract": {}, "ranks": [_rank_record(cfg, rank) for rank in range(2)]},
        "model": {"weight": torch.ones(1)},
        "opt": {},
        "sched": {},
        "return_calibration": {},
    }
    control, resumed, output = (tmp_path / name for name in ("control.pt", "resume.pt", "comparison.json"))
    torch.save(state, control)
    torch.save(state, resumed)
    _COMPARE.main(_COMPARE.Args(control, resumed, output))
    result = json.loads(output.read_text())
    assert result["experiment_id"] == identity
    assert result["passed"]


def test_advantage_weights_apply_to_every_offset() -> None:
    torch.manual_seed(9)
    nll = torch.rand(2, 3, 30, 4)
    weight = torch.tensor([[0.5, 2.0, 1.0], [3.0, 0.25, 1.5]])
    valid = torch.tensor([[True, True, False], [True, True, True]])
    near, far, total = _MODULE.temporal_objective_parts(nll, weight, valid_prefixes=5, valid=valid)
    joint = torch.where(valid[..., None], nll.sum(dim=-1), 0) * weight[..., None] / 30 / 5
    torch.testing.assert_close(near, joint[..., :6].sum())
    torch.testing.assert_close(far, joint[..., 6:].sum())
    torch.testing.assert_close(total, joint.sum())
    attribution = _MODULE.projection_local_target_gradient_l1(
        nll, weight, offsets=tuple(range(1, 31)), valid_prefixes=5, valid=valid
    )
    derivative = torch.where(valid[..., None, None], -torch.expm1(-nll) * weight[..., None, None] / 30 / 5, 0)
    torch.testing.assert_close(
        attribution["diagnostics/projection_local/target_logit_grad_l1/o30/buttons"],
        derivative[..., 29, _MODULE.CONTROLLER_GROUP_NAMES.index("buttons")].sum(),
    )


def test_evaluation_executes_offsets_after_a_21_frame_delay() -> None:
    cfg = _MODULE.TrainConfig()
    assert (cfg.prediction_frames, cfg.delay_frames, cfg.replan_interval_frames) == (23, 21, 2)
    _MODULE._validate_deployment_timing(cfg.prediction_frames, cfg.delay_frames, cfg.replan_interval_frames)
    with pytest.raises(ValueError, match="prediction=23, delay=21"):
        _MODULE.validate_config(replace(cfg, prediction_frames=4, delay_frames=2))
    values = {f"nll_o{offset:02d}": float(offset) for offset in range(1, 31)}
    values.update(
        {
            "loss_unweighted": 0.0,
            "temporal_loss_near_unweighted": 0.0,
            "temporal_loss_far_unweighted": 0.0,
            "exact_frame_acc": 0.0,
            "dense_four_sequence_acc": 0.0,
            "change_f1": 0.0,
            "sampled_transition_rate": 0.0,
        }
    )
    for name in _MODULE.CONTROLLER_GROUP_NAMES:
        values[f"rollout_nll_o23_{name}"] = 0.0
        values[f"exposure_gap_o23_{name}"] = 0.0
    assert _MODULE._validation_wandb_metrics(values, cfg)["executed_nll"] == 22.5


def test_training_batches_keep_the_compiled_feature_keys_without_missing_values() -> None:
    cfg = _MODULE.proxy_config()
    synthetic = _MODULE.synthetic_context(cfg, 2, torch.device("cpu"))
    full_context = _MODULE.synthetic_context(replace(cfg, arch=replace(cfg.arch, L_ctx=286)), 2, torch.device("cpu"))
    columns = {name: value.numpy() for name, value in full_context.features.items() if not name.endswith("_mask")}
    columns["ctx_pad"] = np.zeros(2, dtype=np.int64)
    for name in ("ego_awr_return", "ego_return60"):
        columns[name] = np.zeros((2, 286), dtype=np.float32)
        columns[f"{name}_valid"] = np.ones((2, 286), dtype=bool)
    for player in ("ego", "opp"):
        columns[f"{player}_direction"][:] = 1
    stats = {consolidate_key(name): FeatureStats(mean=0, std=1, min=-1, max=1) for name in columns}
    training = _MODULE._collate_o60_batch(
        ("left", "right"),
        columns,
        stats=stats,
        projection=_MODULE.ITEM_PLAYER_PROJECTION,
        context_length=256,
        return_column="ego_awr_return",
        return_valid_column="ego_awr_return_valid",
    )
    # A batch without missing values must match the warmup graph's keys and order.
    assert list(training.context.features) == list(synthetic.features)
    assert not training.context.features["ego_direction_mask"].any()


def test_warmup_prefixes_share_the_sampler_layout() -> None:
    cfg = _MODULE.proxy_config()
    warmup = _MODULE.synthetic_prefix_positions(cfg, torch.device("cpu"))
    sampled = _MODULE.PrefixSampler(0, "cpu").sample(
        torch.zeros(cfg.local_batch_size, dtype=torch.long),
        length=cfg.arch.L_ctx,
        suffix_start=cfg.arch.direct_loss_start,
    )
    assert (warmup.shape, warmup.stride(), warmup.dtype) == (sampled.shape, sampled.stride(), sampled.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="dense validation attention compiles only on CUDA")
def test_validation_compiles_inside_the_training_recompile_guard() -> None:
    cfg = _small_config()
    model = _MODULE.make_model(cfg).cuda()
    batch = _MODULE.synthetic_awr_batch(cfg, torch.device("cuda"))
    with torch.compiler.set_stance("fail_on_recompile"):
        metrics = _MODULE.validation_wandb_metrics(model, [batch], cfg)
    assert "executed_nll" in metrics
