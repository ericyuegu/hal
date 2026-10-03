"""Frozen scientific and distributed contracts for experiment 060."""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

_SPEC = importlib.util.spec_from_file_location(
    "hal_experiment_060",
    Path(__file__).parents[1] / "experiments" / "060_compute_optimal_action_sequence.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


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
    assert counts["total"] == 193_360_029
    assert _MODULE.compute_equivalent_parameter_count(cfg, counts) == 3_953_315_601


def test_parameter_matched_proxy_arms_change_the_depth_allocation() -> None:
    control = _MODULE.proxy_control_config()
    treatment = _MODULE.proxy_config()

    assert (control.arch.n_layers, control.arch.temporal_layers) == (16, 4)
    assert (treatment.arch.n_layers, treatment.arch.temporal_layers) == (6, 8)
    assert control.arch.head_offsets == treatment.arch.head_offsets == tuple(range(1, 31))
    assert not control.return_conditioning and not treatment.return_conditioning
    assert control.arch.return_embed_dim == treatment.arch.return_embed_dim == 0
    assert control.target_positions == treatment.target_positions == 2**30
    assert control.max_steps == treatment.max_steps == 65_536
    with torch.device("meta"):
        control_model = _MODULE.make_model(control)
        treatment_model = _MODULE.make_model(treatment)
    control_counts = _MODULE.subsystem_parameter_counts(control_model)
    treatment_counts = _MODULE.subsystem_parameter_counts(treatment_model)
    assert control_counts["total"] == 14_665_373
    assert treatment_counts["total"] == 14_693_661
    assert abs(treatment_counts["total"] / control_counts["total"] - 1) < 0.06
    assert _MODULE.compute_equivalent_parameter_count(control, control_counts) == 137_350_289
    assert _MODULE.compute_equivalent_parameter_count(treatment, treatment_counts) == 306_095_121


def test_compute_curve_weights_schedule_and_batch_arithmetic() -> None:
    cfg = _MODULE.TrainConfig()
    _MODULE.validate_config(cfg)
    assert _MODULE.OFFSET_LOSS_WEIGHTS == (1 / 30,) * 30
    assert _MODULE.compute_flops_per_supervised_position(3_953_315_601) == 23_719_893_606
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


def test_training_batches_use_the_live_policy_feature_keys() -> None:
    cfg = _MODULE.proxy_config()
    expected = _MODULE.synthetic_awr_batch(cfg, torch.device("cpu"))
    sparse_context = _MODULE.Context(
        features={name: value for name, value in expected.context.features.items() if not name.endswith("_mask")},
        ctx_pad=expected.context.ctx_pad,
    )
    sparse_batch = _MODULE.TrainBatch(sparse_context, expected.target, ("replay",) * cfg.local_batch_size)

    actual = _MODULE._canonical_training_batch(sparse_batch)

    assert tuple(actual.context.features) == tuple(expected.context.features)
    for name, value in actual.context.features.items():
        torch.testing.assert_close(value, expected.context.features[name])
    assert actual.replay_ids == sparse_batch.replay_ids


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
