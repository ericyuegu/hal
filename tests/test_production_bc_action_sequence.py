import importlib.util
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

_SPEC = importlib.util.spec_from_file_location(
    "hal_experiment_062",
    Path(__file__).parents[1] / "experiments" / "062_production_bc_action_sequence.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _small_config(stick_features: str = "polar") -> Any:
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
        stick_features=stick_features,
        local_batch_size=2,
        microbatch_size=2,
        prediction_frames=3,
        player_vocab_size=2,
        compile_trunk=False,
        compile_temporal=False,
    )


@pytest.mark.parametrize(("arm", "batch", "ranks"), [("b512", 512, 2), ("b1024", 1024, 4), ("b2048", 2048, 8)])
def test_batch_arms_match_in_positions_without_accumulation(arm: str, batch: int, ranks: int) -> None:
    cfg = _MODULE.proxy_config_for_arm(arm)
    _MODULE.validate_config(cfg)
    assert (cfg.batch_size, cfg.world_size) == (batch, ranks)
    assert cfg.local_batch_size == cfg.microbatch_size == 256
    positions = cfg.supervised_positions_per_update
    assert cfg.max_steps * positions == 2**29
    assert cfg.warmup_steps * positions == cfg.val_every * positions == 2**26
    assert cfg.decay_start_update + cfg.decay_duration == cfg.max_steps
    assert cfg.decay_duration * 4 == cfg.max_steps
    assert cfg.stick_features == "polar" and not cfg.state_prediction


def test_stick_control_differs_only_by_stick_features() -> None:
    treatment = _MODULE.proxy_config_for_arm("b1024")
    control = _MODULE.proxy_config_for_arm("b1024_xy")
    _MODULE.validate_config(control)
    assert replace(control, stick_features="polar") == treatment


def test_production_state_head_and_off_arm_configs_are_rejected() -> None:
    cfg = _MODULE.proxy_config_for_arm("b1024")
    with pytest.raises(ValueError, match="production is frozen"):
        _MODULE.validate_config(_MODULE.TrainConfig())
    with pytest.raises(ValueError, match="omits the state head"):
        _MODULE.validate_config(replace(cfg, state_prediction=True))
    with pytest.raises(ValueError, match="arms are frozen"):
        _MODULE.validate_config(replace(cfg, warmup_updates=4096))
    with pytest.raises(ValueError, match="requires a proxy arm"):
        _MODULE.proxy_config_for_arm("none")


def test_polar_stick_features_are_radius_sine_cosine_and_zero_at_neutral() -> None:
    codec = _MODULE.PolarStickCodec(8, main_stick_layout="legacy65")
    plain = _MODULE.DiscreteControllerCodec(8, main_stick_layout="legacy65")
    for name, count in (("main_stick", 65), ("c_stick", 9)):
        indices = torch.arange(count)
        xy = plain.semantic_values(name, indices)
        values = codec.semantic_values(name, indices)
        radius = xy.norm(dim=-1)
        neutral = radius <= _MODULE.STICK_NEUTRAL_RADIUS
        assert int(neutral.sum()) == 1
        assert torch.equal(values[neutral], torch.zeros_like(values[neutral]))
        torch.testing.assert_close(values[~neutral, 0], radius[~neutral])
        torch.testing.assert_close(values[~neutral, 1] * radius[~neutral], xy[~neutral, 1])
        torch.testing.assert_close(values[~neutral, 2] * radius[~neutral], xy[~neutral, 0])
    for name in ("buttons", "triggers"):
        indices = torch.arange(4)
        assert torch.equal(codec.semantic_values(name, indices), plain.semantic_values(name, indices))


def test_stick_arms_share_initialization_except_stick_projections() -> None:
    torch.manual_seed(5)
    polar = _MODULE.make_model(_small_config("polar"))
    after_polar = torch.rand(1)
    torch.manual_seed(5)
    plain = _MODULE.make_model(_small_config("xy"))
    after_plain = torch.rand(1)
    assert torch.equal(after_polar, after_plain)
    assert polar.codec is polar.temporal.codec
    assert isinstance(polar.codec, _MODULE.PolarStickCodec)
    plain_parameters = dict(plain.named_parameters())
    for name, parameter in polar.named_parameters():
        expected = plain_parameters.pop(name)
        if name in ("codec.semantic_projections.main_stick.weight", "codec.semantic_projections.c_stick.weight"):
            assert parameter.shape == (*expected.shape[:-1], 3)
            continue
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
    assert not plain_parameters


def test_proxy_parameter_contract_counts_the_polar_features() -> None:
    torch.manual_seed(0)
    polar = _MODULE.subsystem_parameter_counts(_MODULE.make_model(_MODULE.proxy_config_for_arm("b1024")))
    torch.manual_seed(0)
    plain = _MODULE.subsystem_parameter_counts(_MODULE.make_model(_MODULE.proxy_config_for_arm("b1024_xy")))
    assert polar["total"] - plain["total"] == 64
    assert polar["temporal_decoder"] - plain["temporal_decoder"] == 64
    assert polar["state_head"] == plain["state_head"] == 0


def test_policy_loss_is_unweighted_behavior_cloning_after_awr_would_start() -> None:
    cfg = _small_config()
    torch.manual_seed(1)
    model = _MODULE.make_model(cfg)
    batch = _MODULE.synthetic_awr_batch(cfg, torch.device("cpu"))
    batch = replace(batch, returns=torch.randn_like(batch.returns) * 500)
    trunk, temporal = _MODULE._training_functions(model, cfg)
    _, _, metrics = _MODULE.microbatch_loss(
        model,
        batch,
        cfg,
        step=10 * _MODULE.AWRCalibration.start_update,
        valid_prefixes=4,
        trunk_fn=trunk,
        temporal_fn=temporal,
        prefix_positions=torch.tensor([[4, 7], [5, 6]]),
    )
    weight_keys = [name for name in metrics if "weight" in name and ("mean" in name or "max" in name)]
    assert weight_keys
    for name in weight_keys:
        if name.endswith(("_mean", "_max")) and "correlation" not in name:
            assert math.isclose(float(metrics[name]), 1.0, rel_tol=0, abs_tol=1e-6), name


@pytest.mark.parametrize(
    ("arm", "multiplier"),
    [("b512", 1.0), ("b1024", 1.0), ("b1024_sqrt_lr", math.sqrt(2)), ("b2048", 1.0), ("b2048_sqrt_lr", 2.0)],
)
def test_muon_rate_arms_differ_from_their_batch_control_only_by_the_rate(arm: str, multiplier: float) -> None:
    cfg = _MODULE.proxy_config_for_arm(arm)
    _MODULE.validate_config(cfg)
    assert cfg.muon_lr_multiplier == multiplier
    control = _MODULE.proxy_config_for_arm(f"b{cfg.batch_size}")
    assert replace(cfg, muon_lr_multiplier=1.0) == control
    with pytest.raises(ValueError, match="arms are frozen"):
        _MODULE.validate_config(replace(cfg, muon_lr_multiplier=1.5))


def test_scaled_adam_betas_never_fall_below_their_base_values() -> None:
    for arm in ("b512", "b1024", "b2048"):
        assert _MODULE.scaled_adam_betas(_MODULE.proxy_config_for_arm(arm)) == (0.9, 0.95)
    long_run = replace(_MODULE.proxy_config_for_arm("b512"), target_positions=2**32)
    beta1, beta2 = _MODULE.scaled_adam_betas(long_run)
    assert math.isclose(beta1, 1 - 0.1 / 4) and math.isclose(beta2, 1 - 0.05 / 4)


def test_replay_spacing_matches_across_batch_arms_and_fits_each_rank() -> None:
    gaps = set()
    for arm in ("b512", "b1024", "b2048"):
        cfg = _MODULE.proxy_config_for_arm(arm)
        assert cfg.replay_slots * cfg.world_size == 65_536
        assert cfg.replay_phase_block_batches * cfg.batch_size == 16_384
        rows_per_rank = cfg.train_replays // cfg.world_size
        assert cfg.replay_slots <= rows_per_rank
        gaps.add(cfg.minimum_replay_gap_batches * cfg.batch_size - cfg.batch_size)
    assert len(gaps) == 1


def _gather_worker(rank: int, init_file: str, results: Any) -> None:
    torch.distributed.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        group = torch.distributed.new_group(backend="gloo")
        context = _MODULE.DistributedContext(rank, rank, 2, torch.device("cpu"), group)
        states = _MODULE.gather_rank_checkpoint_states({"rank": rank, "payload": b"x" * (rank + 1) * 2**20}, context)
        if rank == 0:
            assert states is not None
            results.put([len(state["payload"]) for state in states])
    finally:
        torch.distributed.destroy_process_group()


def test_rank_checkpoint_states_gather_over_the_cpu_object_group(tmp_path: Path) -> None:
    context = torch.multiprocessing.get_context("spawn")
    results = context.Queue()
    processes = [
        context.Process(target=_gather_worker, args=(rank, str(tmp_path / "init"), results)) for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=120)
        assert process.exitcode == 0
    assert results.get(timeout=5) == [2**20, 2 * 2**20]


def test_production_config_is_the_selected_batch_on_all_sources_with_awr() -> None:
    cfg = _MODULE.production_config()
    _MODULE.validate_config(cfg)
    assert (cfg.batch_size, cfg.world_size, cfg.local_batch_size, cfg.microbatch_size) == (2048, 4, 512, 512)
    assert cfg.muon_lr * cfg.muon_lr_multiplier == pytest.approx(0.014)
    assert cfg.advantage_weighting and cfg.stick_features == "xy" and not cfg.state_prediction
    assert len(cfg.source_names) == 44 and cfg.val_n_samples == 2048
    assert cfg.max_steps == 204_800 and cfg.max_steps * cfg.supervised_positions_per_update >= cfg.train_frames
    assert cfg.warmup_steps * cfg.supervised_positions_per_update == 2**26
    assert cfg.decay_start_update + cfg.decay_duration == cfg.max_steps and cfg.decay_duration * 4 == cfg.max_steps
    assert cfg.eval_every % cfg.ckpt_every == 0 and cfg.val_every % cfg.ckpt_every == 0
    assert cfg.max_steps % cfg.eval_every == 0 and cfg.decay_start_update % cfg.ckpt_every == 0
    assert cfg.replay_slots <= cfg.train_replays // cfg.world_size
    assert _MODULE.training_fields(replace(cfg, advantage_weighting=False)) != _MODULE.training_fields(cfg)
    with pytest.raises(ValueError, match="production is frozen"):
        _MODULE.validate_config(replace(cfg, advantage_weighting=False))
    with pytest.raises(ValueError, match="proxy arms are frozen|proxy training boundaries"):
        _MODULE.validate_config(replace(_MODULE.proxy_config_for_arm("b2048"), advantage_weighting=True))


def test_production_parameter_contract_drops_the_state_head() -> None:
    torch.manual_seed(0)
    counts = _MODULE.subsystem_parameter_counts(_MODULE.make_model(_MODULE.production_config()))
    assert counts["state_head"] == 0
    assert counts["total"] == 194_554_777 - 1_194_748


@pytest.mark.parametrize(("step", "weighted"), [(0, False), (_MODULE.AWRCalibration.start_update - 1, True)])
def test_advantage_weighting_starts_at_the_o59_update(step: int, weighted: bool) -> None:
    cfg = replace(_small_config(), advantage_weighting=True)
    torch.manual_seed(1)
    model = _MODULE.make_model(cfg)
    batch = _MODULE.synthetic_awr_batch(cfg, torch.device("cpu"))
    batch = replace(batch, returns=torch.randn_like(batch.returns) * 500)
    trunk, temporal = _MODULE._training_functions(model, cfg)
    _, _, metrics = _MODULE.microbatch_loss(
        model,
        batch,
        cfg,
        step=step,
        valid_prefixes=4,
        trunk_fn=trunk,
        temporal_fn=temporal,
        prefix_positions=torch.tensor([[4, 7], [5, 6]]),
    )
    assert bool(metrics["awr/active"]) == weighted
    assert (float(metrics["awr/weight_max"]) > 1.0 + 1e-6) == weighted
