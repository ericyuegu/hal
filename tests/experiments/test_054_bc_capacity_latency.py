"""Contracts for the O54 capacity-latency experiment."""

import importlib.util
import sys
from contextlib import nullcontext
from dataclasses import asdict
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hal.sim.rollout import ObservationRow
from hal.sim.vec import Slot


def _load():
    path = Path(__file__).resolve().parents[2] / "experiments" / "054_bc_capacity_latency.py"
    name = "test_exp054"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


exp = _load()


def test_official_timing_rows_and_offset_weights() -> None:
    expected_delays = (1, 2, 3, 6)
    for width, delay in zip(exp.WIDTHS, expected_delays, strict=True):
        timing = exp.timing_for_width(width)
        assert timing == exp.TimingConfig(delay, delay, 2 * delay)
        weights = exp.offset_weights(exp.Architecture().head_offsets, timing.prediction_frames)
        assert weights == tuple(
            1.0 if offset <= timing.prediction_frames else 0.5 for offset in exp.Architecture().head_offsets
        )


def test_behavior_cloning_objective_normalizes_by_offset_weight_sum() -> None:
    offsets = exp.Architecture().head_offsets
    nll = np.ones((2, len(offsets), exp.CONTROLLER_GROUP_COUNT), dtype=np.float32)
    valid = np.asarray([True, False])

    loss = exp.temporal_objective(
        exp.torch.from_numpy(nll),
        offsets=offsets,
        prediction_frames=4,
        valid_prefixes=1,
        valid=exp.torch.from_numpy(valid),
    )

    assert float(loss) == exp.CONTROLLER_GROUP_COUNT


def test_parameter_and_compute_contracts() -> None:
    expected_counts = {
        256: (3_932_160, 441_072, 202_211, 861_382, 5_436_825),
        512: (34_603_008, 2_901_616, 535_139, 984_518, 39_024_281),
        768: (113_246_208, 9_359_856, 999_139, 1_107_654, 124_712_857),
        1024: (264_241_152, 21_781_872, 1_594_211, 1_230_790, 288_848_025),
    }
    for width, expected in expected_counts.items():
        cfg = exp.config_for_width(width)
        assert cfg.arch.n_layers == exp.TRUNK_DEPTHS[width]
        assert cfg.arch.temporal_layers == exp.TEMPORAL_DEPTHS[width]
        model = exp.GPT(cfg)
        counts = exp.subsystem_parameter_counts(model)
        assert (
            tuple(counts[name] for name in ("trunk", "temporal_decoder", "group_heads", "inputs", "total")) == expected
        )
        assert exp.approximate_training_flops_per_update(cfg, counts) == (
            6 * 512 * 128 * exp.effective_parameter_count(counts)
        )
    assert exp.DATA_UPDATES == (4_096, 8_192, 16_384, 32_768)
    assert exp.DATA_REPLAYS == (65_536, 131_072, 262_144, 524_288)
    assert exp.scientific_checkpoint_updates(exp.config_for_width(768)) == exp.DATA_UPDATES


def test_w1024_microbatches_preserve_the_optimizer_batch() -> None:
    cfg = exp.config_for_width(1024)
    rows = cfg.batch_size
    batch = exp.TrainBatch(
        context=exp.Context(
            features={"row": exp.torch.arange(rows)[:, None]},
            ctx_pad=exp.torch.zeros(rows, dtype=exp.torch.long),
        ),
        target=exp.torch.arange(rows)[:, None],
        replay_ids=tuple(str(index) for index in range(rows)),
    )

    microbatches = exp.training_microbatches(batch, cfg)

    assert exp.train_microbatch_size(cfg) == 256
    assert tuple(item.target.shape[0] for item in microbatches) == (256, 256)
    assert exp.torch.equal(exp.torch.cat([item.target for item in microbatches]), batch.target)
    assert tuple(replay_id for item in microbatches for replay_id in item.replay_ids or ()) == batch.replay_ids
    assert exp.train_microbatch_size(exp.config_for_width(768)) == 512


def test_microbatch_backward_matches_full_batch_gradient(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = exp.config_for_width(1024)
    model = exp.nn.Linear(1, 1, bias=False)
    batch = exp.TrainBatch(
        context=exp.Context(
            features={"row": exp.torch.ones(cfg.batch_size, 1)},
            ctx_pad=exp.torch.zeros(cfg.batch_size, dtype=exp.torch.long),
        ),
        target=exp.torch.arange(cfg.batch_size, dtype=exp.torch.float32)[:, None],
    )

    def fake_loss(model, batch, _cfg, *, valid_prefixes, **_kwargs):
        loss = model.weight.squeeze() * batch.target.sum() / valid_prefixes
        value = loss.detach()
        metrics = {
            "train/loss": value,
            "train/objective": value,
            "stability/button_pre_norm_rms_min": batch.target.amin(),
            "stability/button_input_abs_p999": batch.target.amax(),
            "stability/button_logit_abs_p999": batch.target.amax(),
            "stability/button_margin_mean": batch.target.mean(),
        }
        return loss, batch.target.sum().reshape(1), metrics

    monkeypatch.setattr(exp, "DEVICE", "cpu")
    monkeypatch.setattr(exp, "microbatch_loss", fake_loss)
    nll_sum, metrics = exp._backward_training_batch(
        model,
        batch,
        cfg,
        step=0,
        valid_prefixes=cfg.batch_size,
        trunk_fn=lambda: None,
        temporal_fn=lambda: None,
        phase_timer=None,
    )

    assert model.weight.grad is not None
    assert float(model.weight.grad) == pytest.approx(float(batch.target.mean()))
    assert float(nll_sum) == pytest.approx(float(batch.target.sum()))
    assert float(metrics["train/objective"]) == pytest.approx(
        float(model.weight.detach().squeeze() * batch.target.mean())
    )
    assert float(metrics["stability/button_margin_mean"]) == pytest.approx(float(batch.target.mean()))


def test_final_readout_lr_uses_the_declared_fan_in_scaling() -> None:
    for width in exp.WIDTHS:
        cfg = exp.config_for_width(width)
        roles = exp.optimizer_roles(exp.GPT(cfg), cfg)
        output_roles = [role for role in roles.values() if role.lr_kind == "output"]

        assert output_roles
        assert {role.fan_in_multiplier for role in output_roles} == {width / 256}
        assert {exp._role_lr(role, cfg) for role in output_roles} == {cfg.adam_lr * 256 / width}


def test_data_selection_is_the_frozen_all_44_order() -> None:
    cfg = exp.config_for_width(256)
    selection = exp.data_selection(cfg)

    assert len(selection.sources) == 44
    assert selection.sha256 == cfg.selection_sha256
    assert selection.row_count == cfg.train_replays == 1_295_368
    assert {
        source.source: source.excluded_rows for source in selection.sources if source.excluded_rows
    } == exp.DEDUPLICATED_SOURCE_ROWS
    assert cfg.replay_slots == 65_536
    assert cfg.generations_per_replay * cfg.windows_per_generation == 32


def test_nested_data_phases_are_disjoint_and_have_exact_cumulative_unions() -> None:
    cfg = exp.config_for_width(256)
    phases = [exp.phase_data_selection(cfg, index) for index in range(len(exp.DATA_EXPONENTS))]
    cumulative = [exp.cumulative_data_selection(cfg, exponent) for exponent in exp.DATA_EXPONENTS]

    assert tuple(selection.row_count for selection in phases) == exp.DATA_PHASE_REPLAYS
    assert tuple(selection.row_count for selection in cumulative) == exp.DATA_REPLAYS
    for source_index in range(len(cfg.source_names)):
        phase_ranges = [
            (selection.sources[source_index].start, selection.sources[source_index].stop) for selection in phases
        ]
        assert phase_ranges[0][0] == 0
        assert all(left[1] == right[0] for left, right in zip(phase_ranges[:-1], phase_ranges[1:], strict=True))
        assert tuple(stop for _start, stop in phase_ranges) == tuple(
            selection.sources[source_index].stop for selection in cumulative
        )


def test_nested_loader_checkpoint_discards_a_completed_phase() -> None:
    cfg = exp.config_for_width(256)
    within = exp.nested_loader_checkpoint_state(
        cfg,
        update=4_095,
        phase_index=0,
        loader_state={"cursor": "exact"},
    )
    boundary = exp.nested_loader_checkpoint_state(
        cfg,
        update=4_096,
        phase_index=0,
        loader_state={"cursor": "must-not-leak"},
    )

    assert within["physical_loader"] == {"cursor": "exact"}
    assert boundary["physical_loader"] is None
    resume = {"step": 4_095, "loader": boundary}
    phase_index, physical = exp.resume_data_phase(cfg, resume)
    assert phase_index == 1
    assert physical is None


def test_training_rng_state_restores_the_next_random_draws() -> None:
    exp.torch.manual_seed(54)
    exp.np.random.seed(55)
    exp.random.seed(56)
    state = exp.capture_training_rng_state()

    expected_torch = exp.torch.rand(4)
    expected_numpy = exp.np.random.random(4)
    expected_python = [exp.random.random() for _ in range(4)]
    exp.torch.manual_seed(1)
    exp.np.random.seed(2)
    exp.random.seed(3)

    exp.restore_training_rng_state(state)

    exp.torch.testing.assert_close(exp.torch.rand(4), expected_torch)
    np.testing.assert_array_equal(exp.np.random.random(4), expected_numpy)
    assert [exp.random.random() for _ in range(4)] == expected_python


def test_training_rng_state_is_required_for_resume() -> None:
    with pytest.raises(ValueError, match="complete O54 RNG state"):
        exp.restore_training_rng_state(None)


def test_powerlines_weight_decay_is_fixed_at_d31_within_each_model() -> None:
    short = exp.config_for_width(256, updates=1_000)
    long = exp.config_for_width(256, updates=10_000)
    short_optimizer = exp.make_optimizer(exp.GPT(short), short)
    long_optimizer = exp.make_optimizer(exp.GPT(long), long)

    assert exp.powerlines_weight_decay(2**30, exp.PARAMETER_COUNT_CONTRACTS[512]["total"]) == pytest.approx(1e-4)
    assert short.adam_weight_decay == long.adam_weight_decay
    assert short.adam_weight_decay == pytest.approx(
        exp.powerlines_weight_decay(2**31, exp.PARAMETER_COUNT_CONTRACTS[256]["total"])
    )
    assert {group["lr"] for group in short_optimizer.param_groups} == {
        group["lr"] for group in long_optimizer.param_groups
    }
    assert {group["weight_decay"] for group in short_optimizer.param_groups} == {
        0.0,
        short.adam_weight_decay,
    }
    assert {group["weight_decay"] for group in long_optimizer.param_groups} == {
        0.0,
        long.adam_weight_decay,
    }
    assert exp.lr_schedule(short)(511) == 1.0
    assert exp.lr_schedule(short)(50_000) == 1.0


class _BufferedHarness(exp.BufferedPolicy):
    def _context(self, live):
        for slot in live:
            self._slots[slot].reset_pending = False
        return None

    def _ingest_row(self, slot, row):
        state = self._slots.get(slot)
        reset = row.reset or state is None or row.frame_id < getattr(state, "last_id", row.frame_id)
        self._slots[slot] = SimpleNamespace(reset_pending=reset, last_id=row.frame_id)


@pytest.mark.parametrize("delay", range(1, 7))
def test_buffered_plan_rows_execute_the_preceding_inference(delay: int) -> None:
    calls = 0

    def predict(_context, _committed):
        nonlocal calls
        calls += 1
        offsets = np.arange(1, 2 * delay + 1, dtype=np.float32)
        return np.repeat((100 * calls + offsets)[None, :, None], exp.A_DIM, axis=2)

    policy = _BufferedHarness(
        predict_chunk=predict,
        stats={},
        L_ctx=8,
        L_chunk=2 * delay,
        s=delay,
        d=delay,
        inference_delay_frames=delay,
        device="cpu",
    )
    slot = Slot(0, 1)

    action = np.zeros(exp.A_DIM, dtype=np.float32)
    first = policy.plan_rows({slot: [ObservationRow(0, {}, action, reset=True)]})[slot]
    second = policy.plan_rows({slot: [ObservationRow(1, {}, action)]})[slot]
    reset = policy.plan_rows({slot: [ObservationRow(0, {}, action, reset=True)]})[slot]

    np.testing.assert_array_equal(first[:delay], 0)
    np.testing.assert_array_equal(second[:delay, 0], np.arange(101 + delay, 101 + 2 * delay, dtype=np.float32))
    np.testing.assert_array_equal(reset[:delay], 0)
    assert calls == 3


def test_framewise_staging_matches_spawned_staging() -> None:
    def predictor():
        calls = 0

        def predict(_context, _committed):
            nonlocal calls
            calls += 1
            plan = np.arange(1, 9, dtype=np.float32) + 100 * calls
            return np.repeat(plan[None, :, None], exp.A_DIM, axis=2)

        return predict

    def policy(predict):
        return _BufferedHarness(
            predict_chunk=predict,
            stats={},
            L_ctx=8,
            L_chunk=8,
            s=4,
            d=4,
            inference_delay_frames=4,
            device="cpu",
        )

    slot = Slot(0, 1)
    framewise = policy(predictor())
    framewise._slots[slot] = SimpleNamespace(reset_pending=True)
    framewise._stage([slot])
    framewise._stage([slot])

    spawned = policy(predictor())
    action = np.zeros(exp.A_DIM, dtype=np.float32)
    spawned.plan_rows({slot: [ObservationRow(0, {}, action, reset=True)]})
    second = spawned.plan_rows({slot: [ObservationRow(4, {}, action)]})[slot]

    np.testing.assert_array_equal(np.stack(framewise._executing[slot]), second[:4])


def test_buffered_policy_rejects_overlapping_inference() -> None:
    slot = Slot(0, 1)
    policy = None

    def predict(_context, _committed):
        assert policy is not None
        with pytest.raises(RuntimeError, match="overlap"):
            policy._infer([slot])
        return np.zeros((1, 2, exp.A_DIM), dtype=np.float32)

    policy = _BufferedHarness(
        predict_chunk=predict,
        stats={},
        L_ctx=8,
        L_chunk=2,
        s=1,
        d=1,
        inference_delay_frames=1,
        device="cpu",
    )
    policy._slots[slot] = SimpleNamespace(reset_pending=True)

    policy._infer([slot])


def test_latency_manifest_is_hashed_and_strict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        exp.torch.cuda, "get_device_properties", lambda _device: SimpleNamespace(total_memory=12 * 2**30)
    )
    rows = tuple(
        exp.LatencyRow(width, delay, delay, 2 * delay, 0.001, 1 / 60)
        for width, delay in zip(exp.WIDTHS, (1, 2, 3, 4), strict=True)
    )
    path = tmp_path / "latency.json"
    dolphin = exp.DolphinLoadEvidence("AppRun", "d" * 64)
    probes_path = path.with_suffix(".probes.json")
    probes = tuple(
        exp.LatencyProbe(
            width=row.width,
            inference_delay_frames=row.inference_delay_frames,
            replan_interval_frames=row.replan_interval_frames,
            prediction_frames=row.prediction_frames,
            status="measured",
            samples_seconds=(row.p99_seconds,) * exp.LATENCY_MEASURED_CALLS,
            peak_allocated_bytes=0,
            peak_reserved_bytes=0,
            trial_index=trial,
        )
        for trial in range(exp.LATENCY_TRIALS)
        for row in rows
    )
    raw_digest = exp.write_latency_probe_report(
        probes_path,
        probes,
        "NVIDIA GeForce RTX 3060",
        width_orders=(exp.WIDTHS,) * exp.LATENCY_TRIALS,
        dolphin=dolphin,
    )
    trial_rows = tuple(
        exp.LatencyTrialRow(trial, exp.WIDTHS.index(row.width), row)
        for trial in range(exp.LATENCY_TRIALS)
        for row in rows
    )
    digest = exp.write_latency_manifest(
        path,
        rows,
        trial_rows,
        "NVIDIA GeForce RTX 3060",
        raw_digest,
        dolphin,
    )

    loaded, actual = exp.load_latency_manifest(path)

    assert actual == digest
    assert tuple(loaded) == exp.WIDTHS
    assert exp.latency_bucket_frontier(loaded) == exp.WIDTHS
    document = path.read_text().replace('"p99_seconds": 0.001', '"p99_seconds": 0.002', 1)
    path.write_text(document)
    with pytest.raises(ValueError, match="SHA-256"):
        exp.load_latency_manifest(path)

    diagnostic_rows = (replace(rows[0], batch_size=4), *rows[1:])
    with pytest.raises(ValueError, match="requires eager B1"):
        exp.write_latency_manifest(
            path,
            diagnostic_rows,
            trial_rows,
            "NVIDIA GeForce RTX 3060",
            raw_digest,
            dolphin,
        )


def test_latency_probe_report_preserves_raw_samples(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        exp.torch.cuda, "get_device_properties", lambda _device: SimpleNamespace(total_memory=12 * 2**30)
    )
    probe = exp.LatencyProbe(
        width=512,
        inference_delay_frames=2,
        replan_interval_frames=2,
        prediction_frames=4,
        status="measured",
        samples_seconds=(0.010, 0.020, 0.030),
        peak_allocated_bytes=4 * 2**30,
        peak_reserved_bytes=5 * 2**30,
        measured_calls=3,
    )
    path = tmp_path / "latency.probes.json"

    digest = exp.write_latency_probe_report(path, (probe,), "NVIDIA GeForce RTX 3060")
    payload = exp.json.loads(path.read_text())

    assert payload["sha256"] == digest
    assert payload["device_total_memory_bytes"] == 12 * 2**30
    assert payload["probes"][0]["samples_seconds"] == [0.010, 0.020, 0.030]
    assert probe.percentile_seconds(99) == pytest.approx(0.0298)
    assert probe.meets_deadline
    loaded, actual = exp.load_latency_probe_report(path)
    assert actual == digest
    loaded["probes"][0]["samples_seconds"][0] = 1.0
    path.write_text(exp.json.dumps({**loaded, "sha256": digest}))
    with pytest.raises(ValueError, match="probe SHA-256"):
        exp.load_latency_probe_report(path)


def test_latency_probe_distinguishes_oom_from_a_deadline_miss() -> None:
    probe = exp.LatencyProbe(
        width=512,
        inference_delay_frames=6,
        replan_interval_frames=6,
        prediction_frames=12,
        status="oom",
        samples_seconds=(),
        peak_allocated_bytes=11 * 2**30,
        peak_reserved_bytes=12 * 2**30,
    )

    assert probe.percentile_seconds(99) is None
    assert not probe.meets_deadline


def test_latency_preflight_measures_all_widths_before_rejecting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    measured: list[int] = []

    def benchmark(width: int, **_kwargs):
        measured.append(width)
        if width in (512, 1024):
            raise exp.LatencyProbeFailure(f"width {width} missed")
        delay = exp.INITIAL_DELAYS[width]
        return exp.LatencyRow(width, delay, delay, 2 * delay, 0.001, 1 / 60)

    monkeypatch.setattr(exp.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(exp.torch.cuda, "get_device_name", lambda: "NVIDIA GeForce RTX 3060")
    monkeypatch.setattr(
        exp.torch.cuda, "get_device_properties", lambda _device: SimpleNamespace(total_memory=12 * 2**30)
    )
    monkeypatch.setattr(exp, "benchmark_width_latency", benchmark)
    load = SimpleNamespace(
        evidence=exp.DolphinLoadEvidence("AppRun", "d" * 64),
        raise_if_failed=lambda: None,
    )
    monkeypatch.setattr(exp, "production_dolphin_load", lambda: nullcontext(load))

    with pytest.raises(exp.LatencyProbeFailure, match="width 512 missed"):
        exp.run_latency_preflight(tmp_path / "latency.json")

    assert sorted(measured) == sorted(exp.WIDTHS * exp.LATENCY_TRIALS)


def test_latency_preflight_selects_smallest_passing_timing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def benchmark(width: int, **kwargs):
        trial = kwargs["trial_index"]
        requested = kwargs.get("delay")
        local_delay = 2 if width == 512 else 1
        delay = local_delay if requested is None else requested
        probe = exp.LatencyProbe(
            width=width,
            inference_delay_frames=delay,
            replan_interval_frames=delay,
            prediction_frames=2 * delay,
            status="measured",
            samples_seconds=(0.001,) * exp.LATENCY_MEASURED_CALLS,
            peak_allocated_bytes=0,
            peak_reserved_bytes=0,
            trial_index=trial,
            phase=kwargs["phase"],
        )
        kwargs["on_probe"](probe)
        return exp.LatencyRow(width, delay, delay, 2 * delay, 0.001, 1 / 60)

    monkeypatch.setattr(exp.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(exp.torch.cuda, "get_device_name", lambda: "NVIDIA GeForce RTX 3060")
    monkeypatch.setattr(
        exp.torch.cuda, "get_device_properties", lambda _device: SimpleNamespace(total_memory=12 * 2**30)
    )
    monkeypatch.setattr(exp, "benchmark_width_latency", benchmark)
    load = SimpleNamespace(
        evidence=exp.DolphinLoadEvidence("AppRun", "d" * 64),
        raise_if_failed=lambda: None,
    )
    monkeypatch.setattr(exp, "production_dolphin_load", lambda: nullcontext(load))
    path = tmp_path / "latency.json"

    exp.run_latency_preflight(path)
    rows, _digest = exp.load_latency_manifest(path)

    assert rows[512].inference_delay_frames == 2
    assert all(rows[width].inference_delay_frames == 1 for width in (256, 768, 1024))
    probes, _raw_digest = exp.load_latency_probe_report(path.with_suffix(".probes.json"))
    assert len(probes["width_orders"]) == exp.LATENCY_TRIALS
    assert all(sorted(order) == list(exp.WIDTHS) for order in probes["width_orders"])


def test_study_matrix_uses_one_nested_trajectory_per_architecture() -> None:
    rows = {
        width: exp.LatencyRow(width, delay, delay, 2 * delay, 0.001, 1 / 60)
        for width, delay in zip(exp.WIDTHS, (1, 2, 3, 4), strict=True)
    }

    endpoints = exp.study_endpoints(rows)

    assert [(endpoint.width, endpoint.updates) for endpoint in endpoints] == [
        (256, 32_768),
        (512, 32_768),
        (768, 32_768),
        (1024, 32_768),
    ]
    commands = exp.study_launch_commands(rows, Path("docs/experiments/054_iso_data_latency_manifest.json"))
    assert len(commands) == 4
    assert all(
        command[:8]
        == (
            "uv",
            "run",
            "scripts/launch_modal.py",
            "--gpu",
            "B200",
            "--closed-loop-gpu",
            "RTX-PRO-6000",
            "--app-name",
        )
        for command in commands
    )


def test_checkpoint_contains_no_advantage_or_value_configuration() -> None:
    state = exp._checkpoint_config(exp.config_for_width(256, updates=16_384))

    assert exp.config_from_state(state).arch.d_model == 256
    assert not any("awr" in name or "value" in name or "return" in name for name in state)


def test_experiment_rejects_a_non_bfloat16_treatment() -> None:
    with pytest.raises(ValueError, match="BF16 training and inference"):
        exp.validate_config(replace(exp.config_for_width(256), amp_dtype="float32"))


def test_analysis_rejects_an_evaluation_from_a_different_inference_path() -> None:
    timing = exp.LatencyRow(256, 1, 1, 2, 0.01, 1 / 60)
    pairs, egos, cpus, schedule_sha256 = exp.assert_protocol_diversity(96)
    protocol = asdict(
        exp.EvalProtocol(
            fixed_ego_character=None,
            ego_player_id=exp.MASKED_PLAYER_ID,
            ego_player_code=None,
            opponent_identity_conditioned=False,
            n_matchups=96,
            allowed_cpus=32,
            hardware_wave_bucket=32,
            max_parallel=32,
            max_frames=7_200,
            seed=0,
            cpu_level=9,
            ego_port=1,
            seed_stage=int(exp.PRIOR_SWEEP_SEED_STAGE.value),
            matchup_schedule_sha256=schedule_sha256,
            oriented_pairs=pairs,
            ego_characters=egos,
            cpu_characters=cpus,
            prediction_frames=2,
            delay_frames=1,
            replan_interval_frames=1,
            transport_delay_frames=0,
            timing_model="buffered",
            dtype="torch.bfloat16",
            inference_mode="eager",
            inference_compile_mode="default",
            inference_attention_backend="dense_sdpa",
            compiled_inference_bucket=32,
            checkpoint_sha256="a" * 64,
        )
    )

    exp._validate_analysis_eval_protocol(protocol, timing)
    protocol["dtype"] = "torch.float32"
    with pytest.raises(ValueError, match="frozen O54 protocol"):
        exp._validate_analysis_eval_protocol(protocol, timing)


def test_gameplay_learning_curve_recovers_a_saturating_power() -> None:
    responses = [1.0 + 2.0 * (1 - (positions / 2**28) ** -0.5) for positions in exp.DATA_POSITIONS]

    fit = exp.fit_gameplay_learning_curve(exp.DATA_POSITIONS, responses)

    assert fit.response_at_d28 == pytest.approx(1.0)
    assert fit.asymptotic_gain == pytest.approx(2.0)
    assert fit.exponent == pytest.approx(0.5)
    assert fit.predict(2**34) == pytest.approx(2.75)


def test_analysis_tests_d31_before_reporting_a_d34_winner(tmp_path: Path) -> None:
    latency = {
        width: exp.LatencyRow(width, delay, delay, 2 * delay, 0.001 * delay, 1 / 60)
        for width, delay in zip(exp.WIDTHS, (1, 2, 3, 4), strict=True)
    }
    endpoints = {}
    for endpoint in exp.study_endpoints(latency):
        for data_index, update in enumerate(exp.DATA_UPDATES):
            pair = (endpoint.width, update)
            counts = exp.PARAMETER_COUNT_CONTRACTS[endpoint.width]
            cfg = exp.config_for_width(endpoint.width, updates=update)
            response = endpoint.width / 1_000 + 0.2 * (1 - 2 ** (-0.5 * data_index))
            endpoints[pair] = exp.AnalysisEndpoint(
                run_id=f"run-{endpoint.width}",
                run_name=f"W{endpoint.width}",
                width=endpoint.width,
                updates=update,
                effective_parameters=exp.effective_parameter_count(counts),
                bc_parameters=counts["total"],
                compute=exp.training_compute(cfg, counts),
                net_stock_per_min=response,
                net_stock_lcb=response - 0.1,
            )

    report = exp.write_analysis_artifacts(endpoints, latency, tmp_path)

    assert report["d31_holdout"]["passed"] is True
    assert report["d34"]["reported"] is True
    assert report["d34"]["winner_width"] == 1024
    assert (tmp_path / "endpoints.csv").is_file()
    assert (tmp_path / "iso_data_latency.png").is_file()
    assert (tmp_path / "learning_curves.png").is_file()

    final_key = (256, exp.DATA_UPDATES[-1])
    failed_endpoints = {
        **endpoints,
        final_key: replace(endpoints[final_key], net_stock_per_min=99.0),
    }
    failed = exp.write_analysis_artifacts(failed_endpoints, latency, tmp_path / "failed")
    assert failed["d31_holdout"]["passed"] is False
    assert failed["d34"] == {
        "reported": False,
        "reason": "the through-D=2^30 curve fits did not predict the observed D=2^31 ordering",
    }


def test_learning_curve_bootstrap_resamples_shared_complete_blocks() -> None:
    widths = exp.WIDTHS
    endpoints = {}
    rows = {}
    for rank, width in enumerate(widths, start=1):
        counts = exp.PARAMETER_COUNT_CONTRACTS[width]
        for update in exp.DATA_UPDATES:
            key = (width, update)
            cfg = exp.config_for_width(width, updates=update)
            endpoints[key] = exp.AnalysisEndpoint(
                run_id=f"run-{width}",
                run_name=f"W{width}",
                width=width,
                updates=update,
                effective_parameters=exp.effective_parameter_count(counts),
                bc_parameters=counts["total"],
                compute=exp.training_compute(cfg, counts),
                net_stock_per_min=float(rank),
                net_stock_lcb=float(rank),
            )
            rows[key] = tuple(
                exp.MatchRow(
                    ego_character=boot % 13,
                    opp_character=boot % 14,
                    stage=1,
                    boot_index=boot,
                    match_ordinal=0,
                    active_frames=3_600,
                    total_frames=3_600,
                    damage_dealt=0.0,
                    damage_taken=0.0,
                    stocks_taken=rank,
                    stocks_lost=0,
                )
                for boot in range(96)
            )

    report = exp.bootstrap_learning_curve_analysis(
        endpoints,
        rows,
        widths,
        holdout_passed=True,
        resamples=4,
    )

    assert report["shared_resample_indices"] is True
    assert report["holdout_order_match_fraction"] == 1.0
    assert report["d34_winner_fraction"]["1024"] == 1.0


def test_config_rejects_an_unmeasured_inference_path() -> None:
    cfg = replace(exp.config_for_width(256, updates=1), inference_mode="compiled")

    with pytest.raises(ValueError, match="eager inference path"):
        exp.validate_config(cfg)
