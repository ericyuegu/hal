"""Contracts for the O54 capacity-latency experiment."""

import importlib.util
import sys
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
    expected_delays = (1, 1, 2, 2, 3, 4, 5, 6)
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
    expected_totals = {
        128: 1_692_953,
        256: 5_764_505,
        384: 17_094_169,
        512: 39_024_281,
        640: 70_178_585,
        768: 121_763_737,
        896: 194_172_953,
        1024: 278_362_265,
    }
    for width, expected in expected_totals.items():
        cfg = exp.config_for_width(width)
        assert cfg.arch.n_layers == exp.TRUNK_DEPTHS[width]
        model = exp.GPT(cfg)
        counts = exp.subsystem_parameter_counts(model)
        assert counts["total"] == expected
        assert exp.approximate_training_flops_per_update(cfg, counts) == (
            6 * 512 * 128 * exp.effective_parameter_count(counts)
        )
    assert exp.iso_compute_updates(256) == (16_384, 84_320)
    assert exp.scientific_checkpoint_updates(exp.config_for_width(768)) == (35_072,)
    assert exp.scientific_checkpoint_updates(exp.config_for_width(256)) == (84_320,)


def test_data_selection_is_the_frozen_all_44_order() -> None:
    cfg = exp.config_for_width(256)
    selection = exp.data_selection(cfg)

    assert len(selection.sources) == 44
    assert selection.sha256 == cfg.selection_sha256
    assert selection.row_count == cfg.train_replays == 1_295_370
    assert cfg.replay_slots == 65_536
    assert cfg.generations_per_replay * cfg.windows_per_generation == 32


def test_powerlines_weight_decay_scales_with_duration_and_capacity() -> None:
    short = exp.config_for_width(128, updates=1_000)
    long = exp.config_for_width(128, updates=10_000)
    short_optimizer = exp.make_optimizer(exp.GPT(short), short)
    long_optimizer = exp.make_optimizer(exp.GPT(long), long)

    assert exp.config_for_width(512, updates=16_384).adam_weight_decay == pytest.approx(1e-4)
    assert short.adam_weight_decay > long.adam_weight_decay
    assert short.adam_weight_decay == pytest.approx(
        exp.powerlines_weight_decay(short.target_positions, exp.PARAMETER_COUNT_CONTRACTS[128]["total"])
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


def test_latency_manifest_is_hashed_and_strict(tmp_path: Path) -> None:
    rows = tuple(
        exp.LatencyRow(width, delay, delay, 2 * delay, 0.001, 1 / 60)
        for width, delay in zip(exp.WIDTHS, (1, 1, 2, 2, 3, 4, 5, 6), strict=True)
    )
    path = tmp_path / "latency.json"
    digest = exp.write_latency_manifest(path, rows, "NVIDIA GeForce RTX 3060")

    loaded, actual = exp.load_latency_manifest(path)

    assert actual == digest
    assert tuple(loaded) == exp.WIDTHS
    assert exp.latency_bucket_frontier(loaded) == (256, 512, 640, 768, 896, 1024)
    document = path.read_text().replace('"p99_seconds": 0.001', '"p99_seconds": 0.002', 1)
    path.write_text(document)
    with pytest.raises(ValueError, match="SHA-256"):
        exp.load_latency_manifest(path)

    diagnostic_rows = (replace(rows[0], batch_size=4), *rows[1:])
    with pytest.raises(ValueError, match="requires eager B32"):
        exp.write_latency_manifest(path, diagnostic_rows, "NVIDIA GeForce RTX 3060")


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

    exp.write_latency_probe_report(path, (probe,), "NVIDIA GeForce RTX 3060")
    payload = exp.json.loads(path.read_text())

    assert payload["device_total_memory_bytes"] == 12 * 2**30
    assert payload["probes"][0]["samples_seconds"] == [0.010, 0.020, 0.030]
    assert probe.percentile_seconds(99) == pytest.approx(0.0298)
    assert probe.meets_deadline


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

    def benchmark(width: int, *, on_probe):
        del on_probe
        measured.append(width)
        if width in (512, 1024):
            raise exp.LatencyProbeFailure(f"width {width} missed")
        delay = exp.INITIAL_DELAYS[width]
        return exp.LatencyRow(width, delay, delay, 2 * delay, 0.001, 1 / 60)

    monkeypatch.setattr(exp.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(exp.torch.cuda, "get_device_name", lambda: "NVIDIA GeForce RTX 3060")
    monkeypatch.setattr(exp, "benchmark_width_latency", benchmark)

    with pytest.raises(exp.LatencyProbeFailure, match="width 512 missed; width 1024 missed"):
        exp.run_latency_preflight(tmp_path / "latency.json")

    assert measured == list(exp.WIDTHS)


def test_checkpoint_contains_no_advantage_or_value_configuration() -> None:
    state = exp._checkpoint_config(exp.config_for_width(256, updates=16_384))

    assert exp.config_from_state(state).arch.d_model == 256
    assert not any("awr" in name or "value" in name or "return" in name for name in state)


def test_analysis_requires_a_concave_bracketed_capacity_vertex() -> None:
    optimum, response = exp.fit_quadratic_vertex((100, 1_000, 10_000), (0.0, 1.0, 0.0))

    assert optimum == pytest.approx(1_000)
    assert response == pytest.approx(1.0)
    width, total = exp.vertex_capacity(exp.effective_parameter_count(exp.PARAMETER_COUNT_CONTRACTS[512]))
    assert width == pytest.approx(512)
    assert total == pytest.approx(exp.PARAMETER_COUNT_CONTRACTS[512]["total"])
    with pytest.raises(ValueError, match="not concave"):
        exp.fit_quadratic_vertex((100, 1_000, 10_000), (0.0, -1.0, 0.0))
    with pytest.raises(ValueError, match="not bracketed"):
        exp.fit_quadratic_vertex((100, 1_000, 10_000), (-9.0, -4.0, -1.0))


def test_config_rejects_an_unmeasured_inference_path() -> None:
    cfg = replace(exp.config_for_width(256, updates=1), inference_mode="compiled")

    with pytest.raises(ValueError, match="eager inference path"):
        exp.validate_config(cfg)
