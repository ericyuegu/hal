import copy
import hashlib
import importlib.util
import json
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hal.training.physical_shard_loader import PhysicalRow


def _load_tool(name: str) -> ModuleType:
    path = Path(__file__).parent / "fixtures" / "o59" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


capture = _load_tool("capture_resume_update")
comparison = _load_tool("compare_resume_updates")
performance = _load_tool("measure_training_updates")


def test_resume_capture_comparison_checks_nested_scientific_records() -> None:
    expected = {
        "model": torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
        "optimizer": [{"momentum": torch.tensor([0.25])}],
        "rng": ("MT19937", np.array([1, 2], dtype=np.uint32)),
        "loader": [PhysicalRow("ranked", 2, 7)],
    }
    mismatches: list[str] = []
    assert comparison._compare(expected, copy.deepcopy(expected), "snapshot", mismatches) == 7
    assert not mismatches

    actual = copy.deepcopy(expected)
    actual["optimizer"][0]["momentum"].add_(0.01)
    actual["rng"][1][0] += 1
    actual["loader"][0] = PhysicalRow("ranked", 2, 8)
    comparison._compare(expected, actual, "snapshot", mismatches)
    assert mismatches == [
        "snapshot/optimizer/0/momentum: tensor changed",
        "snapshot/rng/1: array changed",
        "snapshot/loader/0/row: value changed",
    ]


@pytest.mark.parametrize(
    ("expected", "actual"),
    [
        (torch.ones(2), torch.ones(1, 2)),
        (torch.ones(2), torch.ones(2, dtype=torch.float64)),
        (np.ones(2, dtype=np.float32), np.ones(2, dtype=np.float64)),
        ([1, 2], [1]),
        ({"a": 1}, {"a": 1, "b": 2}),
        ((1,), [1]),
    ],
)
def test_resume_capture_comparison_rejects_structure_changes(expected: object, actual: object) -> None:
    mismatches: list[str] = []
    comparison._compare(expected, actual, "snapshot", mismatches)
    assert mismatches


def test_resume_capture_rejects_modified_snapshot(tmp_path: Path) -> None:
    snapshot = tmp_path / "next-update.pt"
    torch.save({"model": torch.tensor([1.0])}, snapshot)
    report = {"snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest()}
    (tmp_path / "report.json").write_text(json.dumps(report))
    values, _ = comparison._read_capture(tmp_path)
    assert torch.equal(values["model"], torch.tensor([1.0]))
    torch.save({"model": torch.tensor([2.0])}, snapshot)
    with pytest.raises(ValueError, match="capture content differs"):
        comparison._read_capture(tmp_path)


def test_resume_gradient_capture_sees_clipped_gradients_before_optimizer_step() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model(torch.ones(1, 2)).sum().backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    expected = model.weight.grad.clone()
    recorder = capture._GradientCapture(model)
    with optimizer.register_step_pre_hook(recorder):
        optimizer.step()
    model.weight.grad.zero_()
    assert torch.equal(recorder.gradients["weight"], expected)
    assert not torch.equal(recorder.gradients["weight"], model.weight.grad)


def test_resume_raw_capture_records_selected_slots_and_rejects_extra_batch() -> None:
    schedule = SimpleNamespace(selected_slots=lambda head: np.array([1, 0]), window_ordinals=np.array([3, 5]))
    ring = SimpleNamespace(
        schedule=schedule,
        fifo_head=4,
        locators=(PhysicalRow("ranked", 2, 8), PhysicalRow("ranked", 3, 9)),
        replay_ids=("left", "right"),
        epochs=np.array([2, 3]),
    )
    recorder = capture._RawBatchCapture(
        SimpleNamespace(_ring=ring, seed=11),
        lambda replay_ids, columns: (replay_ids, columns),
        lambda batch: batch,
    )
    columns = {"frame": np.arange(4).reshape(2, 2)}
    replay_ids = ("right", "left")
    recorder(replay_ids, columns)
    assert recorder.record["seed"] == 11
    assert recorder.record["window_identities"] == [
        {"replay_id": "right", "locator": {"source": "ranked", "shard": 3, "row": 9}, "epoch": 3, "window_ordinal": 3},
        {"replay_id": "left", "locator": {"source": "ranked", "shard": 2, "row": 8}, "epoch": 2, "window_ordinal": 5},
    ]
    with pytest.raises(RuntimeError, match="more than the single boundary batch"):
        recorder(replay_ids, columns)


class _CountingPrefetch:
    def __init__(self) -> None:
        self.queued = 0
        self.staged = False
        self.loaded = 0
        self.consumed = 0

    @property
    def drained(self) -> bool:
        return self.queued == 0 and not self.staged

    def fill_lookahead(self, limit: int) -> None:
        assert not self.staged and self.queued <= limit
        target = min(4, limit)
        self.loaded += target - self.queued
        self.queued = target

    def stage_next(self) -> None:
        assert self.queued > 0 and not self.staged
        self.queued -= 1
        self.staged = True

    def next(self) -> tuple[int, int]:
        assert self.staged
        self.staged = False
        self.consumed += 1
        return self.consumed, 32


@pytest.mark.parametrize("count", (1, 3, 4, 7))
def test_training_measurement_preserves_update_numbers_and_drains_prefetch(count: int) -> None:
    prefetch = _CountingPrefetch()
    calls = []

    def train_step(**values: object) -> SimpleNamespace:
        calls.append(values)
        return SimpleNamespace(nll_sum=torch.tensor(float(values["update"])))

    losses = performance._run_updates(prefetch, train_step, start_step=8193, count=count)
    assert prefetch.drained
    assert prefetch.loaded == prefetch.consumed == count
    assert [call["step"] for call in calls] == list(range(8193, 8193 + count))
    assert [call["update"] for call in calls] == list(range(8194, 8194 + count))
    assert all(call["valid_prefixes"] == 32 and call["prefix_validated_on_cpu"] is True for call in calls)
    assert [float(loss) for loss in losses] == list(range(8194, 8194 + count))


def test_training_measurement_rejects_undrained_prefetch() -> None:
    prefetch = _CountingPrefetch()
    prefetch.fill_lookahead(1)
    with pytest.raises(ValueError, match="start drained"):
        performance._run_updates(prefetch, None, start_step=8193, count=2)


def test_training_measurement_rejects_short_qualification_window() -> None:
    with pytest.raises(ValueError, match="100 warm and 200 measured"):
        performance.measure_training_updates(
            None, None, next_step=8193, batch_size=512, warm_updates=99, measured_updates=200
        )


def test_training_measurement_serializes_every_prediction_head(monkeypatch: pytest.MonkeyPatch) -> None:
    prefetch = _CountingPrefetch()
    head_losses = torch.arange(64, dtype=torch.float32).reshape(16, 4)

    def train_step(**values: object) -> SimpleNamespace:
        return SimpleNamespace(nll_sum=head_losses + int(values["update"]))

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 128)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 256)
    monkeypatch.setattr(torch.compiler, "set_stance", lambda stance: nullcontext())
    monkeypatch.setattr(performance, "_process_counters", lambda: {"cpu_seconds": 0.0})
    monkeypatch.setattr(performance, "_peak_process_tree_rss", lambda: None)
    monkeypatch.setattr(performance, "read_process_tree_memory", lambda pid: {})
    monkeypatch.setattr(performance, "read_cgroup_memory", lambda: {})
    report = performance.measure_training_updates(
        prefetch, train_step, next_step=8193, batch_size=512, warm_updates=100, measured_updates=200
    )
    saved = json.loads(json.dumps(report, allow_nan=False))
    assert prefetch.drained
    assert prefetch.loaded == prefetch.consumed == 300
    assert (saved["first_measured_update"], saved["last_measured_update"]) == (8294, 8493)
    assert saved["nll_sums"] == [(head_losses + update).tolist() for update in range(8294, 8494)]
    assert saved["process_tree_peak_rss_upper_bound_bytes"] is None
