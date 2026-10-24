"""Attribute asynchronous GPU kernels by CPU launch, not GPU execution time."""

import copy
import dataclasses
import json
import sys
from pathlib import Path

import pytest
import torch
from test_065_loop import transformer_trainer

from hal.eval.self_play_checkpoint import EvaluationCheckpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from scripts import o65_actor_trace as actor_trace
    from scripts.o65_actor_trace import stage_seconds
finally:
    sys.path.pop(0)


def test_actor_trace_uses_the_launch_phase_for_late_kernels(tmp_path: Path) -> None:
    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {"name": "o65_actor/trunk_forward", "cat": "user_annotation", "ts": 0, "dur": 10},
                    {"name": "o65_actor/decoder_forward", "cat": "user_annotation", "ts": 10, "dur": 10},
                    {"name": "o65_actor/trunk_forward", "cat": "gpu_user_annotation", "ts": 0, "dur": 10},
                    {"name": "aten::mm", "cat": "cpu_op", "ts": 5, "args": {"External id": 1}},
                    {"name": "gemm", "cat": "kernel", "ts": 15, "dur": 3, "args": {"External id": 1}},
                ]
            }
        )
    )
    assert stage_seconds(trace) == {"trunk_forward": 3e-6}


def test_actor_trace_rejects_unattributed_work(tmp_path: Path) -> None:
    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {"name": "gemm", "cat": "kernel", "ts": 15, "dur": 3, "args": {"External id": 1}},
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="no CPU launch"):
        stage_seconds(trace)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_actor_trace_times_one_captured_backward(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = transformer_trainer("cpu", decoder_precision="bf16")
    prepared = run.prepare(run.engine.collect(0))
    windows = prepared.windows
    index = int(torch.nonzero(windows.slots.valid_WK)[0, 0])
    ids = torch.tensor([index])
    positions = torch.arange(16)[None].expand(len(windows.rows), -1)
    part = windows.take(ids, positions, ids >= 0)
    part = dataclasses.replace(
        part,
        slots=dataclasses.replace(part.slots, forced_queue_WKD4=torch.zeros(1, 16, 3, 4, dtype=torch.long)),
    )
    context, observed = run.policy._packed_context(part.inputs()[:5])
    checkpoint = EvaluationCheckpoint(
        copy.deepcopy(run.policy.model).cuda(), run.policy.statistics, "a" * 64, "b" * 64, {}
    )
    monkeypatch.setattr(actor_trace, "load_evaluation_checkpoint", lambda *_args, **_kwargs: checkpoint)
    inputs = tmp_path / "inputs.pt"
    torch.save(
        {
            "format": "hal.o65.training-probe.v2",
            "provenance": {"checkpoint_sha256": checkpoint.checkpoint_sha256},
            "source_sha256": "c" * 64,
            "training": {"clip_epsilon": 0.2},
            "cohorts": [
                {
                    "inputs": part.inputs(),
                    "context": context,
                    "observed": observed,
                    "behavior": prepared.rollout.table.behavior[part.rows],
                    "advantages": torch.ones_like(part.rows, dtype=torch.float32),
                    "selected": part.slots.valid_WK,
                    "coefficients": torch.zeros_like(part.rows, dtype=torch.float32),
                    "advantage_mean": torch.tensor(0.0),
                    "advantage_sigma": torch.tensor(1.0),
                    "denominator": part.slots.valid_WK.sum().float(),
                }
            ],
        },
        inputs,
    )
    output = tmp_path / "actor"
    actor_trace.main(actor_trace.Args(inputs, windows=1, output=output))
    report = json.loads(output.with_suffix(".report.json").read_text())
    seconds = report["graph_replay_seconds"]
    assert report["queue_capacity"] == 3
    assert report["decisions"] == 16
    assert len(report["graph_replay_samples"]) == 10
    assert all(value > 0 for value in seconds.values())
    assert seconds["total"] == pytest.approx(sum(value for name, value in seconds.items() if name != "total"))
    assert Path(report["graph_replay_trace"]).is_file()
