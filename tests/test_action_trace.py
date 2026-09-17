"""Persisted action-likelihood trace contracts."""

import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from hal.eval.action_trace import ACTION_TRACE_SCHEMA_VERSION
from hal.eval.action_trace import ActionTraceWriter
from hal.training.controller_codec import CONTROLLER_GROUP_NAMES
from hal.training.controller_codec import CONTROLLER_GROUP_VOCABS


def _read_parts(root: Path) -> pa.Table:
    return pa.concat_tables([pq.read_table(path) for path in sorted(root.glob("part-*.parquet"))])


def test_action_trace_records_only_executed_depths_and_their_likelihoods(tmp_path: Path) -> None:
    root = tmp_path / "trace"
    batch = 1
    horizon = 6
    indices = torch.ones(batch, horizon, len(CONTROLLER_GROUP_NAMES), dtype=torch.long)
    logits = []
    for vocab in CONTROLLER_GROUP_VOCABS:
        values = torch.zeros(batch, horizon, vocab)
        values[..., 1] = 2.0
        logits.append(values)
    uniforms = torch.full((horizon, len(CONTROLLER_GROUP_NAMES), batch), 0.75)

    with ActionTraceWriter(root, model="cody", flush_rows=1000) as writer:
        for reset in (True, False, True):
            writer.record_plan(
                decode_seed=17,
                slot_ids=torch.tensor([5]),
                resets=torch.tensor([reset]),
                indices=indices,
                logits=tuple(logits),
                uniforms=uniforms,
                head_offsets=(1, 2, 3, 4, 5, 6),
                temperature=0.5,
                delay_frames=2,
                replan_interval_frames=3,
            )

    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["schema_version"] == ACTION_TRACE_SCHEMA_VERSION
    assert manifest["controller_decode_order"] == ["c_stick", "main_stick", "triggers", "buttons"]

    table = _read_parts(root)
    assert table.num_rows == 3 * 3 * len(CONTROLLER_GROUP_NAMES)
    data = table.to_pydict()
    assert sorted(set(data["plan_depth"])) == [2, 3, 4]
    assert sorted(set(data["execution_frame"])) == [2, 3, 4, 5, 6, 7]
    assert data["generation"].count(0) == 24
    assert data["generation"].count(1) == 12
    assert all(rank == 1 for rank in data["rank"])
    assert [len(values) for values in data["logits"][:4]] == list(CONTROLLER_GROUP_VOCABS)

    first_action = [index for index, frame in enumerate(data["execution_frame"]) if frame == 2][:4]
    joint_log_probability = sum(data["sampled_log_probability"][index] for index in first_action)
    assert all(data["action_log_probability"][index] == pytest.approx(joint_log_probability) for index in first_action)
    assert data["action_probability"][first_action[0]] == pytest.approx(math.exp(joint_log_probability))
    assert data["sampled_probability"][0] > data["base_probability"][0]


def test_action_trace_refuses_to_overwrite_an_existing_directory(tmp_path: Path) -> None:
    root = tmp_path / "trace"
    ActionTraceWriter(root, model="cody").close()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        ActionTraceWriter(root, model="cody")
