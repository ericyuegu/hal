import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from hal.inference.sampling import StreamGroupRng
from hal.models.controller_codec import CONTROLLER_GROUP_NAMES

_SPEC = importlib.util.spec_from_file_location(
    "replay_policy_fault", Path(__file__).resolve().parents[1] / "scripts" / "replay_policy_fault.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
load_fault_inputs = _MODULE.load_fault_inputs


def _write_capsule(path: Path, prefix: int) -> StreamGroupRng:
    rng = StreamGroupRng(17, CONTROLLER_GROUP_NAMES)
    rng.begin([9, 15], [0, 0])
    rng.uniforms("buttons")
    metadata = {
        "schema_version": 2,
        "stream_ids": [9, 15],
        "generations": [1, 1],
        "reset": [False, False],
        "player_id": [4, 7],
        "ctx_pad": [0, 3],
        "desired_return": [19.75, None],
        "temperature": [0.8, 1.1],
        "value_names": ["ego_main_stick_x"],
        "mask_names": ["ego_main_stick_x_mask"],
        "cat_names": ["stage"],
        "emitted_masks": [True],
        "horizon": 4,
        "fixed_prefix_frames": prefix,
        "sampling_seed": 17,
        "sampling_generations": list(rng.generations.items()),
        "sampling_counters": rng.state(),
        "checkpoint_sha256": "a" * 64,
    }
    path.write_text(json.dumps({"schema_version": 1, "policy": metadata}))
    np.savez(
        path.with_suffix(".npz"),
        floats=np.arange(32, dtype=np.float32).reshape(2, 2, 8),
        cats=np.ones((1, 2, 8), dtype=np.int64),
        fixed_actions=np.zeros((2, prefix, 14), dtype=np.float32),
    )
    return rng


@pytest.mark.parametrize("prefix", (0, 2))
def test_fault_replay_preserves_conditioning_prefix_and_rng(tmp_path: Path, prefix: int) -> None:
    capsule = tmp_path / "fault.json"
    control = _write_capsule(capsule, prefix)
    replay = load_fault_inputs(capsule)
    assert replay.fixed_actions.shape == (2, prefix, 14)
    assert replay.player_ids == (4, 7)
    assert replay.desired_returns == (19.75, None)
    assert replay.temperatures == (0.8, 1.1)
    assert replay.checkpoint_sha256 == "a" * 64
    assert replay.context.ctx_pad.tolist() == [0, 3]
    assert torch.equal(replay.context.features["ego_main_stick_x_mask"], torch.arange(16, 32).reshape(2, 8).float())
    restored = replay.sampling()
    restored.begin(replay.stream_ids, tuple(generation - 1 for generation in replay.generations))
    assert torch.equal(control.uniforms("buttons"), restored.uniforms("buttons"))


def test_fault_replay_rejects_old_snapshot_and_wrong_prefix_shape(tmp_path: Path) -> None:
    capsule = tmp_path / "fault.json"
    _write_capsule(capsule, 2)
    payload = json.loads(capsule.read_text())
    payload["policy"]["schema_version"] = 1
    capsule.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="schema-2"):
        load_fault_inputs(capsule)
    payload["policy"]["schema_version"] = 2
    payload["policy"]["fixed_prefix_frames"] = 0
    capsule.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="fixed actions"):
        load_fault_inputs(capsule)
