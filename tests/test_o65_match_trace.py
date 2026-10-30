"""CPU trace capture and viewer/report contracts on complete synthetic matches."""

import dataclasses
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from typing import cast

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from test_065_loop import FakeShard
from test_065_loop import ToyPolicy
from test_065_loop import exp
from test_065_loop import sampler

from hal.sim import msl_trace
from hal.wire import ACTION_DIM
from hal.wire import MASK_INT32

_REPORT_SPEC = importlib.util.spec_from_file_location(
    "o65_trace_report", Path(__file__).resolve().parents[1] / "scripts/o65_trace_report.py"
)
assert _REPORT_SPEC is not None and _REPORT_SPEC.loader is not None
o65_trace_report = importlib.util.module_from_spec(_REPORT_SPEC)
_REPORT_SPEC.loader.exec_module(o65_trace_report)


class StockShard(FakeShard):
    def step(self, inputs: torch.Tensor, active: NDArray[np.bool_]) -> tuple[Any, ...]:
        rows = []
        for actions in inputs.detach().cpu().numpy():
            self.trace.append(actions.copy())
            before = self.stocks.copy()
            self.ticks += active
            self.stocks[:, 0] = 4 - (self.ticks >= 37)
            self.stocks[:, 1] = np.where(self.ticks >= self.length, 0, 4 - (self.ticks >= 180))
            losses = (self.stocks < before).astype(np.int32)
            finals = losses * (self.stocks == 0)
            base = 120 * (losses[:, 1] - losses[:, 0]) + 50 * (finals[:, 1] - finals[:, 0])
            rewards = np.stack((base, -base), -1).astype(np.float32)
            self.returns += np.where(active[:, None], rewards, 0)
            end = (self.ticks >= self.length) & active
            rows.append(self._frames(actions, rewards, end & ~np.bool_(self.timeout), end & np.bool_(self.timeout)))
            if end.any():
                break
        return tuple(rows)


def make_run(
    *,
    length: int = 300,
    warmup: int = 0,
    timeout: bool = False,
    lanes: int = 8,
    shard_type: type[StockShard] = StockShard,
) -> Any:
    torch.manual_seed(19)
    policy = ToyPolicy("cpu")
    # A second group detects accidental capture outside group zero.
    groups = tuple(
        (shard_type(tuple(range(lanes * group, lanes * group + lanes)), length=length, timeout=timeout),)
        for group in range(2)
    )
    engine = exp.RolloutEngine(groups, policy, sampler(), {0: policy.frozen_copy()})
    engine.initialize()
    return exp.Trainer(
        engine,
        exp.TrainConfig(
            actor_optimizer="adam",
            critic_warmup_updates=warmup,
            advantage_scale="batch",
            windows_per_step=2,
            critic_epochs=1,
            snapshot_interval=100,
        ),
    )


def synthetic_trace(path: Path, *, mismatch: bool = False) -> None:
    columns = np.zeros((5, len(exp.native.OBSERVATION_FIELDS)), dtype=np.uint32)
    for name, values, dtype in (
        ("stage", [25] * 5, np.int32),
        ("p1_character", [1] * 5, np.int32),
        ("p2_character", [18] * 5, np.int32),
        ("p1_stock", [2, 2, 1, 1, 1], np.int32),
        ("p2_stock", [1, 1, 1, 1, 0], np.int32),
        ("p1_action", [14, 14, 14, 44, 44], np.int32),
        ("p1_position_x", [0, 1, 2, 3, 4], np.float32),
        ("p1_direction", [1] * 5, np.float32),
        ("p2_direction", [-1] * 5, np.float32),
    ):
        columns[:, exp.native.OBSERVATION_FIELDS.index(name)] = np.array(values, dtype=dtype).view(np.uint32)
    actions = np.zeros((5, 2, ACTION_DIM), dtype=np.float32)
    actions[1, 0, 0] = 0.75
    actions[1, 0, 6] = 1
    actions[1, 0, 4] = 0.5
    rewards = np.array([[0, 0], [0, 0], [-120, 120], [0, 0], [170, -170]], dtype=np.float32)
    if mismatch:
        rewards[2, 0] += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        header=np.asarray(
            json.dumps(
                {
                    "format": "hal.o65.match-trace.v1",
                    "native_columns": exp.native.OBSERVATION_FIELDS,
                    "resolved": {"train": {"seed": 65}},
                    "start_update": 2,
                }
            )
        ),
        frame=np.arange(5),
        columns=columns,
        applied_actions=actions,
        rewards=rewards,
        terminal=np.array([False] * 4 + [True]),
        timeout=np.zeros(5, dtype=np.bool_),
        decision_frame=np.array([0, 2]),
        decision_side=np.array([0, 0]),
        decision_value_mean=np.array([10.0, 20.0]),
        decision_advantage=np.array([1.0, -2.0]),
        decision_lambda=np.array([0.1, 0.1]),
    )


def test_trace_roundtrip_matches_prepare_actor_steps_and_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    traced = make_run()
    control = make_run()
    tracer = exp.MatchTracer(traced.engine, 2, tmp_path, {"train": dataclasses.asdict(traced.config)})
    traced.engine.match_tracer = tracer
    repeat = exp.MatchTracer(control.engine, 2, tmp_path, {})
    assert tracer.matches == repeat.matches
    assert [traced.engine.lane_roles[m] for m in tracer.matches] == [0, None]
    assert all(m in traced.engine.group_matches[0] for m in tracer.matches)
    observed_ratios: dict[tuple[int, int, int, int], float] = {}
    original_step = traced.policy.actor_step

    def observe_step(part: Any, targets: Any, *, clip_epsilon: float) -> Any:
        loss, output = original_step(part, targets, clip_epsilon=clip_epsilon)
        for row, ratio in zip(
            part.metadata[targets.selected_WK].tolist(), loss.joint_ratio_WK[targets.selected_WK].tolist(), strict=True
        ):
            key = tuple(int(row[c]) for c in (exp.MATCH, exp.EPISODE, exp.CLOCK, exp.SIDE))
            observed_ratios[cast(tuple[int, int, int, int], key)] = float(ratio)
        return loss, output

    monkeypatch.setattr(traced.policy, "actor_step", observe_step)
    expected: dict[tuple[int, int, int], tuple[float, ...]] = {}
    for _ in range(6):
        batch, _ = traced.run_update()
        control_batch, _ = control.run_update()
        torch.testing.assert_close(batch.advantages, control_batch.advantages, rtol=0, atol=0)
        torch.testing.assert_close(batch.values, control_batch.values, rtol=0, atol=0)
        meta = batch.rollout.table.integers[batch.rollout.ids]
        for i in range(len(meta)):
            match, episode, clock, side = (int(meta[i, c]) for c in (exp.MATCH, exp.EPISODE, exp.CLOCK, exp.SIDE))
            if match in tracer.matches and episode == 1:
                expected[(match, clock - 384, side)] = (
                    float(batch.values[i]),
                    float(batch.advantages[i]),
                    float(batch.targets[i]),
                    float(batch.opponent_weights[i]),
                    float(batch.selected[i]),
                )
    assert tracer.written == set(tracer.matches)
    assert not tracer.recordings
    for path in sorted(tmp_path.glob("*.npz")):
        header, arrays = o65_trace_report.load_trace(path)
        match = int(header["match"])
        assert header["format"] == "hal.o65.match-trace.v1"
        assert header["start_update"] == 2
        assert header["start_clock"] == 384
        assert header["episode"]["number"] == 1
        assert arrays["columns"].shape == (301, 71)
        assert arrays["applied_actions"].shape == (301, 2, ACTION_DIM)
        assert not arrays["applied_actions"][0].any()
        assert arrays["terminal"][-1] and not arrays["terminal"][:-1].any()
        np.testing.assert_array_equal(arrays["rewards"], o65_trace_report.stock_rewards(arrays["columns"]))
        for i, (frame, side) in enumerate(zip(arrays["decision_frame"], arrays["decision_side"], strict=True)):
            actual = tuple(
                float(arrays[f"decision_{key}"][i])
                for key in ("value_mean", "advantage", "value_target", "opponent_weight", "selected")
            )
            assert actual == expected[(match, int(frame), int(side))]
        trained = arrays["decision_trained"]
        np.testing.assert_array_equal(trained, arrays["decision_selected"])
        assert np.isfinite(arrays["decision_joint_ratio"][trained]).all()
        for frame, side, ratio in zip(
            arrays["decision_frame"][trained],
            arrays["decision_side"][trained],
            arrays["decision_joint_ratio"][trained],
            strict=True,
        ):
            assert float(ratio) == observed_ratios[(match, 1, int(frame) + 384, int(side))]
        assert np.isnan(arrays["decision_joint_ratio"][~trained]).all()
        assert np.isfinite(arrays["decision_reference_log_probs"]).all()
        assert arrays["decision_sampled_factors"].shape[1] == 16
        if header["opponent_slot"] is None:
            assert set(arrays["decision_side"]) == {0, 1}
    for key, tensor in traced.policy.model.state_dict().items():
        torch.testing.assert_close(tensor, control.policy.model.state_dict()[key], rtol=0, atol=0)
    for traced_group, control_group in zip(traced.engine.groups, control.engine.groups, strict=True):
        np.testing.assert_array_equal(traced_group[0].trace, control_group[0].trace)
    assert o65_trace_report.report(tmp_path, tmp_path / "report") == 0


@pytest.mark.parametrize("timeout", [False, True])
def test_warmup_trace_marks_missing_reference_and_untrained_ratio(tmp_path: Path, timeout: bool) -> None:
    run = make_run(length=23, warmup=100, timeout=timeout)
    tracer = exp.MatchTracer(run.engine, 2, tmp_path, {})
    run.engine.match_tracer = tracer
    run.run_update()
    assert len(list(tmp_path.glob("*.npz"))) == 2
    for path in tmp_path.glob("*.npz"):
        _, arrays = o65_trace_report.load_trace(path)
        assert not arrays["decision_trained"].any()
        assert np.isnan(arrays["decision_joint_ratio"]).all()
        assert np.isnan(arrays["decision_reference_log_probs"]).all()
        assert bool(arrays["timeout"][-1]) == timeout
        assert arrays["frame"].tolist() == list(range(24))


def test_tracer_rejects_excess_or_invalid_lane_count(tmp_path: Path) -> None:
    run = make_run()
    for count in (-1, 0, 100):
        with pytest.raises(ValueError, match="trace matches"):
            exp.MatchTracer(run.engine, count, tmp_path, {})
    larger = make_run(lanes=16)
    odd = exp.MatchTracer(larger.engine, 3, tmp_path, {})
    assert [larger.engine.lane_roles[match] for match in odd.matches] == [0, 0, None]


def test_msltrace_is_read_by_the_viewer_and_has_no_torch_dependency(tmp_path: Path) -> None:
    path = tmp_path / "match.npz"
    synthetic_trace(path)
    _, arrays = o65_trace_report.load_trace(path)
    viewer = msl_trace.to_msltrace(arrays["columns"], arrays["applied_actions"])
    assert "stage" not in viewer and "items" not in viewer
    encoded = tmp_path / "match.json"
    encoded.write_text(json.dumps(viewer, allow_nan=False))
    node = shutil.which("node")
    if node:
        reader = Path.home() / "src/melee-sim-light/tools/viewer/msltrace1.js"
        assert reader.is_file(), "required viewer reader fixture is missing"
        result = subprocess.run(
            [
                node,
                "--input-type=module",
                "-e",
                'import fs from "node:fs"; const {mslTraceToReplayData} = await import(process.argv[2]);'
                'const replay = mslTraceToReplayData(JSON.parse(fs.readFileSync(process.argv[1], "utf8")));'
                "console.log(JSON.stringify(replay));",
                str(encoded),
                reader.as_uri(),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        replay = json.loads(result.stdout)
        assert replay["settings"]["stageId"] == 32
        assert replay["settings"]["playerSettings"][1]["internalCharacterIds"] == [18]
        assert len(replay["frames"]) == 5
        players = [frame["players"][0] for frame in replay["frames"]]
        assert [p["state"]["actionStateFrameCounter"] for p in players] == [0, 1, 2, 0, 1]
        assert players[1]["inputs"]["physical"]["a"]
        assert players[1]["inputs"]["processed"]["joystickX"] == 0.75
        assert players[1]["inputs"]["physical"]["lTriggerAnalog"] == 0.5
        assert players[2]["state"]["stocksRemaining"] == 1
    else:
        assert viewer["format"] == "MSLTRACE1" and viewer["schemaVersion"] == 1
        assert {"charId", "actionId", "actionFrame", "x", "y", "facing", "grounded", "percent", "stocks"} <= set(
            msl_trace.PLAYER_FIELDS
        )
        assert cast(dict[str, object], viewer["frames"])["encoding"] == "sparse-delta-v1"
    subprocess.run(
        [sys.executable, "-c", 'import sys; import hal.sim.msl_trace; assert "torch" not in sys.modules'], check=True
    )


def test_report_cli_writes_artifacts_and_fails_on_reward_mismatch(tmp_path: Path) -> None:
    trace_dir, out_dir = tmp_path / "traces", tmp_path / "report"
    path = trace_dir / "match.npz"
    synthetic_trace(path)
    command = [sys.executable, "scripts/o65_trace_report.py", str(trace_dir), str(out_dir)]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "stock rewards match" in result.stdout
    assert (out_dir / "match.msltrace.json").is_file()
    assert (out_dir / "match.png").read_bytes().startswith(b"\x89PNG")
    csv = (out_dir / "match.rewards.csv").read_text()
    assert "2,0,-120.0,-120.0,0.0,True" in csv
    assert "4,0,170.0,170.0,0.0,True" in csv
    synthetic_trace(path, mismatch=True)
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode == 1 and "Stock reward mismatch" in result.stdout
    assert "2,0,-119.0,-120.0,1.0,False" in (out_dir / "match.rewards.csv").read_text()


def test_report_rejects_wrong_identity_and_incomplete_matches(tmp_path: Path) -> None:
    path = tmp_path / "match.npz"
    synthetic_trace(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = dict(archive)
    arrays["header"] = np.asarray(json.dumps({"format": "hal.o65.match-trace.v2"}))
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="identity or schema"):
        o65_trace_report.load_trace(path)
    synthetic_trace(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = dict(archive)
    arrays["terminal"][-1] = False
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="complete match"):
        o65_trace_report.load_trace(path)


def test_stock_reward_masks_simultaneous_losses_and_final_bonus() -> None:
    columns = np.zeros((5, len(exp.native.OBSERVATION_FIELDS)), dtype=np.uint32)
    for side, stocks in ((1, [MASK_INT32, 4, 3, 2, 0]), (2, [MASK_INT32, 4, 3, 3, 2])):
        columns[:, exp.native.OBSERVATION_FIELDS.index(f"p{side}_stock")] = np.array(stocks, np.int32).view(np.uint32)
    np.testing.assert_array_equal(
        o65_trace_report.stock_rewards(columns), [[0, 0], [0, 0], [0, 0], [-120, 120], [-50, 50]]
    )


def test_training_trace_flag_writes_resolved_settings(tmp_path: Path) -> None:
    run = make_run(length=23, warmup=100)
    provenance = {
        key: "a" * (40 if key == "git_sha" else 64)
        for key in (
            "git_sha",
            "checkpoint_sha256",
            "data_manifest_sha256",
            "simulator_sha256",
            "adapter_sha256",
        )
    }
    exp.run_training(run, 1, tmp_path, provenance, trace_matches=2)
    paths = list((tmp_path / "traces").glob("*.npz"))
    assert len(paths) == 2
    resolved = json.loads((tmp_path / "resolved.json").read_text())
    assert resolved["trace_matches"] == 2
    header, _ = o65_trace_report.load_trace(paths[0])
    assert header["resolved"]["train"] == resolved["train"]
    assert run.engine.match_tracer is None


def test_trace_keeps_raw_applied_actions_when_readback_differs(tmp_path: Path) -> None:
    class ReadbackShard(StockShard):
        def _frames(
            self,
            actions: NDArray[np.float32],
            rewards: NDArray[np.float32],
            terminal: NDArray[np.bool_],
            timeout: NDArray[np.bool_],
        ) -> Any:
            frames = super()._frames(actions, rewards, terminal, timeout)
            return dataclasses.replace(
                frames, host_actions=np.zeros_like(actions), actions=torch.zeros_like(frames.actions)
            )

    run = make_run(length=23, warmup=100, shard_type=ReadbackShard)
    run.engine.match_tracer = exp.MatchTracer(run.engine, 2, tmp_path, {})
    run.run_update()
    submitted = np.stack(run.engine.groups[0][0].trace)
    for path in tmp_path.glob("*.npz"):
        header, arrays = o65_trace_report.load_trace(path)
        np.testing.assert_array_equal(arrays["applied_actions"][1:], submitted[23:46, int(header["match"])])
        assert arrays["applied_actions"].any()
