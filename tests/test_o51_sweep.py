"""Exact O51 sweep-arm and launch-state contracts."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest

from hal.data.o51 import D0
from hal.training.o51_sweep import SweepArm
from hal.training.o51_sweep import Treatment
from hal.training.o51_sweep import batch_arms
from hal.training.o51_sweep import decay_arms
from hal.training.o51_sweep import duration_arms
from hal.training.o51_sweep import initialization_screen_arms
from hal.training.o51_sweep import lr_arms
from hal.training.o51_sweep import mid_search_arms
from hal.training.o51_sweep import proxy_transfer_arms
from hal.training.o51_sweep import seed_repeat_arms

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load("test_run_o51_sweep", ROOT / "scripts" / "run_o51_sweep.py")
EXPERIMENT = _load("test_o51_sweep_experiment", ROOT / "experiments" / "051_correct_parameterization.py")


def _flag(argv: tuple[str, ...], name: str) -> str:
    index = argv.index(name)
    return argv[index + 1]


def _all_arms() -> tuple[SweepArm, ...]:
    center = Treatment()
    return (
        *initialization_screen_arms(center),
        *lr_arms(center),
        *decay_arms(center),
        *batch_arms(center),
        *proxy_transfer_arms(center),
        *mid_search_arms(center),
        *seed_repeat_arms(center),
        *duration_arms(center),
    )


def _prepared(arm: SweepArm, *, report: Path | None = None, git_sha: str = "a" * 40):
    reports = {} if report is None else {arm.arm_id: report}
    return RUNNER._prepare_arms(
        (arm,),
        reports,
        git_sha=git_sha,
        gpu="B200",
        disk_gib=2048,
        require_preflight=True,
    )[0]


def test_initialization_screen_has_six_d0_arms_stopped_at_d0_over_8() -> None:
    arms = initialization_screen_arms()

    assert len(arms) == 6
    assert len({arm.arm_id for arm in arms}) == 6
    assert {(arm.treatment.hidden_std_multiplier, arm.treatment.readout_init) for arm in arms} == {
        (hidden, readout) for hidden in (0.5, 1.0, 2.0) for readout in ("zero", "mup-normal")
    }
    assert {arm.level for arm in arms} == {"base"}
    assert {arm.target_positions for arm in arms} == {D0}
    assert {arm.tier_scale for arm in arms} == {1}
    assert {arm.stop_after_update for arm in arms} == {2048}
    assert {
        arm.stop_after_update * arm.treatment.batch_size * 128 for arm in arms if arm.stop_after_update is not None
    } == {D0 // 8}


@pytest.mark.parametrize(
    ("factory", "count"),
    [
        (lr_arms, 9),
        (decay_arms, 9),
        (batch_arms, 8),
        (proxy_transfer_arms, 2),
        (mid_search_arms, 9),
        (seed_repeat_arms, 2),
        (duration_arms, 6),
    ],
)
def test_declared_stage_grid_has_exact_arm_count(factory, count: int) -> None:
    arms = factory(Treatment())

    assert len(arms) == count
    assert len({arm.arm_id for arm in arms}) == count


def test_batch_2048_is_added_only_after_memory_gate() -> None:
    assert {arm.treatment.batch_size for arm in batch_arms(Treatment())} == {128, 256, 512, 1024}
    gated = batch_arms(Treatment(), include_2048=True)
    assert len(gated) == 10
    assert {arm.treatment.batch_size for arm in gated} == {128, 256, 512, 1024, 2048}


def test_every_arm_uses_an_exact_matched_data_endpoint() -> None:
    for arm in _all_arms():
        assert arm.target_positions == arm.tier_scale * D0

    assert {(arm.target_positions, arm.tier_scale) for arm in duration_arms(Treatment())} == {
        (2 * D0, 2),
        (4 * D0, 4),
        (8 * D0, 8),
    }


def test_duration_arms_compare_both_muon_duration_rules() -> None:
    assert {arm.treatment.muon_duration_scaling for arm in duration_arms(Treatment())} == {
        "fixed",
        "inverse-sqrt",
    }


def test_initialization_screen_is_the_only_smoke_stage() -> None:
    screen = initialization_screen_arms()[0]
    screen_argv = screen.argv()
    production = proxy_transfer_arms(Treatment())[0]
    production_argv = production.argv(preflight_report=Path("evidence/proxy-b512.json"))

    assert "--smoke" in screen_argv
    assert _flag(screen_argv, "--smoke-eval-matchups") == "0"
    assert _flag(screen_argv, "--stop-after-update") == "2048"
    assert "--preflight-report" not in screen_argv
    assert "--smoke" not in production_argv
    assert "--smoke-eval-matchups" not in production_argv
    assert _flag(production_argv, "--preflight-report") == "evidence/proxy-b512.json"


def test_train_commands_parse_and_validate_under_o51() -> None:
    for arm in _all_arms():
        report = Path("evidence.json") if arm.requires_preflight else None
        parsed = EXPERIMENT.tyro.cli(EXPERIMENT.Command, args=list(arm.argv(preflight_report=report)[3:]))
        assert isinstance(parsed, EXPERIMENT.TrainArgs)
        cfg = replace(parsed.cfg, arch=EXPERIMENT.MODEL_FAMILY[parsed.level])
        EXPERIMENT.validate_config(cfg)
        assert cfg.target_positions == arm.target_positions
        assert cfg.tier_scale == arm.tier_scale
        assert cfg.adam_eps == 1e-12


@pytest.mark.parametrize(
    "changes",
    [
        {"muon_lr": 0.0},
        {"adam_lr": float("nan")},
        {"adam_weight_decay": -0.1},
        {"num_workers": 12},
        {"predownload": 31},
        {"shuffle_block_size": 1024},
    ],
)
def test_treatment_rejects_values_outside_the_o51_search(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        Treatment(**changes)


def test_treatment_loads_partial_overrides_and_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "treatment.json"
    path.write_text('{"muon_lr": 0.014}\n')
    assert Treatment.load(path).muon_lr == 0.014

    path.write_text('{"unknown": 1}\n')
    with pytest.raises(ValueError, match="unknown fields"):
        Treatment.load(path)


def test_sweep_arm_rejects_a_mismatched_data_endpoint() -> None:
    with pytest.raises(ValueError, match="matched D/U endpoint"):
        replace(lr_arms(Treatment())[0], tier_scale=2)


def test_production_preflight_map_supports_override_and_fallback(tmp_path: Path) -> None:
    path = tmp_path / "reports.json"
    arm = lr_arms(Treatment())[0]
    path.write_text(json.dumps({"*": "default.json", arm.arm_id: "selected.json"}))

    reports = RUNNER._preflight_reports(path)

    assert RUNNER._report_for(arm, reports, required=True) == Path("selected.json")
    assert RUNNER._report_for(lr_arms(Treatment())[1], reports, required=True) == Path("default.json")


def test_production_launch_requires_preflight_before_any_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(RUNNER, "_git_sha", lambda: "a" * 40)
    monkeypatch.setattr(
        RUNNER.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("missing preflight evidence started a subprocess"),
    )

    with pytest.raises(ValueError, match="needs a preflight report"):
        RUNNER.launch(RUNNER.LaunchArgs(stage="lr", state=tmp_path / "state.jsonl", dry_run=True))


def test_production_dry_run_injects_preflight_without_writing_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    reports = tmp_path / "reports.json"
    reports.write_text('{"*": "pyproject.toml"}\n')
    state = tmp_path / "state.jsonl"
    monkeypatch.setattr(RUNNER, "_git_sha", lambda: "a" * 40)

    RUNNER.launch(
        RUNNER.LaunchArgs(
            stage="lr",
            state=state,
            preflight_reports=reports,
            max_arms=1,
            dry_run=True,
        )
    )

    command = json.loads(capsys.readouterr().out)["command"]
    assert "--preflight-report" in command
    assert "pyproject.toml" in command
    assert "--smoke" not in command
    assert not state.exists()
    assert not state.with_suffix(".jsonl.lock").exists()


def test_preflight_report_must_be_a_tracked_image_file(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text("{}\n")

    with pytest.raises(ValueError, match="relative to the repository"):
        RUNNER._validated_preflight_reports({"*": report})


def test_successful_launch_journals_before_and_after_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = tmp_path / "state.jsonl"
    args = RUNNER.LaunchArgs(stage="initialization-screen", state=state)
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(RUNNER, "_git_sha", lambda: "a" * 40)

    def submit(command: tuple[str, ...], **_kwargs) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="submitted Modal App ap-test123, Function call fc-test456\n",
            stderr="",
        )

    monkeypatch.setattr(RUNNER.subprocess, "run", submit)

    RUNNER.launch(args)

    lines = [json.loads(line) for line in state.read_text().splitlines()]
    assert [line["event"] for line in lines] == ["launching", "launched"] * 6
    assert all(lines[index]["attempt_id"] == lines[index + 1]["attempt_id"] for index in range(0, 12, 2))
    assert lines[1]["app_id"] == "ap-test123"
    assert lines[1]["function_call_id"] == "fc-test456"
    assert len(calls) == 6
    assert all("--smoke" in call for call in calls)

    monkeypatch.setattr(
        RUNNER.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("a completed arm launched twice"),
    )
    RUNNER.launch(args)


def test_unresolved_launch_attempt_blocks_an_automatic_retry(tmp_path: Path) -> None:
    arm = initialization_screen_arms()[0]
    prepared = _prepared(arm)
    record = RUNNER._event_payload(prepared, "a" * 40, "attempt-1", "launching")
    RUNNER._append(tmp_path / "state.jsonl", record)

    previous = RUNNER._records(tmp_path / "state.jsonl")

    with pytest.raises(RuntimeError, match="reconcile Modal"):
        RUNNER._pending_arms((prepared,), previous)


def test_changed_launch_spec_cannot_reuse_an_arm_id(tmp_path: Path) -> None:
    arm = initialization_screen_arms()[0]
    prepared = _prepared(arm)
    launched = RUNNER._event_payload(prepared, "a" * 40, "attempt-1", "launching")
    completed = RUNNER._event_payload(prepared, "a" * 40, "attempt-1", "launched")
    completed.update(app_id="ap-one", function_call_id="fc-one")
    RUNNER._append(tmp_path / "state.jsonl", launched)
    RUNNER._append(tmp_path / "state.jsonl", completed)
    changed = _prepared(arm, git_sha="b" * 40)

    with pytest.raises(RuntimeError, match="different launch specification"):
        RUNNER._pending_arms((changed,), RUNNER._records(tmp_path / "state.jsonl"))


def test_launch_state_rejects_a_truncated_record(tmp_path: Path) -> None:
    state = tmp_path / "state.jsonl"
    state.write_text('{"schema_version": 2')

    with pytest.raises(ValueError, match="invalid launch state"):
        RUNNER._records(state)


def test_launch_state_lock_rejects_a_second_writer(tmp_path: Path) -> None:
    state = tmp_path / "state.jsonl"

    def acquire_second_lock() -> None:
        with RUNNER._state_lock(state):
            pytest.fail("two launchers acquired the same state lock")

    with RUNNER._state_lock(state), pytest.raises(RuntimeError, match="another O51 sweep launcher"):
        acquire_second_lock()


def test_launch_args_keep_the_selected_b200_and_disk_floor() -> None:
    RUNNER._validate_launch_args(RUNNER.LaunchArgs(stage="initialization-screen"))
    with pytest.raises(ValueError, match="B200"):
        RUNNER._validate_launch_args(RUNNER.LaunchArgs(stage="initialization-screen", gpu="H100"))
    with pytest.raises(ValueError, match="2048 GiB"):
        RUNNER._validate_launch_args(RUNNER.LaunchArgs(stage="initialization-screen", disk_gib=1024))
