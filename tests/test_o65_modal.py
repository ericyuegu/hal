import dataclasses
import importlib.util
import json
import shlex
import sys
from pathlib import Path

import pytest
import tyro

_SPEC = importlib.util.spec_from_file_location("o65_modal", Path(__file__).parents[1] / "scripts/launch_o65_modal.py")
assert _SPEC is not None and _SPEC.loader is not None
launcher = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = launcher
_SPEC.loader.exec_module(launcher)
Request = launcher.Request
command = launcher.command
select_cpus = launcher.select_cpus


def test_budget_reserves_failures_and_reconciles_completed_calls(tmp_path: Path) -> None:
    path = tmp_path / "budget.jsonl"
    assert launcher.reserved_seconds(path) == 0
    launcher.record_budget(path, "first", 900, "reserve")
    launcher.record_budget(path, "first", -600, "completed")
    launcher.record_budget(path, "failed", 900, "reserve")
    assert launcher.reserved_seconds(path) == 1200


@pytest.mark.parametrize(
    "change",
    [
        {"format": "old"},
        {"extra": 1},
        {"reserved_gpu_seconds": float("nan")},
        {"reserved_gpu_seconds": True},
        {"reserved_gpu_seconds": -1},
    ],
)
def test_budget_rejects_invalid_or_unversioned_records(tmp_path: Path, change: dict[str, object]) -> None:
    path = tmp_path / "budget.jsonl"
    record = {"format": launcher.BUDGET_FORMAT, "name": "test", "reason": "reserve", "reserved_gpu_seconds": 900}
    record.update(change)
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError):
        launcher.reserved_seconds(path)


@pytest.mark.parametrize(
    "change", [{}, {"format": "old"}, {"exit_code": True}, {"elapsed_seconds": float("inf")}, {"finished_unix": 99}]
)
def test_result_duration_validates_identity(tmp_path: Path, change: dict[str, object]) -> None:
    path = tmp_path / "result.json"
    record = {
        "format": launcher.RESULT_FORMAT,
        "exit_code": 0,
        "started_unix": 100,
        "finished_unix": 103,
        "elapsed_seconds": 3,
    }
    record.update(change)
    path.write_text(json.dumps(record))
    if not change:
        assert launcher.result_seconds(path, 0) == 3
    else:
        with pytest.raises(ValueError, match="result identity"):
            launcher.result_seconds(path, 0)


def test_download_paths_stay_inside_the_named_run() -> None:
    for path in ("run/profile/timed.json", "/run/profile/timed.json"):
        assert launcher.relative_result_path("run", path) == Path("profile/timed.json")
    for path in ("other/file", "run/../outside", "run"):
        with pytest.raises(ValueError):
            launcher.relative_result_path("run", path)
    with pytest.raises(ValueError, match="run name"):
        launcher.relative_result_path("../outside", "../outside/file")


def test_image_delivery_commands_match_the_real_cli() -> None:
    spec = importlib.util.spec_from_file_location(
        "o65_modal_delivery", Path(__file__).parents[1] / "scripts/o65_delivery.py"
    )
    assert spec is not None and spec.loader is not None
    delivery = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = delivery
    spec.loader.exec_module(delivery)
    parsed = []
    for line in launcher.image_commands("a" * 40):
        argv = shlex.split(line)
        if "scripts.o65_delivery" in argv:
            parsed.append(tyro.cli(delivery.Args, args=argv[3:]))
    assert [args.action for args in parsed] == ["install", "verify"]
    assert all(args.output == Path("delivery") for args in parsed)


@pytest.mark.parametrize(
    "field,value",
    [
        ("matches", 64),
        ("matches", True),
        ("name", "../x"),
        ("git_sha", "HEAD"),
        ("timeout_seconds", 0),
        ("created_unix", float("inf")),
        ("created_unix", True),
    ],
)
def test_modal_request_rejects_invalid_config(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(Request("test", "a" * 40), **{field: value})


def test_modal_sdk_rejects_unqualified_versions() -> None:
    launcher.require_modal_version("1.5.3")
    with pytest.raises(ValueError, match="compatibility tests"):
        launcher.require_modal_version("1.5.4")


def test_cpu_selection_uses_distinct_available_ids() -> None:
    assert select_cpus(set(range(1, 37, 2))) == tuple(range(1, 35, 2))
    with pytest.raises(ValueError, match="17 available CPU IDs"):
        select_cpus(set(range(16)))


def test_attempts_share_one_absolute_reservation_deadline() -> None:
    request = Request("test", "a" * 40, timeout_seconds=900, created_unix=100)
    assert launcher.remaining_seconds(request, 200) == 900
    assert launcher.remaining_seconds(request, 800) == 560
    with pytest.raises(ValueError, match="expired"):
        launcher.remaining_seconds(request, 1300)


def test_volume_staging_copies_checkpoint_bytes_and_excludes_caches(tmp_path: Path) -> None:
    source, target = tmp_path / "local", tmp_path / "volume"
    checkpoint = source / "profile/checkpoints/actor.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"durable actor")
    (checkpoint.parent / "linked.pt").hardlink_to(checkpoint)
    for name in ("tmp", "inductor", "triton"):
        (source / name).mkdir()
        (source / name / "cache").write_bytes(b"ephemeral")
    launcher.stage_results(source, target)
    copied = target / "profile/checkpoints/actor.pt"
    assert copied.read_bytes() == b"durable actor"
    assert (copied.parent / "linked.pt").read_bytes() == copied.read_bytes()
    assert copied.stat().st_ino != checkpoint.stat().st_ino
    assert not any((target / name).exists() for name in ("tmp", "inductor", "triton"))


def test_interrupted_staging_keeps_the_previous_checkpoint_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, destination = tmp_path / "source", tmp_path / "volume"
    source.mkdir()
    (source / "pilot-identity.json").write_text("source identity")
    (source / "old.pt").write_bytes(b"old state")
    (source / "update-00000032.json").write_text("old.pt")
    launcher.stage_results(source, destination)
    (source / "new.pt").write_bytes(b"new state")
    (source / "update-00000064.json").write_text("new.pt")
    copied = []
    original = launcher.copy_result

    def interrupt(source_path: str, destination_path: str) -> str:
        copied.append(Path(source_path).name)
        if Path(source_path).name == "update-00000064.json":
            raise InterruptedError("preemption before manifest publication")
        return original(source_path, destination_path)

    monkeypatch.setattr(launcher, "copy_result", interrupt)
    with pytest.raises(InterruptedError):
        launcher.stage_results(source, destination)
    assert copied[0] == "pilot-identity.json"
    assert copied.index("new.pt") < copied.index("update-00000064.json")
    assert (destination / "old.pt").read_bytes() == b"old state"
    assert not (destination / "update-00000064.json").exists()


def test_failed_pilot_preserves_the_published_resume_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import Mock

    results, work = tmp_path / "results", tmp_path / "work"
    published = results / "pilot/pilot"
    published.mkdir(parents=True)
    (published / "checkpoint").write_text("published update 32")

    def run_child(request: Request, output: Path, started: float) -> int:
        (output / "pilot/checkpoint").write_text("unpublished update 64")
        return 1

    monkeypatch.setattr(launcher, "RESULTS", results)
    monkeypatch.setattr(launcher, "WORK", work)
    monkeypatch.setattr(launcher, "run_child", run_child)
    monkeypatch.setattr(launcher.modal.Volume, "from_name", Mock(return_value=Mock()))
    assert launcher.run_remote(Request("pilot", "a" * 40, mode="pilot")) == 1
    assert (published / "checkpoint").read_text() == "published update 32"
    assert (results / "pilot/last-attempt/pilot/checkpoint").read_text() == "unpublished update 64"


def test_learner_migration_uses_read_only_source_volume_and_new_run() -> None:
    source = "o65-pilot-3d4ad115/pilot/train/checkpoints/update-00000256.json"
    request = Request("o65-fixed-resume", "a" * 40, mode="pilot", resume_learner=source)
    argv = command(request, tuple(range(16)))
    assert argv[argv.index("--resume-learner") + 1] == f"/o65-results/{source}"
    assert argv[argv.index("--output") + 1] == "/opt/o65-work/o65-fixed-resume/pilot"
    for path in ("../" + source, "/" + source, source.replace("256", "255"), "o65-fixed-resume/" + source):
        with pytest.raises(ValueError):
            dataclasses.replace(request, resume_learner=path)
