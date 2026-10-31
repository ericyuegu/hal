import contextlib
import dataclasses
import hashlib
import json
import shlex
from pathlib import Path
from unittest.mock import Mock

import pytest
import tyro
from test_o65_modal import launcher

pilot = launcher.pilot


def requests(sweep: str = "collapse") -> tuple[launcher.Request, ...]:
    return tuple(
        launcher.Request(
            f"{sweep}-{arm.name}", "a" * 40, mode="ablation", timeout_seconds=9000, arm=arm.name, sweep=sweep
        )
        for arm in pilot.ARMS
    )


def test_ledger_reserves_all_arms_and_refuses_over_65_dollars(tmp_path: Path) -> None:
    path = tmp_path / "budget.jsonl"
    launcher.reserve_ablation(path, requests())
    original = path.read_bytes()
    assert len(original.splitlines()) == 5
    with path.open() as log:
        balances = launcher.read_ablation_budget(log)
    assert set(balances) == {request.name for request in requests()}
    assert (
        sum(balance.reserved for balance in balances.values()) * launcher.compute_rate() + launcher.BUILD_RESERVE_USD
        <= 65
    )
    with pytest.raises(ValueError, match=r"\$65 budget"):
        launcher.reserve_ablation(path, requests("second"))
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="already reserved"):
        launcher.reserve_ablation(path, requests())


def test_ledger_settles_each_arm_once_including_failed_calls(tmp_path: Path) -> None:
    path = tmp_path / "budget.jsonl"
    launcher.reserve_ablation(path, requests())
    assert launcher.settle_ablation(path, "collapse-old", 600) == 960
    original = path.read_bytes()
    assert launcher.settle_ablation(path, "collapse-old", 600) == 960
    assert path.read_bytes() == original
    assert launcher.settle_ablation(path, "collapse-reward", 20000) == 9360
    for elapsed in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="elapsed"):
            launcher.settle_ablation(path, "collapse-scale", elapsed)


def test_deadline_starts_in_the_container_and_survives_rescheduling(tmp_path: Path) -> None:
    request = dataclasses.replace(requests()[0], created_unix=100)
    started = launcher.first_start(request, tmp_path, 20000)
    assert launcher.remaining_seconds(request, 20000, first_started=started) == 9000
    restarted = launcher.first_start(request, tmp_path, 21000)
    assert restarted == started
    assert launcher.remaining_seconds(request, 21000, first_started=restarted) == 8000
    with pytest.raises(ValueError, match="expired"):
        launcher.remaining_seconds(request, 29000, first_started=restarted)
    with pytest.raises(ValueError, match="first start"):
        launcher.remaining_seconds(request, 20000)
    with pytest.raises(ValueError, match="identity"):
        launcher.first_start(dataclasses.replace(request, git_sha="b" * 40), tmp_path, 21000)


def test_compile_cache_is_seeded_once_and_writes_stay_in_the_arm(tmp_path: Path) -> None:
    shared, old, reward = (tmp_path / name for name in ("shared", "old", "reward"))
    shared.mkdir()
    (shared / "artifact").write_bytes(b"shared seed")
    launcher.seed_compile_cache(shared, old)
    launcher.seed_compile_cache(shared, reward)
    (old / "artifact").write_bytes(b"old artifact")
    launcher.seed_compile_cache(shared, old)
    assert (old / "artifact").read_bytes() == b"old artifact"
    assert (reward / "artifact").read_bytes() == (shared / "artifact").read_bytes() == b"shared seed"
    for request in requests():
        argv = launcher.command(request, tuple(range(16)))
        args = tyro.cli(pilot.Args, args=argv[2:])
        assert args.compile_cache == launcher.WORK / f"{request.name}-compile-cache"
        assert args.publish_directory == launcher.RESULTS / request.name / "pilot"
        assert args.r2_prefix == f"o65/pilot-results/{request.name}"
        assert args.arm == request.arm and args.sweep == "collapse"


def test_ablation_dry_run_prints_and_writes_all_full_argv_without_modal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for name in ("App", "Secret", "Volume", "FunctionCall"):
        monkeypatch.setattr(launcher.modal, name, Mock(side_effect=AssertionError("dry run contacted Modal")))
    monkeypatch.setattr(launcher, "build_image", Mock(side_effect=AssertionError("dry run built a Modal image")))
    launcher.main(launcher.Args(mode="ablation", sweep="collapse", output=tmp_path, dry_run=True))
    output = capsys.readouterr().out
    plan = json.loads((tmp_path / "collapse-dry-run.json").read_text())
    assert len(plan["arms"]) == 5
    for entry in plan["arms"]:
        assert entry["request"]["timeout_seconds"] == 9000
        argv = entry["trainer_argv"]
        assert shlex.join(argv) in output
        assert argv[:3] == ["/opt/venv/bin/python", "/opt/o65-hal/experiments/065_selfplay_ppo.py", "train"]
        assert argv[argv.index("--updates") + 1] == "672"
        assert argv[argv.index("--training.seed") + 1] == "65"
        assert argv[argv.index("--training.critic-warmup-updates") + 1] == "32"
        assert argv[argv.index("--trace-matches") + 1] == "4"
        assert argv[argv.index("--publication.r2-prefix") + 1] == f"o65/pilot-results/{entry['request']['name']}"
    assert not list(tmp_path.glob("*-call.json"))


def test_interrupted_cache_copy_leaves_no_partial_arm_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shared, own = tmp_path / "shared", tmp_path / "old"
    shared.mkdir()
    (shared / "seed").write_bytes(b"shared")
    original = launcher.shutil.copytree

    def interrupt(source: Path, destination: Path) -> None:
        destination.mkdir()
        (destination / "partial").write_bytes(b"unfinished")
        raise InterruptedError("copy interrupted")

    monkeypatch.setattr(launcher.shutil, "copytree", interrupt)
    with pytest.raises(InterruptedError):
        launcher.seed_compile_cache(shared, own)
    assert not own.exists()
    assert not own.with_name(f".{own.name}.partial").exists()
    assert (shared / "seed").read_bytes() == b"shared"
    monkeypatch.setattr(launcher.shutil, "copytree", original)
    launcher.seed_compile_cache(shared, own)
    assert (own / "seed").read_bytes() == b"shared"


def test_unknown_arm_is_rejected_before_any_modal_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    modal_call = Mock(side_effect=AssertionError("contacted Modal"))
    monkeypatch.setattr(launcher.modal.Secret, "from_name", modal_call)
    # Direct dataclass construction must validate the same names as the CLI.
    with pytest.raises(ValueError, match="unknown ablation arm"):
        launcher.main(
            dataclasses.replace(
                launcher.Args(mode="ablation", sweep="collapse", output=tmp_path, dry_run=True), arms=("typo",)
            )
        )
    modal_call.assert_not_called()
    with pytest.raises(SystemExit):
        tyro.cli(launcher.Args, args=["--mode", "ablation", "--sweep", "collapse", "--arms", "typo"])


def test_launch_builds_one_image_detaches_and_reserves_before_every_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, budget = tmp_path / "output", tmp_path / "budget.jsonl"
    sha = "a" * 40

    def git(argv: list[str], **_: object) -> str:
        if argv[1] == "status":
            return ""
        if argv[1] == "ls-remote":
            return sha + " refs/heads/test"
        return "test" if argv[1] == "branch" else sha

    monkeypatch.setattr(launcher.subprocess, "check_output", git)
    secret, volume, image, app, worker = (Mock() for _ in range(5))
    app.run.return_value = contextlib.nullcontext()
    app.app_id = "ap-test"
    app.function.return_value = lambda function: worker
    builds = Mock(return_value=image)
    monkeypatch.setattr(launcher, "build_image", builds)
    monkeypatch.setattr(launcher.modal, "App", Mock(return_value=app))
    monkeypatch.setattr(launcher.modal.Secret, "from_name", Mock(return_value=secret))
    monkeypatch.setattr(launcher.modal.Volume, "from_name", Mock(return_value=volume))
    monkeypatch.setattr(launcher.modal, "enable_output", contextlib.nullcontext)
    spawned: list[str] = []

    def spawn(request: launcher.Request) -> Mock:
        with budget.open() as log:
            assert len(launcher.read_ablation_budget(log)) == 5
        spawned.append(request.name)
        return Mock(object_id="fc-" + request.name)

    worker.spawn.side_effect = spawn
    launcher.launch_ablation(launcher.Args(mode="ablation", sweep="collapse", output=output), budget)
    builds.assert_called_once_with(sha, secret)
    app.run.assert_called_once_with(detach=True)
    assert app.function.call_args.kwargs["max_containers"] == 5
    assert app.function.call_args.kwargs["timeout"] == 9000
    assert spawned == [request.name for request in requests()]
    assert len(list(output.glob("*-call.json"))) == 5
    for path in output.glob("*-call.json"):
        assert json.loads(path.read_text())["call_id"].startswith("fc-collapse-")
    worker.get.assert_not_called()


def test_collect_waits_downloads_hashes_and_does_not_double_credit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    budget = tmp_path / "budget.jsonl"
    arms = requests()
    launcher.reserve_ablation(budget, arms)
    plan = {
        "format": "hal.o65.modal-ablation-plan.v1",
        "sweep": "collapse",
        "arms": [{"request": dataclasses.asdict(request)} for request in arms],
    }
    (tmp_path / "collapse-plan.json").write_text(json.dumps(plan))
    for request in arms:
        (tmp_path / f"{request.name}-call.json").write_text(
            json.dumps(
                {
                    "format": "hal.o65.modal-call.v1",
                    "app_id": "ap-test",
                    "call_id": "fc-" + request.name,
                    "request": dataclasses.asdict(request),
                }
            )
        )
    handles: list[Mock] = []

    def from_id(call_id: str) -> Mock:
        assert call_id.startswith("fc-collapse-")
        call = Mock()
        call.get.return_value = 0
        handles.append(call)
        return call

    downloads: list[str] = []

    def download(name: str, output: Path) -> None:
        downloads.append(name)
        directory = output / name
        directory.mkdir(exist_ok=True)
        (directory / "artifact").write_bytes(name.encode())
        (directory / "result.json").write_text(
            json.dumps(
                {
                    "format": launcher.RESULT_FORMAT,
                    "exit_code": 0,
                    "started_unix": 20000,
                    "finished_unix": 20600,
                    "elapsed_seconds": 600,
                }
            )
        )

    monkeypatch.setattr(launcher.modal.FunctionCall, "from_id", from_id)
    monkeypatch.setattr(launcher, "download", download)
    launcher.collect_ablation("collapse", tmp_path, budget)
    original = budget.read_bytes()
    original_checksums = (tmp_path / "collapse-old/SHA256SUMS").read_bytes()
    launcher.collect_ablation("collapse", tmp_path, budget)
    assert budget.read_bytes() == original
    assert (tmp_path / "collapse-old/SHA256SUMS").read_bytes() == original_checksums
    assert downloads == [request.name for request in arms] * 2
    assert len(handles) == 10
    for handle in handles:
        handle.get.assert_called_once_with()
    for request in arms:
        expected = hashlib.sha256(request.name.encode()).hexdigest()
        assert f"{expected}  artifact\n" in (tmp_path / request.name / "SHA256SUMS").read_text()
    with budget.open() as log:
        assert all(balance.charged == 960 for balance in launcher.read_ablation_budget(log).values())
