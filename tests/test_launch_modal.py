import importlib.util
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import boto3
import pytest
from botocore.exceptions import ClientError

_SPEC = importlib.util.spec_from_file_location(
    "hal_launch_modal", Path(__file__).parents[1] / "scripts" / "launch_modal.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

Args = _MODULE.Args
RunState = _MODULE.RunState
explicit_resume = _MODULE.explicit_resume
function_resources = _MODULE.function_resources
gpu_request = _MODULE.gpu_request
plan_attempt = _MODULE.plan_attempt
preflight_git = _MODULE.preflight_git
preflight_modal = _MODULE.preflight_modal
read_state = _MODULE.read_state
redact_argv = _MODULE.redact_argv
retry_policy = _MODULE.retry_policy
validate_args = _MODULE.validate_args
write_state = _MODULE.write_state

EXPERIMENT = "experiments/028_onehot_controller.py"


def test_defaults_request_l40s_with_burst_cpu_and_ephemeral_ssd() -> None:
    args = Args(cmd=["uv", "run", EXPERIMENT])

    assert function_resources(args) == {
        "gpu": "L40S",
        "cpu": (8.0, 16.0),
        "memory": 64 * 1024,
        "ephemeral_disk": 512 * 1024,
        "timeout": 24 * 60 * 60,
        "startup_timeout": 30 * 60,
        "cloud": None,
        "region": None,
    }


def test_gpu_request_accepts_fallbacks_and_rejects_empty() -> None:
    assert gpu_request("L40S,A100-80GB") == ["L40S", "A100-80GB"]
    with pytest.raises(SystemExit, match="at least one"):
        gpu_request(" , ")


def test_validate_args_requires_experiment_for_auto_resume() -> None:
    validate_args(Args(cmd=["uv", "run", EXPERIMENT]))
    validate_args(Args(cmd=["python", "train.py"], auto_resume=False))

    with pytest.raises(SystemExit, match="automatic recovery"):
        validate_args(Args(cmd=["python", "train.py"]))


def test_explicit_resume_supports_both_forms_and_rejects_ambiguity() -> None:
    assert explicit_resume(("python", EXPERIMENT, "--resume", "run-1")) == "run-1"
    assert explicit_resume(("python", EXPERIMENT, "--resume=run-1")) == "run-1"
    with pytest.raises(SystemExit, match="more than once"):
        explicit_resume(("python", EXPERIMENT, "--resume=one", "--resume", "two"))
    with pytest.raises(SystemExit, match="invalid"):
        explicit_resume(("python", EXPERIMENT, "--resume", "../escape"))


def test_plan_attempt_resumes_only_after_checkpoint_exists() -> None:
    command = ("uv", "run", EXPERIMENT)
    running = RunState(status="running", run_name="run-1")

    resumed = plan_attempt(running, command, auto_resume=True, checkpoint_found=True)
    assert resumed.action == "run"
    assert resumed.argv == (*command, "--resume", "run-1")
    assert resumed.state == running

    fresh = plan_attempt(running, command, auto_resume=True, checkpoint_found=False)
    assert fresh.argv == command
    assert fresh.state == RunState(status="running")


def test_plan_attempt_does_not_rerun_terminal_states() -> None:
    command = ("uv", "run", EXPERIMENT)
    failed = RunState(status="failed", run_name="run-1", exit_code=2)
    succeeded = RunState(status="succeeded", run_name="run-1")

    assert plan_attempt(failed, command, auto_resume=True, checkpoint_found=True).action == "fail"
    assert plan_attempt(succeeded, command, auto_resume=True, checkpoint_found=True).action == "complete"


def test_explicit_resume_is_not_duplicated() -> None:
    command = ("uv", "run", EXPERIMENT, "--resume=run-1")
    attempt = plan_attempt(None, command, auto_resume=True, checkpoint_found=False)

    assert attempt.argv == command
    assert attempt.state.run_name == "run-1"


def test_state_round_trip_and_validation(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    state = RunState(status="failed", run_name="run-1", exit_code=7)

    write_state(path, state)

    assert read_state(path) == state
    assert set(path.read_text().split('"')) >= {"status", "run_name", "exit_code", "schema"}
    with pytest.raises(ValueError, match="must include"):
        RunState(status="failed")


def test_checkpoint_detection_handles_not_found_and_propagates_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class Client:
        def __init__(self, code: str | None) -> None:
            self.code = code

        def head_object(self, **_kwargs: str) -> None:
            if self.code is not None:
                raise ClientError({"Error": {"Code": self.code}}, "HeadObject")

    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://example.invalid")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "id")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("AWS_BUCKET", "bucket")
    monkeypatch.setattr(boto3, "client", lambda *_args, **_kwargs: Client("404"))
    assert not _MODULE._checkpoint_exists("run-1")

    monkeypatch.setattr(boto3, "client", lambda *_args, **_kwargs: Client(None))
    assert _MODULE._checkpoint_exists("run-1")

    monkeypatch.setattr(boto3, "client", lambda *_args, **_kwargs: Client("AccessDenied"))
    with pytest.raises(ClientError):
        _MODULE._checkpoint_exists("run-1")


def test_preflight_git_rejects_dirty_tree_and_pushes_unpublished_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_MODULE, "_git", lambda *_args: " M file.py")
    with pytest.raises(SystemExit, match="dirty"):
        preflight_git()

    values = {
        ("status", "--porcelain"): "",
        ("rev-parse", "HEAD"): "a" * 40,
        ("branch", "-r", "--contains", "a" * 40): "",
        ("rev-parse", "--abbrev-ref", "HEAD"): "feature",
    }
    calls: list[list[str]] = []
    monkeypatch.setattr(_MODULE, "_git", lambda *args: values[args])
    monkeypatch.setattr(_MODULE.subprocess, "run", lambda argv, **_kwargs: calls.append(argv))

    assert preflight_git() == "a" * 40
    assert calls == [["git", "push", "origin", "feature"]]


def test_modal_preflight_verifies_client_and_required_secret_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[object] = []
    client = SimpleNamespace(hello=lambda: events.append("hello"))

    class ClientFactory:
        @staticmethod
        def from_env() -> object:
            events.append("client")
            return client

    class SecretHandle:
        def hydrate(self, *, client: object) -> None:
            events.append(("hydrate", client))

    class SecretFactory:
        @staticmethod
        def from_name(name: str, *, required_keys: list[str]) -> SecretHandle:
            events.append((name, tuple(required_keys)))
            return SecretHandle()

    monkeypatch.setattr(_MODULE.modal, "Client", ClientFactory)
    monkeypatch.setattr(_MODULE.modal, "Secret", SecretFactory)

    found_client, _secret = preflight_modal("hal")

    assert found_client is client
    assert events == ["client", "hello", ("hal", _MODULE.REQUIRED_SECRET_KEYS), ("hydrate", client)]


def test_retry_policy_uses_requested_attempt_count_without_delay() -> None:
    retries = retry_policy(10)

    assert retries.max_retries == 10
    assert retries.initial_delay.total_seconds() == 0


def test_redact_argv_hides_conventional_secret_values() -> None:
    rendered = redact_argv(("python", "job.py", "--token", "token-value", "--password=hunter2", "--name", "visible"))

    assert "token-value" not in rendered
    assert "hunter2" not in rendered
    assert "visible" in rendered


def test_dry_run_does_not_create_a_volume_or_app(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_MODULE, "preflight_git", lambda: "a" * 40)
    monkeypatch.setattr(_MODULE, "preflight_modal", lambda _name: (object(), object()))
    monkeypatch.setattr(
        _MODULE.modal.Volume,
        "from_name",
        lambda *_args, **_kwargs: pytest.fail("dry run created a Modal Volume"),
    )

    _MODULE.main(Args(cmd=["uv", "run", EXPERIMENT], dry_run=True))


def test_image_separates_dependency_and_source_layers(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple] = []

    class FakeImage:
        def add_local_file(self, local: Path, remote: str, *, copy: bool) -> FakeImage:
            events.append(("file", local, remote, copy))
            return self

        def add_local_dir(self, local: Path, remote: str, *, copy: bool, ignore: object) -> FakeImage:
            events.append(("dir", local, remote, copy, ignore))
            return self

        def workdir(self, path: str) -> FakeImage:
            events.append(("workdir", path))
            return self

        def run_commands(self, command: str) -> FakeImage:
            events.append(("run", command))
            return self

    fake = FakeImage()
    monkeypatch.setattr(_MODULE.modal.Image, "from_registry", lambda tag: events.append(("base", tag)) or fake)
    monkeypatch.setattr(_MODULE.modal.FilePatternMatcher, "from_file", lambda path: ("ignore", path))

    assert _MODULE._image("example/image:tag") is fake
    dependency_run = events.index(("run", f"UV_INDEX_URL={_MODULE.PYPI_INDEX} uv sync --locked --no-install-project"))
    source_copy = next(index for index, event in enumerate(events) if event[0] == "dir")
    project_run = events.index(
        ("run", f"UV_INDEX_URL={_MODULE.PYPI_INDEX} uv sync --locked --offline --no-build-isolation")
    )

    assert events[:4] == [
        ("base", "example/image:tag"),
        ("file", _MODULE.ROOT / "pyproject.toml", str(_MODULE.REMOTE_ROOT / "pyproject.toml"), True),
        ("file", _MODULE.ROOT / "uv.lock", str(_MODULE.REMOTE_ROOT / "uv.lock"), True),
        ("workdir", str(_MODULE.REMOTE_ROOT)),
    ]
    assert dependency_run < source_copy < project_run


def test_training_failure_is_persisted_and_not_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    states: list[RunState] = []
    monkeypatch.setattr(_MODULE, "REMOTE_ROOT", tmp_path)
    monkeypatch.setattr(_MODULE, "_commit_state", lambda _path, state, _volume: states.append(state))
    command = (
        sys.executable,
        "-c",
        "print('[ckpt] writing checkpoints to runs/run-1', flush=True); raise SystemExit(3)",
    )

    with pytest.raises(RuntimeError, match="terminal failure"):
        _MODULE._run_training(
            command,
            RunState(status="running"),
            env=dict(_MODULE.os.environ),
            path=tmp_path / "state.json",
            volume_name="test-volume",
            stall_s=10,
        )

    assert states[-1] == RunState(status="failed", run_name="run-1", exit_code=3)


def test_training_interrupt_preserves_recoverable_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    states: list[RunState] = []
    timers: list[threading.Timer] = []
    scheduled = False

    def fake_signal(sig: object, handler: object) -> object:
        nonlocal scheduled
        if sig == _MODULE.signal.SIGTERM and callable(handler) and not scheduled:
            scheduled = True
            timer = threading.Timer(0.5, handler, args=(_MODULE.signal.SIGTERM, None))
            timers.append(timer)
            timer.start()
        return _MODULE.signal.SIG_DFL

    monkeypatch.setattr(_MODULE, "REMOTE_ROOT", tmp_path)
    monkeypatch.setattr(_MODULE, "_commit_state", lambda _path, state, _volume: states.append(state))
    monkeypatch.setattr(_MODULE.signal, "signal", fake_signal)
    command = (
        sys.executable,
        "-c",
        "import time; print('[ckpt] writing checkpoints to runs/run-1', flush=True); time.sleep(30)",
    )

    with pytest.raises(RuntimeError, match="was interrupted"):
        _MODULE._run_training(
            command,
            RunState(status="running"),
            env=dict(_MODULE.os.environ),
            path=tmp_path / "state.json",
            volume_name="test-volume",
            stall_s=10,
        )
    for timer in timers:
        timer.join()

    assert states
    assert states[-1] == RunState(status="running", run_name="run-1")
