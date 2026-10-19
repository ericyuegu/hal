import dataclasses
import importlib.util
import json
import shlex
import sys
from pathlib import Path

import pytest
import torch
import tyro
from test_065_loop import exp
from test_065_loop import transformer_trainer

from hal.wire import ACTION_DIM

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
        ("windows", 0),
        ("decisions", True),
        ("matches", 64),
        ("name", "../x"),
        ("git_sha", "HEAD"),
        ("timeout_seconds", 0),
    ],
)
def test_modal_request_rejects_invalid_config(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(Request("test", "a" * 40), **{field: value})


def test_modal_commands_are_bounded_and_have_no_external_sink() -> None:
    profile = command(Request("test", "a" * 40, windows=32, decisions=256), tuple(range(16)))
    assert profile[2] == "profile"
    assert profile[profile.index("--matches-per-shard") + 1] == "64"
    assert profile[profile.index("--replay-chunks.windows") + 1] == "32"
    assert profile[profile.index("--replay-chunks.decisions") + 1] == "256"
    smoke = command(Request("test", "a" * 40, mode="smoke", matches=16), tuple(range(16)))
    assert smoke[2] == "train"
    assert smoke[smoke.index("--updates") + 1] == "1"
    assert smoke[smoke.index("--groups") + 1] == "1"
    assert smoke[smoke.index("--evaluation-every") + 1] == "0"
    assert not any("wandb" in arg or "upload" in arg for arg in profile + smoke)


def test_cpu_selection_uses_distinct_available_ids() -> None:
    assert select_cpus(set(range(1, 37, 2))) == tuple(range(1, 35, 2))
    with pytest.raises(ValueError, match="17 available CPU IDs"):
        select_cpus(set(range(16)))


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


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_replay_chunks_reject_invalid_sizes(value: int) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        exp.ReplayChunks(value, 128)
    with pytest.raises(ValueError, match="positive integers"):
        exp.ReplayChunks(16, value)


@pytest.mark.parametrize("chunks", [exp.ReplayChunks(1, 1), exp.ReplayChunks(2, 3)])
def test_replay_chunking_preserves_rows_features_and_gradients(chunks: exp.ReplayChunks, tmp_path: Path) -> None:
    run = transformer_trainer("cpu", development=True)
    policy = run.policy
    match = run.engine.matches[0]
    states = []
    for stream, side, clock in ((1, 1, 16), (0, 0, 0), (1, 1, 4), (2, 0, 8)):
        player, opponent = match.episode.players[side], match.episode.players[1 - side]
        states.append(
            exp.DecisionState(
                stream,
                0,
                side,
                match.episode,
                clock,
                (0, stream, 0, 0),
                tuple(match.history) * (clock + 1),
                torch.zeros(player.timing.delay, 4, dtype=torch.long),
                torch.zeros(opponent.timing.delay, 4, dtype=torch.long),
            )
        )
    actions = policy.model.codec.quantize(torch.rand(4, 4, ACTION_DIM))
    expected = policy.evaluate(states, actions)
    (-expected.log_probs.sum()).backward()
    gradients = tuple(None if p.grad is None else p.grad.clone() for p in policy.actor_parameters())
    policy.model.zero_grad(set_to_none=True)
    treatment = exp.TransformerPolicy(policy.model, policy.schedule, policy.statistics, replay_chunks=chunks)
    recorder = exp.UpdateProfile(tmp_path, torch.device("cpu"), traces=False)
    treatment.profile = recorder
    with torch.no_grad():
        actual = treatment.evaluate(states, actions)
    assert recorder.calls["outside/trunk_chunk"] == (3 + chunks.windows - 1) // chunks.windows
    assert recorder.calls["outside/decoder_chunk"] == (4 + chunks.decisions - 1) // chunks.decisions
    torch.testing.assert_close(actual.log_probs, expected.log_probs, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual.features.trunk, expected.features.trunk, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual.features.decoder, expected.features.decoder, rtol=2e-5, atol=2e-6)
    for first, second in zip(actual.distributions, expected.distributions, strict=True):
        torch.testing.assert_close(first, second, rtol=2e-5, atol=2e-6)
    recorder.calls.clear()
    actor = treatment.evaluate(states, actions)
    assert recorder.calls == {"outside/trunk_chunk": 1, "outside/decoder_chunk": 1}
    (-actor.log_probs.sum()).backward()
    for gradient, parameter in zip(gradients, treatment.actor_parameters(), strict=True):
        if gradient is None:
            assert parameter.grad is None
        else:
            assert parameter.grad is not None
            torch.testing.assert_close(parameter.grad, gradient, rtol=3e-4, atol=5e-6)
    assert treatment.frozen_copy().replay_chunks == chunks


def test_profile_cli_records_chunk_sizes(tmp_path: Path) -> None:
    cli = exp.GpuProfileCli(
        tmp_path / "final.pt",
        tmp_path / "data",
        tmp_path / "output",
        cpus=(0,),
        replay_chunks=exp.ReplayChunks(32, 256),
    )
    assert cli.native().replay_chunks == exp.ReplayChunks(32, 256)


def test_resume_identity_binds_no_grad_chunk_sizes() -> None:
    run = transformer_trainer("cpu", development=True)
    original = exp._configuration_digest(run)
    run.policy.replay_chunks = exp.ReplayChunks(32, 256)
    assert exp._configuration_digest(run) != original
