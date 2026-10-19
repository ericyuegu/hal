import dataclasses
import importlib.util
import sys
from pathlib import Path

import pytest
import torch
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


def test_cpu_selection_uses_distinct_physical_cores(tmp_path: Path) -> None:
    for cpu in range(36):
        topology = tmp_path / f"cpu{cpu}/topology"
        topology.mkdir(parents=True)
        (topology / "physical_package_id").write_text("0")
        (topology / "core_id").write_text(str(cpu % 18))
    assert select_cpus(set(range(36)), tmp_path) == tuple(range(17))
    with pytest.raises(ValueError, match="17 available physical cores"):
        select_cpus(set(range(16)), tmp_path)


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
