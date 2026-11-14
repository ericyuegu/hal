"""The fox-ditto arm: fixed sampler conditions and the per-update KL limit."""

import dataclasses
from pathlib import Path

import pytest
import torch
from test_065_loop import exp
from test_065_loop import sampler
from test_065_loop import trainer
from test_o65_ablation import flag_values
from test_o65_ablation import pilot

ROOT = Path(__file__).resolve().parents[1]


def test_fixed_conditions_replace_the_sampled_ones() -> None:
    sampling = sampler()
    config = dataclasses.replace(sampling.config, percentile_fixed=0.95, lambda_min=0.1, lambda_max=0.1)
    schedule = exp.ScheduleConfig((exp.DelayReplan(8, 4),), sampling.schedule.supervised_offsets)
    fixed = exp.EpisodeSampler(config, schedule, sampling.rng)
    players = [player for match in range(32) for player in fixed.sample(match, 0, None).players]
    assert {player.percentile for player in players} == {0.95}
    assert {round(player.log_lambda, 12) for player in players} == {round(exp.math.log(0.1), 12)}
    assert {player.timing for player in players} == {exp.DelayReplan(8, 4)}


def test_kl_limit_stops_the_actor_steps_of_an_update() -> None:
    free = trainer(groups=1)
    _, unlimited = free.run_update()
    assert unlimited["policy/actor_steps_taken"] > 2
    limited = trainer(groups=1)
    limited.config = dataclasses.replace(limited.config, update_kl_limit=1e-9)
    _, metrics = limited.run_update()
    # The KL is measured before each step, so the step that crosses the limit is followed by one more forward pass.
    assert metrics["policy/actor_steps_taken"] == 2
    assert metrics["policy/old_kl_growth"] > 1e-9


def test_fox_ditto_arm_command() -> None:
    command = pilot.training_command(
        Path("delivery"),
        Path("train"),
        tuple(range(16)),
        matches=64,
        groups=4,
        updates=2048,
        checkpoint_every=32,
        warmup=32,
        arm="fox-ditto",
    )
    flags = flag_values(command)
    expected = {
        "--sampler": str(pilot.ROOT / "experiments/o65/fox-ditto-sampler.json"),
        "--training.actor-lr": "0.0001",
        "--training.windows-per-step": "128",
        "--training.actor-microbatches": "1",
        "--training.update-kl-limit": "0.0004",
        "--lambda-fixed": "0.1",
        "--percentile-fixed": "0.95",
        "--timing": "8",
        "--training.lane-layout": "self_play_heavy",
        "--training.league-selection": "latest",
    }
    assert {flag: flags[flag] for flag in expected} == expected
    assert command[command.index("--timing") + 1 : command.index("--timing") + 3] == ["8", "4"]
    fox = exp.read_native_sampler(ROOT / "experiments/o65/fox-ditto-sampler.json", lambda_fixed=0.1)
    assert [matchup.characters for matchup in fox.matchups] == [(1, 1)]
    assert (fox.lambda_min, fox.lambda_max) == (0.1, 0.1)


def test_fox_exploit_arm_command() -> None:
    opponent = Path("/o65-results") / str(pilot.get_arm("fox-exploit").opponent)
    command = pilot.training_command(
        Path("delivery"),
        Path("train"),
        tuple(range(16)),
        matches=64,
        groups=4,
        updates=640,
        checkpoint_every=32,
        warmup=32,
        arm="fox-exploit",
        opponent=opponent,
    )
    flags = flag_values(command)
    expected = {
        "--sampler": str(pilot.ROOT / "experiments/o65/fox-ditto-sampler.json"),
        "--training.update-kl-limit": "0.0004",
        "--training.lane-layout": "frozen_opponent",
        "--training.league-selection": "frozen",
        "--training.damage-reward": "0.25",
        "--opponent": str(opponent),
    }
    assert {flag: flags[flag] for flag in expected} == expected
    assert float(flags["--training.gamma"]) == pytest.approx(2 ** (-1 / 1800))


def test_kl_ignores_categories_outside_the_policy_support() -> None:
    reference = torch.tensor([0.5, 0.25, 0.25]).log()
    policy = torch.tensor([0.5, 0.5, 0.0]).log().requires_grad_()
    kl = exp.categorical_kl(policy, reference)
    assert kl.item() == pytest.approx(0.5 * torch.log(torch.tensor(2.0)).item())
    kl.backward()
    assert torch.isfinite(policy.grad).all()
    with pytest.raises(ValueError, match="cover the policy's support"):
        exp.categorical_kl(reference, policy.detach())


def test_timeout_arm_differs_from_fox_exploit_by_the_timeout_flag() -> None:
    def command(arm: pilot.ArmName) -> list[str]:
        return list(
            pilot.training_command(
                Path("delivery"),
                Path("train"),
                tuple(range(16)),
                matches=64,
                groups=4,
                updates=640,
                checkpoint_every=32,
                warmup=32,
                arm=arm,
                opponent=Path("opponent.json"),
            )
        )

    timeout = command("fox-exploit-timeout")
    assert "--training.timeout-loss" in timeout
    timeout.remove("--training.timeout-loss")
    assert timeout == command("fox-exploit")


def test_delay16_arm_differs_from_fox_ditto_by_the_timing_and_the_reference() -> None:
    def command(arm: pilot.ArmName, reference: Path | None) -> list[str]:
        return list(
            pilot.training_command(
                Path("delivery"),
                Path("train"),
                tuple(range(16)),
                matches=64,
                groups=4,
                updates=pilot.get_arm(arm).updates,
                checkpoint_every=32,
                warmup=32,
                arm=arm,
                reference=reference,
            )
        )

    delayed = command("fox-ditto-delay16", Path("dense.pt"))
    base = command("fox-ditto", None)
    changed = {
        flag
        for flag in ("--checkpoint", "--timing", "--updates")
        if flag_values(delayed)[flag] != flag_values(base)[flag]
    }
    assert changed == {"--checkpoint", "--timing", "--updates"}
    assert delayed[delayed.index("--timing") + 1 : delayed.index("--timing") + 3] == ["16", "4"]
    assert len(delayed) == len(base)
    with pytest.raises(ValueError, match="own reference"):
        command("fox-ditto-delay16", None)
