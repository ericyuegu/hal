import dataclasses
import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest
import tyro
from test_o65_modal import launcher

pilot = launcher.pilot


def arm_command(name: pilot.ArmName) -> list[str]:
    return pilot.training_command(
        Path("delivery"),
        Path("train"),
        tuple(range(16)),
        matches=64,
        groups=4,
        updates=672,
        checkpoint_every=32,
        warmup=32,
        arm=name,
    )


def flag_values(command: list[str]) -> dict[str, str]:
    return {flag: command[index + 1] for index, flag in enumerate(command) if flag.startswith("--")}


def test_old_arm_matches_the_collapsed_configuration() -> None:
    command = arm_command("old")
    flags = flag_values(command)
    expected = {
        "--training.lane-layout": "self_play_heavy",
        "--training.league-selection": "latest",
        "--training.snapshot-interval": "256",
        "--training.opponent-weight-power": "0",
        "--training.gamma": "0.99855",
        "--training.damage-reward": "1",
        "--training.critic": "residual_scalar",
        "--training.critic-hidden-dim": "128",
        "--training.critic-depth": "0",
        "--training.critic-warmup-updates": "32",
        "--training.value-loss-coefficient": "0",
        "--training.advantage-scale": "batch",
        "--percentile-alpha": "5",
        "--percentile-beta": "1.3",
        "--trace-matches": "4",
        "--updates": "672",
        "--groups": "4",
        "--matches-per-shard": "64",
        "--training.windows-per-step": "512",
        "--training.actor-microbatches": "4",
        "--training.seed": "65",
        "--training.head-start-frames": "1200",
        "--training.checkpoint-interval": "32",
    }
    assert {flag: flags[flag] for flag in expected} == expected
    assert len([item for item in command if item.startswith("--")]) == len(flags)
    assert command[command.index("--cpus") + 1 : command.index("--groups")] == [str(cpu) for cpu in range(16)]
    assert pilot.OLD.__dataclass_params__.frozen
    assert not hasattr(pilot.OLD, "__dict__")


@pytest.mark.parametrize(
    "name,changes",
    [
        ("reward", {"--training.gamma": str(2 ** (-1 / 1800)), "--training.damage-reward": "0"}),
        (
            "critic",
            {"--training.critic": "hl_gauss", "--training.critic-hidden-dim": "512", "--training.critic-depth": "3"},
        ),
        ("scale", {"--training.advantage-scale": "warmup"}),
        (
            "matchup",
            {
                "--training.lane-layout": "anchor_heavy",
                "--training.league-selection": "pfsp",
                "--training.snapshot-interval": "64",
                "--training.opponent-weight-power": "1",
            },
        ),
    ],
)
def test_each_arm_changes_only_its_treatment(name: pilot.ArmName, changes: dict[str, str]) -> None:
    old, new = flag_values(arm_command("old")), flag_values(arm_command(name))
    assert new == old | changes
    assert {flag: value for flag, value in new.items() if old[flag] != value} == changes


def test_arms_reject_unknown_names_and_changed_invariants() -> None:
    with pytest.raises(ValueError, match="unknown ablation arm"):
        pilot.get_arm("typo")
    with pytest.raises(ValueError, match="four groups"):
        pilot.training_command(
            Path("delivery"),
            Path("train"),
            tuple(range(16)),
            matches=64,
            groups=4,
            updates=2000,
            checkpoint_every=32,
            warmup=32,
            arm="old",
        )
    with pytest.raises(ValueError, match="full training"):
        pilot.Args(Path("run"), tiny=True, arm="old")
    assert tyro.cli(pilot.Args, args=["--output", "run", "--arm", "reward"]).arm == "reward"


def test_persisted_pilot_identity_binds_the_arm_and_sweep(tmp_path: Path) -> None:
    pilot.bind_pilot_identity(tmp_path, "a" * 40, tiny=False, arm="old", sweep="collapse")
    pilot.bind_pilot_identity(tmp_path, "a" * 40, tiny=False, arm="old", sweep="collapse")
    for arm, sweep in (("reward", "collapse"), ("old", "another")):
        with pytest.raises(ValueError, match="differ"):
            pilot.bind_pilot_identity(tmp_path, "a" * 40, tiny=False, arm=arm, sweep=sweep)


@pytest.mark.parametrize("warming,actor_steps,update", [(0, 1, 672), (1, 0, 672), (0, 0, 672), (0, 1, 32)])
def test_arm_runs_finish_ppo_and_skip_held_out(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    warming: int,
    actor_steps: int,
    update: int,
) -> None:
    available = tuple(sorted(os.sched_getaffinity(0)))
    monkeypatch.setattr(pilot.os, "sched_getaffinity", lambda _: set(available) | set(range(17)))
    monkeypatch.setattr(pilot.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(pilot.torch.cuda, "get_device_name", lambda: "RTX PRO 6000")
    monkeypatch.setenv("WANDB_API_KEY", "test")
    monkeypatch.setattr(pilot, "compact_checkpoints", Mock())
    commands: list[list[str]] = []

    def execute(command: list[str], log: Path, environment: dict[str, str]) -> None:
        commands.append(command)
        if command[1] != str(pilot.ROOT / "experiments/065_selfplay_ppo.py"):
            return
        assert environment["WANDB_RUN_GROUP"] == "collapse"
        assert environment["WANDB_NAME"] == "collapse-old"
        assert flag_values(command)["--training.critic-warmup-updates"] == "32"
        run = tmp_path / "pilot/train"
        run.mkdir()
        (run / "metrics.jsonl").write_text(
            json.dumps({"global_step": update, "train/warming": warming, "train/actor_steps": actor_steps}) + "\n"
        )

    monkeypatch.setattr(pilot, "execute", execute)
    args = pilot.Args(tmp_path / "pilot", local_delivery=tmp_path / "delivery", arm="old", sweep="collapse")
    if warming or not actor_steps or update != 672:
        with pytest.raises(RuntimeError, match="PPO"):
            pilot.main(args)
    else:
        pilot.main(args)
    assert not any("eval_o65.py" in " ".join(command) for command in commands)
    assert not (args.output / "held-out-consumed.json").exists()


def test_default_pilot_command_keeps_its_current_settings() -> None:
    command = pilot.training_command(
        Path("delivery"),
        Path("train"),
        tuple(range(16)),
        matches=64,
        groups=4,
        updates=2000,
        checkpoint_every=32,
        warmup=128,
    )
    flags = flag_values(command)
    assert flags["--training.advantage-scale"] == "warmup"
    assert flags["--training.critic-warmup-updates"] == "128"
    assert "--training.gamma" not in flags
    assert "--percentile-alpha" not in flags
    assert dataclasses.replace(pilot.OLD, name="scale", advantage_scale="warmup") == pilot.get_arm("scale")
