"""Local checks for the O65 torchrun launch and explicit B200 budget."""

from pathlib import Path
from typing import Any

import pytest


def test_single_gpu_pilot_and_torchrun_commands(tmp_path: Path) -> None:
    from test_o65_pilot import _MODULE as pilot

    kwargs: dict[str, Any] = dict(matches=4, groups=1, updates=3, checkpoint_every=1, warmup=0)
    single = pilot.training_command(tmp_path, tmp_path, (2, 4), **kwargs)
    multi = pilot.training_command(tmp_path, tmp_path, (2, 4), gpus=2, **kwargs)
    assert single[1].endswith("065_selfplay_ppo.py")
    assert multi[1:8] == [
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node",
        "2",
        single[1],
    ]


def test_b200_command_and_separate_explicit_budget(tmp_path: Path) -> None:
    from test_o65_modal import launcher

    request = launcher.Request("ddp-check", "a" * 40, mode="ddp", matches=16, gpus=2, ddp_updates=3)
    command = launcher.command(request, tuple(range(8)))
    assert command[command.index("--gpus") + 1] == "2"
    assert command[command.index("--updates") + 1] == "3"
    assert "--tiny" in command
    ledger = tmp_path / "b200.jsonl"
    launcher.b200_budget(ledger, "first", 4.0, 6.0, reserve=False)
    assert ledger.read_text() == ""
    launcher.b200_budget(ledger, "first", 4.0, 6.0, reserve=True)
    with pytest.raises(ValueError, match="budget exhausted"):
        launcher.b200_budget(ledger, "second", 3.0, 6.0, reserve=True)
    assert len(ledger.read_text().splitlines()) == 1
    with pytest.raises(ValueError, match="finite and positive"):
        launcher.b200_budget(ledger, "invalid", 1.0, float("nan"), reserve=False)
    # Missing caps are rejected before building or allocating the remote worker.
    with pytest.raises(ValueError, match="b200-max-usd"):
        launcher.main(launcher.Args(mode="ddp", gpus=2, matches=16, dry_run=True, output=tmp_path))


def test_pilot_identity_rejects_different_world_size(tmp_path: Path) -> None:
    from test_o65_pilot import _MODULE as pilot

    pilot.bind_pilot_identity(tmp_path, "a" * 40, tiny=True, world_size=2, updates=3)
    with pytest.raises(ValueError, match="source or inputs differ"):
        pilot.bind_pilot_identity(tmp_path, "a" * 40, tiny=True, world_size=4, updates=3)
