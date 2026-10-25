"""Local checks for the O65 torchrun launch and explicit B200 budget."""

import dataclasses
import json
import os
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import tyro

from scripts import o65_production


def test_resumed_attempt_preserves_discarded_metrics_and_has_separate_history(tmp_path: Path) -> None:
    from test_o65_pilot import write_boundary

    output = tmp_path / "logical-run/pilot"
    checkpoint = write_boundary(output / "train/checkpoints", 32)
    metrics = output / "train/metrics.jsonl"
    original = b'{"global_step": 32, "loss": 1}\n{"global_step": 33, "loss": 2}\n'
    metrics.write_bytes(original)
    first = o65_production.begin_attempt(output, "a" * 40, checkpoint)
    assert first.previous_attempt_run_id == "logical-run"
    assert first.resume_update == 32 and first.source == "a" * 40
    assert first.checkpoint_sha256 == o65_production.sha256(checkpoint)
    archive = output / "attempts" / first.run_id
    assert (archive / "metrics-before-resume.jsonl").read_bytes() == original
    assert first.prior_metrics_sha256 == o65_production.sha256(archive / "metrics-before-resume.jsonl")
    assert metrics.read_bytes() == original.splitlines(keepends=True)[0]
    metrics.write_bytes(metrics.read_bytes() + b'{"global_step": 33, "loss": 3}\n')
    second = o65_production.begin_attempt(output, "b" * 40, checkpoint)
    assert second.run_id != first.run_id
    assert second.previous_attempt_run_id == first.run_id
    assert second.group == first.group == "logical-run"
    assert (archive / "metrics-before-resume.jsonl").read_bytes() == original
    assert (
        (output / "attempts" / second.run_id / "metrics-before-resume.jsonl")
        .read_bytes()
        .endswith(b'{"global_step": 33, "loss": 3}\n')
    )
    assert json.loads((archive / "lineage.json").read_text())["source"] == "a" * 40
    assert json.loads((output / "run-attempt.json").read_text())["source"] == "b" * 40


def test_explicit_source_transition_preserves_settings_and_resume_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_065_loop import exp
    from test_o65_pilot import write_boundary

    args = production_args(tmp_path)
    source = "a" * 40
    commands: list[list[str]] = []
    environments: list[dict[str, str]] = []

    def execute(command: list[str], **kwargs: object) -> None:
        commands.append(command)
        environments.append(dict(kwargs["env"]))

    monkeypatch.setenv("WANDB_API_KEY", "test")
    args.checkpoint.write_bytes(b"architecture checkpoint")
    monkeypatch.setattr(o65_production, "O64_SHA256", o65_production.sha256(args.checkpoint))
    monkeypatch.setattr(o65_production.subprocess, "check_output", lambda *args, **kwargs: source + "\n")
    monkeypatch.setattr(o65_production.subprocess, "run", execute)
    monkeypatch.setattr(o65_production.self_play_publication, "publish", lambda args: None)
    o65_production.main(args)
    checkpoint = write_boundary(args.output / "train/checkpoints", 32)
    source = "b" * 40
    with pytest.raises(ValueError, match="identity differs"):
        o65_production.main(args)
    resumed = dataclasses.replace(
        args, resume_source="a" * 40, resume_manifest_sha256=o65_production.sha256(checkpoint)
    )
    with pytest.raises(ValueError, match="identity differs"):
        o65_production.main(dataclasses.replace(resumed, windows_per_step=2))
    unexpected = write_boundary(args.output / "train/checkpoints", 33)
    before = {path.relative_to(args.output): path.read_bytes() for path in args.output.rglob("*") if path.is_file()}
    command_count = len(commands)
    with pytest.raises(ValueError, match="checkpoint manifest differs from the requested SHA-256"):
        o65_production.main(resumed)
    assert len(commands) == command_count
    assert {
        path.relative_to(args.output): path.read_bytes() for path in args.output.rglob("*") if path.is_file()
    } == before
    unexpected.unlink()
    o65_production.main(dataclasses.replace(resumed, verify_resume=True))
    command = commands[-1]
    parsed = tyro.cli(exp.NativeCli, args=command[command.index("train") + 1 :])
    assert parsed.resume == checkpoint and parsed.resume_source == "a" * 40
    preparation = tyro.cli(exp.NativeCli, args=commands[-2][commands[-2].index("train") + 1 :])
    assert preparation.prepare_resume_check and preparation.resume_check == parsed.resume_check
    assert preparation.publication is None and preparation.wandb_project is None
    assert not parsed.prepare_resume_check and parsed.publication is not None
    history = [json.loads(line) for line in (args.output / "source-transitions.jsonl").read_text().splitlines()]
    assert len(history) == 1 and history[0]["update"] == 32
    assert history[0]["previous"]["git_sha"] == "a" * 40
    assert history[0]["current"]["git_sha"] == "b" * 40
    assert o65_production.resume_settings(resumed) == o65_production.resume_settings(args)
    assert environments[0]["WANDB_RUN_ID"] != environments[-1]["WANDB_RUN_ID"]
    assert environments[-1]["WANDB_RESUME"] == "never"
    lineage = json.loads(environments[-1]["WANDB_NOTES"])
    assert lineage["source"] == "b" * 40 and lineage["resume_update"] == 32
    assert lineage["previous_attempt_run_id"] == environments[0]["WANDB_RUN_ID"]
    assert len(commands) == 3
    assert not (args.output / "held-out-consumed.json").exists()
    assert not (args.output / "held-out").exists()


@pytest.mark.parametrize("endpoint", [65, 512, 1024])
def test_training_endpoint_can_extend_without_changing_resume_identity(tmp_path: Path, endpoint: int) -> None:
    from test_065_loop import exp
    from test_o65_modal import launcher

    args = production_args(tmp_path)
    extended = dataclasses.replace(args, updates=endpoint)
    assert o65_production.resume_settings(extended) == o65_production.resume_settings(args)
    command = o65_production.training_command(extended)
    parsed = tyro.cli(exp.NativeCli, args=command[command.index("train") + 1 :])
    assert parsed.updates == endpoint
    assert parsed.training == tyro.cli(exp.NativeCli, args=o65_production.training_command(args)[9:]).training
    request = launcher.Request(
        "extended",
        "a" * 40,
        mode="ddp",
        gpus=2,
        matches=16,
        windows_per_step=1,
        damage_weight=0.1,
        game_win_weight=120,
        ddp_updates=endpoint,
    )
    assert tyro.cli(o65_production.Args, args=launcher.command(request, (0, 1))[3:]).updates == endpoint


def test_legacy_identity_migrates_only_execution_endpoint(tmp_path: Path) -> None:
    settings = o65_production.resume_settings(production_args(tmp_path))
    current = {
        "format": "hal.o65.production-identity.v2",
        "git_sha": "b" * 40,
        "checkpoint_sha256": "c" * 64,
        "settings": settings,
        "inputs": {"sampler": "fixed"},
    }
    previous = current | {
        "format": "hal.o65.production-identity.v1",
        "git_sha": "a" * 40,
        "settings": settings | {"updates": 256},
    }
    o65_production.validate_resume_identity(previous, current, "a" * 40)
    for changed in (
        previous | {"inputs": {"sampler": "changed"}},
        previous | {"settings": previous["settings"] | {"windows_per_step": 2}},
        previous | {"git_sha": "c" * 40},
        previous | {"format": "unknown"},
    ):
        with pytest.raises(ValueError, match="identity differs"):
            o65_production.validate_resume_identity(changed, current, "a" * 40)


def evaluation_args(tmp_path: Path) -> o65_production.EvaluationArgs:
    from test_o65_pilot import write_boundary

    output = tmp_path / "evaluation"
    checkpoint = write_boundary(output / "train/checkpoints", 512)
    reference = tmp_path / "reference.pt"
    reference.write_bytes(b"reference")
    return o65_production.EvaluationArgs(
        output,
        tmp_path / "delivery",
        checkpoint,
        o65_production.sha256(checkpoint),
        reference,
        tmp_path / "published",
        "hal-o65-profile",
        "o65/pilot-results/test",
    )


def test_final_evaluation_claims_selected_checkpoint_and_preserves_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = evaluation_args(tmp_path)
    monkeypatch.setattr(o65_production, "O64_SHA256", o65_production.sha256(args.reference_checkpoint))
    publications = []
    monkeypatch.setattr(
        o65_production.self_play_publication,
        "publish",
        lambda _: publications.append((args.output / "held-out-consumed.json").read_bytes()),
    )

    def fail(command: list[str], **kwargs: object) -> None:
        assert command[command.index("--checkpoint") + 1] == str(args.checkpoint)
        assert len(publications) == 1
        (args.output / "held-out").mkdir()
        (args.output / "held-out/partial.json").write_text('{"games": 1}')
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(o65_production.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        o65_production.evaluate(args)
    claim = json.loads(publications[0])
    assert claim["update"] == 512 and claim["checkpoint_sha256"] == args.checkpoint_sha256
    assert len(publications) == 2 and publications[0] == publications[1]
    assert (args.output / "held-out/partial.json").is_file()
    with pytest.raises(ValueError, match="already exist"):
        o65_production.evaluate(args)
    assert len(publications) == 2


def test_launcher_passes_explicit_resume_source(tmp_path: Path) -> None:
    from test_o65_modal import launcher

    request = launcher.Request(
        "resume",
        "b" * 40,
        mode="ddp",
        gpus=2,
        matches=16,
        windows_per_step=1,
        damage_weight=0.1,
        game_win_weight=120,
        resume_source="a" * 40,
        resume_manifest_sha256="b" * 64,
    )
    command = launcher.command(request, (0, 1))
    parsed = tyro.cli(o65_production.Args, args=command[3:])
    assert parsed.resume_source == "a" * 40
    assert parsed.resume_manifest_sha256 == "b" * 64
    with pytest.raises(ValueError, match="full Git SHA"):
        dataclasses.replace(request, resume_source="short")
    with pytest.raises(ValueError, match="checkpoint manifest SHA-256"):
        dataclasses.replace(request, resume_manifest_sha256=None)


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

    request = launcher.Request(
        "ddp-check",
        "a" * 40,
        mode="ddp",
        matches=16,
        gpus=2,
        ddp_updates=2,
        windows_per_step=1,
        measurement=True,
        damage_weight=0.1,
        game_win_weight=120,
    )
    command = launcher.command(request, (0, 1))
    assert command[command.index("--gpus") + 1] == "2"
    assert command[command.index("--updates") + 1] == "2"
    assert "--measurement" in command
    assert "--publish-directory" not in command
    assert command[command.index("--checkpoint") + 1].endswith("o64-inputs/step-0098304.pt")
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


def production_args(tmp_path: Path, *, measurement: bool = False) -> o65_production.Args:
    return o65_production.Args(
        output=tmp_path / "pilot",
        delivery=tmp_path / "delivery",
        checkpoint=tmp_path / "o64.pt",
        cpus=tuple(sorted(os.sched_getaffinity(0)))[:2],
        gpus=2,
        matches=16,
        windows_per_step=1,
        measurement=measurement,
        updates=2 if measurement else 256,
        publish_directory=None if measurement else tmp_path / "published",
        publish_volume=None if measurement else "hal-o65-profile",
        r2_prefix=None if measurement else "o65/pilot-results/test",
    )


@pytest.mark.parametrize("measurement", [False, True])
def test_o64_commands_parse_the_native_cli(tmp_path: Path, measurement: bool) -> None:
    from test_065_loop import exp

    args = production_args(tmp_path, measurement=measurement)
    command = o65_production.training_command(args)
    parsed = tyro.cli(exp.NativeCli, args=command[command.index("train") + 1 :])
    assert parsed.cadence == "production" and parsed.rollout_frames == 512
    assert parsed.training.reward_weights == exp.native.RewardWeights(0.1, 120, 120)
    assert parsed.groups == 4 and parsed.training.snapshot_interval == 64
    assert parsed.training.windows_per_step == 1 and parsed.training.actor_microbatches == 1
    assert parsed.training.decisions_per_window == "all"
    assert parsed.matches_per_shard == 16
    assert parsed.measurement == measurement
    assert parsed.training.critic_warmup_updates == (0 if measurement else 32)
    assert (parsed.publication is None) == measurement


def test_o64_warm_cache_command_and_resume_identity(tmp_path: Path) -> None:
    from test_065_loop import exp

    args = production_args(tmp_path)
    warm = dataclasses.replace(args, compile_cache=tmp_path / "packed-cache", require_compile_cache=True)
    assert o65_production.resume_settings(args) == o65_production.resume_settings(warm)
    command = o65_production.training_command(warm)
    parsed = tyro.cli(exp.NativeCli, args=command[command.index("train") + 1 :])
    assert parsed.require_compile_cache
    assert parsed.compile_cache == tmp_path / "packed-cache"


def test_one_gpu_preparation_builds_the_cache_used_by_production_rank_zero(tmp_path: Path) -> None:
    from test_065_loop import exp
    from test_o65_modal import launcher

    request = launcher.Request(
        "cache-preparation",
        "a" * 40,
        mode="ddp",
        gpus=1,
        matches=1024,
        windows_per_step=512,
        actor_microbatches=8,
        ddp_updates=6,
        measurement=True,
        cold_cache=True,
        damage_weight=0.1,
        game_win_weight=120,
    )
    cpus = tuple(sorted(os.sched_getaffinity(0)))[:8]
    args = tyro.cli(o65_production.Args, args=launcher.command(request, cpus)[3:])
    command = o65_production.training_command(args)
    parsed = tyro.cli(exp.NativeCli, args=command[command.index("train") + 1 :])
    assert parsed.compile_cache == launcher.RESULTS / "compile-cache/rank0"
    assert parsed.measurement and not parsed.require_compile_cache
    assert parsed.training.windows_per_step == 512 and parsed.training.actor_microbatches == 8
    warm = dataclasses.replace(request, cold_cache=False)
    assert "--require-compile-cache" in launcher.command(warm, cpus)
    with pytest.raises(ValueError, match="one-GPU measurement"):
        dataclasses.replace(request, gpus=2)
    with pytest.raises(ValueError, match="one GPU is allowed for measurement"):
        dataclasses.replace(request, measurement=False)


def test_cache_source_import_is_explicit_and_measurement_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from test_065_loop import exp

    command = o65_production.training_command(production_args(tmp_path, measurement=True))
    config = tyro.cli(exp.NativeCli, args=command[command.index("train") + 1 :])
    imported = dataclasses.replace(
        config,
        compile_cache=tmp_path / "cache",
        compile_cache_seed=tmp_path / "seed",
        compile_cache_seed_source="a" * 64,
    )
    with pytest.raises(ValueError, match="cross-source cache seeds"):
        dataclasses.replace(imported, measurement=False)
    with pytest.raises(ValueError, match="cross-source cache seeds"):
        dataclasses.replace(imported, compile_cache_seed_source=None)
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="cross-source cache seeds"):
        dataclasses.replace(imported)


def test_o64_final_evaluation_uses_frozen_384_game_grid(tmp_path: Path) -> None:
    from torch._inductor.codegen.triton import TritonKernel
    from torch._inductor.codegen.triton import TritonScheduling
    from torch.compiler import config as compiler_config

    from hal.inference.inductor_compat import CACHE_KEY_TAG
    from scripts import eval_o65

    command = o65_production.evaluation_command(evaluation_args(tmp_path))

    def compiled_evaluation(*args: object, **kwargs: object) -> None:
        assert compiler_config.cache_key_tag == CACHE_KEY_TAG
        assert TritonScheduling.kernel_type is not TritonKernel

    with (
        patch.object(TritonScheduling, "kernel_type", TritonKernel),
        patch.object(compiler_config, "cache_key_tag", ""),
        patch.object(eval_o65, "evaluate", autospec=True, side_effect=compiled_evaluation) as evaluate,
    ):
        eval_o65.main(tyro.cli(eval_o65.Args, args=command[2:]))
    settings = evaluate.call_args.kwargs
    assert settings["simulators"] == ("msl",) and settings["split"] == "held_out"
    protocol = settings["protocol"]
    assert len(protocol.held_out.starts) * len(protocol.held_out.seeds) * 2 == 384
    assert all(
        (condition.delay, condition.replan, condition.lambda_value, condition.player_id) == (18, 2, 1, 0)
        for pair in settings["condition_grid"]
        for condition in pair
    )


def test_o64_launch_requires_capacity_and_measurement_is_explicit(tmp_path: Path) -> None:
    from test_o65_modal import launcher

    request = launcher.Request(
        "o64",
        "a" * 40,
        mode="ddp",
        matches=1024,
        gpus=8,
        windows_per_step=16,
        damage_weight=0.1,
        game_win_weight=120,
    )
    with pytest.raises(ValueError, match="windows-per-step"):
        dataclasses.replace(request, windows_per_step=None)
    with pytest.raises(ValueError, match="positive integer"):
        dataclasses.replace(request, ddp_updates=0)
    with pytest.raises(ValueError, match="outcome reward"):
        dataclasses.replace(request, damage_weight=1)
    with pytest.raises(ValueError, match="matches per GPU"):
        dataclasses.replace(request, matches=1)
    with pytest.raises(ValueError, match="requires --matches"):
        launcher.main(launcher.Args(mode="ddp", gpus=2, dry_run=True, b200_max_usd=100, output=tmp_path))


def test_measurement_sweep_has_fresh_processes_and_no_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from test_o65_modal import launcher

    check_budget = launcher.b200_budget
    monkeypatch.setattr(
        launcher,
        "b200_budget",
        lambda path, name, dollars, cap, *, reserve: check_budget(
            tmp_path / "budget.jsonl", name, dollars, cap, reserve=reserve
        ),
    )
    launcher.main(
        launcher.Args(
            mode="ddp",
            name="o64-measure",
            gpus=8,
            matches=16,
            windows_per_step=1,
            ddp_updates=2,
            measurement=True,
            measurement_sizes=("16/1", "64/4", "256/16"),
            dry_run=True,
            b200_max_usd=1000,
            output=tmp_path,
        )
    )
    plan = json.loads((tmp_path / "o64-measure-dry-run.json").read_text())
    commands = plan["measurement_commands"]
    assert len(commands) == 3 and plan["gpu"] == "B200:8"
    outputs = {command[command.index("--output") + 1] for command in commands}
    assert len(outputs) == 3
    for command in commands:
        assert "--measurement" in command and "--publish-directory" not in command
        assert "--require-compile-cache" in command
        parsed = tyro.cli(o65_production.Args, args=command[3:])
        assert parsed.measurement and parsed.updates == 2
    assert plan["checkpoint"]["sha256"] == o65_production.O64_SHA256


def test_corrupt_staged_o64_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "step-0098304.pt").write_bytes(b"incorrect")
    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        o65_production.stage_checkpoint(tmp_path)


def test_native_match_count_is_per_rank_before_process_group_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_065_loop import exp

    monkeypatch.setenv("WORLD_SIZE", "2")
    cpus = tuple(sorted(os.sched_getaffinity(0)))[:4]
    config = exp.NativeCli(
        checkpoint=tmp_path / "bc.pt",
        sampler=tmp_path / "sampler.json",
        output=tmp_path / "run",
        data_dir=tmp_path,
        cpus=cpus,
        updates=1,
        active_matches_per_gpu=8,
    )
    assert config.active_matches_per_gpu == 8
    with pytest.raises(ValueError, match="active matches"):
        dataclasses.replace(config, active_matches_per_gpu=4)


def test_resume_settings_ignore_virtual_cpu_ids(tmp_path: Path) -> None:
    args = production_args(tmp_path)
    moved = dataclasses.replace(args, cpus=(100, 101))
    assert o65_production.resume_settings(args) == o65_production.resume_settings(moved)
    assert o65_production.resume_settings(dataclasses.replace(args, windows_per_step=2)) != (
        o65_production.resume_settings(args)
    )


def test_rank_visible_devices_put_each_rank_gpu_first() -> None:
    from hal.training.self_play_distributed import rank_visible_devices

    assert rank_visible_devices(None, 0, 3) == "0,1,2"
    assert rank_visible_devices(None, 2, 3) == "2,0,1"
    assert rank_visible_devices("GPU-a,GPU-b,GPU-c,GPU-d", 1, 4) == "GPU-b,GPU-a,GPU-c,GPU-d"
    with pytest.raises(ValueError, match="own visible CUDA device"):
        rank_visible_devices("0,1", 0, 3)
