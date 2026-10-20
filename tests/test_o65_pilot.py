import dataclasses
import hashlib
import importlib.util
import json
import os
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import pytest
import tyro

from hal.training import self_play_publication

_SPEC = importlib.util.spec_from_file_location("o65_pilot", Path(__file__).parents[1] / "experiments/o65/pilot.py")
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_final_evaluation_dispatches_a_paired_full_game_grid(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("eval_o65", Path(__file__).parents[1] / "scripts/eval_o65.py")
    assert spec is not None and spec.loader is not None
    eval_o65 = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = eval_o65
    spec.loader.exec_module(eval_o65)
    command = _MODULE.evaluation_command(
        tmp_path / "delivery",
        tmp_path / "checkpoint.json",
        Path("experiments/o65/pilot-evaluation.json"),
        tmp_path / "results",
    )
    with patch.object(eval_o65, "evaluate", autospec=True) as evaluate:
        eval_o65.main(tyro.cli(eval_o65.Args, args=command[2:]))
    evaluate.assert_called_once()
    settings = evaluate.call_args.kwargs
    protocol = settings["protocol"]
    grid = settings["condition_grid"]
    matches = [
        (start, seed, port, conditions)
        for conditions in grid
        for start in protocol.held_out.starts
        for seed in protocol.held_out.seeds
        for port in (1, 2)
    ]
    assert settings["split"] == "held_out" and settings["simulators"] == ("msl",)
    assert len(matches) == 384
    assert all((condition.delay, condition.replan) == (8, 4) for *_, pair in matches for condition in pair)
    assert protocol.max_frames - 123 - 1 == 8 * 60 * 60


def write_boundary(directory: Path, update: int) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    shards = []
    for role, suffix, content in (
        ("training", "rank0.pt", str(update).encode()),
        ("simulator_0_0", "sim.zlib", b"shared simulator"),
        ("actor", "actor.pt", f"actor-{update}".encode()),
    ):
        digest = hashlib.sha256(content).hexdigest()
        name = f"{digest}.{suffix}"
        (directory / name).write_bytes(content)
        shards.append({"name": name, "sha256": digest, "role": role})
    path = directory / f"update-{update:08d}.json"
    path.write_text(
        json.dumps(
            {
                "format": "hal.o65.training.v7",
                "version": 7,
                "experiment": "065_selfplay_ppo",
                "update": update,
                "configuration_sha256": "c" * 64,
                "environment": {
                    key: "test"
                    for key in (
                        "python",
                        "torch",
                        "numpy",
                        "cuda",
                        "device",
                        "world_size",
                        "tf32_matmul",
                        "tf32_cudnn",
                    )
                },
                "provenance": {
                    "git_sha": "a" * 40,
                    "checkpoint_sha256": "b" * 64,
                    "data_manifest_sha256": "c" * 64,
                    "simulator_sha256": "d" * 64,
                    "adapter_sha256": "e" * 64,
                },
                "shards": shards,
            }
        )
    )
    return path


def test_compact_checkpoints_keeps_latest_actors_and_shared_shards(tmp_path: Path) -> None:
    directory = tmp_path / "checkpoints"
    old = write_boundary(directory, 32)
    original = old.read_bytes()
    boundaries = [write_boundary(directory, update) for update in (256, 288, 320)]
    keep = boundaries[-1:]
    _MODULE.compact_checkpoints(directory, 320)
    assert sorted(directory.glob("update-*.json")) == keep
    assert (tmp_path / "retired-checkpoints" / old.name).read_bytes() == original
    shards = json.loads(original)["shards"]
    assert not (directory / shards[0]["name"]).exists()
    assert (directory / shards[1]["name"]).is_file()
    assert (directory / shards[2]["name"]).is_file()
    for path in keep:
        for shard in json.loads(path.read_text())["shards"]:
            assert (directory / shard["name"]).is_file()
    _MODULE.compact_checkpoints(directory, 320)
    write_boundary(directory, 352)
    _MODULE.compact_checkpoints(directory, 352)
    assert not keep[-1].exists()
    assert len(list(directory.glob("*.rank0.pt"))) == 1
    assert len(list(directory.glob("*.actor.pt"))) == 5


def test_compact_checkpoints_validates_before_removing_files(tmp_path: Path) -> None:
    directory = tmp_path / "checkpoints"
    old = write_boundary(directory, 32)
    latest = write_boundary(directory, 64)
    latest.write_text("{}")
    with pytest.raises(ValueError, match="manifest schema"):
        _MODULE.compact_checkpoints(directory, 64)
    assert old.exists()
    write_boundary(directory, 64)
    with pytest.raises(ValueError, match="latest checkpoint is missing"):
        _MODULE.compact_checkpoints(directory, 96)
    assert old.exists()
    renamed = latest.with_name("update-00000065.json")
    latest.rename(renamed)
    with pytest.raises(ValueError, match="filename differs"):
        _MODULE.compact_checkpoints(directory, 64)
    assert old.exists()
    renamed.rename(latest)
    history = tmp_path / "retired-checkpoints"
    history.mkdir()
    (history / old.name).write_text("conflict")
    with pytest.raises(FileExistsError):
        _MODULE.compact_checkpoints(directory, 64)
    assert old.exists()


def test_latest_resume_rejects_a_corrupt_latest_instead_of_falling_back(tmp_path: Path) -> None:
    directory = tmp_path / "checkpoints"
    write_boundary(directory, 32)
    latest = write_boundary(directory, 64)
    assert _MODULE.latest_checkpoint(directory) == latest
    manifest = json.loads(latest.read_text())
    (directory / manifest["shards"][0]["name"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt shard"):
        _MODULE.latest_checkpoint(directory)


def test_publication_preserves_shards_before_manifests_and_compaction(tmp_path: Path) -> None:
    module = self_play_publication
    stage_state = module.stage_state

    source, destination = tmp_path / "source", tmp_path / "volume"
    checkpoint = source / "train/checkpoints"
    first = write_boundary(checkpoint, 32)
    hashes = stage_state(source, destination)
    assert "train/checkpoints/update-00000032.json" in hashes
    assert _MODULE.latest_checkpoint(destination / "train/checkpoints").name == first.name
    write_boundary(checkpoint, 64)
    _MODULE.compact_checkpoints(checkpoint, 64)
    stage_state(source, destination)
    assert not (destination / "train/checkpoints" / first.name).exists()
    assert (destination / "train/retired-checkpoints" / first.name).exists()
    assert _MODULE.latest_checkpoint(destination / "train/checkpoints").name == "update-00000064.json"


def test_resume_trims_uncommitted_metrics_and_preserves_evidence(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"global_step":32}\n{"global_step":33}\n{"global_')
    _MODULE.trim_interrupted_metrics(path, 32)
    assert path.read_text() == '{"global_step":32}\n'
    evidence = list(tmp_path.glob("metrics-interrupted-*.jsonl"))
    assert len(evidence) == 1 and evidence[0].read_text().endswith('{"global_')


def test_latest_resume_rejects_a_renamed_manifest(tmp_path: Path) -> None:
    path = write_boundary(tmp_path, 32)
    path.rename(path.with_name("update-00000064.json"))
    with pytest.raises(ValueError, match="filename differs"):
        _MODULE.latest_checkpoint(tmp_path)


def test_publication_rejects_missing_and_overlapping_source_trees(tmp_path: Path) -> None:
    module = self_play_publication
    source = tmp_path / "pilot"
    source.mkdir()
    checkpoint = source / "state.pt"
    checkpoint.write_bytes(b"preserve")
    for destination in (tmp_path, source, source / "child"):
        with pytest.raises(ValueError, match="separate trees"):
            module.stage_state(source, destination)
    with pytest.raises(ValueError, match="source directory is missing"):
        module.stage_state(tmp_path / "absent", source)
    assert checkpoint.read_bytes() == b"preserve"


def test_published_held_out_consumption_blocks_a_retry(tmp_path: Path) -> None:
    module = self_play_publication
    source, volume = tmp_path / "local", tmp_path / "volume"
    source.mkdir()
    consumed = _MODULE.consume_held_out(source, 2000, tmp_path / "protocol.json")
    (source / ".o65-held-out-claim.json").write_text("claim")
    module.stage_state(source, volume)
    assert (volume / consumed.name).read_bytes() == consumed.read_bytes()
    assert (volume / ".o65-held-out-claim.json").read_text() == "claim"
    with pytest.raises(RuntimeError, match="already consumed"):
        _MODULE.consume_held_out(volume, 2000, tmp_path / "protocol.json")


def test_completed_pilot_cannot_resume_under_different_source(tmp_path: Path) -> None:
    _MODULE.bind_pilot_identity(tmp_path, "a" * 40, tiny=False)
    write_boundary(tmp_path / "train/checkpoints", 2000)
    _MODULE.bind_pilot_identity(tmp_path, "a" * 40, tiny=False)
    with pytest.raises(ValueError, match="source or inputs differ"):
        _MODULE.bind_pilot_identity(tmp_path, "b" * 40, tiny=False)
    assert _MODULE.latest_checkpoint(tmp_path / "train/checkpoints").name == "update-00002000.json"


@pytest.mark.parametrize("fail", [False, True])
def test_publication_leaves_training_free_and_reports_worker_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    from test_065_loop import PROVENANCE
    from test_065_loop import exp
    from test_065_loop import trainer

    started, release = threading.Event(), threading.Event()
    caller = threading.get_ident()
    source = tmp_path / "pilot"
    source.mkdir()
    (source / "pilot-identity.json").write_text("fixed identity")
    destination = tmp_path / "volume"
    published: list[int] = []

    def publish(args: self_play_publication.Args) -> None:
        assert threading.get_ident() != caller
        started.set()
        assert release.wait(timeout=30)
        if fail:
            raise OSError("publication failed")
        self_play_publication.stage_state(args.source, args.destination)
        checkpoint = _MODULE.latest_checkpoint(destination / "train/checkpoints")
        update = json.loads(checkpoint.read_text())["update"]
        rows = (destination / "train/metrics.jsonl").read_text().splitlines()
        assert json.loads(rows[-1])["global_step"] == update
        published.append(update)

    monkeypatch.setattr(self_play_publication, "publish", publish)
    publisher = self_play_publication.CheckpointPublisher(
        self_play_publication.Args(source, destination, "test", "o65/pilot-results/test")
    )
    run = trainer(groups=1, anchor_rate=0)
    run.config = dataclasses.replace(run.config, checkpoint_interval=1)
    try:
        exp.run_training(run, 2, source / "train", PROVENANCE, publisher=publisher)
        assert started.wait(timeout=5)
        assert run.update == 2 and not release.is_set()
        # The first snapshot's shards survive local compaction at update two.
        assert len(publisher.pending) == 2
        with pytest.raises(RuntimeError, match="queue is full"):
            publisher.submit(source / "train/checkpoints/update-00000002.json")
    finally:
        release.set()
        if fail:
            with pytest.raises(OSError, match="publication failed"):
                publisher.close()
            with pytest.raises(OSError, match="publication failed"):
                publisher.check()
        else:
            publisher.close()
    assert published == ([] if fail else [1, 2])


@pytest.mark.parametrize("resume", [False, True])
def test_pilot_dispatches_one_trainer_with_in_process_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resume: bool
) -> None:
    from test_065_loop import exp

    output = tmp_path / "pilot"
    commands: list[exp.NativeCli] = []
    previous = None
    if resume:
        output.mkdir()
        sha = _MODULE.subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        _MODULE.bind_pilot_identity(output, sha, tiny=True)
        previous = write_boundary(output / "train/checkpoints", 2)
        (output / "train/metrics.jsonl").write_text('{"global_step":2,"train/warming":0}\n')

    def execute(command: list[str], log: Path, environment: dict[str, str]) -> None:
        if command[1] != str(_MODULE.ROOT / "experiments/065_selfplay_ppo.py"):
            return
        config = tyro.cli(exp.NativeCli, args=command[3:])
        assert environment["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
        commands.append(config)
        write_boundary(config.output / "checkpoints", config.updates)
        (config.output / "metrics.jsonl").write_text('{"global_step":4,"train/warming":0}\n')

    monkeypatch.setattr(_MODULE, "execute", execute)
    _MODULE.main(
        _MODULE.Args(
            output,
            local_delivery=tmp_path / "delivery",
            tiny=True,
            wandb=False,
            publish_directory=tmp_path / "volume",
            publish_volume="test",
            r2_prefix="o65/pilot-results/test",
            resume=resume,
        )
    )
    assert len(commands) == 1
    assert commands[0].updates == 4 and commands[0].training.checkpoint_interval == 1
    assert commands[0].resume == previous
    assert commands[0].publication == self_play_publication.Args(
        output, tmp_path / "volume", "test", "o65/pilot-results/test"
    )


def test_training_rejects_publication_of_another_run(tmp_path: Path) -> None:
    from test_065_loop import exp

    with pytest.raises(ValueError, match="source must contain this run"):
        exp.NativeCli(
            tmp_path / "checkpoint.pt",
            tmp_path / "sampler.json",
            tmp_path / "other/train",
            tmp_path / "data",
            (min(os.sched_getaffinity(0)),),
            1,
            publication=self_play_publication.Args(
                tmp_path / "pilot", tmp_path / "volume", "test", "o65/pilot-results/test"
            ),
        )


def test_verify_only_rejects_unused_publication(tmp_path: Path) -> None:
    from test_065_loop import exp

    with pytest.raises(ValueError, match="verify-only does not publish"):
        exp.NativeCli(
            tmp_path / "checkpoint.pt",
            tmp_path / "sampler.json",
            tmp_path / "pilot/train",
            tmp_path / "data",
            (min(os.sched_getaffinity(0)),),
            1,
            resume=tmp_path / "resume.json",
            verify_resume=True,
            verify_only=True,
            publication=self_play_publication.Args(
                tmp_path / "pilot", tmp_path / "volume", "test", "o65/pilot-results/test"
            ),
        )
