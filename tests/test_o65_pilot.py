import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location("o65_pilot", Path(__file__).parents[1] / "experiments/o65/pilot.py")
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


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
    spec = importlib.util.spec_from_file_location(
        "o65_pilot_state", Path(__file__).parents[1] / "scripts/o65_pilot_state.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
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
    spec = importlib.util.spec_from_file_location(
        "o65_pilot_state_guard", Path(__file__).parents[1] / "scripts/o65_pilot_state.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
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
    spec = importlib.util.spec_from_file_location(
        "o65_pilot_state_claim", Path(__file__).parents[1] / "scripts/o65_pilot_state.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
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
