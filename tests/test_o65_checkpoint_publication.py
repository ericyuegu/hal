"""Exercise distributed progress while checkpoint storage is blocked."""

import dataclasses
import json
import threading
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_o65_pilot import write_boundary

from hal.training import self_play_publication as publication


def _distributed_worker(rank: int, rendezvous: str, directory: str) -> None:
    from test_065_loop import PROVENANCE
    from test_065_loop import exp
    from test_065_loop import trainer

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=90))
    source = Path(directory) / "pilot"
    output = source / "train"
    output.mkdir(parents=True, exist_ok=True)
    started, release = threading.Event(), threading.Event()
    publisher = (
        publication.CheckpointPublisher(
            publication.Args(source, Path(directory) / "volume", "test", "o65/pilot-results/test")
        )
        if rank == 0
        else None
    )

    def publish(args: publication.Args) -> None:
        started.set()
        assert release.wait(timeout=30), "training did not advance during the upload"
        publication.stage_state(args.source, args.destination)

    run = trainer(groups=1)
    run.config = dataclasses.replace(run.config, checkpoint_interval=1)
    update = run.run_update

    def run_update():
        uploading = run.update == 1 and rank == 0
        if uploading:
            assert started.wait(timeout=10)
        result = update()
        if uploading:
            assert publisher is not None and publisher.pending and not publisher.pending[0].done()
            release.set()
        return result

    check = exp.DistributedCheckpointWorker.check

    def check_after_local_write(worker: exp.DistributedCheckpointWorker, *, wait: bool = False) -> None:
        # Control file readiness so this test isolates overlap with network I/O.
        # The rank-worker test separately blocks disk writes during an update.
        check(worker, wait=True)

    try:
        with (
            patch.object(publication, "publish", publish),
            patch.object(run, "run_update", run_update),
            patch.object(exp.DistributedCheckpointWorker, "check", check_after_local_write),
        ):
            exp.run_training(run, 2, output, PROVENANCE, publisher=publisher)
            if publisher is not None:
                publisher.close()
        assert run.update == 2
        # New source provenance is explicit; all other exact-resume checks stay active.
        checkpoint = output / "checkpoints/update-00000002.json"
        newer = PROVENANCE | {"git_sha": "e" * 40}
        with pytest.raises(ValueError, match="environment or checkpoint"):
            exp.restore_checkpoint(run, checkpoint, newer)
        exp.restore_checkpoint(run, checkpoint, newer, source_git_sha=PROVENANCE["git_sha"])
        expected, _ = run.run_update()
        expected_weights = {name: value.clone() for name, value in run.policy.model.state_dict().items()}
        cold_engine = exp.RolloutEngine(run.engine.groups, run.policy, run.engine.sampler, {0: run.reference})
        cold = exp.Trainer(cold_engine, run.config)
        assert not cold.engine.matches
        exp.restore_checkpoint(cold, checkpoint, newer, source_git_sha=PROVENANCE["git_sha"])
        resumed, _ = cold.run_update()
        from test_065_loop import assert_batch_equal

        assert_batch_equal(expected, resumed)
        for name, value in cold.policy.model.state_dict().items():
            torch.testing.assert_close(value, expected_weights[name], atol=0, rtol=0)
        dist.barrier()
    finally:
        release.set()
        if publisher is not None:
            publisher.close()
        dist.destroy_process_group()


def test_distributed_training_advances_during_upload_and_resumes(tmp_path: Path) -> None:
    mp.spawn(_distributed_worker, args=(f"file://{tmp_path / 'rendezvous'}", str(tmp_path)), nprocs=2, join=True)
    manifest = tmp_path / "volume/train/checkpoints/update-00000002.json"
    assert json.loads(manifest.read_text())["update"] == 2
    rows = (tmp_path / "volume/train/metrics.jsonl").read_text().splitlines()
    assert [json.loads(row)["global_step"] for row in rows] == [1, 2]


def test_publication_bounds_snapshots_by_waiting_for_oldest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "pilot"
    checkpoint = write_boundary(source / "train/checkpoints", 1)
    entered, release, submitted = threading.Event(), threading.Event(), threading.Event()

    def publish(args: publication.Args) -> None:
        entered.set()
        assert release.wait(timeout=10)

    monkeypatch.setattr(publication, "publish", publish)
    publisher = publication.CheckpointPublisher(
        publication.Args(source, tmp_path / "volume", "test", "o65/pilot-results/test")
    )

    def submit() -> None:
        publisher.submit(checkpoint)
        submitted.set()

    try:
        publisher.submit(checkpoint)
        assert entered.wait(timeout=5)
        publisher.submit(checkpoint)
        thread = threading.Thread(target=submit)
        thread.start()
        assert not submitted.wait(timeout=0.1)
        release.set()
        thread.join(timeout=10)
        assert submitted.is_set()
    finally:
        release.set()
        publisher.close()


@pytest.mark.parametrize("fail", [False, True])
def test_r2_uploads_overlap_and_publish_manifest_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    source = tmp_path / "pilot"
    write_boundary(source / "train/checkpoints", 1)
    barrier = threading.Barrier(3, timeout=10)
    uploaded: list[str] = []

    class Client:
        def upload_file(self, source: str, bucket: str, key: str, **kwargs: object) -> None:
            name = Path(key).name
            if name.endswith((".pt", ".zlib")):
                barrier.wait()
                if fail:
                    raise OSError("upload failed")
            if name.startswith("update-"):
                assert len(uploaded) == 3
            if name == "publication.json":
                assert len(uploaded) == 4
            uploaded.append(name)

    class Volume:
        def commit(self) -> None:
            pass

    monkeypatch.setattr(publication.r2, "client", lambda **kwargs: Client())
    monkeypatch.setattr(publication.r2, "bucket", lambda: "test")
    monkeypatch.setattr(publication.modal.Volume, "from_name", lambda name: Volume())
    args = publication.Args(source, tmp_path / "volume", "test", "o65/pilot-results/test")
    if fail:
        with pytest.raises(OSError, match="upload failed"):
            publication.publish(args)
        assert not uploaded
    else:
        publication.publish(args)
        assert uploaded[-1] == "publication.json"
