"""Exercise distributed progress while checkpoint storage is blocked."""

import dataclasses
import hashlib
import json
import sys
import threading
from datetime import timedelta
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs
from urllib.parse import urlsplit

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from boto3.exceptions import S3UploadFailedError
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


@pytest.mark.parametrize("failure", ["transient_part", "persistent_part", "publication_record"])
def test_r2_retries_multipart_requests_and_commits_only_complete_uploads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    source = tmp_path / "pilot"
    checkpoint = write_boundary(source / "train/checkpoints", 1)
    large = source / "optimizer.pt"
    # Cross the production part size. A serial multipart upload cannot pass the barrier.
    with large.open("wb") as stream:
        stream.truncate(65 * 1024**2)
    expected = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source.rglob("*") if path.is_file()
    }
    uploaded: dict[str, str] = {}
    parts: dict[int, str] = {}
    attempts: dict[int, int] = {}
    server_errors: list[str] = []
    aborted = threading.Event()
    overlapping_parts = threading.Event()
    lock = threading.Lock()
    record_attempts = 0
    commits = 0

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: object) -> None:
            pass

        def reply(self, status: int = 200, body: bytes = b"") -> None:
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("ETag", '"test-etag"')
            self.end_headers()
            self.wfile.write(body)

        def consume(self) -> str:
            digest = hashlib.sha256()
            remaining = int(self.headers["Content-Length"])
            while remaining:
                chunk = self.rfile.read(min(remaining, 1024**2))
                assert chunk
                digest.update(chunk)
                remaining -= len(chunk)
            return digest.hexdigest()

        def do_POST(self) -> None:
            query = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
            self.consume()
            if "uploads" in query:
                self.reply(
                    body=b"<InitiateMultipartUploadResult><UploadId>test</UploadId></InitiateMultipartUploadResult>"
                )
            else:
                with lock:
                    assert set(parts) == {1, 2}
                    assert parts[1] == hashlib.sha256(bytes(64 * 1024**2)).hexdigest()
                    assert parts[2] == hashlib.sha256(bytes(1024**2)).hexdigest()
                    uploaded[large.name] = expected[large.name]
                self.reply(
                    body=b"<CompleteMultipartUploadResult><ETag>test-etag</ETag></CompleteMultipartUploadResult>"
                )

        def do_PUT(self) -> None:
            nonlocal record_attempts
            parsed = urlsplit(self.path)
            name = Path(parsed.path).name
            query = parse_qs(parsed.query)
            digest = self.consume()
            if "partNumber" in query:
                part = int(query["partNumber"][0])
                with lock:
                    attempts[part] = attempts.get(part, 0) + 1
                    attempt = attempts[part]
                    if len(attempts) == 2:
                        overlapping_parts.set()
                assert overlapping_parts.wait(timeout=10), "multipart requests did not overlap"
                if part == 1 and (failure == "persistent_part" or attempt == 1):
                    self.reply(502, b"<Error><Code>BadGateway</Code></Error>")
                    return
                with lock:
                    parts[part] = digest
            else:
                with lock:
                    if name == checkpoint.name:
                        assert set(uploaded) == expected.keys() - {checkpoint.name}
                    if name == "publication.json":
                        record_attempts += 1
                        assert uploaded == expected
                        if failure == "publication_record":
                            self.reply(502, b"<Error><Code>BadGateway</Code></Error>")
                            return
                    uploaded[name] = digest
            self.reply()

        def do_DELETE(self) -> None:
            aborted.set()
            self.reply(204)

    class Volume:
        def commit(self) -> None:
            nonlocal commits
            commits += 1

    class Server(ThreadingHTTPServer):
        def handle_error(self, request: object, client_address: tuple[str, int]) -> None:
            # socketserver catches handler assertions; SDK retries must not hide them.
            with lock:
                server_errors.append(repr(sys.exception()))
            super().handle_error(request, client_address)

    with Server(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        monkeypatch.setenv("AWS_ENDPOINT_URL", f"http://127.0.0.1:{server.server_port}")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
        monkeypatch.setenv("AWS_BUCKET", "test")
        monkeypatch.setattr(publication.modal.Volume, "from_name", lambda name: Volume())
        args = publication.Args(source, tmp_path / "volume", "test", "o65/pilot-results/test")
        previous = args.destination / "publication.json"
        try:
            if failure == "transient_part":
                publication.publish(args)
                assert attempts == {1: 2, 2: 1}
                assert record_attempts == 1 and commits == 2
                assert json.loads(previous.read_text())["files"]["optimizer.pt"] == expected[large.name]
                assert uploaded["publication.json"] == hashlib.sha256(previous.read_bytes()).hexdigest()
            else:
                with pytest.raises(S3UploadFailedError, match="BadGateway"):
                    publication.publish(args)
                assert not previous.exists() and commits == 1
                if failure == "persistent_part":
                    assert attempts[1] == 4 and aborted.is_set()
                    assert checkpoint.name not in uploaded and record_attempts == 0
                else:
                    assert record_attempts == 4 and "publication.json" not in uploaded
            # R2 failure must not remove the committed Volume recovery checkpoint.
            assert (args.destination / checkpoint.relative_to(source)).read_bytes() == checkpoint.read_bytes()
        finally:
            server.shutdown()
            thread.join(timeout=10)
    assert not server_errors, server_errors
