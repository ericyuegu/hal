"""Launch and inspect policy-world-v8 rematerialization jobs on Modal."""

from __future__ import annotations

import json
import subprocess
import uuid
from collections import Counter
from dataclasses import asdict
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Final
from typing import Literal
from typing import cast

import modal
import tyro

from hal import streams
from hal.scripts.rematerialize_policy_world_v8 import POLICY_ID
from hal.scripts.rematerialize_policy_world_v8 import Boto3ObjectStore
from hal.scripts.rematerialize_policy_world_v8 import CorpusJob
from hal.scripts.rematerialize_policy_world_v8 import independent_manifest_counts
from hal.scripts.rematerialize_policy_world_v8 import rematerialize_corpus

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
REMOTE_ROOT: Final[Path] = Path("/opt/hal")
IMAGE: Final[str] = "ghcr.io/ericyuegu/hal:cuda13"
PYPI_INDEX: Final[str] = "https://pypi.org/simple"
PILOT_CORPUS: Final[str] = "professional-rapm-policy-world-v8"
SECRET_KEYS: Final[tuple[str, ...]] = (
    "AWS_ENDPOINT_URL",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_BUCKET",
)
# Modal 1.5.3 rejects the requested 32 GiB ephemeral disk because the current
# service minimum is 512 GiB. Restore 32 GiB if Modal accepts it in the future.
WORKER_EPHEMERAL_DISK_MIB: Final[int] = 512 * 1024


@dataclass(frozen=True, slots=True)
class Args:
    command: Literal["launch", "status"] = "launch"
    """Launch jobs or read publication markers and run records."""

    corpora: tuple[str, ...] = ()
    """Exact v8 corpus names. Empty selects all 44 corpora."""

    run_id: str | None = None
    """Stable run identifier. Launch generates one when omitted; status then shows the latest record."""

    pilot: bool = False
    """Run only professional RapM, stage it, and compare an independent manifest-only count."""

    staging_only: bool = False
    """Audit staging output without publishing it."""

    wait: bool = False
    """Wait for mapped jobs. The default submits a detached spawn_map."""

    dry_run: bool = False
    """Print the exact deterministic job list without creating a Modal App."""

    secret: str = "hal"
    """Modal Secret containing the R2 AWS variables."""

    image: str = IMAGE
    """Dependency image on which the committed source is installed."""

    app_name: str | None = None
    """Optional Modal App name."""


@dataclass(frozen=True, slots=True)
class StatusQuery:
    name: str
    final_prefix: str
    run_id: str | None


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def committed_git_sha() -> str:
    if _git("status", "--porcelain"):
        raise SystemExit("working tree is dirty; commit the rematerializer before launching it")
    return _git("rev-parse", "HEAD")


def _prefix(source: streams.StreamSource) -> str:
    scheme = "s3://"
    if not source.remote.startswith(scheme):
        raise ValueError(f"{source.name}: expected an s3:// source")
    _, separator, key = source.remote.removeprefix(scheme).partition("/")
    if not separator or not key:
        raise ValueError(f"{source.name}: invalid source URI {source.remote!r}")
    return key.rstrip("/")


def _v8_name(source: streams.StreamSource) -> str:
    suffix = "-policy-world-v7"
    if not source.name.endswith(suffix):
        raise ValueError(f"cannot derive v8 name from {source.name!r}")
    return source.name.removesuffix(suffix) + "-policy-world-v8"


def enumerate_corpus_jobs(run_id: str, git_sha: str) -> tuple[CorpusJob, ...]:
    """Return all jobs in stable registry order."""
    jobs: list[CorpusJob] = []
    for source in streams.POLICY_WORLD_V7_SOURCES:
        name = _v8_name(source)
        source_prefix = _prefix(source)
        jobs.append(
            CorpusJob(
                name=name,
                source_prefix=source_prefix,
                staging_prefix=f"processed/_staging/{POLICY_ID}/{run_id}/{name}",
                final_prefix=source_prefix.removesuffix("mds-policy-world-v7") + "mds-policy-world-v8",
                run_id=run_id,
                git_sha=git_sha,
            )
        )
    return tuple(jobs)


def _select_jobs(jobs: tuple[CorpusJob, ...], names: tuple[str, ...]) -> tuple[CorpusJob, ...]:
    if not names:
        return jobs
    if len(names) != len(set(names)):
        raise ValueError("--corpora contains a duplicate name")
    by_name = {job.name: job for job in jobs}
    unknown = sorted(set(names) - set(by_name))
    if unknown:
        raise ValueError(f"unknown v8 corpora: {unknown}")
    requested = set(names)
    return tuple(job for job in jobs if job.name in requested)


def _image(tag: str) -> modal.Image:
    ignore = modal.FilePatternMatcher.from_file(ROOT / ".dockerignore")
    dependencies = (
        modal.Image.from_registry(tag)
        .add_local_file(ROOT / "pyproject.toml", str(REMOTE_ROOT / "pyproject.toml"), copy=True)
        .add_local_file(ROOT / "uv.lock", str(REMOTE_ROOT / "uv.lock"), copy=True)
        .workdir(str(REMOTE_ROOT))
        .run_commands(f"UV_INDEX_URL={PYPI_INDEX} uv sync --locked --no-install-project")
    )
    return (
        dependencies.add_local_dir(ROOT, str(REMOTE_ROOT), copy=True, ignore=ignore)
        .workdir(str(REMOTE_ROOT))
        .run_commands(f"UV_INDEX_URL={PYPI_INDEX} uv sync --locked --offline --no-build-isolation")
    )


def _store() -> Boto3ObjectStore:
    from hal import r2

    return Boto3ObjectStore(r2.client(), r2.bucket())


def _remote_rematerialize(job: CorpusJob, *, staging_only: bool) -> dict[str, object]:
    scratch = Path("/tmp") / POLICY_ID / job.name
    return rematerialize_corpus(_store(), job, scratch, staging_only=staging_only)


def _remote_count(job: CorpusJob) -> dict[str, object]:
    return independent_manifest_counts(_store(), job)


def _remote_status(query: StatusQuery) -> dict[str, object]:
    store = _store()
    success = store.head(f"{query.final_prefix}/_SUCCESS") is not None
    record: dict[str, object] | None = None
    record_key: str | None = None
    if query.run_id is not None:
        candidate = f"processed/_runs/{POLICY_ID}/{query.run_id}/{query.name}.json"
        if store.head(candidate) is not None:
            record_key = candidate
            record = json.loads(store.read_bytes(candidate))
    else:
        suffix = f"/{query.name}.json"
        candidates = sorted(key for key in store.list(f"processed/_runs/{POLICY_ID}/") if key.endswith(suffix))
        for candidate in candidates:
            value = json.loads(store.read_bytes(candidate))
            if record is None or str(value.get("updated_at", "")) > str(record.get("updated_at", "")):
                record_key = candidate
                record = value
    return {
        "corpus": query.name,
        "published": success,
        "success_key": f"{query.final_prefix}/_SUCCESS" if success else None,
        "record_key": record_key,
        "record": record,
    }


def _modal(secret_name: str) -> tuple[modal.Client, modal.Secret]:
    client = modal.Client.from_env()
    client.hello()
    secret = modal.Secret.from_name(secret_name, required_keys=list(SECRET_KEYS))
    secret.hydrate(client=client)
    return client, secret


def _app_name(args: Args, sha: str) -> str:
    if args.app_name is not None:
        return args.app_name
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"hal-policy-world-v8-{stamp}-{sha[:7]}"


def _print_jobs(jobs: tuple[CorpusJob, ...]) -> None:
    print(json.dumps([asdict(job) for job in jobs], indent=2, sort_keys=True))


def _queries(jobs: tuple[CorpusJob, ...], run_id: str | None) -> tuple[StatusQuery, ...]:
    return tuple(StatusQuery(job.name, job.final_prefix, run_id) for job in jobs)


def _status_summary(results: list[dict[str, object]]) -> dict[str, object]:
    states: Counter[str] = Counter()
    sources: dict[str, dict[str, object]] = {}
    for result in results:
        raw_record = result.get("record")
        if not isinstance(raw_record, dict):
            states["no-record"] += 1
            continue
        record = cast(dict[str, object], raw_record)
        states[str(record.get("state", "unknown"))] += 1
        raw_audit = record.get("audit")
        if isinstance(raw_audit, dict):
            sources[str(result["corpus"])] = cast(dict[str, object], raw_audit)

    def audit_count(audit: dict[str, object], name: str) -> int:
        value = audit.get(name)
        if not isinstance(value, int):
            raise ValueError(f"status audit has invalid {name}")
        return value

    def train_replays(audit: dict[str, object]) -> int:
        raw_rows = audit.get("rows")
        if not isinstance(raw_rows, dict):
            raise ValueError("status audit has invalid rows")
        return audit_count(cast(dict[str, object], raw_rows), "train")

    return {
        "corpora": len(results),
        "published": sum(bool(result["published"]) for result in results),
        "states": dict(sorted(states.items())),
        "audited": len(sources),
        "retained": sum(audit_count(audit, "retained") for audit in sources.values()),
        "train_replays": sum(train_replays(audit) for audit in sources.values()),
        "rejections": sum(audit_count(audit, "rejections") for audit in sources.values()),
        "train_frames": sum(audit_count(audit, "train_frames") for audit in sources.values()),
        "sources": sources,
    }


def main(args: Args) -> None:
    if args.pilot and args.command != "launch":
        raise SystemExit("--pilot is valid only with --command launch")
    if args.pilot and args.corpora:
        raise SystemExit("--pilot selects professional RapM; do not also pass --corpora")
    if args.pilot and not args.staging_only:
        raise SystemExit("the pilot must use --staging-only")
    if args.pilot and not args.wait:
        raise SystemExit("the pilot must use --wait so the independent count can be compared")

    sha = committed_git_sha() if args.command == "launch" else _git("rev-parse", "HEAD")
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    all_jobs = enumerate_corpus_jobs(run_id, sha)
    jobs = _select_jobs(all_jobs, (PILOT_CORPUS,) if args.pilot else args.corpora)
    if args.dry_run:
        _print_jobs(jobs)
        return

    client, secret = _modal(args.secret)
    name = _app_name(args, sha)
    app = modal.App(name=name, tags={"provider": "modal", "git_sha": sha, "run_id": run_id})
    image = _image(args.image)
    worker = app.function(
        image=image,
        secrets=[secret],
        include_source=False,
        serialized=True,
        cpu=8.0,
        memory=16 * 1024,
        ephemeral_disk=WORKER_EPHEMERAL_DISK_MIB,
        timeout=24 * 60 * 60,
        retries=modal.Retries(max_retries=2, initial_delay=0.0),
        max_containers=16,
        single_use_containers=True,
        name="rematerialize",
    )(_remote_rematerialize)
    counter = app.function(
        image=image,
        secrets=[secret],
        include_source=False,
        serialized=True,
        cpu=2.0,
        memory=4 * 1024,
        timeout=60 * 60,
        name="manifest-count",
    )(_remote_count)
    status = app.function(
        image=image,
        secrets=[secret],
        include_source=False,
        serialized=True,
        cpu=1.0,
        memory=1024,
        timeout=15 * 60,
        name="status",
    )(_remote_status)

    with modal.enable_output(), app.run(name=name, client=client, detach=args.command == "launch" and not args.wait):
        print(f"Modal App {app.app_id}; run_id={run_id}; corpora={len(jobs)}", flush=True)
        if args.command == "status":
            results = list(status.map(_queries(jobs, args.run_id), order_outputs=True))
            for result in results:
                print(json.dumps(result, sort_keys=True), flush=True)
            print(json.dumps({"summary": _status_summary(results)}, sort_keys=True), flush=True)
            return
        if args.pilot:
            job = jobs[0]
            materialized = worker.spawn(job, staging_only=True).get()
            independent = counter.spawn(job).get()
            if materialized["rows"] != independent["retained"]:
                raise RuntimeError(
                    f"pilot count mismatch: rematerialized={materialized['rows']} independent={independent['retained']}"
                )
            if materialized["rejections"] != independent["rejected_total"]:
                raise RuntimeError(
                    "pilot rejection mismatch: "
                    f"rematerialized={materialized['rejections']} independent={independent['rejected_total']}"
                )
            print(json.dumps({"pilot": materialized, "independent": independent}, sort_keys=True), flush=True)
        elif args.wait:
            for result in worker.map(jobs, kwargs={"staging_only": args.staging_only}, order_outputs=True):
                print(json.dumps(result, sort_keys=True), flush=True)
        else:
            worker.spawn_map(jobs, kwargs={"staging_only": args.staging_only})
            print(f"submitted detached spawn_map: https://modal.com/apps/{app.app_id}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
