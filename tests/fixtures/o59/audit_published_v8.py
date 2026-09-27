"""Inventory and inspect immutable published v8 corpora without rewriting them."""

import argparse
import configparser
import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Final
from typing import Literal
from typing import cast

from hal import r2
from hal import streams
from hal.data.policy_world_v8 import DRAFT_PUBLICATION_GIT_SHA
from hal.data.policy_world_v8 import DRAFT_PUBLICATION_RUN_ID
from hal.data.policy_world_v8 import DRAFT_PUBLISHED_CORPORA
from hal.data.policy_world_v8 import Boto3ObjectStore
from hal.data.policy_world_v8 import CorpusJob
from hal.data.policy_world_v8 import audit_dataset
from hal.data.policy_world_v8 import inspect_source

DEFAULT_RCLONE_CONFIG: Final[Path] = Path.home() / ".config/rclone/rclone.conf"


@dataclass(frozen=True, slots=True)
class Args:
    mode: Literal["inventory", "metadata", "full"]
    output: Path
    rclone_config: Path = DEFAULT_RCLONE_CONFIG
    corpus: str | None = None
    scratch_root: Path | None = None


def _configure_r2(path: Path) -> None:
    settings = configparser.ConfigParser()
    if not settings.read(path) or "r2" not in settings:
        raise ValueError(f"R2 credentials are absent from {path}")
    values = settings["r2"]
    os.environ["AWS_ENDPOINT_URL"] = values["endpoint"]
    os.environ["AWS_ACCESS_KEY_ID"] = values["access_key_id"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = values["secret_access_key"]
    os.environ["AWS_BUCKET"] = "hal"


def _prefix(uri: str) -> str:
    if not uri.startswith("s3://hal/"):
        raise ValueError(f"unexpected corpus URI: {uri}")
    return uri.removeprefix("s3://hal/")


def _job(
    store: Boto3ObjectStore,
    name: str,
    source_prefix: str,
    final_prefix: str,
) -> tuple[CorpusJob, dict[str, Any], str]:
    marker_bytes = store.read_bytes(f"{final_prefix}/_SUCCESS")
    marker = cast(dict[str, Any], json.loads(marker_bytes))
    if set(marker) == {
        "bytes",
        "failures",
        "final",
        "objects",
        "published_at",
        "rows",
        "shards",
        "staging",
    }:
        if marker["final"] != f"r2:hal/{final_prefix}" or marker["failures"] != 0:
            raise ValueError(f"{name}: full-MDS publication marker differs")
        staging_prefix = f"processed/_staging/policy-world-v8/inspection-only/{name}"
        run_id = "inspection-only"
        git_sha = "0" * 40
    else:
        if marker.get("schema_version") != 1 or marker.get("corpus") != name:
            raise ValueError(f"{name}: invalid publication marker identity")
        if marker.get("final_prefix") != final_prefix or marker.get("source", {}).get("prefix") != source_prefix:
            raise ValueError(f"{name}: publication marker points at different source or final corpus")
        staging_prefix = str(marker["staging_prefix"])
        run_id = str(marker["run_id"])
        git_sha = str(marker["git_sha"])
    job = CorpusJob(
        name=name,
        source_prefix=source_prefix,
        staging_prefix=staging_prefix,
        final_prefix=final_prefix,
        run_id=run_id,
        git_sha=git_sha,
    )
    if name in DRAFT_PUBLISHED_CORPORA and (
        job.run_id != DRAFT_PUBLICATION_RUN_ID or job.git_sha != DRAFT_PUBLICATION_GIT_SHA
    ):
        raise ValueError(f"{name}: draft publication provenance differs")
    return job, marker, hashlib.sha256(marker_bytes).hexdigest()


def main(args: Args) -> None:
    _configure_r2(args.rclone_config)
    store = Boto3ObjectStore(r2.client(), r2.bucket())
    v7 = {
        source.name.removesuffix("-policy-world-v7") + "-policy-world-v8": source
        for source in streams.POLICY_WORLD_V7_SOURCES
    }
    names = tuple(source.name for source in streams.POLICY_WORLD_V8_SOURCES)
    if len(names) != 44 or set(names) != set(v7):
        raise ValueError("current v7/v8 source registries do not match the 44-source contract")
    if args.corpus is not None and args.corpus not in names:
        raise ValueError(f"unknown published v8 corpus {args.corpus}")
    if args.mode == "full" and args.corpus is None:
        raise ValueError("full row audit requires one explicit --corpus")
    if args.mode == "full" and args.scratch_root is None:
        raise ValueError("full row audit requires an explicit disk-backed --scratch-root")

    results: list[dict[str, object]] = []
    started = time.monotonic()
    for source in streams.POLICY_WORLD_V8_SOURCES:
        if args.corpus is not None and source.name != args.corpus:
            continue
        source_prefix = _prefix(v7[source.name].remote)
        final_prefix = _prefix(source.remote)
        job, marker, marker_sha = _job(store, source.name, source_prefix, final_prefix)
        objects = store.list(final_prefix + "/")
        data_objects = {key: value for key, value in objects.items() if key != f"{final_prefix}/_SUCCESS"}
        audit = cast(dict[str, Any], marker.get("audit", marker))
        compressed_bytes = sum(value.size for value in data_objects.values())
        if len(data_objects) != audit["objects"] or compressed_bytes != audit["bytes"]:
            raise ValueError(f"{source.name}: publication marker object count or bytes differ")
        result: dict[str, object] = {
            "name": source.name,
            "source_prefix": source_prefix,
            "final_prefix": final_prefix,
            "marker_sha256": marker_sha,
            "run_id": job.run_id,
            "git_sha": job.git_sha,
            "marker_format": "full-mds" if "audit" not in marker else "policy-world-v8",
            "job_identity_for_inspection_only": "audit" not in marker,
            "draft_exception": source.name in DRAFT_PUBLISHED_CORPORA,
            "published_audit": audit,
            "compressed_bytes": compressed_bytes,
            "data_objects": len(data_objects),
        }
        if args.mode in ("metadata", "full"):
            inspected = inspect_source(store, job)
            marker_source = cast(dict[str, Any] | None, marker.get("source"))
            identity = inspected.identity
            expected_marker_source = (
                {key: identity[key] for key in ("prefix", "manifest_sha256", "index_sha256", "success_sha256", "rows")}
                if source.name in DRAFT_PUBLISHED_CORPORA
                else identity
            )
            if marker_source is not None and marker_source != expected_marker_source:
                raise ValueError(f"{source.name}: publication marker source identity differs")
            if inspected.selection.retained_counts != audit["rows"]:
                raise ValueError(f"{source.name}: retained selection differs from published row counts")
            result["source_identity"] = identity
            result["selection_input_rows"] = inspected.selection.input_counts
            result["selection_retained_rows"] = inspected.selection.retained_counts
            result["selection_rejections"] = len(inspected.selection.rejections)
            if args.mode == "full":
                scratch_root = args.scratch_root
                assert scratch_root is not None
                scratch_root.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(prefix=f"hal-v8-audit-{source.name}-", dir=scratch_root) as temp:
                    recomputed = audit_dataset(
                        store,
                        final_prefix,
                        job=job,
                        source=inspected,
                        scratch=Path(temp),
                    )
                expected = (
                    {key: audit[key] for key in ("rows", "objects", "bytes")} if "audit" not in marker else audit
                )
                actual = {key: recomputed[key] for key in expected}
                if actual != expected:
                    raise ValueError(f"{source.name}: recomputed row audit differs from publication marker")
                result["recomputed_audit"] = recomputed
        results.append(result)
        print(
            f"{len(results)}/{1 if args.corpus else len(names)} {source.name}: {compressed_bytes:,} bytes", flush=True
        )

    report = {
        "mode": args.mode,
        "selected_corpus": args.corpus,
        "source_count": len(results),
        "source_order": names,
        "total_compressed_bytes": sum(cast(int, row["compressed_bytes"]) for row in results),
        "total_published_train_rows": sum(
            cast(int, cast(dict[str, Any], row["published_audit"])["rows"]["train"]) for row in results
        ),
        "elapsed_seconds": time.monotonic() - started,
        "scope": (
            "read-only full row, metadata, checksum, and statistics audit of one published v8 corpus"
            if args.mode == "full"
            else "read-only publication marker/object inventory and source inspection; no v8 shard rows were downloaded"
        ),
        "sources": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {key: report[key] for key in ("mode", "source_count", "total_compressed_bytes", "elapsed_seconds")},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("inventory", "metadata", "full"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rclone-config", type=Path, default=DEFAULT_RCLONE_CONFIG)
    parser.add_argument("--corpus")
    parser.add_argument("--scratch-root", type=Path)
    namespace = parser.parse_args()
    main(Args(namespace.mode, namespace.output, namespace.rclone_config, namespace.corpus, namespace.scratch_root))
