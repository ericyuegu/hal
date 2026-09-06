"""Stage, audit, and immutably publish the policy-world-v8 corpus."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any
from typing import Final

import tyro
from loguru import logger

from hal import r2
from hal import streams
from hal.data.index import read_jsonl
from hal.data.schema import POLICY_WORLD_V8_MDS_COLUMNS
from hal.scripts.filter_policy_world_mds import DEFAULT_SCRATCH
from hal.scripts.filter_policy_world_mds import audit_policy_world_v8
from hal.scripts.filter_policy_world_mds import filter_policy_world_mds
from hal.scripts.publish_mds import publish_mds

V7_SUFFIX: Final[str] = "/mds-policy-world-v7"
V8_SUFFIX: Final[str] = "/mds-policy-world-v8"
RANK_ONE_SOURCE: Final[str] = "ranked-anonymized-1-policy-world-v7"
AKLO_SOURCE: Final[str] = "professional-aklo-policy-world-v7"
OBSOLETE_RANK_ONE_PREFIX: Final[str] = "r2:hal/processed/ranked-anonymized-1/mds-policy-world-v8"
EXPECTED_V7_ROWS: Final[int] = 1_326_988


def _git_sha() -> str:
    repo = Path(__file__).resolve().parents[2]
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _rclone_source(source: streams.StreamSource) -> str:
    prefix = "s3://hal/"
    if not source.remote.startswith(prefix) or not source.remote.endswith(V7_SUFFIX):
        raise ValueError(f"unexpected policy-world-v7 source URI {source.remote!r}")
    return f"r2:hal/{source.remote.removeprefix(prefix)}"


def _v8_name(source: streams.StreamSource) -> str:
    if not source.name.endswith("-v7"):
        raise ValueError(f"unexpected policy-world-v7 source name {source.name!r}")
    return f"{source.name.removesuffix('-v7')}-v8"


def _final_prefix(source: streams.StreamSource) -> str:
    return f"{_rclone_source(source).removesuffix(V7_SUFFIX)}{V8_SUFFIX}"


def _staging_parent(source: streams.StreamSource, staging_root: str, git_sha: str) -> str:
    relative = _rclone_source(source).removeprefix("r2:hal/processed/").removesuffix(V7_SUFFIX)
    return f"{staging_root.rstrip('/')}/{git_sha}/{relative}"


def _staging_attempt(source: streams.StreamSource, staging_root: str, git_sha: str, scratch: Path) -> str:
    parent = _staging_parent(source, staging_root, git_sha)
    for attempt in range(1, 1_000):
        candidate = f"{parent}/attempt-{attempt:03d}/mds-policy-world-v8"
        objects = r2.list_files(candidate)
        if not objects:
            return candidate
        if "selection.json" not in objects:
            logger.warning(f"ignoring incomplete v8 staging attempt: {candidate}")
            continue
        try:
            audit_policy_world_v8(candidate, scratch=scratch)
        except FileNotFoundError, RuntimeError, ValueError:
            logger.warning(f"ignoring invalid v8 staging attempt: {candidate}")
            continue
        return candidate
    raise RuntimeError(f"no free staging attempt below {parent}")


def _selected_sources(names: tuple[str, ...]) -> tuple[streams.StreamSource, ...]:
    available = {source.name: source for source in streams.POLICY_WORLD_V7_SOURCES}
    if not names:
        return streams.POLICY_WORLD_V7_SOURCES
    unknown = sorted(set(names) - set(available))
    if unknown:
        raise ValueError(f"unknown policy-world-v7 source names: {unknown}")
    return tuple(available[name] for name in names)


def _rank_one_metadata(
    path: Path,
    staging_root: str,
    git_sha: str,
) -> tuple[str, dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"ranked-1 supplemental index not found: {path}")
    sha256 = _sha256(path)
    destination = f"{staging_root.rstrip('/')}/{git_sha}/_metadata/ranked-1-index.{sha256}.jsonl"
    objects = r2.list_files(destination)
    if objects and objects != [Path(destination).name]:
        raise FileExistsError(f"unexpected objects at immutable metadata path {destination}: {objects}")
    if not objects:
        r2.copy_file(path, destination)
    return destination, {"remote": destination, "sha256": sha256, "bytes": path.stat().st_size}


def _remote_rank_one_metadata(path: str) -> tuple[str, dict[str, object]]:
    name = Path(path).name
    if not name.startswith("ranked-1-index.") or not name.endswith(".jsonl"):
        raise ValueError("remote ranked-1 metadata must use its SHA-256-pinned filename")
    sha256 = name.removeprefix("ranked-1-index.").removesuffix(".jsonl")
    if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
        raise ValueError("remote ranked-1 metadata filename has an invalid SHA-256")
    objects = r2.list_files(path)
    if objects != [name]:
        raise FileNotFoundError(f"immutable ranked-1 metadata object differs at {path}: {objects}")
    size = json.loads(r2.run_rclone("size", path, "--json"))
    if int(size.get("count", -1)) != 1 or int(size.get("bytes", -1)) < 1:
        raise ValueError(f"remote ranked-1 metadata size is invalid: {size}")
    return path, {"remote": path, "sha256": sha256, "bytes": int(size["bytes"])}


def stage_rank_one_metadata(path: Path, staging_root: str) -> tuple[str, dict[str, object]]:
    """Upload the local supplemental index before starting a Modal worker."""
    return _rank_one_metadata(path, staging_root, _git_sha())


def _resolve_rank_one_metadata(
    local: Path,
    remote: str | None,
    staging_root: str,
    git_sha: str,
) -> tuple[str, dict[str, object]]:
    if remote is not None:
        return _remote_rank_one_metadata(remote)
    return _rank_one_metadata(local, staging_root, git_sha)


def _report_row(source: streams.StreamSource, prefix: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_name": source.name,
        "name": _v8_name(source),
        "source": _rclone_source(source),
        "remote": prefix.replace("r2:hal/", "s3://hal/", 1),
        "rows": result["rows"],
        "frames": result["frames"],
        "source_rows": result["source_rows"],
        "rejections": result["rejections"],
        "rejection_reasons": result["rejection_reasons"],
        "ranks": result["ranks"],
        "bytes": result.get("bytes"),
        "object_hashes": result.get("object_hashes"),
        "columns": POLICY_WORLD_V8_MDS_COLUMNS,
    }


def _audit_manifest_uniqueness(prefixes: dict[str, str], scratch: Path) -> dict[str, object]:
    seen: dict[str, tuple[str, str]] = {}
    split_rows: Counter[str] = Counter()
    with tempfile.TemporaryDirectory(dir=scratch, prefix="v8-manifests-") as name:
        root = Path(name)
        for source_name, prefix in prefixes.items():
            manifest = root / f"{source_name}.jsonl"
            r2.copy_file(f"{prefix.rstrip('/')}/manifest.jsonl", manifest)
            for entry in read_jsonl(manifest):
                if entry.annotation is None:
                    raise ValueError(f"{source_name}: output manifest contains an unannotated row")
                if entry.sha1 is None or len(entry.sha1) != 40:
                    raise ValueError(f"{source_name}: {entry.path} has no valid SHA-1")
                previous = seen.get(entry.sha1)
                if previous is not None:
                    raise ValueError(
                        f"unexpected cross-corpus duplicate SHA-1 {entry.sha1}: "
                        f"{previous[0]} {previous[1]} and {source_name} {entry.path}"
                    )
                seen[entry.sha1] = (source_name, entry.path)
                split_rows[entry.annotation.split] += 1
    return {"unique_sha1": len(seen), "rows": {split: split_rows[split] for split in ("train", "val", "test")}}


def _corpus_summary(
    manifest_audit: dict[str, object],
    source_audits: dict[str, dict[str, Any]],
) -> dict[str, object]:
    input_rows = sum(
        int(count) for audit_result in source_audits.values() for count in audit_result["source_rows"].values()
    )
    if input_rows != EXPECTED_V7_ROWS:
        raise ValueError(f"44-source v7 input has {input_rows} rows, expected {EXPECTED_V7_ROWS}")
    observed: Counter[str] = Counter()
    imputed: Counter[str] = Counter()
    sides = 0
    for audit_result in source_audits.values():
        ranks = audit_result["ranks"]
        observed.update(ranks["observed"])
        imputed.update(ranks["imputed"])
        sides += int(ranks["sides"])
    return {
        **manifest_audit,
        "source_rows": input_rows,
        "ranks": {
            "observed": dict(sorted(observed.items())),
            "imputed": dict(sorted(imputed.items())),
            "sides": sides,
        },
    }


def verify_and_delete_obsolete_rank_one_prefix() -> list[str]:
    """Delete only the authorized incomplete rank-1 prefix after the pilot gate."""
    prefix = OBSOLETE_RANK_ONE_PREFIX
    if prefix != "r2:hal/processed/ranked-anonymized-1/mds-policy-world-v8":
        raise AssertionError("obsolete rank-1 deletion target changed")
    objects = r2.list_files(prefix)
    if "_SUCCESS" in objects:
        raise ValueError(f"refusing to delete a successful artifact at {prefix}")
    if objects:
        logger.warning(f"deleting {len(objects)} objects from authorized incomplete prefix {prefix}")
        r2.run_rclone("purge", prefix)
    return objects


@dataclass(frozen=True, slots=True)
class PolicyWorldV8ScaleupConfig:
    sources: tuple[str, ...] = ()
    staging_root: str = "r2:hal/processed/_staging/policy-world-v8"
    scratch: Path = DEFAULT_SCRATCH
    report: Path = Path("data/builds/policy-world-v8/publication.json")
    rank_one_metadata_index: Path = Path("data/processed/ranked-anonymized-1/index.jsonl")
    rank_one_metadata_remote: str | None = None
    publish: bool = False
    max_concurrent_modal_jobs: int = 8


def scaleup_policy_world_v8(cfg: PolicyWorldV8ScaleupConfig) -> dict[str, Any]:
    if not 1 <= cfg.max_concurrent_modal_jobs <= 8:
        raise ValueError("max_concurrent_modal_jobs must be in 1..8")
    selected = _selected_sources(cfg.sources)
    if cfg.publish and len(selected) != len(streams.POLICY_WORLD_V7_SOURCES):
        raise ValueError("immutable publication requires the complete 44-source corpus")
    git_sha = _git_sha()
    cfg.scratch.mkdir(parents=True, exist_ok=True)
    rank_one_metadata: str | None = None
    metadata_identity: dict[str, object] | None = None
    if any(source.name == RANK_ONE_SOURCE for source in selected):
        rank_one_metadata, metadata_identity = _resolve_rank_one_metadata(
            cfg.rank_one_metadata_index,
            cfg.rank_one_metadata_remote,
            cfg.staging_root,
            git_sha,
        )

    staged: dict[str, str] = {}
    audited: dict[str, dict[str, Any]] = {}
    for position, source in enumerate(selected, start=1):
        final = _final_prefix(source)
        final_objects = r2.list_files(final)
        if final_objects:
            if "_SUCCESS" not in final_objects:
                raise FileExistsError(f"nonempty final prefix lacks _SUCCESS: {final}")
            result = audit_policy_world_v8(final, scratch=cfg.scratch)
            staged[source.name] = final
        else:
            staging = _staging_attempt(source, cfg.staging_root, git_sha, cfg.scratch)
            if "selection.json" not in r2.list_files(staging):
                logger.info(f"[{position}/{len(selected)}] building {_v8_name(source)}")
                filter_policy_world_mds(
                    _rclone_source(source),
                    staging,
                    source_name=source.name,
                    scratch=cfg.scratch,
                    metadata_index=rank_one_metadata if source.name == RANK_ONE_SOURCE else None,
                )
            result = audit_policy_world_v8(staging, scratch=cfg.scratch)
            staged[source.name] = staging
        audited[source.name] = result
        cfg.report.parent.mkdir(parents=True, exist_ok=True)
        cfg.report.write_text(
            json.dumps(
                {
                    "git_sha": git_sha,
                    "rank_one_metadata": metadata_identity,
                    "sources": {
                        name: _report_row(next(item for item in selected if item.name == name), staged[name], value)
                        for name, value in audited.items()
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    aggregate = None
    if len(staged) == len(streams.POLICY_WORLD_V7_SOURCES):
        aggregate = _corpus_summary(_audit_manifest_uniqueness(staged, cfg.scratch), audited)
    if cfg.publish:
        if aggregate is None:
            raise AssertionError("complete corpus audit was not run")
        for source in selected:
            final = _final_prefix(source)
            if staged[source.name] != final:
                publish_mds(
                    staged[source.name],
                    final,
                    purge_staging=True,
                    audit_fn=partial(audit_policy_world_v8, scratch=cfg.scratch),
                )
                staged[source.name] = final
        final_prefixes = {source.name: _final_prefix(source) for source in selected}
        for source in selected:
            audited[source.name] = audit_policy_world_v8(_final_prefix(source), scratch=cfg.scratch)
        aggregate = _corpus_summary(_audit_manifest_uniqueness(final_prefixes, cfg.scratch), audited)

    report = {
        "git_sha": git_sha,
        "rank_one_metadata": metadata_identity,
        "aggregate": aggregate,
        "sources": {
            source.name: _report_row(source, staged[source.name], audited[source.name]) for source in selected
        },
    }
    cfg.report.parent.mkdir(parents=True, exist_ok=True)
    cfg.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    logger.info(f"audited {len(audited)}/{len(selected)} policy-world-v8 sources")
    return report


@dataclass(frozen=True, slots=True)
class PolicyWorldV8PilotConfig:
    staging_root: str = "r2:hal/processed/_staging/policy-world-v8"
    scratch: Path = DEFAULT_SCRATCH
    report: Path = Path("data/builds/policy-world-v8/pilot.json")
    rank_one_metadata_index: Path = Path("data/processed/ranked-anonymized-1/index.jsonl")
    rank_one_metadata_remote: str | None = None
    ranked_train_rows: int = 16_384


def build_policy_world_v8_pilot(cfg: PolicyWorldV8PilotConfig) -> dict[str, Any]:
    if cfg.ranked_train_rows < 1:
        raise ValueError("ranked_train_rows must be positive")
    git_sha = _git_sha()
    sources_by_name = {source.name: source for source in streams.POLICY_WORLD_V7_SOURCES}
    selected = (sources_by_name[AKLO_SOURCE], sources_by_name[RANK_ONE_SOURCE])
    rank_one_metadata, metadata = _resolve_rank_one_metadata(
        cfg.rank_one_metadata_index,
        cfg.rank_one_metadata_remote,
        cfg.staging_root,
        git_sha,
    )
    results: dict[str, dict[str, Any]] = {}
    for source in selected:
        pilot_root = f"{cfg.staging_root.rstrip('/')}/pilot"
        staging = _staging_attempt(source, pilot_root, git_sha, cfg.scratch)
        if "selection.json" not in r2.list_files(staging):
            limits = {"train": cfg.ranked_train_rows} if source.name == RANK_ONE_SOURCE else None
            filter_policy_world_mds(
                _rclone_source(source),
                staging,
                source_name=source.name,
                scratch=cfg.scratch,
                metadata_index=rank_one_metadata if source.name == RANK_ONE_SOURCE else None,
                retained_limits=limits,
            )
        results[source.name] = {
            "prefix": staging,
            "audit": audit_policy_world_v8(staging, scratch=cfg.scratch),
        }
    report = {"git_sha": git_sha, "rank_one_metadata": metadata, "sources": results}
    cfg.report.parent.mkdir(parents=True, exist_ok=True)
    cfg.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(scaleup_policy_world_v8(tyro.cli(PolicyWorldV8ScaleupConfig)), indent=2, sort_keys=True))
