"""Resumable production runner for ranked-anonymized policy-world datasets."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import tyro
from loguru import logger

from hal.data.archive import archive_member_path
from hal.data.archive import list_archive_slps
from hal.data.index import read_jsonl
from hal.scripts.build_index import build_index
from hal.scripts.filter import FilterConfig
from hal.scripts.filter import run as filter_replays
from hal.scripts.materialize import process_replays
from hal.scripts.publish_mds import audit
from hal.scripts.publish_mds import publish_mds


def _rclone_objects(prefix: str) -> list[str]:
    result = subprocess.run(
        ["rclone", "lsf", prefix, "--recursive", "--files-only"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"rclone lsf failed for {prefix}: {detail}")
    return [line for line in result.stdout.splitlines() if line]


def _failure_paths(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with path.open() as handle:
        return {str(json.loads(line)["path"]) for line in handle if line.strip()}


def _validate_index_accounting(archive: Path, index: Path, failures: Path) -> dict[str, int]:
    members = list_archive_slps(archive)
    expected = {archive_member_path(archive, member) for member in members}
    successes = {entry.path for entry in read_jsonl(index)}
    failed = _failure_paths(failures)
    foreign = (successes | failed) - expected
    missing = expected - successes - failed
    if foreign:
        raise ValueError(f"{archive.name}: index/failure ledger contains {len(foreign)} foreign paths")
    if missing:
        raise ValueError(f"{archive.name}: {len(missing)} members are absent from index and failure ledger")
    return {"members": len(expected), "indexed": len(successes), "failures": len(failed)}


def _empty_attempt(staging_root: str, rank: int) -> str:
    base = f"{staging_root.rstrip('/')}/ranked-anonymized-{rank}"
    for attempt in range(1, 1_000):
        candidate = f"{base}/attempt-{attempt:03d}/mds-policy-world-v7"
        objects = _rclone_objects(candidate)
        if not objects:
            return candidate
        if "projection.json" in objects:
            try:
                audit(candidate)
            except RuntimeError, ValueError:
                logger.warning(f"ignoring incomplete staged attempt: {candidate}")
            else:
                return candidate
    raise RuntimeError(f"no free staging attempt below {base}")


@dataclass
class RankedScaleupConfig:
    ranks: tuple[int, ...] = (2, 3, 4, 5, 6)
    raw_root: Path = Path("data/raw")
    build_root: Path = Path("data/builds/policy-world-20260816")
    staging_root: str = "r2:hal/processed/_staging/policy-world-20260816"
    final_root: str = "r2:hal/processed"
    workers: int = 10
    queue_size: int = 8
    index_tmpfs: Path = Path("/dev/shm/hal_ranked_index")
    materialize_tmpfs: Path = Path("/dev/shm/hal_ranked_materialize")


def scaleup_ranked(cfg: RankedScaleupConfig) -> None:
    for rank in cfg.ranks:
        if rank not in range(1, 7):
            raise ValueError(f"rank archive number must be 1..6, got {rank}")
        archives = sorted(cfg.raw_root.glob(f"ranked-anonymized-{rank}-*.7z"))
        if len(archives) != 1:
            raise ValueError(f"rank {rank}: expected one archive, found {archives}")
        archive = archives[0]
        root = cfg.build_root / f"ranked-anonymized-{rank}"
        root.mkdir(parents=True, exist_ok=True)
        index = root / "index.jsonl"
        failures = root / "index.failures.jsonl"
        marker = root / "index-complete.json"
        if not marker.is_file():
            try:
                accounting = _validate_index_accounting(archive, index, failures)
            except FileNotFoundError, ValueError:
                build_index(
                    output=index,
                    archive=archive,
                    incremental=index.exists(),
                    compute_sha1=True,
                    with_stats=True,
                    workers=cfg.workers,
                    tmpfs_root=cfg.index_tmpfs / f"rank-{rank}",
                    queue_size=cfg.queue_size,
                    failure_log=failures,
                )
                accounting = _validate_index_accounting(archive, index, failures)
            marker.write_text(json.dumps(accounting, indent=2, sort_keys=True) + "\n")
        else:
            accounting = json.loads(marker.read_text())
        logger.info(f"rank {rank}: index accounting {accounting}")

        paths = root / "paths.txt"
        kept = filter_replays(FilterConfig(index=index, output=paths))
        logger.info(f"rank {rank}: quality filter kept {kept}")

        final = f"{cfg.final_root.rstrip('/')}/ranked-anonymized-{rank}/mds-policy-world-v7"
        final_objects = _rclone_objects(final)
        if final_objects:
            if "_SUCCESS" not in final_objects:
                raise FileExistsError(f"rank {rank}: nonempty final prefix lacks _SUCCESS: {final}")
            logger.info(f"rank {rank}: already published: {final}")
            continue

        staging = _empty_attempt(cfg.staging_root, rank)
        staged_objects = _rclone_objects(staging)
        if "projection.json" not in staged_objects:
            process_replays(
                paths_file=paths,
                index=index,
                output=staging,
                workers=cfg.workers,
                tmpfs_root=cfg.materialize_tmpfs / f"rank-{rank}",
                queue_size=cfg.queue_size,
                replay_format="policy-world",
            )
        logger.info(f"rank {rank}: staged audit {audit(staging)}")
        publish_mds(staging, final, purge_staging=True)


if __name__ == "__main__":
    scaleup_ranked(tyro.cli(RankedScaleupConfig))
