"""Resumable production runner for per-player professional policy-world data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro
from loguru import logger

from hal.scripts.materialize import process_replays
from hal.scripts.prepare_professional import PrepareProfessionalConfig
from hal.scripts.prepare_professional import load_sources
from hal.scripts.prepare_professional import prepare_professional
from hal.scripts.publish_mds import audit
from hal.scripts.publish_mds import publish_mds
from hal.scripts.scaleup_ranked import _rclone_objects


def _empty_attempt(staging_root: str, slug: str) -> str:
    base = f"{staging_root.rstrip('/')}/professional/{slug}"
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
class ProfessionalScaleupConfig:
    raw_root: Path = Path("/home/ericgu/data/raw")
    manifest: Path = Path("/home/ericgu/data/raw/dropbox-pro-replays-manifest.jsonl")
    build_root: Path = Path("data/builds/policy-world-20260816/professional")
    staging_root: str = "r2:hal/processed/_staging/policy-world-20260816"
    final_root: str = "r2:hal/processed"
    players: tuple[str, ...] = ()
    workers: int = 10
    queue_size: int = 8
    min_owner_coverage: float = 0.5
    rank_mode: Literal["owner", "corpus"] = "owner"
    index_tmpfs: Path = Path("/dev/shm/hal_professional_index")
    materialize_tmpfs: Path = Path("/dev/shm/hal_professional_materialize")


def scaleup_professional(cfg: ProfessionalScaleupConfig) -> None:
    grouped = load_sources(cfg.raw_root, cfg.manifest)
    slugs = cfg.players or tuple(grouped)
    unknown = sorted(set(slugs) - set(grouped))
    if unknown:
        raise ValueError(f"unknown professional player slugs: {unknown}")

    for slug in slugs:
        final = f"{cfg.final_root.rstrip('/')}/professional/{slug}/mds-policy-world-v7"
        final_objects = _rclone_objects(final)
        if final_objects:
            if "_SUCCESS" not in final_objects:
                raise FileExistsError(f"{slug}: nonempty final prefix lacks _SUCCESS: {final}")
            logger.info(f"{slug}: already published: {final}")
            continue

        prepare_professional(
            PrepareProfessionalConfig(
                raw_root=cfg.raw_root,
                manifest=cfg.manifest,
                build_root=cfg.build_root,
                player=slug,
                workers=cfg.workers,
                queue_size=cfg.queue_size,
                tmpfs_root=cfg.index_tmpfs,
                rank_mode=cfg.rank_mode,
            )
        )
        root = cfg.build_root / slug
        report = json.loads((root / "professional-report.json").read_text())
        rank_report = report["rank"]
        selected = int(rank_report["selected_replays"])
        labeled = int(rank_report["owner_labeled_replays"])
        coverage = labeled / max(1, selected)
        if not rank_report["owner_tokens"] or coverage < cfg.min_owner_coverage:
            raise ValueError(
                f"{slug}: owner identity coverage {labeled}/{selected} ({coverage:.1%}) is below "
                f"the {cfg.min_owner_coverage:.1%} publication floor; inspect {root / 'professional-report.json'}"
            )

        staging = _empty_attempt(cfg.staging_root, slug)
        staged_objects = _rclone_objects(staging)
        if "projection.json" not in staged_objects:
            process_replays(
                paths_file=root / "paths.txt",
                index=root / "deduped-index.jsonl",
                output=staging,
                workers=cfg.workers,
                tmpfs_root=cfg.materialize_tmpfs / slug,
                queue_size=cfg.queue_size,
                replay_format="policy-world",
                rank_overrides=root / "rank-overrides.jsonl",
            )
        logger.info(f"{slug}: staged audit {audit(staging)}")
        publish_mds(staging, final, purge_staging=True)


if __name__ == "__main__":
    scaleup_professional(tyro.cli(ProfessionalScaleupConfig))
