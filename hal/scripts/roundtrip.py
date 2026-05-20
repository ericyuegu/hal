"""CLI: replay an MDS row through Dolphin and compare gamestate to truth.

Compose ``Session`` + ``MdsControllerSource`` (per port) + ``drive`` + ``diff``
for one round-trip validation run. Truth is the original .slp re-read via
peppi-py (``Trajectory.from_slp``); the live capture comes from libmelee.

Run:
    python -m hal.scripts.roundtrip \
        --mds-dir /path/to/mds \
        --replay-uuid 12345678 \
        --iso /path/to/melee.iso \
        --dolphin-path ~/data/dolphin/exiai/squashfs-root/AppRun \
        --max-frames 600
"""

from pathlib import Path

import tyro
from loguru import logger
from streaming import StreamingDataset

from hal.data.index import ReplayIndexEntry
from hal.data.index import read_jsonl
from hal.data.index import resolve_replay_path
from hal.sim.diff import diff
from hal.sim.loop import drive
from hal.sim.session import ReplayMatchup
from hal.sim.session import Session
from hal.sim.sources import ControllerSource
from hal.sim.sources import InternalControllerSource
from hal.sim.sources import MdsControllerSource
from hal.sim.trajectory import Trajectory


def _load_manifest_entry(manifest_path: Path, replay_uuid: int | None) -> ReplayIndexEntry:
    entries = list(read_jsonl(manifest_path))
    if not entries:
        raise FileNotFoundError(f"no entries in {manifest_path}")
    if replay_uuid is None:
        # Fall back to the first entry — convenient for dev runs over a tiny
        # MDS where every entry is interesting.
        return entries[0]
    for entry in entries:
        if entry.annotation is not None and entry.annotation.replay_uuid == replay_uuid:
            return entry
    raise KeyError(f"replay_uuid={replay_uuid} not found in {manifest_path}")


def _read_mds_row(mds_dir: Path, split: str, row_idx: int) -> dict:
    """Single-row read out of an MDS shard. StreamingDataset indexes over the
    union of shards in a split; row_idx is the position within that split."""
    ds = StreamingDataset(
        local=str(mds_dir / split),
        remote=None,
        shuffle=False,
        batch_size=1,
        allow_unsafe_types=False,
    )
    return ds[row_idx]


def roundtrip(
    mds_dir: Path,
    iso: Path,
    dolphin_path: Path,
    *,
    manifest: Path | None = None,
    replay_uuid: int | None = None,
    max_frames: int = 600,
) -> int:
    """Round-trip validation: returns 0 on PASS, 1 on FAIL.

    Args:
        mds_dir: directory containing train/val/test/manifest.jsonl produced
            by ``hal.scripts.materialize``.
        iso: path to the Melee ISO.
        dolphin_path: path to the Dolphin executable / AppRun.
        manifest: override path to manifest.jsonl. Defaults to
            ``mds_dir/manifest.jsonl``.
        replay_uuid: pick a specific replay by uuid. If None, picks the first
            entry in the manifest.
        max_frames: bit-exact comparison window. ~600 ≈ 10s of game time.
    """
    manifest_path = manifest or (mds_dir / "manifest.jsonl")
    entry = _load_manifest_entry(manifest_path, replay_uuid)
    if entry.annotation is None:
        raise ValueError(f"entry for {entry.path} has no Stage3Annotation")
    logger.info(
        f"replay {entry.path}  uuid={entry.annotation.replay_uuid:#x}  "
        f"split={entry.annotation.split}  row_idx={entry.annotation.mds_row_idx}  "
        f"frames={entry.annotation.frame_count_actual}"
    )

    sample = _read_mds_row(mds_dir, entry.annotation.split, entry.annotation.mds_row_idx)

    matchup = ReplayMatchup.from_replay(entry)
    logger.info(
        f"matchup: stage={matchup.stage} players={[(p.port, p.character.name) for p in matchup.players]} "
        f"port_to_mds={matchup.port_to_mds_prefix}"
    )

    sources: dict[int, ControllerSource] = {}
    for port, prefix in matchup.port_to_mds_prefix.items():
        sources[port] = MdsControllerSource(columns=sample, port_prefix=prefix)
    for player in matchup.players:
        sources.setdefault(player.port, InternalControllerSource())

    with Session(iso_path=iso, dolphin_path=dolphin_path) as s:
        live = drive(s, matchup, sources, max_frames=max_frames)

    truth = Trajectory.from_slp(resolve_replay_path(entry)).take(max_frames)
    report = diff(live, truth, max_frames=max_frames)
    logger.info(report.summary())
    for d in report.divergences[:5]:
        logger.info(f"  divergence: {d}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(tyro.cli(roundtrip))
