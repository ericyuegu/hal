"""Build a shard-level subset of an MDS dataset, hardlinked (no data is copied).

Keeps every ``--every``-th train shard and all of the other splits, then rewrites
each split's ``index.json`` over the kept shards. The result is a smaller
dataset for local screening runs that costs no extra disk: the shard files are
hardlinks to the source's, so deleting either copy keeps the other readable.

The manifest is deliberately NOT carried over. Dropping shards renumbers nothing
but invalidates every ``Stage3Annotation.mds_row_idx``, so a copied manifest
would address the wrong rows; the subset is a training input only.

Usage:
    python -m hal.scripts.subset_mds --src <mds> --out <mds-sub4> --every 4
"""

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import tyro
from loguru import logger

# Splits thinned by ``--every``. Everything else present in the source is kept
# whole: val and test are already small and a screening run must still see all
# of them, or its numbers stop comparing with the full dataset's.
THINNED_SPLITS: tuple[str, ...] = ("train",)


def _hardlink(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError as e:
        raise OSError(f"cannot hardlink {src} -> {dst}; --out must be on the same filesystem as --src") from e


def _link_shard(src_dir: Path, out_dir: Path, info: dict[str, Any]) -> dict[str, Any]:
    """Hardlink one shard's files, and return its index entry for the kept files.

    A shard can be on disk uncompressed, compressed, or both. Whatever is there
    is linked; an entry that claims a compressed copy which is not on disk is
    rewritten as uncompressed so the reader looks for the file that exists.
    """
    out = deepcopy(info)
    raw = src_dir / info["raw_data"]["basename"]
    zip_data = info.get("zip_data")
    zipped = src_dir / zip_data["basename"] if zip_data else None

    if raw.is_file():
        _hardlink(raw, out_dir / raw.name)
    if zipped is not None and zipped.is_file():
        _hardlink(zipped, out_dir / zipped.name)
    elif raw.is_file():
        out["zip_data"] = None
        out["compression"] = None
    else:
        raise FileNotFoundError(f"shard {info['raw_data']['basename']} is on disk in neither form under {src_dir}")
    return out


def _subset_split(src: Path, out: Path, split: str, every: int) -> tuple[int, int]:
    index = json.loads((src / split / "index.json").read_text())
    stride = every if split in THINNED_SPLITS else 1
    kept = index["shards"][::stride]
    if not kept:
        raise ValueError(f"{split}: --every {every} kept no shards out of {len(index['shards'])}")

    (out / split).mkdir(parents=True)
    shards = [_link_shard(src / split, out / split, info) for info in kept]
    (out / split / "index.json").write_text(json.dumps({**index, "shards": shards}))
    return len(shards), sum(int(s["samples"]) for s in shards)


def subset_mds(src: Path, out: Path, *, every: int = 4) -> None:
    """Hardlink every ``every``-th train shard of ``src`` into ``out``."""
    if every < 1:
        raise ValueError(f"--every must be >= 1; got {every}")
    if out.exists():
        raise FileExistsError(f"{out} already exists; choose a fresh --out")
    splits = [d.name for d in sorted(src.iterdir()) if (d / "index.json").is_file()]
    if not splits:
        raise FileNotFoundError(f"{src} holds no MDS split (no <split>/index.json)")

    out.mkdir(parents=True)
    for split in splits:
        n_shards, n_samples = _subset_split(src, out, split, every)
        logger.info(f"{split}: {n_shards} shards, {n_samples} rows")

    stats = src / "stats.json"
    if stats.is_file():
        _hardlink(stats, out / "stats.json")
    else:
        logger.warning(f"{stats} not found; the subset has no stats.json and training cannot normalize")

    logger.info(f"subset (every {every} of {THINNED_SPLITS}): {src} -> {out}")


if __name__ == "__main__":
    tyro.cli(subset_mds)
