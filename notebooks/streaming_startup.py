"""Probe StreamingDataset startup cost for the prod MDS, via the real make_loader chain.

The cloud run hung because the default py1e shuffle_block_size (~4M samples) exceeds the
whole dataset (112k samples), so the loader buffered ~everything before the first batch.
This sweeps shuffle_block_size / cache_limit / shm to find a config that yields the first
batch after pulling only a few GB. Run it in the cloud image via compose so /dev/shm and
deps match vast:

    HAL_SHM_SIZE=2gb docker compose -f docker/compose.yaml run --rm hal \
        uv run notebooks/streaming_startup.py --shuffle-block-size 2000 --cache-limit-gb 50

Each run uses a fresh temp cache dir, so the reported bytes are exactly this config's
startup download. AWS_* (R2) creds come from the host shell through compose.
"""

import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import tyro
from loguru import logger

from hal import streams
from hal.training.dataloader import make_loader
from hal.training.stats import load_consolidated_stats


@dataclass(frozen=True)
class Args:
    shuffle_block_size: int = 2000
    """py1e shuffle unit in samples. Default streaming value (~4M) buffers the whole
    112k-sample dataset; a few shards' worth (631 samples/shard) starts fast."""
    cache_limit_gb: int = 50
    """StreamingDataset shard-cache cap; evicts past this so disk stays bounded."""
    num_workers: int = 8
    batch_size: int = 512
    n_batches: int = 5
    L_ctx: int = 256
    L_chunk: int = 16


def _du_gb(path: Path) -> float:
    out = subprocess.run(["du", "-sb", str(path)], capture_output=True, text=True).stdout
    return int(out.split()[0]) / 1e9


def _shm_used_gb() -> float:
    total, used, _ = shutil.disk_usage("/dev/shm")
    return used / 1e9


def main(args: Args) -> None:
    src = streams.RANKED_ANONYMIZED_1
    streams.pull_stats(src)
    stats = load_consolidated_stats(src.local_root / "stats.json")

    Path("/opt/hal/data").mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="streamtune_", dir="/opt/hal/data"))
    logger.info(
        f"config: shuffle_block_size={args.shuffle_block_size} cache_limit={args.cache_limit_gb}gb "
        f"num_workers={args.num_workers} batch_size={args.batch_size} | fresh cache {tmp}"
    )
    loader = make_loader(
        data_root=str(tmp),
        split="train",
        stats=stats,
        L_ctx=args.L_ctx,
        L_chunk=args.L_chunk,
        batch_size=args.batch_size,
        seed=0,
        remote=src.remote,
        cache_limit=f"{args.cache_limit_gb}gb",
        shuffle_block_size=args.shuffle_block_size,
        num_workers=args.num_workers,
        prefetch_factor=4,
    )
    try:
        t0 = time.time()
        it = iter(loader)
        next(it)
        t_first = time.time() - t0
        gb_first, shm_first = _du_gb(tmp), _shm_used_gb()
        logger.success(f"first batch: {t_first:.1f}s | pulled {gb_first:.1f} GB | /dev/shm used {shm_first:.2f} GB")

        for _ in range(args.n_batches - 1):
            next(it)
        t_n = time.time() - t0
        logger.info(
            f"{args.n_batches} batches: {t_n:.1f}s ({t_n / args.n_batches:.1f}s/batch) | "
            f"pulled {_du_gb(tmp):.1f} GB | /dev/shm peak {_shm_used_gb():.2f} GB"
        )
    finally:
        del loader
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
