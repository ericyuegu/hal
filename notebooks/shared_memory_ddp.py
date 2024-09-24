# %%
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from loguru import logger

from hal.training.config import DataConfig
from hal.training.config import TrainConfig
from hal.training.dataset import InMemoryDataset
from hal.training.dataset import load_filtered_parquet_as_tensordict
from hal.training.distributed import print
from hal.training.mem_utils import MemoryMonitor


def _worker(rank: int, world_size: int, dataset: torch.utils.data.Dataset) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    time.sleep(1)
    while True:
        for sample in dataset:
            time.sleep(0.000001)
            result = sample


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def train() -> None:
    train_config = TrainConfig(
        n_gpus=2, debug=True, arch="GPTv1-4-4", data=DataConfig(data_dir="/opt/projects/hal2/data/dev")
    )
    split = "train"
    monitor = MemoryMonitor()
    print(monitor.table())

    logger.info(f"Loading {split} dataset")
    input_path = Path(train_config.data.data_dir) / f"{split}.parquet"
    td = load_filtered_parquet_as_tensordict(input_path, train_config.data)
    td.share_memory_()
    print(monitor.table())

    logger.info(f"Creating {split} dataloader")
    dataset = InMemoryDataset(
        tensordict=td,
        stats_path=train_config.data.stats_path,
        data_config=train_config.data,
        embed_config=train_config.embedding,
    )
    world_size = 4

    # Initialize the process group in the main process
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=0, world_size=world_size)

    time.sleep(1)
    ctx = torch.multiprocessing.start_processes(
        _worker, args=(world_size, dataset), nprocs=world_size, join=False, start_method="forkserver"
    )

    pids = ctx.pids()
    all_pids = [None] * world_size
    dist.all_gather_object(all_pids, pids)
    print(f"All PIDs: {all_pids}")
    monitor = MemoryMonitor(all_pids)
    print(monitor.table())

    try:
        for k in range(100):
            # Print memory (of all processes) in the main process only.
            print(monitor.table())
            time.sleep(1)
    finally:
        ctx.join()
        dist.destroy_process_group()  # Add this line to clean up


if __name__ == "__main__":
    train()
