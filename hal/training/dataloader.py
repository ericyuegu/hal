from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler

from hal.training.config import TrainConfig
from hal.training.dataset import MmappedParquetDataset


def create_dataloaders(
    train_config: TrainConfig, rank: Optional[int], world_size: Optional[int]
) -> Tuple[DataLoader, DataLoader]:
    data_dir = Path(train_config.data.data_dir)
    stats_path = data_dir / "stats.json"

    dataloaders: List[DataLoader] = []
    for split in ("train", "val"):
        is_train = split == "train"
        dataset = MmappedParquetDataset(
            input_path=data_dir / f"{split}.parquet",
            stats_path=stats_path,
            data_config=train_config.data,
        )
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, seed=train_config.seed, drop_last=is_train
        )
        dataloader: DataLoader[MmappedParquetDataset] = DataLoader(
            dataset,
            batch_size=train_config.local_batch_size,
            shuffle=is_train,
            sampler=sampler,
            num_workers=train_config.dataworker.data_workers_per_gpu,
            pin_memory=torch.cuda.is_available(),
            # collate_fn=train_config.dataworker.collate_fn,  # TODO
            prefetch_factor=train_config.dataworker.prefetch_factor,
            # collate_fn=train_config.dataworker.collate_fn,  # TODO
            persistent_workers=True,
        )
        dataloaders.append(dataloader)

    return dataloaders[0], dataloaders[1]
