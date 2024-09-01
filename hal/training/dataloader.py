from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torch.utils.data import Sampler

from hal.training.config import TrainConfig
from hal.training.dataset import MmappedParquetDataset
from hal.training.dataset import SizedDataset


class RepeatFirstBatchSampler(Sampler):
    """For debugging"""

    def __init__(self, dataset: SizedDataset, batch: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.batch_indices = torch.randint(0, len(dataset), (batch,)).tolist()

    def __iter__(self):
        while True:
            yield from iter(self.batch_indices)


def create_dataloaders(
    train_config: TrainConfig, rank: Optional[int], world_size: Optional[int]
) -> Tuple[DataLoader, DataLoader]:
    data_dir = Path(train_config.data.data_dir)
    stats_path = data_dir / "stats.json"
    is_distributed = rank is not None and world_size is not None and world_size > 1

    dataloaders: List[DataLoader] = []
    for split in ("train", "val"):
        is_train = split == "train"
        # Dataset
        dataset = MmappedParquetDataset(
            input_path=data_dir / f"{split}.parquet",
            stats_path=stats_path,
            data_config=train_config.data,
            embed_config=train_config.embedding,
        )
        # Sampler
        debug_repeat_batch = train_config.data.debug_repeat_batch
        shuffle = is_train and not (is_distributed or debug_repeat_batch)
        if debug_repeat_batch:
            sampler = RepeatFirstBatchSampler(dataset, batch=train_config.local_batch_size)
        else:
            sampler = (
                DistributedSampler(
                    dataset, num_replicas=world_size, rank=rank, seed=train_config.seed, drop_last=is_train
                )
                if is_distributed
                else None
            )
        # Dataloader
        dataloader: DataLoader[MmappedParquetDataset] = DataLoader(
            dataset,
            batch_size=train_config.local_batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=train_config.dataworker.data_workers_per_gpu,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=train_config.dataworker.prefetch_factor,
            persistent_workers=True,
        )
        dataloaders.append(dataloader)

    return dataloaders[0], dataloaders[1]
