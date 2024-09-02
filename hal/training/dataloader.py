from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple

import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import DistributedSampler
from torch.utils.data import Sampler

from hal.training.config import DataConfig
from hal.training.config import TrainConfig
from hal.training.dataset import InMemoryDataset
from hal.training.dataset import load_filtered_parquet_as_tensordict


class RepeatFirstBatchSampler(Sampler):
    """For debugging"""

    def __init__(self, dataset: Dataset, batch: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.batch_indices = torch.randint(0, len(dataset), (batch,)).tolist()

    def __iter__(self):
        while True:
            yield from iter(self.batch_indices)


def create_tensordicts(data_config: DataConfig) -> Tuple[TensorDict, TensorDict]:
    data_dir = Path(data_config.data_dir)
    tds: List[TensorDict] = []
    for split in ("train", "val"):
        input_path = data_dir / f"{split}.parquet"
        td = load_filtered_parquet_as_tensordict(input_path, data_config)
        td.share_memory_()
        tds.append(td)
    return tds[0], tds[1]


def create_dataloaders(
    train_td: TensorDict, val_td: TensorDict, train_config: TrainConfig, rank: Optional[int], world_size: Optional[int]
) -> Tuple[DataLoader, DataLoader]:
    data_dir = Path(train_config.data.data_dir)
    stats_path = data_dir / "stats.json"
    is_distributed = rank is not None and world_size is not None and world_size > 1

    dataloaders: List[DataLoader] = []
    for split in ("train", "val"):
        is_train = split == "train"
        # Dataset
        dataset = InMemoryDataset(
            tensordict=train_td if is_train else val_td,
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
        num_workers = 1 if train_config.debug else train_config.dataworker.data_workers_per_gpu

        # Dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=train_config.local_batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=torch.stack,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=train_config.dataworker.prefetch_factor,
            persistent_workers=True,
        )
        dataloaders.append(dataloader)

    return dataloaders[0], dataloaders[1]
