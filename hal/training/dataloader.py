from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple

from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler

from hal.data.dataset import MmappedParquetDataset
from hal.training.config import TrainConfig


def create_dataloaders(
    train_config: TrainConfig, rank: Optional[int], world_size: Optional[int]
) -> Tuple[DataLoader, DataLoader]:
    data_dir = Path(train_config.data.data_dir)
    stats_path = data_dir / "stats.json"

    dataloaders: List[DataLoader] = []
    for split in ("train", "val"):
        dataset = MmappedParquetDataset(
            input_path=data_dir / f"{split}.parquet",
            stats_path=stats_path,
            input_len=train_config.data.input_len,
            target_len=train_config.data.target_len,
        )
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        dataloader: DataLoader[MmappedParquetDataset] = DataLoader(
            dataset,
            batch_size=train_config.local_batch_size,
            shuffle=True if split == "train" else False,
            sampler=sampler,
            num_workers=train_config.dataworker.data_workers_per_gpu,
            # collate_fn=train_config.dataworker.collate_fn,  # TODO
            prefetch_factor=train_config.dataworker.prefetch_factor,
            # collate_fn=train_config.dataworker.collate_fn,  # TODO
        )
        dataloaders.append(dataloader)

    return tuple(dataloaders)
