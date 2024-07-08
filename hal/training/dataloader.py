from pathlib import Path
from typing import Tuple

from data.dataset import MmappedParquetDataset
from torch.utils.data import DataLoader

from hal.training.config import TrainConfig


def create_dataloaders(dataset_dir: Path, train_config: TrainConfig) -> Tuple[DataLoader, DataLoader]:
    stats_path = dataset_dir / "stats.json"
    for split in ("train", "val"):
        dataset = MmappedParquetDataset(
            str(dataset_dir / f"{split}.parquet"),
            str(stats_path),
            train_config.dataset.input_len,
            train_config.target_len,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=train_config.local_batch_size,
            num_workers=train_config.data_workers_per_gpu,
            prefetch_factor=train_config.prefetch_factor,
        )
