from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader

from hal.training.config import TrainConfig
from hal.training.streaming_dataset import HALStreamingDataset


def get_dataloaders(config: TrainConfig) -> Tuple[DataLoader, DataLoader]:
    batch_size = config.local_batch_size
    train_dir = Path(config.data.data_dir) / "train"
    val_dir = Path(config.data.data_dir) / "val"
    train_dataset = HALStreamingDataset(
        local=str(train_dir),
        remote=None,
        batch_size=batch_size,
        shuffle=True,
        data_config=config.data,
        embed_config=config.embedding,
        stats_path=config.data.stats_path,
    )
    val_dataset = HALStreamingDataset(
        local=str(val_dir),
        remote=None,
        batch_size=batch_size,
        shuffle=False,
        data_config=config.data,
        embed_config=config.embedding,
        stats_path=config.data.stats_path,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader
