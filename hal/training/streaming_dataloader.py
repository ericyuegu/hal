from pathlib import Path
from typing import Sequence
from typing import Tuple

import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader

from hal.training.config import TrainConfig
from hal.training.streaming_dataset import HALStreamingDataset


def collate_tensordicts(batch: Sequence[TensorDict]) -> TensorDict:
    # Custom collate function for TensorDict because PyTorch type routing doesn't know about it yet
    # Use tensordict's built-in compatibility with torch.stack
    return torch.stack(batch)  # type: ignore


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
        embedding_config=config.embedding,
        stats_path=config.data.stats_path,
    )
    val_dataset = HALStreamingDataset(
        local=str(val_dir),
        remote=None,
        batch_size=batch_size,
        shuffle=False,
        data_config=config.data,
        embedding_config=config.embedding,
        stats_path=config.data.stats_path,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_tensordicts,
        num_workers=config.dataworker.data_workers_per_gpu,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=config.dataworker.prefetch_factor,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_tensordicts,
        num_workers=config.dataworker.data_workers_per_gpu,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=config.dataworker.prefetch_factor,
    )

    return train_loader, val_loader
