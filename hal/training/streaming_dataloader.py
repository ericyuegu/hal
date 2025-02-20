from copy import deepcopy
from pathlib import Path
from typing import Sequence
from typing import Tuple

import torch
from streaming import StreamingDataLoader
from tensordict import TensorDict

from hal.training.config import TrainConfig
from hal.training.streaming_dataset import HALStreamingDataset


def collate_tensordicts(batch: Sequence[TensorDict]) -> TensorDict:
    # Custom collate function for TensorDict because PyTorch type routing doesn't know about it yet
    # Use tensordict's built-in compatibility with torch.stack
    return torch.stack(batch)  # type: ignore


def get_dataloaders(config: TrainConfig) -> Tuple[StreamingDataLoader, StreamingDataLoader]:
    batch_size = config.local_batch_size

    train_streams = None
    val_streams = None
    local_dir = None
    if config.data.streams:
        train_streams = config.data.get_streams()
        val_streams = deepcopy(train_streams)  # StreamingDataset mutates streams passed in
    else:
        local_dir = config.data.data_dir

    train_dataset = HALStreamingDataset(
        streams=train_streams,
        local=local_dir,
        remote=None,
        batch_size=batch_size,
        shuffle=True,
        data_config=config.data,
        split="train",
        num_canonical_nodes=1,  # fix to single node training
    )

    val_dataset = HALStreamingDataset(
        streams=val_streams,
        local=local_dir,
        remote=None,
        batch_size=batch_size,
        shuffle=False,
        data_config=config.data,
        split="val",
        num_canonical_nodes=1,
    )

    train_loader = StreamingDataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_tensordicts,
        num_workers=config.dataworker.data_workers_per_gpu,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=config.dataworker.prefetch_factor,
    )
    val_loader = StreamingDataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_tensordicts,
        num_workers=config.dataworker.data_workers_per_gpu,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=config.dataworker.prefetch_factor,
    )

    return train_loader, val_loader


def save_dataloader_state(loader: StreamingDataLoader, path: Path) -> None:
    """Checkpoint the dataloader state to disk."""
    state = loader.state_dict()
    with path.open("wb") as f:
        torch.save(state, f)


def load_dataloader_state(loader: StreamingDataLoader, path: Path) -> None:
    """Load checkpointed dataloader state from disk."""
    state = torch.load(path)
    loader.load_state_dict(state)
