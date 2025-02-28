from pathlib import Path
from typing import Sequence
from typing import Tuple

import torch
from loguru import logger
from streaming import Stream
from streaming import StreamingDataLoader
from streaming import StreamingDataset
from streaming.base.util import clean_stale_shared_memory
from tensordict import TensorDict

from hal.training.config import TrainConfig
from hal.training.distributed import barrier
from hal.training.distributed import get_device_id
from hal.training.distributed import is_master
from hal.training.distributed import log_if_master
from hal.training.streaming_dataset import HALStreamingDataset


def collate_tensordicts(batch: Sequence[TensorDict]) -> TensorDict:
    # Custom collate function for TensorDict because PyTorch type routing doesn't know about it yet
    # Use tensordict's built-in compatibility with torch.stack
    return torch.stack(batch)  # type: ignore


def get_dataloaders(config: TrainConfig) -> Tuple[StreamingDataLoader, StreamingDataLoader]:
    batch_size = config.local_batch_size

    if is_master():
        logger.info("Cleaning stale shared memory for StreamingDataset")
        clean_stale_shared_memory()
    barrier()

    train_streams = None
    val_streams = None
    local_dir = None
    if config.data.streams:
        # Get original streams
        original_streams = config.data.get_streams()

        # Pre-download data on rank 0 only
        if is_master():
            rank = get_device_id()
            # Force download by using original streams with remote paths
            for stream in original_streams:
                if stream.remote is not None:
                    log_if_master(f"Rank {rank}: Pre-downloading data from {stream.remote}...")
                    temp_dataset = StreamingDataset(streams=[stream], batch_size=1, shuffle=False)
                    # Trigger download of a few samples
                    for i, _ in enumerate(temp_dataset):
                        if i > 10:  # Just enough to start downloads
                            break
                    log_if_master(f"Rank {rank}: Pre-downloading {stream.remote} complete")

        # Wait for rank 0 to finish downloading
        barrier()

        # Create new streams with remote=None for all ranks
        # This is the key to avoiding the "reused local directory" error
        train_streams = []
        val_streams = []
        for stream in original_streams:
            train_stream = Stream(
                remote=None,  # Important: set remote to None
                local=stream.local,
                proportion=stream.proportion,
                keep_zip=stream.keep_zip if hasattr(stream, "keep_zip") else False,
                split="train",
            )
            val_stream = Stream(
                remote=None,  # Important: set remote to None
                local=stream.local,
                proportion=stream.proportion,
                keep_zip=stream.keep_zip if hasattr(stream, "keep_zip") else False,
                split="val",
            )
            train_streams.append(train_stream)
            val_streams.append(val_stream)
    else:
        local_dir = config.data.data_dir

    train_dataset = HALStreamingDataset(
        streams=train_streams,
        local=local_dir,
        remote=None,
        batch_size=batch_size,
        shuffle=True,
        data_config=config.data,
        num_canonical_nodes=1,  # fix to single node training
    )

    val_dataset = HALStreamingDataset(
        streams=val_streams,
        local=local_dir,
        remote=None,
        batch_size=batch_size,
        shuffle=False,
        data_config=config.data,
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
