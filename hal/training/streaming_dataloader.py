from pathlib import Path
from typing import Sequence
from typing import Tuple

import torch
from loguru import logger
from streaming import Stream
from streaming import StreamingDataLoader
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


def pre_download_streams(config: TrainConfig) -> None:
    original_streams = config.data.get_streams()
    # Pre-download data on rank 0 only
    if is_master():
        rank = get_device_id()
        # Force download by using original streams with remote paths
        for stream in original_streams:
            if stream.remote is not None:
                for split in ["train", "val"]:
                    if Path(stream.local).exists():
                        log_if_master(f"Rank {rank}: {split} split already exists in {stream.local}, skipping")
                        continue
                    log_if_master(f"Rank {rank}: Pre-downloading {split} split from {stream.remote}...")
                    temp_stream = Stream(
                        remote=stream.remote,
                        local=stream.local,
                        proportion=stream.proportion,
                        keep_zip=stream.keep_zip,
                    )
                    temp_dataset = HALStreamingDataset(
                        streams=[temp_stream],
                        local=None,
                        remote=None,
                        batch_size=1,
                        shuffle=False,
                        data_config=config.data,
                        split=split,
                    )
                    # Trigger download of a few samples
                    for i, _ in enumerate(temp_dataset):
                        if i > 10:
                            break


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
        # Create new streams with remote=None for all ranks
        # Mosaic Streaming unfortunately filelocks the local directory if remote != None to prevent race conditions
        # To avoid this but also avoid downloading duplicate data, we pre-download the data on rank 0 if it does not exist, then set remote=None for all streams
        # Streams are also modified in-place after passing to StreamingDataset, so we create deep copies
        pre_download_streams(config)
        barrier()

        original_streams = config.data.get_streams()
        train_streams = []
        val_streams = []
        for stream in original_streams:
            train_stream = Stream(
                remote=None,  # Important: set remote to None
                local=stream.local,
                proportion=stream.proportion,
                keep_zip=stream.keep_zip,
            )
            val_stream = Stream(
                remote=None,  # Important: set remote to None
                local=stream.local,
                proportion=stream.proportion,
                keep_zip=stream.keep_zip,
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
        split="train",  # assign 'split' on dataset level to apply to all mixed streams
    )

    val_dataset = HALStreamingDataset(
        streams=val_streams,
        local=local_dir,
        remote=None,
        batch_size=batch_size,
        shuffle=False,
        data_config=config.data,
        num_canonical_nodes=1,
        split="val",
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
