#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A torchrun-compatible script for distributed training with PyTorch.
This replaces the custom DDP initialization in the trainer.

Usage:
    torchrun --nproc_per_node=NUM_GPUS debug_ddp.py [args]
"""

import argparse
import os
from pathlib import Path
from typing import Union

import torch
import torch.distributed as dist
from loguru import logger
from streaming import StreamingDataLoader

# We're ignoring the tensordict linter error as it's just missing stubs
from tensordict import TensorDict  # type: ignore

from hal.training.config import TrainConfig
from hal.training.config import create_parser_for_attrs_class
from hal.training.config import parse_args_to_attrs_instance
from hal.training.trainer import Trainer


def setup_distributed() -> bool:
    """
    Initialize the distributed environment using environment variables set by torchrun.
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        logger.warning("Not using distributed mode")
        return False

    # Set by torchrun
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # Set the device
    torch.cuda.set_device(local_rank)

    # Initialize the process group
    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)

    logger.info(f"Initialized process group: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    return True


def cleanup_distributed() -> None:
    """
    Clean up the distributed environment.
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def get_device() -> str:
    """
    Get the device for the current process.
    """
    if dist.is_initialized():
        return f"cuda:{dist.get_rank()}"

    if torch.cuda.is_available():
        return "cuda:0"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_master() -> bool:
    """
    Check if the current process is the master process.
    """
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def log_if_master(message: str) -> None:
    """
    Log a message only if the current process is the master process.
    """
    if is_master():
        logger.info(message)


def wrap_model_distributed(
    model: torch.nn.Module,
) -> Union[torch.nn.Module, torch.nn.parallel.DistributedDataParallel]:
    """
    Wrap the model with DistributedDataParallel if distributed training is enabled.
    """
    device = get_device()
    model = model.to(device)

    if dist.is_initialized():
        local_rank = int(os.environ["LOCAL_RANK"])
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    return model


def get_world_size() -> int:
    """
    Get the world size for distributed training.
    """
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


class ConcreteTrainer(Trainer):
    """
    A concrete implementation of the abstract Trainer class.
    """

    def loss(self, pred: TensorDict, target: TensorDict) -> TensorDict:
        """
        Compute the loss between prediction and target.
        """
        # Implement your loss computation here
        # This is a placeholder implementation
        return pred

    def forward_loop(self, batch: TensorDict) -> TensorDict:
        """
        Forward pass through the model.
        """
        # Implement your forward pass here
        # This is a placeholder implementation
        return self.model(batch)


def create_data_loader(config: TrainConfig, is_train: bool = True) -> StreamingDataLoader:
    """
    Create a data loader for training or validation.

    Args:
        config: The training configuration
        is_train: Whether to create a training or validation data loader

    Returns:
        A StreamingDataLoader instance
    """
    # This is a simplified implementation
    # In a real scenario, you would use your actual data loading logic

    if config.data.streams:
        streams = config.data.get_streams()
    else:
        # Use data_dir if streams not specified
        # Import here to avoid potential circular imports
        try:
            from data.streams import LocalShardedStream

            data_dir = Path(config.data.data_dir)
            stream_dir = data_dir / ("train" if is_train else "val")
            streams = [LocalShardedStream(str(stream_dir))]
        except ImportError:
            # Fallback if LocalShardedStream is not available
            logger.error("LocalShardedStream not found. Please specify streams in config.")
            raise

    # Create the data loader
    batch_size = config.local_batch_size

    # If using DDP, each process should only see a subset of the data
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Configure the loader for distributed training
        loader = StreamingDataLoader(
            streams,
            batch_size=batch_size,
            num_workers=config.dataworker.data_workers_per_gpu,
            prefetch_factor=config.dataworker.prefetch_factor,
            drop_last=True,
            distributed=True,
            world_size=world_size,
            rank=rank,
        )
    else:
        # Single process data loading
        loader = StreamingDataLoader(
            streams,
            batch_size=batch_size,
            num_workers=config.dataworker.data_workers_per_gpu,
            prefetch_factor=config.dataworker.prefetch_factor,
            drop_last=True,
        )

    return loader


def main() -> None:
    """
    Main function for distributed training.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Distributed training with torchrun")
    create_parser_for_attrs_class(TrainConfig, parser)
    args = parser.parse_args()

    # Create config
    config = parse_args_to_attrs_instance(TrainConfig, args)

    # Set up distributed training
    is_distributed = setup_distributed()

    try:
        # Set up CUDA
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        # Override n_gpus in config with actual world size
        if is_distributed:
            config.n_gpus = get_world_size()
        else:
            config.n_gpus = 1

        # Create data loaders
        train_loader = create_data_loader(config, is_train=True)
        val_loader = create_data_loader(config, is_train=False)

        # Create trainer
        trainer = ConcreteTrainer(config, train_loader, val_loader)

        # Train
        trainer.train_loop(train_loader, val_loader)

    finally:
        # Clean up
        cleanup_distributed()


if __name__ == "__main__":
    main()
