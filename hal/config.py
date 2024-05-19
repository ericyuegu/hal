from pathlib import Path
from typing import Optional, Sequence, Tuple, Literal

import attr
import torch

from hal.types import DEVICES, EVAL_MODE, EVAL_STAGES


@attr.s(auto_attribs=True, frozen=True)
class DatasetConfig:
    """Training & eval dataset metadata."""
    data_dir: str
    meta_path: Optional[str] = None
    test_ratio: float = 0.1
    # comma-separated lists of characters, or "all"
    allowed_characters: str = 'all'
    allowed_opponents: str = 'all'
    seed: int = 0


@attr.s(auto_attribs=True, frozen=True)
class RolloutConfig:
    """Number of gamestate frames for each training example ('rollout')."""
    input_frame_count: int = 64
    target_frame_count: int = 16


@attr.s(auto_attribs=True, frozen=True)
class EmbeddingConfig:
    normalization_fn: str
    analog_discretization_fn: str


@attr.s(auto_attribs=True, frozen=True)
class DataloaderConfig:
    data_workers_per_gpu: int
    prefetch_factor: float
    collate_fn: Optional[str] = None


@attr.s(auto_attribs=True, frozen=True)
class DataConfig:
    """
    Dataset, preprocessing, and embedding configs.

    Useful for input/output data consistency across training and eval.
    """
    dataset: DatasetConfig
    rollout: RolloutConfig
    embed: EmbeddingConfig


@attr.s(auto_attribs=True, frozen=True)
class TrainerConfig:
    # Used for launching DDP training
    n_gpus: int

    # Configs
    data_config: DataConfig
    loader_config: DataloaderConfig

    # Hyperparameters
    local_batch_size: int = 1024
    lr: float = 3e-4
    train_samples: int = 2 ** 20
    val_samples: int = 2 ** 16
    num_checkpoints: int = 4
    report_len: int = int(train_samples / 8)
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    wd: float = 1e-2
    debug: bool = False


@attr.s(auto_attribs=True, frozen=True)
class ClosedLoopEvalConfig:
    data_config: DataConfig
    model_arch: torch.Module
    model_path: Path
    opponent: EVAL_MODE = "cpu"
    opponent_model_arch: Optional[torch.Module] = None
    opponent_model_path: Optional[Path] = None
    # Which device to load model(s) for inference
    device: DEVICES = "cpu"
    # Comma-separated lists of stages, or "all"
    stage: EVAL_STAGES = "all"
