import argparse
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Type

import attr
import torch

from hal.constants import DEVICES
from hal.constants import EVAL_MODE
from hal.constants import EVAL_STAGES


@attr.s(auto_attribs=True, frozen=True)
class DatasetConfig:
    """Training & eval dataset metadata."""

    data_dir: str
    stats_path: Optional[str]
    input_preprocessing_fn: str
    target_preprocessing_fn: str
    # Number of input and target frames in example/rollout
    input_len: int
    target_len: int
    seed: int = 42


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
class TempConfig:
    # Used for launching DDP training
    n_gpus: int

    # Configs
    data_config: DataConfig
    loader_config: DataloaderConfig

    # Hyperparameters
    local_batch_size: int = 1024
    lr: float = 3e-4
    train_samples: int = 2**20
    val_samples: int = 2**16
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


@attr.s(auto_attribs=True, frozen=True)
class TrainConfig:
    n_gpus: int

    arch: str = "lstm-v0-256-2"

    loss_fn: str = "ce"
    local_batch_size: int = 1024
    num_data_workers: int = 4  # per gpu
    lr: float = 3e-4
    n_samples: int = 2**27
    n_val_samples: int = 2**17
    keep_ckpts: int = 8
    report_len: int = 2**24
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    wd: float = 1e-2
    debug: bool = False


def create_parser_for_attrs_class(
    cls: Type[Any], parser: argparse.ArgumentParser, prefix: str = ""
) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser()

    for field in attr.fields(cls):
        arg_name = f"--{prefix}{field.name}".replace("_", "-")

        if attr.has(field.type):
            # If the field is another attrs class, recurse
            create_parser_for_attrs_class(field.type, parser, f"{prefix}{field.name}.")
        else:
            # Otherwise, add it as a regular argument
            parser.add_argument(
                arg_name,
                type=field.type,
                help=field.metadata.get("help", ""),
                default=field.default if field.default is not attr.NOTHING else None,
                required=field.default is attr.NOTHING,
            )

    return parser


def parse_args_to_attrs_instance(cls: Type[Any], args: argparse.Namespace, prefix: str = "") -> Any:
    kwargs: Dict[str, Any] = {}

    for field in attr.fields(cls):
        arg_name = f"{prefix}{field.name}"

        if attr.has(field.type):
            # If the field is another attrs class, recurse
            kwargs[field.name] = parse_args_to_attrs_instance(field.type, args, f"{arg_name}.")
        else:
            # Otherwise, get the value from args
            value = getattr(args, arg_name.replace(".", "_"))
            if value is not None:
                kwargs[field.name] = value

    return cls(**kwargs)
