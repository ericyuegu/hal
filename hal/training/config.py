import argparse
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Type

import attr


@attr.s(auto_attribs=True, frozen=True)
class ReplayFilter:
    """Filter for replay."""

    replay_uuid: Optional[str] = None
    stage: Optional[str] = None
    character: Optional[str] = None


@attr.s(auto_attribs=True, frozen=True)
class DataConfig:
    """Training & eval dataset & preprocessing."""

    data_dir: str
    input_preprocessing_fn: str = "inputs_v0"
    target_preprocessing_fn: str = "targets_v0"
    # Number of input and target frames in example/rollout
    input_len: int = 60
    target_len: int = 5
    replay_filter: ReplayFilter = ReplayFilter()
    include_both_players: bool = True
    truncate_rollouts_to_replay_end: bool = False


@attr.s(auto_attribs=True, frozen=True)
class DataworkerConfig:
    data_workers_per_gpu: int = 4
    prefetch_factor: int = 2
    collate_fn: Optional[str] = None


@attr.s(auto_attribs=True, frozen=True)
class BaseConfig:
    n_gpus: int
    # TODO(eric): store true by default
    debug: bool


@attr.s(auto_attribs=True, frozen=True)
class TrainConfig(BaseConfig):
    # Model
    arch: str

    # Data
    data: DataConfig
    dataworker: DataworkerConfig
    seed: int = 42

    # Hyperparams
    loss_fn: str = "ce"
    local_batch_size: int = 1024
    lr: float = 3e-4
    n_samples: int = 2**27
    n_val_samples: int = 2**17
    keep_ckpts: int = 8
    report_len: int = 2**20
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    wd: float = 1e-2


def create_parser_for_attrs_class(
    cls: Type[Any], parser: argparse.ArgumentParser, prefix: str = ""
) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser()

    for field in attr.fields(cls):
        arg_name = f"--{prefix}{field.name}"

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
                # action="store_true" if field.type == bool else "store",
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
            value = getattr(args, arg_name)
            if value is not None:
                kwargs[field.name] = value

    return cls(**kwargs)


# @attr.s(auto_attribs=True, frozen=True)
# class ClosedLoopEvalConfig:
#     data_config: DatasetConfig
#     model_arch: torch.Module
#     model_path: Path
#     opponent: EVAL_MODE = "cpu"
#     opponent_model_arch: Optional[torch.Module] = None
#     opponent_model_path: Optional[Path] = None
#     # Which device to load model(s) for inference
#     device: DEVICES = "cpu"
#     # Comma-separated lists of stages, or "all"
#     stage: EVAL_STAGES = "all"
