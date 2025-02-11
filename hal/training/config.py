import argparse
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Type

import attr

from hal.constants import IDX_BY_ACTION
from hal.constants import IDX_BY_CHARACTER
from hal.constants import IDX_BY_STAGE
from hal.constants import INCLUDED_CHARACTERS
from hal.constants import INCLUDED_STAGES


@attr.s(auto_attribs=True, frozen=True)
class ReplayFilter:
    """Filter for replay."""

    replay_uuid: Optional[str] = None
    stage: Optional[str] = attr.ib(
        default=None, validator=attr.validators.optional(attr.validators.in_(INCLUDED_STAGES))
    )
    ego_character: Optional[str] = attr.ib(
        default=None, validator=attr.validators.optional(attr.validators.in_(INCLUDED_CHARACTERS))
    )
    opponent_character: Optional[str] = attr.ib(
        default=None, validator=attr.validators.optional(attr.validators.in_(INCLUDED_CHARACTERS))
    )


@attr.s(auto_attribs=True, frozen=True)
class DataConfig:
    """Training & eval dataset & preprocessing."""

    data_dir: str = "/opt/projects/hal2/data/dev"

    # Number of input and target frames in example
    seq_len: int = 256
    replay_filter: ReplayFilter = ReplayFilter()

    # Debugging
    debug_repeat_batch: bool = False
    debug_save_batch: bool = False

    @property
    def stats_path(self) -> Path:
        return Path(self.data_dir) / "stats.json"


@attr.s(auto_attribs=True, frozen=True)
class DataworkerConfig:
    data_workers_per_gpu: int = 8
    prefetch_factor: int = 2
    collate_fn: Optional[str] = None


@attr.s(auto_attribs=True, frozen=True)
class EmbeddingConfig:
    input_preprocessing_fn: str = "inputs_v0"
    target_preprocessing_fn: str = "targets_v0"
    pred_postprocessing_fn: str = "preds_v0"

    stage_embedding_dim: int = 4
    character_embedding_dim: int = 12
    action_embedding_dim: int = 32

    num_stages: int = len(IDX_BY_STAGE)
    num_characters: int = len(IDX_BY_CHARACTER)
    num_actions: int = len(IDX_BY_ACTION)

    num_buttons: Optional[int] = None
    num_main_stick_clusters: Optional[int] = None
    num_c_stick_clusters: Optional[int] = None
    num_shoulder_clusters: Optional[int] = None

    def __attrs_post_init__(self) -> None:
        from hal.preprocess.target_embedding_sizes import TARGETS_EMBEDDING_SIZES

        target_sizes = TARGETS_EMBEDDING_SIZES[self.target_preprocessing_fn]
        object.__setattr__(self, "num_buttons", target_sizes.get("buttons", None))
        object.__setattr__(self, "num_main_stick_clusters", target_sizes.get("main_stick", None))
        object.__setattr__(self, "num_c_stick_clusters", target_sizes.get("c_stick", None))
        object.__setattr__(self, "num_shoulder_clusters", target_sizes.get("shoulder", None))


@attr.s(auto_attribs=True, frozen=True)
class EvalConfig:
    n_workers: int = 16
    closed_loop_eval_every_n: int = 2**21


@attr.s(auto_attribs=True, frozen=True)
class BaseConfig:
    n_gpus: int
    debug: bool


@attr.s(auto_attribs=True, frozen=True)
class TrainConfig(BaseConfig):
    # Model
    arch: str

    # Data
    data: DataConfig = DataConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    dataworker: DataworkerConfig = DataworkerConfig()
    seed: int = 42

    # Eval
    eval: EvalConfig = EvalConfig()

    # Hyperparams
    loss_fn: str = "ce"
    local_batch_size: int = 256
    lr: float = 3e-4
    n_samples: int = 2**24
    n_val_samples: int = 2**14
    keep_ckpts: int = 2**4
    report_len: int = 2**18
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    wd: float = 1e-2


def create_parser_for_attrs_class(
    cls: Type[Any], parser: argparse.ArgumentParser, prefix: str = ""
) -> argparse.ArgumentParser:
    for field in attr.fields(cls):
        arg_name = f"--{prefix}{field.name}"

        if attr.has(field.type):
            # If the field is another attrs class, recurse
            create_parser_for_attrs_class(field.type, parser, f"{prefix}{field.name}.")
        else:
            # Otherwise, add it as a regular argument
            if field.type == bool:
                parser.add_argument(
                    arg_name,
                    action="store_true",
                    help=field.metadata.get("help", ""),
                    default=field.default if field.default is not attr.NOTHING else False,
                    required=field.default is attr.NOTHING and arg_name != "--debug",
                )
            else:
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
