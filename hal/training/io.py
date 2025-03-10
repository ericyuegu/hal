import json
import os
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import attr
import torch
import torch.nn
from loguru import logger
from streaming import StreamingDataLoader
from tensordict import TensorDict
from yasoo import deserialize
from yasoo import serialize

import wandb
from hal.preprocess.preprocessor import Preprocessor
from hal.training.config import BaseConfig
from hal.training.config import TrainConfig
from hal.training.config import ValueTrainerConfig
from hal.training.distributed import is_master
from hal.training.models.registry import Arch
from hal.training.streaming_dataloader import load_dataloader_state
from hal.training.streaming_dataloader import save_dataloader_state
from hal.training.utils import get_git_repo_root

ARTIFACT_DIR_ROOT = "runs"


def get_path_friendly_datetime() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_exp_name(config) -> str:
    return "_".join(
        f"{k}@{v}"
        for k, v in sorted(attr.asdict(config).items())
        if k
        in (
            "arch",
            "dataset",
            "local_batch_size",
            "n_samples",
            "input_preprocessing_fn",
            "target_preprocessing_fn",
            "input_len",
            "target_len",
        )
    )


def get_artifact_dir(*args) -> Path:
    artifact_dir = get_git_repo_root().joinpath(ARTIFACT_DIR_ROOT, get_path_friendly_datetime(), *args)
    Path.mkdir(artifact_dir, parents=True, exist_ok=True)
    return artifact_dir


def get_log_dir(*args) -> Path:
    log_dir = get_git_repo_root().joinpath("logs", get_path_friendly_datetime(), *args)
    Path.mkdir(log_dir, parents=True, exist_ok=True)
    return log_dir


def get_default_dolphin_path() -> str:
    dolphin_path = os.environ.get("DEFAULT_DOLPHIN_PATH", None)
    assert dolphin_path is not None, "DEFAULT_DOLPHIN_PATH environment variable must be set"
    return dolphin_path


def get_default_melee_iso_path() -> str:
    melee_path = os.environ.get("DEFAULT_MELEE_ISO_PATH", None)
    assert melee_path is not None, "DEFAULT_MELEE_ISO_PATH environment variable must be set"
    return melee_path


MODEL_FILE_MATCH: str = "*.pt"
MODEL_FILE_FORMAT: str = "%012d.pt"
TRAIN_LOADER_STATE_FILENAME: str = "train_loader_state_%012d.pth"
VAL_LOADER_STATE_FILENAME: str = "val_loader_state_%012d.pth"
CONFIG_FILENAME: str = "config.json"


def load_config_from_artifact_dir(artifact_dir: Path) -> TrainConfig:
    with open(artifact_dir / "config.json", "r", encoding="utf-8") as f:
        config: Union[TrainConfig, ValueTrainerConfig] = deserialize(json.load(f))  # type: ignore
    return config


def override_stats_path(config: TrainConfig, stats_path: Path) -> TrainConfig:
    return attr.evolve(
        config,
        data=attr.evolve(config.data, stats_path=str(stats_path.absolute())),
    )


def load_model_from_artifact_dir(
    artifact_dir: Path,
    idx: Optional[int] = None,
    device: str | torch.device = "cpu",
    stats_path_override: Optional[Path] = None,
) -> Tuple[torch.nn.Module, TrainConfig]:
    config = load_config_from_artifact_dir(artifact_dir)
    if stats_path_override is not None:
        config = override_stats_path(config, stats_path_override)
    preprocessor = Preprocessor(data_config=config.data)
    model = Arch.get(config.arch, preprocessor=preprocessor)
    ckpt = Checkpoint(model, config, artifact_dir, keep_ckpts=config.keep_ckpts)
    ckpt.restore(idx=idx, device=device)
    return ckpt.model, config


def find_latest_idx(artifact_dir: Path) -> int:
    all_ckpts = artifact_dir.glob(MODEL_FILE_MATCH)
    try:
        filename = max(str(x) for x in all_ckpts)
        idx = int(Path(filename).stem.split(".")[0])
        return idx
    except ValueError:
        return 0


@attr.s(auto_attribs=True, frozen=True)
class Checkpoint:
    model: torch.nn.Module
    config: BaseConfig
    artifact_dir: Path
    keep_ckpts: int

    def restore(
        self,
        idx: Optional[int] = None,
        device: str | torch.device = "cpu",
        train_loader: Optional[StreamingDataLoader] = None,
        val_loader: Optional[StreamingDataLoader] = None,
    ) -> Tuple[int, Optional[Path]]:
        if idx is None:
            idx = find_latest_idx(self.artifact_dir)
            if idx == 0:
                return 0, None
        ckpt = self.artifact_dir / (MODEL_FILE_FORMAT % idx)
        logger.info(f"Resuming checkpoint from: {ckpt}")
        with ckpt.open("rb") as f:
            state_dict = torch.load(f, map_location=device)
            # Remove 'module.' prefix if it exists
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)

        train_state_file = self.artifact_dir / (TRAIN_LOADER_STATE_FILENAME % idx)
        if train_loader is not None and train_state_file.exists():
            load_dataloader_state(train_loader, train_state_file)
            logger.info(f"Loaded train loader state from {train_state_file}")

        val_state_file = self.artifact_dir / (VAL_LOADER_STATE_FILENAME % idx)
        if val_loader is not None and val_state_file.exists():
            load_dataloader_state(val_loader, val_state_file)
            logger.info(f"Loaded val loader state from {val_state_file}")

        return idx, ckpt

    def save(
        self,
        idx: int,
        train_loader: StreamingDataLoader,
        val_loader: StreamingDataLoader,
    ) -> None:
        if not is_master():  # only save master's state
            return
        self.artifact_dir.mkdir(exist_ok=True, parents=True)
        config_path = self.artifact_dir / CONFIG_FILENAME
        with config_path.open("w") as f:
            json.dump(serialize(self.config), f)
        ckpt = self.artifact_dir / (MODEL_FILE_FORMAT % idx)
        with ckpt.open("wb") as f:
            torch.save(self.model.state_dict(), f)

        save_dataloader_state(train_loader, self.artifact_dir / (TRAIN_LOADER_STATE_FILENAME % idx))
        save_dataloader_state(val_loader, self.artifact_dir / (VAL_LOADER_STATE_FILENAME % idx))

        old_ckpts = sorted(self.artifact_dir.glob(MODEL_FILE_MATCH), key=str)
        for ckpt_file in old_ckpts[: -self.keep_ckpts]:
            logger.debug(f"Deleting old checkpoint: {ckpt_file}")
            ckpt_file.unlink()

    def save_file(self, model: torch.nn.Module, filename: str) -> None:
        if not is_master():  # only save master's state
            return
        self.artifact_dir.mkdir(exist_ok=True, parents=True)
        with (self.artifact_dir / filename).open("wb") as f:
            torch.save(model.state_dict(), f)


@attr.s(auto_attribs=True, frozen=True)
class WandbConfig:
    project: str
    train_config: Dict[str, Any]
    tags: List[str]
    name: str
    model: torch.nn.Module

    @classmethod
    def create(cls, model: torch.nn.Module, train_config: BaseConfig) -> Optional["WandbConfig"]:
        if not os.getenv("WANDB_API_KEY"):
            logger.info("W&B run not initiated because WANDB_API_KEY not set.")
            return None
        if train_config.debug:
            logger.info("Debug mode, skipping W&B.")
            return None

        model_name = model.model.__class__.__name__
        config = {"model_name": model_name, **vars(train_config)}
        tags = [model_name]

        name_path: Path = model.log_dir
        name = name_path.stem

        return cls(project="hal", train_config=config, tags=tags, name=name, model=model)


class DummyWriter:
    def __init__(self, wandb_config: Optional[WandbConfig]) -> None:
        pass

    def watch(self, model: torch.nn.Module, **kwargs) -> None:
        """Hooks into torch model to collect gradients and the topology."""

    def log(self, summary_dict: TensorDict | Dict[str, Any], step: int, commit: bool = True) -> None:
        """Add on event to the event file."""

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.close()


class Writer:
    def __init__(self, wandb_config: WandbConfig) -> None:
        self.wandb_config = wandb_config
        if is_master():
            wandb.init(
                project=wandb_config.project,
                config=wandb_config.train_config,
                tags=wandb_config.tags,
                name=wandb_config.name,
            )
            train_config = wandb_config.train_config
            log_freq = train_config["report_len"] // (train_config["local_batch_size"] * train_config["n_gpus"])
            wandb.watch(wandb_config.model, log="all", log_freq=log_freq)

    def log(self, summary_dict: TensorDict | Dict[str, Any], step: int, commit: bool = True) -> None:
        """Add on event to the event file."""
        wandb.log(summary_dict, step=step, commit=commit)

    def close(self) -> None:
        wandb.finish()

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.close()

    @classmethod
    def create(cls, wandb_config: Optional["WandbConfig"] = None) -> "Writer":
        if is_master() and wandb_config is not None:
            return cls(wandb_config)
        else:
            return DummyWriter(wandb_config)  # type: ignore
