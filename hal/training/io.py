import json
import os
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type

import attr
import torch
import torch.nn
from loguru import logger
from tensordict import TensorDict
from yasoo import deserialize
from yasoo import serialize

import wandb
from hal.training.config import BaseConfig
from hal.training.config import TrainConfig
from hal.training.distributed import is_master
from hal.training.utils import get_git_repo_root
from hal.training.zoo.models.registry import Arch


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
    artifact_dir = get_git_repo_root().joinpath("runs", get_path_friendly_datetime(), *args)
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


def log_if_master(message: Any) -> None:
    if is_master():
        logger.info(message)


FILE_MATCH: str = "*.pth"
FILE_FORMAT: str = "%012d.pth"
CONFIG_FILENAME: str = "config.json"


def load_model_from_artifact_dir(artifact_dir: Path, idx: Optional[int] = None) -> Tuple[torch.nn.Module, TrainConfig]:
    with open(artifact_dir / "config.json", "r", encoding="utf-8") as f:
        config: TrainConfig = deserialize(json.load(f))  # type: ignore
    model = Arch.get(config.arch, config=config)
    idx = Checkpoint.find_latest_idx(Path(artifact_dir)) if idx is None else idx
    ckpt_path = artifact_dir / (FILE_FORMAT % idx)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt)
    return model, config


@attr.s(auto_attribs=True, frozen=True)
class Checkpoint:
    model: torch.nn.Module
    config: BaseConfig
    artifact_dir: Path
    keep_ckpts: int

    @staticmethod
    def find_latest_idx(artifact_dir: Path) -> int:
        all_ckpts = artifact_dir.glob(FILE_MATCH)
        try:
            filename = max(str(x) for x in all_ckpts)
            idx = int(Path(filename).stem.split(".")[0])
            return idx
        except ValueError:
            return 0

    def restore(self, idx: Optional[int] = None, device: str = "cpu") -> Tuple[int, Optional[Path]]:
        if idx is None:
            idx = self.find_latest_idx(self.artifact_dir)
            if idx == 0:
                return 0, None
        ckpt = self.artifact_dir / (FILE_FORMAT % idx)
        logger.info(f"Resuming checkpoint from: {ckpt}")
        with ckpt.open("rb") as f:
            self.model.load_state_dict(torch.load(f, map_location=device))
        return idx, ckpt

    def save(self, idx: int) -> None:
        if not is_master():  # only save master's state
            return
        self.artifact_dir.mkdir(exist_ok=True, parents=True)
        config_path = self.artifact_dir / CONFIG_FILENAME
        with config_path.open("w") as f:
            json.dump(serialize(self.config), f)
        ckpt = self.artifact_dir / (FILE_FORMAT % idx)
        with ckpt.open("wb") as f:
            torch.save(self.model.state_dict(), f)
        old_ckpts = sorted(self.artifact_dir.glob(FILE_MATCH), key=str)
        for ckpt in old_ckpts[: -self.keep_ckpts]:
            ckpt.unlink()

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

    def plot_confusion_matrix(
        self,
        probs: Optional[Sequence[Sequence]] = None,
        y_true: Optional[Sequence] = None,
        preds: Optional[Sequence] = None,
        class_names: Optional[Sequence[str]] = None,
        title: Optional[str] = None,
    ) -> wandb.viz.CustomChart:
        pass

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

    def plot_confusion_matrix(
        self,
        probs: Optional[Sequence[Sequence]] = None,
        y_true: Optional[Sequence] = None,
        preds: Optional[Sequence] = None,
        class_names: Optional[Sequence[str]] = None,
        title: Optional[str] = None,
    ) -> wandb.viz.CustomChart:
        return wandb.plot.confusion_matrix(
            probs=probs, y_true=y_true, preds=preds, class_names=class_names, title=title
        )

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
