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

import wandb
from hal.training.config import TrainConfig
from hal.training.distributed import is_master
from hal.training.utils import get_git_repo_root


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


@attr.s(auto_attribs=True, frozen=True)
class Checkpoint:
    model: torch.nn.Module
    logdir: Path
    keep_ckpts: int
    FILE_MATCH: str = "*.pth"
    FILE_FORMAT: str = "%012d.pth"

    @staticmethod
    def checkpoint_idx(filename: str) -> int:
        return int(os.path.basename(filename).split(".")[0])

    def restore(self, idx: Optional[int] = None, device: str = "cpu") -> Tuple[int, Optional[Path]]:
        if idx is None:
            all_ckpts = self.logdir.glob(self.FILE_MATCH)
            try:
                idx = self.checkpoint_idx(max(str(x) for x in all_ckpts))
            except ValueError:
                return 0, None
        ckpt = self.logdir / (self.FILE_FORMAT % idx)
        logger.info(f"Resuming checkpoint from: {ckpt}")
        with ckpt.open("rb") as f:
            self.model.load_state_dict(torch.load(f, map_location=device))
        return idx, ckpt

    def save(self, idx: int) -> None:
        if not is_master():  # only save master's state
            return
        self.logdir.mkdir(exist_ok=True, parents=True)
        ckpt = self.logdir / (self.FILE_FORMAT % idx)
        with ckpt.open("wb") as f:
            torch.save(self.model.state_dict(), f)
        old_ckpts = sorted(self.logdir.glob(self.FILE_MATCH), key=str)
        for ckpt in old_ckpts[: -self.keep_ckpts]:
            ckpt.unlink()

    def save_file(self, model: torch.nn.Module, filename: str) -> None:
        if not is_master():  # only save master's state
            return
        self.logdir.mkdir(exist_ok=True, parents=True)
        with (self.logdir / filename).open("wb") as f:
            torch.save(model.state_dict(), f)


@attr.s(auto_attribs=True, frozen=True)
class WandbConfig:
    project: str
    config: Dict[str, Any]
    tags: List[str]
    name: str
    model: torch.nn.Module

    @classmethod
    def create(cls, model: torch.nn.Module, exp_config: TrainConfig) -> Optional["WandbConfig"]:
        if not os.getenv("WANDB_API_KEY"):
            logger.info("W&B run not initiated because WANDB_API_KEY not set.")
            return None
        if exp_config.debug:
            logger.info("Debug mode, skipping W&B.")
            return None

        model_name = model.model.__class__.__name__
        config = {"model_name": model_name, **vars(exp_config)}
        tags = [model_name]

        name_path: Path = model.log_dir
        name = name_path.stem

        return cls(project="hal", config=config, tags=tags, name=name, model=model)


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
                config=wandb_config.config,
                tags=wandb_config.tags,
                name=wandb_config.name,
            )
            wandb.watch(wandb_config.model, log="all")

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
