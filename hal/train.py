# %%
import argparse
import pickle
import random
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from time import time
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Tuple
from typing import Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from hal.data.constants import ACTION_BY_IDX
from hal.data.dataset import MmappedParquetDataset
from hal.training.distributed import auto_distribute
from hal.training.distributed import get_device_id
from hal.training.distributed import get_world_size
from hal.training.distributed import is_master
from hal.training.distributed import print
from hal.training.distributed import trange
from hal.training.distributed import wrap_multiprocessing
from hal.training.io import Checkpoint
from hal.training.io import WandbConfig
from hal.training.io import Writer
from hal.training.io import get_artifact_dir
from hal.training.io import get_log_dir
from hal.utils import get_exp_name
from hal.utils import move_tensors_to_device
from hal.utils import repeater
from hal.utils import report_module_weights
from hal.utils import time_format

dataset_path = "/opt/projects/hal2/data/dev/train.parquet"
stats_path = "/opt/projects/hal2/data/dev/stats.json"
dataset = MmappedParquetDataset(input_path=dataset_path, stats_path=stats_path, input_len=60, target_len=10)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

# %%
dataloader_iter = iter(dataloader)
inputs, targets = next(dataloader_iter)

inputs

# %%
gamestate = inputs["gamestate"]
gamestate[2, :30]

# %%
action = inputs["action_other"]
action[2, :30]
# %%
for idx in action[1, :30]:
    print(ACTION_BY_IDX[idx])
# %%


import numpy as np
import pyarrow.parquet as pq
import torch

LOSS_KEY = "loss/loss_total"


@dataclass
class Config:
    arch: str = "lstm-v0-256-2"
    dataset: str = "data/ranked"
    preprocessing_fn: str = "independent_axes"
    input_len: int = 120
    target_len: int = 1
    num_analog_discretized_values: int = 17
    loss_fn: str = "ce"
    local_batch_size: int = 1024
    num_data_workers: int = 4  # per gpu
    lr: float = 3e-4
    n_samples: int = 2**27
    n_val_samples: int = 2**17
    keep_ckpts: int = 8
    report_len: int = int(n_samples / keep_ckpts)
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    wd: float = 1e-2
    debug: bool = False


class Trainer(torch.nn.Module):
    train_op: Callable[..., Dict[str, Tensor]]
    model: torch.nn.Module
    TRAIN_LR = (1 / 16, 1), (1 - 4 / 16, 1), (1 - 2 / 16, 0.1), (1 - 1 / 16, 0.01)

    @property
    def device(self) -> str:
        for x in self.model.parameters():
            return x.device

    @property
    def log_dir(self) -> Path:
        params = get_exp_name(self.config)
        return get_log_dir(f"{self.__class__.__name__}({self.config.arch})", self.config.dataset, params)

    @property
    def artifact_dir(self) -> Path:
        params = get_exp_name(self.config)
        return get_artifact_dir(f"{self.__class__.__name__}({self.config.arch})", self.config.dataset, params)

    def __init__(self, arch: str, config: Config) -> None:
        super().__init__()
        self.config = config
        model = Arch.get(
            arch, input_size=get_num_input_floats(), num_analog_values=config.num_analog_discretized_values
        )
        self.model = wrap_model(model)  # Needed for .backward and to wrap into a module for saving
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            betas=self.config.betas,
            eps=self.config.eps,
            weight_decay=self.config.wd,
        )
        batch_size = get_world_size() * self.config.local_batch_size
        self.scheduler = CosineAnnealingLR(self.opt, T_max=int(config.n_samples / batch_size), eta_min=1e-6)

    def __str__(self) -> str:
        return "\n".join(
            (
                f'{" Model ":-^80}',
                str(self.model),
                f'{" Parameters ":-^80}',
                report_module_weights(self.model),
                f'{" Config ":-^80}',
                "\n".join(f"{k:20s}: {v}" for k, v in vars(self.config).items()),
            )
        )

    @staticmethod
    def cross_entropy_loss(
        pred: dict[str, Tensor], target: dict[str, Tensor], reduction: str = "mean"
    ) -> dict[str, Tensor]:
        losses = {f"loss/{k}": F.cross_entropy(pred[k], target[k], reduction=reduction) for k in target.keys()}
        return losses

    @staticmethod
    def focal_loss(pred: dict[str, Tensor], target: dict[str, Tensor], gamma: float = 2.0) -> dict[str, Tensor]:
        ce_losses = Trainer.cross_entropy_loss(pred, target, reduction="none")
        focal_losses = {}
        for k, loss in ce_losses.items():
            pt = torch.exp(-loss)
            focal_losses[f"loss/{k}"] = ((1 - pt) ** gamma * loss).mean()
        return focal_losses

    @staticmethod
    def calculate_confusion_matrix(pred: dict[str, Tensor], targets: dict[str, Tensor]):
        cm_dict = {}
        for k, pred_action in pred.items():
            pred_action_id = pred_action.argmax(dim=-1)
            target_action_id = targets[k].argmax(dim=-1)
            cm_dict[f"confusion_matrix/{k}"] = (pred_action_id, target_action_id)
        return cm_dict

    def loss_fn(self, pred: dict[str, Tensor], target: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.loss_fn == "ce":
            return self.cross_entropy_loss(pred, target)
        elif self.config.loss_fn == "focal":
            return self.focal_loss(pred, target)

    def train_step(self, batch: tuple[dict[str, Tensor], dict[str, Tensor]], writer: Writer, step: int) -> None:
        batch = move_tensors_to_device(batch, self.device)
        inputs, targets = batch
        self.opt.zero_grad(set_to_none=True)
        pred, _ = self.model(inputs)
        loss = self.loss_fn(pred, targets)
        loss_total = sum(loss.values())
        loss_total.backward()
        self.opt.step()
        self.scheduler.step()

        loss[LOSS_KEY] = loss_total
        metrics_dict = {f"train/{k}": v.item() for k, v in loss.items()}
        metrics_dict["lr/lr"] = self.scheduler.get_last_lr()
        writer.log(metrics_dict, step=step, commit=False)

    def val_step(self, inputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        with torch.no_grad():
            pred, _ = self.model(inputs)
            loss = self.loss_fn(pred, targets)
            conf_matrix = self.calculate_confusion_matrix(pred, targets)
        metrics_dict = {k: v.item() for k, v in loss.items()} | conf_matrix
        return metrics_dict

    def save_batch_to_disk(self, batch: tuple[dict[str, Tensor], ...], step: int) -> None:
        save_batch_path = self.artifact_dir / "training_samples" / f"{step}.pkl"
        Path.mkdir(save_batch_path.parent, exist_ok=True, parents=True)
        with open(save_batch_path, "wb") as f:
            pickle.dump(batch, f)
        print(f"Saved example to {save_batch_path}")

    def validate(self, val_loader: Iterable, batch_size: int, n_val_samples: int, writer: Writer, step: int) -> None:
        val_iter = iter(val_loader)
        device = self.device
        n_val_samples = min(len(val_loader), n_val_samples)
        range_iter = trange(
            0,
            n_val_samples,
            batch_size,
            leave=False,
            unit="samples",
            unit_scale=batch_size,
            desc=f"Validating at {step / (1 << 20):.2f}M samples",
        )
        concat_metrics = defaultdict(list)

        for i in range_iter:
            batch = next(val_iter)
            batch = move_tensors_to_device(batch, device)
            if i == 0:
                self.save_batch_to_disk(batch, step=step)
            metrics_dict = self.val_step(*batch)
            metrics_dict = move_tensors_to_device(metrics_dict, "cpu", non_blocking=False)
            for k, v in metrics_dict.items():
                concat_metrics[k].append(v)

        loss_dict = {f"val/{k}": sum(v) / len(v) for k, v in concat_metrics.items() if "loss" in k}
        loss_total = sum(v for k, v in loss_dict.items() if "loss" in k) / len(loss_dict)
        loss_dict[f"val/{LOSS_KEY}"] = loss_total
        writer.log(loss_dict, step=step, commit=False)

        conf_matrix_dict = {}
        for k, list_tuple_pred_target in concat_metrics.items():
            if "confusion_matrix" in k:
                pred_action_ids, target_action_ids = zip(*list_tuple_pred_target)
                pred_action_ids = torch.cat(pred_action_ids, dim=-1).tolist()
                target_action_ids = torch.cat(target_action_ids, dim=-1).tolist()
                if "button" in k:
                    class_names = list(MAP_IDX_TO_BUTTON.values())
                else:
                    class_names = get_discretized_analog_axis_values(
                        self.config.num_analog_discretized_values
                    ).tolist()
                conf_matrix_dict[f"val/{k}"] = writer.plot_confusion_matrix(
                    preds=pred_action_ids, y_true=target_action_ids, class_names=class_names, title=k
                )
        writer.log(conf_matrix_dict, step=step, commit=False)

    def train_loop(
        self,
        train_loader: Iterable,
        val_loader: Iterable,
        local_batch_size: int,
        n_samples: int,
        n_val_samples: int,
        report_len: int,
        keep_ckpts: int,
    ) -> None:
        print(self)
        print(f"artifact_dir: {self.artifact_dir}")
        wandb_config = WandbConfig.create(self, self.config) if is_master() else None
        assert report_len % local_batch_size == 0
        assert n_samples % report_len == 0
        batch_size = get_world_size() * local_batch_size
        train_loader = repeater(train_loader)
        ckpt = Checkpoint(self, self.artifact_dir, keep_ckpts)
        start = ckpt.restore()[0]
        if start:
            print(f"Resuming training at {start} ({start / (1 << 20):.2f}M samples)")

        with Writer.create(wandb_config) as writer:
            for i in range(start, n_samples, report_len):
                self.train()
                range_iter = trange(
                    i,
                    i + report_len,
                    batch_size,
                    leave=False,
                    unit="samples",
                    unit_scale=batch_size,
                    desc=f"Training stage {i / report_len}/{n_samples / report_len}",
                )
                t0 = time()
                for samples in range_iter:
                    self.train_step(next(train_loader), writer=writer, step=samples)

                t1 = time()
                writer.log({"throughput/samples_per_sec_train": report_len / (t1 - t0)}, step=samples, commit=False)
                self.validate(
                    val_loader, batch_size=local_batch_size, n_val_samples=n_val_samples, writer=writer, step=samples
                )
                t2 = time()
                writer.log({"throughput/samples_per_sec_val": n_val_samples / (t2 - t1)}, step=samples, commit=True)
                ckpt.save(samples)

                print(
                    f"{samples / (1 << 20):.2f}M/{n_samples / (1 << 20):.2f}M samples, "
                    f"time left {time_format((t2 - t0) * (n_samples - samples) / report_len)}"
                )

        ckpt.save_file(self.model, "model.ckpt")


def get_dataset(in_memory_datasets: list[Tensor], input_len: int, target_len: int) -> tuple[InMemoryDataset, ...]:
    return tuple(
        InMemoryDataset(tensor=tensor, input_len=input_len, target_len=target_len) for tensor in in_memory_datasets
    )


def make_data_loaders(
    train: InMemoryDataset,
    val: InMemoryDataset,
    preprocessing_fn: Callable,
    batch_size: int,
    num_data_workers: int = 1,
    **kwargs,
) -> tuple[DataLoader, DataLoader]:
    train_loader = MaybeDistributedDataLoader(
        train,
        batch_size=batch_size,
        collate_fn=preprocessing_fn,
        shuffle=True,
        num_workers=num_data_workers,
        world_size=get_world_size(),
        rank=get_device_id(),
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if num_data_workers > 1 else None,
        **kwargs,
    )
    val_loader = MaybeDistributedDataLoader(
        val,
        batch_size=batch_size,
        collate_fn=preprocessing_fn,
        shuffle=False,
        num_workers=num_data_workers,
        world_size=get_world_size(),
        rank=get_device_id(),
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if num_data_workers > 1 else None,
        **kwargs,
    )
    return train_loader, val_loader


@auto_distribute
def main(config: Config, in_memory_datasets: list[Tensor], seed: int = 894756923) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    train_ds, val_ds = get_dataset(in_memory_datasets, input_len=config.input_len, target_len=config.target_len)
    preprocessing_fn = partial(
        get_preprocessing_fn(config.preprocessing_fn),
        input_len=config.input_len,
        num_analog_discretized_values=config.num_analog_discretized_values,
        dataset_path=config.dataset,
    )
    train_loader, val_loader = make_data_loaders(
        train_ds, val_ds, preprocessing_fn, config.local_batch_size, config.num_data_workers
    )
    trainer = Trainer(config.arch, config=config)
    trainer.train_loop(
        train_loader,
        val_loader,
        local_batch_size=config.local_batch_size,
        n_samples=config.n_samples,
        n_val_samples=config.n_val_samples,
        report_len=config.report_len,
        keep_ckpts=config.keep_ckpts,
    )


def parse_cli() -> Config:
    parser = argparse.ArgumentParser()
    for field, typ in Config.__dataclass_fields__.items():
        parser.add_argument(f"--{field}", type=typ.type, default=typ.default, required=False)
    config = Config(**vars(parser.parse_args()))
    return config


def load_datasets_to_memory(dataset_filepath: Union[str, Path]) -> list[Tensor]:
    in_memory_datasets = []
    for split in ["train", "val"]:
        path = Path(dataset_filepath) / f"{split}.parquet"
        print(f"Loading dataset {path} to memory")
        tensor = torch.tensor(pq.read_table(path).to_pandas().to_numpy())
        print(f"Moving to shared memory")
        tensor.share_memory_()
        in_memory_datasets.append(tensor)
    return in_memory_datasets


if __name__ == "__main__":
    exp_config = parse_cli()
    datasets = load_datasets_to_memory(exp_config.dataset)
    # download_dataset(exp_config.dataset)
    # automatically distribute on CUDA if available
    # pass positional args and call wrapped fn; (kwargs not accepted)
    wrap_multiprocessing(main, exp_config, datasets)()
