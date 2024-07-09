import argparse
import pickle
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch
from loguru import logger
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.types import Number
from torch.utils.data import DataLoader

from hal.training.config import TrainConfig
from hal.training.config import create_parser_for_attrs_class
from hal.training.config import parse_args_to_attrs_instance
from hal.training.dataloader import create_dataloaders
from hal.training.distributed import auto_distribute
from hal.training.distributed import get_world_size
from hal.training.distributed import is_master
from hal.training.distributed import maybe_wrap_model_distributed
from hal.training.distributed import print
from hal.training.distributed import trange
from hal.training.distributed import wrap_multiprocessing
from hal.training.io import Checkpoint
from hal.training.io import WandbConfig
from hal.training.io import Writer
from hal.training.io import get_artifact_dir
from hal.training.io import get_exp_name
from hal.training.zoo.embed.registry import ModelInputs
from hal.training.zoo.embed.registry import ModelOutputs
from hal.training.zoo.models.registry import Arch
from hal.utils import move_tensors_to_device
from hal.utils import repeater
from hal.utils import report_module_weights
from hal.utils import time_format


class Trainer(torch.nn.Module):
    model: Union[torch.nn.Module, torch.nn.parallel.DistributedDataParallel]

    @property
    def device(self) -> str:
        return str(next(self.model.parameters()).device)

    @property
    def artifact_dir(self) -> Path:
        params = get_exp_name(self.config)
        return get_artifact_dir(params)

    def __init__(self, config: TrainConfig, train_loader: DataLoader, val_loader: DataLoader) -> None:
        super().__init__()
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.samples = 0

        model = Arch.get(self.config.arch)  # TODO input shapes
        self.model = maybe_wrap_model_distributed(model)  # Needed for .backward and to wrap into a module for saving
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

    def loss_fn(self, pred: ModelOutputs, target: ModelOutputs) -> dict[str, Tensor]:
        raise NotImplementedError()

    def train_op(self, inputs: ModelInputs, targets: ModelOutputs) -> dict[str, Tensor]:
        raise NotImplementedError()

    def train_step(self, batch: Tuple[ModelInputs, ModelOutputs], writer: Writer, step: int) -> None:
        batch = move_tensors_to_device(batch, self.device)
        inputs, targets = batch
        metrics = self.train_op(inputs, targets)
        writer.log(metrics, step=step, commit=False)

    def train_loop(
        self,
        train_loader: Iterable[Tuple[ModelInputs, ModelOutputs]],
        val_loader: Iterable[Tuple[ModelInputs, ModelOutputs]],
        local_batch_size: int,
        n_samples: int,
        n_val_samples: int,
        report_len: int,
        keep_ckpts: int,
    ) -> None:
        logger.info(self)
        logger.info(f"{self.artifact_dir=}")
        assert report_len % local_batch_size == 0
        assert n_samples % report_len == 0

        wandb_config = WandbConfig.create(self, self.config) if is_master() else None
        batch_size = get_world_size() * local_batch_size
        train_loader = repeater(train_loader)

        ckpt = Checkpoint(model=self, logdir=self.artifact_dir, keep_ckpts=keep_ckpts)
        resume_idx = ckpt.restore()[0]
        if resume_idx:
            logger.info(f"Resuming training at {resume_idx} ({resume_idx / (1 << 20):.2f}M samples)")

        with Writer.create(wandb_config) as writer:
            for i in range(resume_idx, n_samples, report_len):
                self.train()
                range_iter = trange(
                    start=i,
                    stop=i + report_len,
                    step=batch_size,
                    leave=False,
                    unit="samples",
                    unit_scale=batch_size,
                    desc=f"Training stage {i / report_len}/{n_samples / report_len}",
                )
                t0 = time.perf_counter()
                for samples in range_iter:
                    self.train_step(next(train_loader), writer=writer, step=samples)
                    self.samples = samples
                t1 = time.perf_counter()
                writer.log(
                    {"throughput/samples_per_sec_train": report_len / (t1 - t0)}, step=self.samples, commit=False
                )

                self.validate(
                    val_loader,
                    batch_size=local_batch_size,
                    n_val_samples=n_val_samples,
                    writer=writer,
                    step=self.samples,
                )
                t2 = time.perf_counter()
                writer.log(
                    {"throughput/samples_per_sec_val": n_val_samples / (t2 - t1)}, step=self.samples, commit=True
                )
                ckpt.save(self.samples)

                logger.info(
                    f"{self.samples / (1 << 20):.2f}M/{n_samples / (1 << 20):.2f}M samples, "
                    f"time left {time_format((t2 - t0) * (n_samples - self.samples) / report_len)}"
                )

        ckpt.save_file(self.model, "model.ckpt")

    def val_step(self, inputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        self.eval()
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
            if i == 0 and self.config.debug:
                self.save_batch_to_disk(batch, step=step)
            metrics_dict = self.val_step(*batch)
            metrics_dict = move_tensors_to_device(metrics_dict, "cpu", non_blocking=False)
            for k, v in metrics_dict.items():
                concat_metrics[k].append(v)

        loss_dict = {f"val/{k}": sum(v) / len(v) for k, v in concat_metrics.items() if "loss" in k}
        loss_total = sum(v for k, v in loss_dict.items() if "loss" in k)
        loss_dict["val/loss_total"] = loss_total
        writer.log(loss_dict, step=step, commit=False)

        # TODO confusion matrix debugging
        # conf_matrix_dict = {}
        # for k, list_tuple_pred_target in concat_metrics.items():
        #     if "confusion_matrix" in k:
        #         pred_action_ids, target_action_ids = zip(*list_tuple_pred_target)
        #         pred_action_ids = torch.cat(pred_action_ids, dim=-1).tolist()
        #         target_action_ids = torch.cat(target_action_ids, dim=-1).tolist()
        #         if "button" in k:
        #             class_names = list(MAP_IDX_TO_BUTTON.values())
        #         else:
        #             class_names = get_discretized_analog_axis_values(
        #                 self.config.num_analog_discretized_values
        #             ).tolist()
        #         conf_matrix_dict[f"val/{k}"] = writer.plot_confusion_matrix(
        #             preds=pred_action_ids, y_true=target_action_ids, class_names=class_names, title=k
        #         )
        # writer.log(conf_matrix_dict, step=step, commit=False)


class RecurrentTrainer(Trainer):
    def train_op(self, inputs: ModelInputs, targets: ModelOutputs) -> dict[str, Number]:
        self.opt.zero_grad(set_to_none=True)
        pred = self.model(inputs)
        loss_by_head = self.loss_fn(pred, targets)
        loss_total: Tensor = sum(loss_by_head.values())
        loss_total.backward()
        self.opt.step()
        self.scheduler.step()

        loss_by_head["train/loss_total"] = loss_total
        metrics_dict = {f"train/{k}": v.item() for k, v in loss_by_head.items()}
        metrics_dict["lr"] = self.scheduler.get_lr()
        return metrics_dict


@auto_distribute
def main(
    rank: Optional[int],
    world_size: Optional[int],
    train_config: TrainConfig,
) -> None:
    seed = train_config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_loader, val_loader = create_dataloaders(train_config, rank=rank, world_size=world_size)
    trainer = Trainer(config=train_config, train_loader=train_loader, val_loader=val_loader)
    trainer.train_loop(
        train_loader,
        val_loader,
        local_batch_size=train_config.local_batch_size,
        n_samples=train_config.n_samples,
        n_val_samples=train_config.n_val_samples,
        report_len=train_config.report_len,
        keep_ckpts=train_config.keep_ckpts,
    )


def parse_cli() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser = create_parser_for_attrs_class(TrainConfig, parser)
    args = parser.parse_args()
    return parse_args_to_attrs_instance(TrainConfig, args)


if __name__ == "__main__":
    train_config = parse_cli()
    # pass positional args and call wrapped fn; (kwargs not accepted)
    wrap_multiprocessing(main, train_config)()
