import abc
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import Union

import torch
from loguru import logger
from tensordict import TensorDict
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from hal.training.config import TrainConfig
from hal.training.distributed import get_world_size
from hal.training.distributed import is_master
from hal.training.distributed import maybe_wrap_model_distributed
from hal.training.distributed import trange
from hal.training.io import Checkpoint
from hal.training.io import WandbConfig
from hal.training.io import Writer
from hal.training.io import get_artifact_dir
from hal.training.io import get_exp_name
from hal.training.io import get_log_dir
from hal.training.utils import repeater
from hal.training.utils import report_module_weights
from hal.training.utils import time_format
from hal.training.zoo.models.registry import Arch

MetricsDict = Dict[str, float]


class Trainer(torch.nn.Module, abc.ABC):
    model: Union[torch.nn.Module, torch.nn.parallel.DistributedDataParallel]

    @property
    def device(self) -> str:
        return str(next(self.model.parameters()).device)

    @property
    def artifact_dir(self) -> Path:
        params = get_exp_name(self.config)
        return get_artifact_dir(params)

    @property
    def log_dir(self) -> Path:
        params = get_exp_name(self.config)
        return get_log_dir(params)

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

    @abc.abstractmethod
    def loss(self, pred: TensorDict, target: TensorDict) -> TensorDict:
        ...

    @abc.abstractmethod
    def train_op(self, batch: TensorDict) -> TensorDict:
        ...

    @abc.abstractmethod
    def val_op(self, batch: TensorDict) -> TensorDict:
        ...

    def train_step(self, batch: TensorDict, writer: Writer, step: int) -> None:
        batch = batch.to(self.device)
        metrics = self.train_op(batch)
        writer.log(metrics, step=step, commit=False)

    def train_loop(
        self,
        train_loader: Iterable[TensorDict],
        val_loader: Iterable[TensorDict],
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
        val_loader = repeater(val_loader)

        ckpt = Checkpoint(model=self, logdir=self.artifact_dir, keep_ckpts=keep_ckpts)
        resume_idx = ckpt.restore()[0]
        if resume_idx:
            logger.info(f"Resuming training at {resume_idx} ({resume_idx / (1 << 20):.2f}M samples)")

        with Writer.create(wandb_config) as writer:
            for i in range(resume_idx, n_samples, report_len):
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

    def save_batch_to_disk(self, batch: TensorDict, step: int) -> None:
        save_batch_dir = self.artifact_dir / "training_samples" / f"{step}"
        Path.mkdir(save_batch_dir, exist_ok=True, parents=True)
        batch.save(str(save_batch_dir))
        logger.info(f"Saved example to {save_batch_dir}")

    def validate(
        self,
        val_loader: Iterator[TensorDict],
        batch_size: int,
        n_val_samples: int,
        writer: Writer,
        step: int,
    ) -> None:
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
            batch = next(val_loader)
            batch = batch.to(device, non_blocking_pin=True)
            if i == 0 and self.config.debug:
                self.save_batch_to_disk(batch, step=step)
            metrics_dict = self.val_op(batch)
            metrics_dict = metrics_dict.to("cpu")
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
