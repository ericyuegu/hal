import abc
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import Union

import torch
from tensordict import TensorDict
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from hal.data.constants import STICK_XY_CLUSTER_CENTERS_V0
from hal.data.constants import TARGET_FEATURES_TO_ONE_HOT_ENCODE
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
from hal.training.io import log_if_master
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
        assert self.config.report_len % self.config.local_batch_size == 0
        assert self.config.n_samples % self.config.report_len == 0
        self.samples = 0

        model = Arch.get(self.config.arch, config=self.config)
        self.model = maybe_wrap_model_distributed(model)
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            betas=self.config.betas,
            eps=self.config.eps,
            weight_decay=self.config.wd,
        )
        batch_size = get_world_size() * self.config.local_batch_size
        self.scheduler = CosineAnnealingLR(self.opt, T_max=int(config.n_samples / batch_size), eta_min=1e-6)
        self.ckpt = Checkpoint(
            model=self.model, config=self.config, artifact_dir=self.artifact_dir, keep_ckpts=self.config.keep_ckpts
        )

    def __str__(self) -> str:
        return "\n".join(
            (
                "\n",
                f'{" Model ":-^80}',
                str(self.model),
                f'{" Parameters ":-^80}',
                report_module_weights(self.model),
                f'{" Config ":-^80}',
                "\n".join(f"{k:20s}: {v}" for k, v in vars(self.config).items()),
            )
        )

    def _restore_checkpoint(self) -> int:
        resume_idx, _ = self.ckpt.restore()
        if resume_idx > 0:
            log_if_master(f"Resuming training at {resume_idx} ({resume_idx / (1 << 20):.2f}M samples)")
        return resume_idx

    @abc.abstractmethod
    def loss(self, pred: TensorDict, target: TensorDict) -> TensorDict:
        ...

    @abc.abstractmethod
    def train_op(self, batch: TensorDict) -> MetricsDict:
        ...

    @abc.abstractmethod
    def val_op(self, batch: TensorDict) -> MetricsDict:
        ...

    def train_step(self, batch: TensorDict, writer: Writer, step: int) -> None:
        batch = batch.to(self.device, non_blocking=True)
        metrics = self.train_op(batch)
        writer.log(metrics, step=step, commit=False)

    def train_loop(self, train_loader: Iterable[TensorDict], val_loader: Iterable[TensorDict]) -> None:
        log_if_master(self)
        log_if_master(f"Saving to {str(self.artifact_dir)}")

        wandb_config = WandbConfig.create(self, self.config) if is_master() else None
        batch_size = get_world_size() * self.config.local_batch_size
        train_loader = repeater(train_loader)
        val_loader = repeater(val_loader)
        resume_idx = self._restore_checkpoint()

        with Writer.create(wandb_config) as writer:
            for i in range(resume_idx, self.config.n_samples, self.config.report_len):
                self.train()
                range_iter = trange(
                    i,
                    i + self.config.report_len,
                    batch_size,
                    leave=False,
                    unit="samples",
                    unit_scale=batch_size,
                    desc=f"Training stage {i / self.config.report_len}/{self.config.n_samples / self.config.report_len}",
                )
                t0 = time.perf_counter()

                for samples in range_iter:
                    self.train_step(next(train_loader), writer=writer, step=samples)
                    self.samples = samples
                t1 = time.perf_counter()
                writer.log(
                    {"throughput/samples_per_sec_train": self.config.report_len / (t1 - t0)},
                    step=self.samples,
                    commit=False,
                )

                self.validate(val_loader, writer=writer, step=self.samples)
                t2 = time.perf_counter()
                writer.log(
                    {"throughput/samples_per_sec_val": self.config.n_val_samples / (t2 - t1)},
                    step=self.samples,
                    commit=True,
                )
                self.ckpt.save(self.samples)

                log_if_master(
                    f"{self.samples / (1 << 20):.2f}M/{self.config.n_samples / (1 << 20):.2f}M samples, "
                    f"time left {time_format((t2 - t0) * (self.config.n_samples - self.samples) / self.config.report_len)}"
                )

        self.ckpt.save_file(self.model, "model.ckpt")

    def save_batch_to_disk(self, batch: TensorDict, step: int) -> None:
        save_batch_dir = self.artifact_dir / "training_samples" / f"{step}"
        Path.mkdir(save_batch_dir, exist_ok=True, parents=True)
        batch.save(str(save_batch_dir))
        log_if_master(f"Saved example to {save_batch_dir}")

    def validate(
        self,
        val_loader: Iterator[TensorDict],
        writer: Writer,
        step: int,
    ) -> None:
        self.eval()
        range_iter = trange(
            0,
            self.config.n_val_samples,
            self.config.local_batch_size,
            leave=False,
            unit="samples",
            unit_scale=self.config.local_batch_size,
            desc=f"Validating at {step / (1 << 20):.2f}M samples",
        )
        concat_metrics = defaultdict(list)

        for i in range_iter:
            batch = next(val_loader)
            batch = batch.to(self.device, non_blocking=True)
            if i == 0 and self.config.debug:
                self.save_batch_to_disk(batch, step=step)
            metrics_dict = self.val_op(batch)
            for k, v in metrics_dict.items():
                concat_metrics[k].append(v)

        loss_dict = {f"val/{k}": sum(v) / len(v) for k, v in concat_metrics.items() if "loss" in k}
        loss_total = sum(v for k, v in loss_dict.items() if "loss" in k)
        loss_dict["val/loss_total"] = loss_total
        writer.log(loss_dict, step=step, commit=False)


class CategoricalBCTrainer(Trainer):
    """
    Trains behavior cloning with cross-entropy loss for all controller inputs.
    """

    def loss(self, pred: TensorDict, target: TensorDict) -> TensorDict:
        loss_dict: TensorDict = TensorDict({})
        loss_fns = {"buttons": F.cross_entropy, "main_stick": F.cross_entropy, "c_stick": F.cross_entropy}

        # Calculate and log losses for each controller input
        for control, loss_fn in loss_fns.items():
            # Calculate per-frame losses
            frame_losses = loss_fn(pred[control], target[control], reduction="none")

            # Loss for each class
            for t in range(frame_losses.shape[1]):
                if control == "buttons":
                    loss_dict[f"_{control}_loss_{TARGET_FEATURES_TO_ONE_HOT_ENCODE[t]}"] = frame_losses[:, t].mean()
                else:
                    loss_dict[f"_{control}_loss_{STICK_XY_CLUSTER_CENTERS_V0[t]}"] = frame_losses[:, t].mean()
            mean_loss = frame_losses.mean()
            loss_dict[f"loss_{control}"] = mean_loss

        return loss_dict

    def _forward_loop(self, batch: TensorDict) -> TensorDict:
        inputs: TensorDict = batch["inputs"]
        targets: TensorDict = batch["targets"]

        input_len = self.config.data.input_len
        target_len = self.config.data.target_len

        preds = []
        for i in range(target_len):
            pred = self.model(inputs[:, i : i + input_len])
            preds.append(pred)

        preds_td: TensorDict = torch.stack(preds, dim=1)  # type: ignore
        targets_td = targets[:, input_len : input_len + target_len]

        loss_by_head = self.loss(preds_td, targets_td)

        return loss_by_head

    def train_op(self, batch: TensorDict) -> MetricsDict:
        self.opt.zero_grad(set_to_none=True)
        loss_by_head = self._forward_loop(batch)

        loss_total = sum(v for k, v in loss_by_head.items() if k.startswith("loss"))
        loss_total.backward()  # type: ignore
        self.opt.step()
        self.scheduler.step()

        loss_by_head["loss_total"] = loss_total  # type: ignore
        metrics_dict = {f"train/{k}": v.item() for k, v in loss_by_head.items()}
        metrics_dict["lr"] = self.scheduler.get_lr()  # type: ignore
        return metrics_dict

    def val_op(self, batch: TensorDict) -> MetricsDict:
        with torch.no_grad():
            loss_by_head = self._forward_loop(batch)
            loss_total = sum(v.detach() for k, v in loss_by_head.items() if k.startswith("loss"))

        loss_by_head["loss_total"] = loss_total  # type: ignore
        metrics_dict = {f"val/{k}": v.item() for k, v in loss_by_head.items()}
        return metrics_dict
