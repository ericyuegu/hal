import abc
import multiprocessing as mp
import time
from collections import defaultdict
from pathlib import Path
from queue import Empty
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import Union

import torch
from loguru import logger
from tensordict import TensorDict
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from hal.eval.eval import run_closed_loop_evaluation
from hal.eval.eval_helper import EpisodeStats
from hal.training.config import TrainConfig
from hal.training.distributed import get_world_size
from hal.training.distributed import is_master
from hal.training.distributed import log_if_master
from hal.training.distributed import maybe_wrap_model_distributed
from hal.training.distributed import trange
from hal.training.io import Checkpoint
from hal.training.io import WandbConfig
from hal.training.io import Writer
from hal.training.io import get_artifact_dir
from hal.training.io import get_exp_name
from hal.training.io import get_log_dir
from hal.training.models.registry import Arch
from hal.training.optim import create_optimizer
from hal.training.preprocess.preprocessor import Preprocessor
from hal.training.utils import repeater
from hal.training.utils import report_module_weights
from hal.training.utils import time_format

MetricsDict = Dict[str, float]


class Trainer(torch.nn.Module, abc.ABC):
    model: Union[torch.nn.Module, torch.nn.parallel.DistributedDataParallel]

    @property
    def device(self) -> str:
        return str(next(self.model.parameters()).device)

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

        self.preprocessor = Preprocessor(data_config=config.data, embedding_config=config.embedding)
        self.samples = 0
        self.artifact_dir = get_artifact_dir(get_exp_name(self.config))

        logger.info(f"Initializing model {self.config.arch}")
        model = Arch.get(self.config.arch, preprocessor=self.preprocessor)
        self.model = maybe_wrap_model_distributed(model)
        self.opt = create_optimizer(
            self.model,
            weight_decay=self.config.wd,
            learning_rate=self.config.lr,
            betas=self.config.betas,
            device_type=self.device,
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
        if self.samples == 0 and self.config.data.debug_save_batch:
            self.save_batch_to_disk(batch, step=step)
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
                    i + batch_size,
                    i + self.config.report_len + batch_size,  # end at even multiple of report_len
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

                # Save checkpoint & configs before validation & closed loop eval
                self.ckpt.save(self.samples)

                self.validate(val_loader, writer=writer, step=self.samples)
                t2 = time.perf_counter()
                writer.log(
                    {"throughput/samples_per_sec_val": self.config.n_val_samples / (t2 - t1)},
                    step=self.samples,
                    commit=True,
                )

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

        should_closed_loop_eval = step % self.config.eval.closed_loop_eval_every_n == 0
        if should_closed_loop_eval:
            logger.debug("Starting closed loop evaluation")
            eval_stats_queue: mp.Queue = mp.Queue()
            eval_process = mp.Process(
                target=run_closed_loop_evaluation,
                kwargs=dict(
                    artifact_dir=self.artifact_dir,
                    n_workers=self.config.eval.n_workers,
                    checkpoint_idx=step,
                    eval_stats_queue=eval_stats_queue,
                    player="p1",
                ),
            )
            eval_process.start()

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

        for _ in range_iter:
            batch = next(val_loader)
            batch = batch.to(self.device, non_blocking=True)
            metrics_dict = self.val_op(batch)
            for k, v in metrics_dict.items():
                concat_metrics[k].append(v)

        loss_dict = {k: sum(v) / len(v) for k, v in concat_metrics.items() if "loss" in k}
        loss_total = sum(v for k, v in loss_dict.items() if "loss" in k)
        loss_dict["val/loss_total"] = loss_total

        if should_closed_loop_eval:
            try:
                # We wait so wandb can commit the metrics
                logger.debug("Waiting for closed loop evaluation")
                # Standard match time limit is 8 minutes, plus buffer for setup/teardown
                closed_loop_eval_stats: EpisodeStats = eval_stats_queue.get(block=True, timeout=60 * 8 + 30)
                loss_dict.update(closed_loop_eval_stats.to_wandb_dict(prefix="closed_loop_eval", player="p1"))
            except Empty:
                logger.warning("Closed loop evaluation stats not available")
            finally:
                eval_process.join(timeout=1.0)
                if eval_process.is_alive():
                    eval_process.kill()

        writer.log(loss_dict, step=step, commit=False)


class CategoricalBCTrainer(Trainer, abc.ABC):
    """
    Trains behavior cloning with cross-entropy loss for all controller inputs.
    """

    def loss(self, pred: TensorDict, target: TensorDict) -> TensorDict:
        loss_dict: TensorDict = TensorDict({})
        loss_fns = {
            "buttons": F.cross_entropy,
            "main_stick": F.cross_entropy,
            "c_stick": F.cross_entropy,
            "shoulder": F.cross_entropy,
        }

        for target_feature, loss_fn in loss_fns.items():
            if target_feature in pred and target_feature in target:
                frame_losses = loss_fn(pred[target_feature], target[target_feature])
                loss_dict[f"loss_{target_feature}"] = frame_losses

        return loss_dict

    @abc.abstractmethod
    def _forward_loop(self, batch: TensorDict) -> TensorDict:
        ...

    def train_op(self, batch: TensorDict) -> MetricsDict:
        self.opt.zero_grad(set_to_none=True)
        loss_by_head = self._forward_loop(batch)

        loss_total = sum(v for k, v in loss_by_head.items() if k.startswith("loss"))
        loss_total.backward()  # type: ignore
        self.opt.step()

        loss_by_head["loss_total"] = loss_total  # type: ignore
        metrics_dict = {f"train/{k}": v.item() for k, v in loss_by_head.detach().to("cpu").items()}
        metrics_dict["lr/lr"] = self.scheduler(self.samples / self.config.n_samples)
        return metrics_dict

    def val_op(self, batch: TensorDict) -> MetricsDict:
        with torch.no_grad():
            loss_by_head = self._forward_loop(batch)
            loss_total = sum(v.detach() for k, v in loss_by_head.items() if k.startswith("loss"))

        loss_by_head["loss_total"] = loss_total  # type: ignore
        metrics_dict = {f"val/{k}": v.item() for k, v in loss_by_head.items()}
        return metrics_dict
