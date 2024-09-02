import argparse
import random
from typing import Iterable
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from tensordict import TensorDict
from torch.nn import functional as F
from training.trainer import Trainer

from hal.training.config import TrainConfig
from hal.training.config import create_parser_for_attrs_class
from hal.training.config import parse_args_to_attrs_instance
from hal.training.dataloader import create_dataloaders
from hal.training.dataloader import create_tensordicts
from hal.training.distributed import auto_distribute
from hal.training.distributed import get_device_id
from hal.training.distributed import get_world_size
from hal.training.distributed import wrap_multiprocessing


class RecurrentTrainer(Trainer):
    """
    Trainer for deep models with recurrent blocks.
    """

    def loss(self, pred: TensorDict, target: TensorDict) -> TensorDict:
        loss_dict: TensorDict = TensorDict({})
        loss_fns = {"buttons": F.cross_entropy, "main_stick": F.cross_entropy, "c_stick": F.cross_entropy}

        # Calculate and log losses for each controller input
        for control, loss_fn in loss_fns.items():
            # Calculate per-frame losses
            frame_losses = loss_fn(pred[control], target[control], reduction="none")

            for t in range(frame_losses.shape[1]):
                loss_dict[f"_{control}_loss_frame_{t}"] = frame_losses[:, t].mean()

            mean_loss = frame_losses.mean()
            loss_dict[f"loss_{control}"] = mean_loss

        return loss_dict

    def _teacher_forcing_loop(self, batch: TensorDict) -> TensorDict:
        inputs: TensorDict = batch["inputs"]
        targets: TensorDict = batch["targets"]

        # Warmup trajectory without calculating loss
        warmup_len = self.config.data.input_len
        target_len = self.config.data.target_len
        warmup_inputs = inputs[:, :warmup_len]

        hidden: Iterable[Optional[Tuple[torch.Tensor, torch.Tensor]]]
        _, hidden = self.model(warmup_inputs)

        # Teacher forcing
        preds = []
        for i in range(target_len):
            pred, hidden = self.model(inputs[:, warmup_len + i : warmup_len + i + 1], hidden)
            preds.append(pred)

        preds_td: TensorDict = torch.stack(preds, dim=1)  # type: ignore
        targets_td = targets[:, warmup_len : warmup_len + target_len]

        loss_by_head = self.loss(preds_td, targets_td)

        return loss_by_head

    def train_op(self, batch: TensorDict) -> TensorDict:
        self.opt.zero_grad(set_to_none=True)
        loss_by_head = self._teacher_forcing_loop(batch)
        loss_total = torch.tensor(sum(v for k, v in loss_by_head.items() if k.startswith("loss")))
        loss_total.backward()
        self.opt.step()
        self.scheduler.step()

        loss_by_head["loss_total"] = loss_total
        metrics_dict = TensorDict({f"train/{k}": v.item() for k, v in loss_by_head.items()}, device="cpu")  # type: ignore
        metrics_dict["lr"] = self.scheduler.get_lr()
        return metrics_dict

    def val_op(self, batch: TensorDict) -> TensorDict:
        self.eval()
        with torch.no_grad():
            loss_by_head = self._teacher_forcing_loop(batch)
            loss_total = torch.tensor(sum(v for k, v in loss_by_head.items() if k.startswith("loss")))

        loss_by_head["loss_total"] = loss_total
        metrics_dict = TensorDict({f"val/{k}": v.item() for k, v in loss_by_head.items()}, device="cpu")  # type: ignore
        return metrics_dict


@auto_distribute
def main(
    train_config: TrainConfig,
    train_td: TensorDict,
    val_td: TensorDict,
) -> None:
    rank = get_device_id()
    seed = train_config.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_loader, val_loader = create_dataloaders(
        train_td, val_td, train_config, rank=rank, world_size=get_world_size()
    )
    trainer = RecurrentTrainer(config=train_config, train_loader=train_loader, val_loader=val_loader)
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
    config = parse_cli()
    train_data, val_data = create_tensordicts(config.data)
    # pass positional args and call wrapped fn; (kwargs not accepted)
    wrapped_train = wrap_multiprocessing(main, config, train_data, val_data)
    wrapped_train()
