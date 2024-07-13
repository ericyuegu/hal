import argparse
import random
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.types import Number
from training.trainer import Trainer

from hal.training.config import TrainConfig
from hal.training.config import create_parser_for_attrs_class
from hal.training.config import parse_args_to_attrs_instance
from hal.training.dataloader import create_dataloaders
from hal.training.distributed import auto_distribute
from hal.training.distributed import wrap_multiprocessing


def slice_tensor_dict(tensor_dict: Dict[str, Tensor], start: int, end: int) -> Dict[str, Tensor]:
    return {k: v[:, start:end] for k, v in tensor_dict.items()}


def stack_tensor_dict(tensor_dicts: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    if not tensor_dicts:
        return {}
    return {k: torch.stack([d[k] for d in tensor_dicts], dim=-2) for k in tensor_dicts[0].keys()}


class RecurrentTrainer(Trainer):
    def loss_fn(self, pred: Dict[str, Tensor], target: Dict[str, Tensor]) -> Dict[str, Tensor]:
        loss_dict: Dict[str, Tensor] = {}

        # Calculate per-frame losses
        button_loss = F.cross_entropy(pred["buttons"], target["buttons"], reduction="none")
        main_stick_loss = F.cross_entropy(pred["main_stick"], target["main_stick"], reduction="none")
        c_stick_loss = F.cross_entropy(pred["c_stick"], target["c_stick"], reduction="none")

        # Log per-frame losses
        for t in range(button_loss.shape[1]):
            loss_dict[f"_button_loss_frame_{t}"] = button_loss[:, t].mean()
            loss_dict[f"_main_stick_loss_frame_{t}"] = main_stick_loss[:, t].mean()
            loss_dict[f"_c_stick_loss_frame_{t}"] = c_stick_loss[:, t].mean()

        # Calculate mean losses across all frames
        mean_button_loss = button_loss.mean()
        mean_main_stick_loss = main_stick_loss.mean()
        mean_c_stick_loss = c_stick_loss.mean()

        loss_dict["loss_button"] = mean_button_loss
        loss_dict["loss_main_stick"] = mean_main_stick_loss
        loss_dict["loss_c_stick"] = mean_c_stick_loss

        return loss_dict

    def train_op(self, inputs: Dict[str, Tensor], targets: Dict[str, Tensor]) -> Dict[str, Number]:
        self.opt.zero_grad(set_to_none=True)

        # Warmup trajectory without calculating loss
        warmup_len = self.config.data.input_len
        target_len = self.config.data.target_len
        warmup_inputs = slice_tensor_dict(inputs, 0, warmup_len)
        warmup_pred_dict: Dict[str, Tensor] = self.model(warmup_inputs)

        # Teacher forcing
        hidden, cell = warmup_pred_dict["hidden"], warmup_pred_dict["cell"]
        preds = []
        for i in range(self.config.data.target_len):
            pred_dict = self.model(slice_tensor_dict(inputs, warmup_len + i, warmup_len + i + 1), hidden, cell)
            hidden, cell = pred_dict["hidden"], pred_dict["cell"]
            preds.append(pred_dict)

        preds = stack_tensor_dict(preds)
        targets = slice_tensor_dict(targets, warmup_len, warmup_len + target_len)

        loss_by_head = self.loss_fn(preds, targets)
        loss_total = sum(v for k, v in loss_by_head.items() if k.startswith("loss"))
        loss_total.backward()
        self.opt.step()
        self.scheduler.step()

        loss_by_head["loss_total"] = loss_total
        metrics_dict = {f"train/{k}": v.item() for k, v in loss_by_head.items()}
        metrics_dict["lr"] = self.scheduler.get_lr()
        return metrics_dict

    def val_op(self, inputs: Dict[str, Tensor], targets: Dict[str, Tensor]) -> Dict[str, float]:
        self.eval()
        with torch.no_grad():
            # Warmup trajectory
            warmup_len = self.config.data.input_len
            target_len = self.config.data.target_len
            warmup_inputs = slice_tensor_dict(inputs, 0, warmup_len)
            warmup_pred_dict = self.model(warmup_inputs)

            # Teacher forcing
            hidden, cell = warmup_pred_dict["hidden"], warmup_pred_dict["cell"]
            preds = []
            for i in range(target_len):
                pred_dict = self.model(slice_tensor_dict(inputs, warmup_len + i, warmup_len + i + 1), hidden, cell)
                hidden, cell = pred_dict["hidden"], pred_dict["cell"]
                preds.append(pred_dict)

            preds = stack_tensor_dict(preds)
            targets = slice_tensor_dict(targets, warmup_len, warmup_len + target_len)

            loss_by_head = self.loss_fn(preds, targets)
            loss_total = sum(v for k, v in loss_by_head.items() if k.startswith("loss"))

        loss_by_head["loss_total"] = loss_total
        metrics_dict = {k: v.item() for k, v in loss_by_head.items()}
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
    train_config = parse_cli()
    # pass positional args and call wrapped fn; (kwargs not accepted)
    wrap_multiprocessing(main, train_config)()
