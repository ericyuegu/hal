import argparse
import random
from typing import Dict
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


class RecurrentTrainer(Trainer):
    def train_op(self, inputs: Dict[str, Tensor], targets: Dict[str, Tensor]) -> Dict[str, Number]:
        self.opt.zero_grad(set_to_none=True)
        pred = self.model(inputs)
        # unroll over target len
        loss_by_head = self.loss_fn(pred, targets)
        loss_total = sum(loss_by_head.values())
        loss_total.backward()
        self.opt.step()
        self.scheduler.step()

        loss_by_head["train/loss_total"] = loss_total
        metrics_dict = {f"train/{k}": v.item() for k, v in loss_by_head.items()}
        metrics_dict["lr"] = self.scheduler.get_lr()
        return metrics_dict

    def loss_fn(self, pred: Dict[str, Tensor], target: Dict[str, Tensor]) -> Dict[str, Tensor]:
        button_loss = F.cross_entropy(pred["buttons"], target["buttons"])
        main_stick_loss = F.cross_entropy(pred["main_stick"], target["main_stick"])
        c_stick_loss = F.cross_entropy(pred["c_stick"], target["c_stick"])
        return {"button_loss": button_loss, "main_stick_loss": main_stick_loss, "c_stick_loss": c_stick_loss}


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
