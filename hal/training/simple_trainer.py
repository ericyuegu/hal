import argparse
import random

import numpy as np
import torch
from tensordict import TensorDict
from training.streaming_dataloader import get_dataloaders

from hal.training.config import TrainConfig
from hal.training.config import create_parser_for_attrs_class
from hal.training.config import parse_args_to_attrs_instance
from hal.training.distributed import auto_distribute
from hal.training.distributed import get_device_id
from hal.training.distributed import wrap_multiprocessing
from hal.training.trainer import CategoricalBCTrainer


class SimpleTrainer(CategoricalBCTrainer):
    """
    Trains behavior cloning by passing in full sequences of input.
    """

    def _forward_loop(self, batch: TensorDict) -> TensorDict:
        inputs: TensorDict = batch["inputs"]
        targets: TensorDict = batch["targets"]

        # No warm-up inputs, just next-token prediction for the entire sequence
        pred = self.model(inputs)
        B, T, *_ = pred.shape
        loss_by_head = self.loss(pred.reshape(B * T, -1).squeeze(), targets.reshape(B * T, -1).squeeze())

        return loss_by_head


@auto_distribute
def main(train_config: TrainConfig) -> None:
    rank = get_device_id()
    seed = train_config.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_loader, val_loader = get_dataloaders(train_config)
    trainer = SimpleTrainer(config=train_config, train_loader=train_loader, val_loader=val_loader)
    trainer.train_loop(train_loader, val_loader)


def parse_cli() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser = create_parser_for_attrs_class(TrainConfig, parser)
    args = parser.parse_args()
    return parse_args_to_attrs_instance(TrainConfig, args)


if __name__ == "__main__":
    config = parse_cli()
    # pass positional args and call wrapped fn; (kwargs not accepted)
    wrapped_train = wrap_multiprocessing(main, config)
    wrapped_train()
