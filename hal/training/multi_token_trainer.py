import argparse
import random

import numpy as np
import torch
from streaming import StreamingDataLoader
from tensordict import TensorDict
from torch.nn import functional as F

from hal.training.config import TrainConfig
from hal.training.config import create_parser_for_attrs_class
from hal.training.config import parse_args_to_attrs_instance
from hal.training.distributed import auto_distribute
from hal.training.distributed import get_device_id
from hal.training.distributed import wrap_multiprocessing
from hal.training.streaming_dataloader import get_dataloaders
from hal.training.trainer import Trainer


class MultiTokenTrainer(Trainer):
    """
    Trains behavior cloning using cross-entropy loss on multiple-token prediction.
    """

    def __init__(
        self, config: TrainConfig, train_loader: StreamingDataLoader, val_loader: StreamingDataLoader
    ) -> None:
        super().__init__(config, train_loader, val_loader)
        assert self.preprocessor.target_config.multi_token_heads is not None
        self.multi_token_heads = self.preprocessor.target_config.multi_token_heads

    def loss(self, pred: TensorDict, target: TensorDict) -> TensorDict:
        loss_dict: TensorDict = TensorDict({})

        loss_fns = {
            "shoulder": F.cross_entropy,
            "c_stick": F.cross_entropy,
            "main_stick": F.cross_entropy,
            "buttons": F.cross_entropy,
        }

        for target_feature, loss_fn in loss_fns.items():
            for frame in self.multi_token_heads:
                feature_name = f"{target_feature}_{frame}"

                if feature_name in pred and feature_name in target:
                    frame_losses = loss_fn(pred[feature_name], target[feature_name])
                    loss_dict[f"loss_{feature_name}"] = frame_losses

        return loss_dict

    def forward_loop(self, batch: TensorDict) -> TensorDict:
        inputs: TensorDict = batch["inputs"]
        targets: TensorDict = batch["targets"]

        # No warm-up inputs, just next-token prediction for the entire sequence
        pred = self.model(inputs)
        B, L, *_ = pred.shape
        # Important! Reshape the batch to 2D for proper CE loss calculation
        loss_by_head = self.loss(pred.reshape(B * L, -1).squeeze(), targets.reshape(B * L, -1).squeeze())

        return loss_by_head


@auto_distribute
def main(train_config: TrainConfig) -> None:
    rank = get_device_id()
    seed = train_config.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_loader, val_loader = get_dataloaders(train_config)
    trainer = MultiTokenTrainer(config=train_config, train_loader=train_loader, val_loader=val_loader)
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
