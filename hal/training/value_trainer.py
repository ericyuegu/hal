import argparse
import random

import attr
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


@attr.s(auto_attribs=True, frozen=True)
class ValueTrainerConfig(TrainConfig):
    value_fn_loss_weight: float = 0.5
    advantage_weighted_loss: bool = False


class ValueTrainer(Trainer):
    """
    Trains behavior cloning using cross-entropy loss on next-/multi-token prediction and value function loss.
    """

    def __init__(
        self, config: ValueTrainerConfig, train_loader: StreamingDataLoader, val_loader: StreamingDataLoader
    ) -> None:
        super().__init__(config, train_loader, val_loader)
        self.config = config
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

        # Multi-token prediction
        for target_feature, loss_fn in loss_fns.items():
            feature_losses = []

            for frame in self.multi_token_heads:
                feature_name = f"{target_feature}_{frame}"

                if feature_name in pred and feature_name in target:
                    frame_loss = loss_fn(pred[feature_name], target[feature_name])
                    loss_dict[f"loss_{feature_name}"] = frame_loss
                    feature_losses.append(frame_loss)

            if feature_losses:
                loss_dict[f"loss_{target_feature}"] = torch.mean(torch.stack(feature_losses)).detach()

        # Value function loss
        loss_dict["loss_value"] = F.mse_loss(pred["value"].squeeze(-1), target["value"])

        return loss_dict

    def sum_losses(self, loss_by_head: TensorDict) -> torch.Tensor:
        loss_total = sum(
            self.config.value_fn_loss_weight * v if k == "loss_value" else v
            for k, v in loss_by_head.items()
            if k.startswith("loss")
        )
        return loss_total


@auto_distribute
def main(train_config: ValueTrainerConfig) -> None:
    rank = get_device_id()
    seed = train_config.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_loader, val_loader = get_dataloaders(train_config)
    trainer = ValueTrainer(config=train_config, train_loader=train_loader, val_loader=val_loader)
    trainer.train_loop(train_loader, val_loader)


def parse_cli() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser = create_parser_for_attrs_class(ValueTrainerConfig, parser)
    args = parser.parse_args()
    return parse_args_to_attrs_instance(ValueTrainerConfig, args)


if __name__ == "__main__":
    config = parse_cli()
    # pass positional args and call wrapped fn; (kwargs not accepted)
    wrapped_train = wrap_multiprocessing(main, config)
    wrapped_train()
