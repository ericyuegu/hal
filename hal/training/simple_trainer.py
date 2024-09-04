import argparse
import random

import numpy as np
import torch
from tensordict import TensorDict
from torch.nn import functional as F

from hal.data.constants import STICK_XY_CLUSTER_CENTERS_V0
from hal.data.constants import TARGET_FEATURES_TO_ONE_HOT_ENCODE
from hal.training.config import TrainConfig
from hal.training.config import create_parser_for_attrs_class
from hal.training.config import parse_args_to_attrs_instance
from hal.training.dataloader import create_dataloaders
from hal.training.dataloader import create_tensordicts
from hal.training.distributed import auto_distribute
from hal.training.distributed import get_device_id
from hal.training.distributed import get_world_size
from hal.training.distributed import wrap_multiprocessing
from hal.training.trainer import MetricsDict
from hal.training.trainer import Trainer


class SimpleTrainer(Trainer):
    """
    Trains behavior cloning by passing the entire sequence at once.
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

    def train_op(self, batch: TensorDict) -> MetricsDict:
        self.opt.zero_grad(set_to_none=True)
        loss_by_head = self.model(batch)

        loss_weights = self.config.loss_weights
        weighted_losses = {
            "loss_buttons": loss_by_head["loss_buttons"] * loss_weights.buttons,
            "loss_main_stick": loss_by_head["loss_main_stick"] * loss_weights.main_stick,
            "loss_c_stick": loss_by_head["loss_c_stick"] * loss_weights.c_stick,
        }
        loss_total = sum(weighted_losses.values())
        loss_total.backward()  # type: ignore
        self.opt.step()
        self.scheduler.step()

        loss_by_head["loss_total"] = loss_total  # type: ignore
        metrics_dict = {f"train/{k}": v.item() for k, v in loss_by_head.items()}
        metrics_dict["lr"] = self.scheduler.get_lr()  # type: ignore
        return metrics_dict

    def val_op(self, batch: TensorDict) -> MetricsDict:
        self.eval()
        with torch.no_grad():
            loss_by_head = self._teacher_forcing_loop(batch)
            loss_total = sum(v.detach() for k, v in loss_by_head.items() if k.startswith("loss"))

        loss_by_head["loss_total"] = loss_total  # type: ignore
        metrics_dict = {f"val/{k}": v.item() for k, v in loss_by_head.items()}
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
    trainer = SimpleTrainer(config=train_config, train_loader=train_loader, val_loader=val_loader)
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
