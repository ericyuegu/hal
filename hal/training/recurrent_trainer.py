import argparse
import random
from typing import Iterable
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from tensordict import TensorDict

from hal.training.config import TrainConfig
from hal.training.config import create_parser_for_attrs_class
from hal.training.config import parse_args_to_attrs_instance
from hal.training.dataloader import create_dataloaders
from hal.training.dataloader import create_tensordicts
from hal.training.distributed import auto_distribute
from hal.training.distributed import get_device_id
from hal.training.distributed import get_world_size
from hal.training.distributed import wrap_multiprocessing
from hal.training.trainer import CategoricalBCTrainer


class RecurrentTrainer(CategoricalBCTrainer):
    """
    Trainer for deep models with recurrent blocks.
    """

    def _forward_loop(self, batch: TensorDict) -> TensorDict:
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
            # Select the i-th input after warmup for all samples in the batch
            current_input = inputs[:, warmup_len + i].unsqueeze(1)
            pred, hidden = self.model(current_input, hidden)
            preds.append(pred)

        preds_td: TensorDict = torch.stack(preds, dim=1)  # type: ignore
        targets_td = targets[:, warmup_len : warmup_len + target_len]

        loss_by_head = self.loss(preds_td, targets_td)

        return loss_by_head


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
    trainer.train_loop(train_loader, val_loader)


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
