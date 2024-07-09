# %%
from training.config import DataConfig
from training.config import DataworkerConfig
from training.config import TrainConfig

from hal.training.dataloader import create_dataloaders

train_config = TrainConfig(
    n_gpus=1,
    debug=True,
    arch="",
    data=DataConfig(
        data_dir="/opt/projects/hal2/data/dev",
        input_preprocessing_fn="inputs_v0",
        target_preprocessing_fn="targets_v0",
    ),
    dataworker=DataworkerConfig(),
)
train_loader, val_loader = create_dataloaders(train_config, rank=None, world_size=None)

# %%
x = next(iter(train_loader))
x
# %%
