# %%
import numpy as np
from training.config import DataConfig
from training.config import DataworkerConfig
from training.config import TrainConfig

from hal.training.dataloader import create_dataloaders

np.set_printoptions(threshold=np.inf)

train_config = TrainConfig(
    n_gpus=1,
    debug=True,
    arch="",
    data=DataConfig(
        data_dir="/opt/projects/hal2/data/dev",
        input_preprocessing_fn="inputs_v0",
        target_preprocessing_fn="targets_v0",
        input_len=10,
        target_len=100,
    ),
    dataworker=DataworkerConfig(),
)
train_loader, val_loader = create_dataloaders(train_config, rank=None, world_size=None)
train_iter = iter(train_loader)
# %%
for i, (x, y) in enumerate(train_iter):
    if i > 10:
        break
    print(y["buttons"][0])
# %%
x["gamestate"][0].shape

# %%
y["buttons"].shape
# %%
y["buttons"].squeeze()[0]

# %%
x.keys()
# %%
x["ego_character"].shape
# %%
for k, v in x.items():
    print(k, v.shape)
# %%
