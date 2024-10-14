# %%
from pathlib import Path

import numpy as np

from hal.training.config import DataConfig
from hal.training.config import DataworkerConfig
from hal.training.config import TrainConfig
from hal.training.dataset import MmappedParquetDataset
from hal.training.tensordict_dataloader import create_tensordict_dataloaders

# %%
np.set_printoptions(threshold=np.inf)

train_config = TrainConfig(
    n_gpus=1,
    debug=True,
    arch="",
    data=DataConfig(
        data_dir="/opt/projects/hal2/data/partial",
        input_preprocessing_fn="inputs_v0",
        target_preprocessing_fn="targets_v0",
        input_len=60,
        target_len=5,
    ),
    dataworker=DataworkerConfig(),
)
# %%
train_loader, val_loader = create_tensordict_dataloaders(train_config, rank=None, world_size=None)
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
dataset = MmappedParquetDataset(
    input_path=Path("/opt/projects/hal2/data/partial/train.parquet"),
    stats_path=Path("/opt/projects/hal2/data/partial/stats.json"),
    data_config=train_config.data,
)
# %%
len(dataset)
# %%
x, y = dataset[0]
# %%
x["gamestate"][0]
