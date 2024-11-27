# %%
from pathlib import Path

from hal.training.config import DataConfig
from hal.training.config import EmbeddingConfig
from hal.training.config import TrainConfig
from hal.training.streaming_dataloader import get_dataloaders
from hal.training.streaming_dataset import HALStreamingDataset

# %%
ds = HALStreamingDataset(
    local="/opt/projects/hal2/data/mang0/train",
    remote=None,
    batch_size=4,
    shuffle=False,
    data_config=DataConfig(),
    embedding_config=EmbeddingConfig(),
    stats_path=Path("/opt/projects/hal2/data/mang0/stats.json"),
)

# %%
x = super(HALStreamingDataset, ds).__getitem__(0)
x
# %%
for value in x.values():
    print(f"{type(value)} {value.shape}")

# %%
ds[0]

# %%
import torch

torch.stack([ds[0]["inputs"], ds, dim=0)
# %%
import os

os.environ["TORCH_LOGS"] = "not_implemented"
# %%
config = TrainConfig(
    n_gpus=1,
    debug=True,
    arch="",
    data=DataConfig(data_dir="/opt/projects/hal2/data/mang0"),
    embedding=EmbeddingConfig(),
    local_batch_size=4,
)
train_loader, val_loader = get_dataloaders(config)

# %%
i = 0
for batch in train_loader:
    print(batch)
    i += 1
    if i > 10:
        break
    

# %%
config