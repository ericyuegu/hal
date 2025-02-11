# %%
from tensordict import TensorDict
import torch
import pandas as pd
from pathlib import Path

from streaming import StreamingDataset
from hal.preprocess.target_configs import preprocess_targets_v1
from hal.training.config import DataConfig
from hal.training.config import DataConfig
from hal.training.config import TrainConfig
from hal.training.streaming_dataloader import get_dataloaders
from hal.preprocess.transformations import preprocess_controller_inputs_fine_shoulder
from hal.training.streaming_dataset import HALStreamingDataset

torch.set_printoptions(precision=4, sci_mode=False, linewidth=120, threshold=torch.inf)

# %%
mds_path = "/opt/projects/hal2/data/ranked/train"
ds = StreamingDataset(local=mds_path, batch_size=1, shuffle=True)

# %%
x = ds[346]
td = TensorDict(x, batch_size=(len(x["frame"]),))
# %%
y = preprocess_controller_inputs_fine_shoulder(td, "p1")
# %%
y["shoulder"].sum(dim=0)

# %%
ds = HALStreamingDataset(
    local="/opt/projects/hal2/data/mang0/val",
    remote=None,
    batch_size=4,
    shuffle=False,
    data_config=DataConfig(data_dir="/opt/projects/hal2/data/mang0"),
    embedding_config=DataConfig(input_preprocessing_fn="inputs_v0"),
)

# %%
x = super(HALStreamingDataset, ds).__getitem__(0)
x

# %%
x = ds[3]["inputs"]
x

# %%
ds.preprocessor.input_shapes_by_head

# %%
pd.DataFrame(x.numpy()).describe()
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