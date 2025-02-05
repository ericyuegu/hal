# %%
from pathlib import Path

import torch
from tensordict import TensorDict

from hal.constants import ACTION_BY_IDX
from hal.training.config import DataConfig
from hal.training.io import load_model_from_artifact_dir
from hal.training.streaming_dataset import HALStreamingDataset

torch.set_printoptions(threshold=torch.inf)
# %%
ACTION_BY_IDX

# %%
artifact_dir = Path("/opt/projects/hal2/runs/2025-02-04_13-53-10/arch@GPTv1-4-4_local_batch_size@32_n_samples@262144/")
model, config = load_model_from_artifact_dir(artifact_dir)

# %%
x = TensorDict.load(artifact_dir / "training_samples/32/inputs")
y = TensorDict.load(artifact_dir / "training_samples/32/targets")

# %%
y_hat = model(x)

# %%
y_hat["buttons"][0]
# %%
y_hat["buttons"][0].argmax(dim=-1)
# %%
y["buttons"][0].argmax(dim=-1)

# %%
# Load closed loop replay and run it through the model
replay_dir = Path("/opt/projects/hal2/data/multishine_eval/test")

data_config = DataConfig(
    data_dir="/opt/projects/hal2/data/multishine_eval",
    seq_len=28800,
)
test_dataset = HALStreamingDataset(
    local=str(replay_dir),
    remote=None,
    batch_size=1,
    shuffle=False,
    data_config=data_config,
    embedding_config=config.embedding,
)

# %%
test_dataset[0]

# %%
x = test_dataset[0]["inputs"][:256].unsqueeze(0)
y_hat = model(x)

# %%
x["ego_action"][0]
# %%
predicted_buttons = y_hat["buttons"][0].argmax(dim=-1)
predicted_buttons
# %%
y_hat["buttons"][0][:64]
# %%
actual_buttons = test_dataset[0]["targets"]["buttons"][:256].argmax(dim=-1)
actual_buttons
# %%
predicted_main_stick = y_hat["main_stick"][0].argmax(dim=-1)
predicted_main_stick
# %%
actual_main_stick = test_dataset[0]["targets"]["main_stick"][:256].argmax(dim=-1)
actual_main_stick
# %%
replay_dir = Path("/opt/projects/hal2/data/multishine_eval_argmax/test")

data_config = DataConfig(
    data_dir="/opt/projects/hal2/data/multishine_eval_argmax",
    seq_len=28800,
)
test_dataset = HALStreamingDataset(
    local=str(replay_dir),
    remote=None,
    batch_size=1,
    shuffle=False,
    data_config=data_config,
    embedding_config=config.embedding,
)
# %%
x = test_dataset[0]["inputs"][:256].unsqueeze(0)
y_hat = model(x)
# %%
predicted_buttons = y_hat["buttons"][0].argmax(dim=-1)
predicted_buttons
# %%
actual_buttons = test_dataset[0]["targets"]["buttons"][:256].argmax(dim=-1)
actual_buttons
# %%
predicted_main_stick = y_hat["main_stick"][0].argmax(dim=-1)
predicted_main_stick
# %%
actual_main_stick = test_dataset[0]["targets"]["main_stick"][:256].argmax(dim=-1)
actual_main_stick
# %%
model
