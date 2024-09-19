# %%
from pathlib import Path

import torch
from tensordict import TensorDict

from hal.training.io import load_model_from_artifact_dir

artifact_dir = Path(
    "/opt/projects/hal2/runs/2024-09-19_06-18-33/arch@MLPDebug-512-8_local_batch_size@32_n_samples@1048576"
)

model, train_config = load_model_from_artifact_dir(artifact_dir)
model.eval()

input_len = train_config.data.input_len
target_len = train_config.data.target_len

# %%
batch = TensorDict.load(artifact_dir / "training_samples" / "0")

# %%
batch["inputs"].shape

# %%
batch["targets"].shape

# %%
# x = batch["inputs"][0, :input_len].unsqueeze(0)
# y = batch["targets"][0, input_len : input_len + 1].unsqueeze(0)
x = batch["inputs"][:, :input_len]
y = batch["targets"][:, input_len : input_len + 1]

# %%
y["buttons"].squeeze(-2).shape

# %%
y_hat_buttons = model(x)["buttons"]
y_hat_buttons.shape

# %%
torch.nn.functional.cross_entropy(y_hat_buttons, y["buttons"].squeeze(-2))

# %%
# overfit on single example
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

for i in range(1000):
    optimizer.zero_grad()
    y_hat_buttons = model(x)["buttons"]
    loss = criterion(y_hat_buttons, y["buttons"].squeeze(-2))
    print(loss.item())
    loss.backward()
    optimizer.step()

# %%
# overfit on single example multiple heads
model, train_config = load_model_from_artifact_dir(artifact_dir)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()
heads = ["buttons", "main_stick", "c_stick"]

for i in range(1000):
    optimizer.zero_grad()
    y_hat = model(x)
    total_loss = 0
    for head in heads:
        loss = criterion(y_hat[head], y[head].squeeze(-2))
        total_loss += loss
    print(total_loss.item())
    total_loss.backward()
    optimizer.step()

# %%
# overfit on single example multiple heads multiple frames
model, train_config = load_model_from_artifact_dir(artifact_dir)
model.train()
input_len = train_config.data.input_len
target_len = train_config.data.target_len
batch = TensorDict.load(artifact_dir / "training_samples" / "0")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()
heads = ["buttons", "main_stick", "c_stick"]
frames = 5

for i in range(1000):
    optimizer.zero_grad()
    total_loss = 0

    for t in range(frames):
        x_t = batch["inputs"][:, t : t + input_len]
        y_t = batch["targets"][:, t + input_len : t + input_len + 1]
        for head in heads:
            y_hat = model(x_t)
            loss = criterion(y_hat[head], y_t[head].squeeze(-2))
            total_loss += loss
    print(total_loss.item())
    total_loss.backward()
    optimizer.step()
