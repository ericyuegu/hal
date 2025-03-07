# %%
import time

import torch

# %%
from hal.training.config import DataConfig
from hal.training.streaming_dataset import HALStreamingDataset

torch.set_printoptions(threshold=float("inf"), precision=3)

# # %%
# mds_path = "/opt/projects/hal2/data/ranked/diamond/train"
# ds = StreamingDataset(local=mds_path, batch_size=1, shuffle=True)
# # %%
# x = ds[0]
# y = add_reward_to_episode(x)
# # %%
# y["p1_stock"]
# # %%
# np.where(y["p1_reward"] != 0)
# # %%
# y["p2_stock"]
# np.where(y["p2_reward"] != 0)
# # %%
# y["p1_stock"][1437:1439]
# # %%
# len(y["p1_stock"])
# # %%
# len(y["p1_reward"])
# %%
data_dir = "/opt/projects/hal2/data/top_players/Cody"

data = DataConfig(
    data_dir=data_dir,
    seq_len=7400,
    gamma=0.999,
    input_preprocessing_fn="baseline_controller_fine_main_analog_shoulder",
    target_preprocessing_fn="frame_1_and_12_value",
    pred_postprocessing_fn="frame_1",
)

ds = HALStreamingDataset(
    streams=None,
    local=data_dir,
    remote=None,
    batch_size=1,
    shuffle=False,
    data_config=data,
    split="test",
    debug=True,
)

# %%
ds.preprocessor.frame_offsets_by_input
# %%
ds.preprocessor.frame_offsets_by_target
# %%

# %%
x = ds[0]
x

# %%
p1_stocks = x["inputs"]["gamestate"][:, 1]
p2_stocks = x["inputs"]["gamestate"][:, 10]
# %%
torch.where(p2_stocks == 0.5)
# %%
x["targets"]["returns"][6400:6412]

# %%
torch.where(returns != 0)

# %%
p1_stocks[1515:1520]
# %%
p2_stocks[3035:3045]
# %%
rewards = torch.zeros(10000)
rewards[6400] = 1
rewards[2000] = -1
rewards[5100] = -1
rewards[1400] = 1


# %%
def compute_returns(rewards) -> torch.Tensor:
    """Calculate the return for a given episode."""
    gamma = 0.999
    returns = torch.zeros_like(rewards, dtype=torch.float32)
    running_return = 0

    # Work backwards through the trajectory
    for t in reversed(range(len(rewards))):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return

    return returns


def compute_returns_vectorized(rewards) -> torch.Tensor:
    """
    Vectorized version of discounted returns.
    Exactly matches the per-step logic:
        R[t] = r[t] + gamma * r[t+1] + gamma^2 * r[t+2] + ...
    """
    gamma = 0.999
    rewards = rewards  # shape: [T] (or possibly more dims)
    T = rewards.shape[0]
    device = rewards.device
    dtype = rewards.dtype

    # 1) Multiply each reward by gamma^t
    #    w[k] = gamma^k * r[k]
    t_range = torch.arange(T, device=device, dtype=dtype)
    w = (gamma**t_range) * rewards

    # 2) Take a cumulative sum of w:
    #    z[k] = sum_{i=0..k} w[i]
    z = torch.cumsum(w, dim=0)

    # 3) Convert z back to standard returns:
    #    R[0] = z[T-1]
    #    R[t] = gamma^{-t} * (z[T-1] - z[t-1])   for t > 0
    returns = torch.empty_like(rewards)
    returns[0] = z[-1]  # R[0]
    if T > 1:
        # For t in [1..T-1]
        t_idx = torch.arange(1, T, device=device)
        returns[1:] = (z[-1] - z[t_idx - 1]) * (gamma ** (-t_idx))

    return returns


t0 = time.perf_counter()
returns = compute_returns(rewards)
t1 = time.perf_counter()
returns_vectorized = compute_returns_vectorized(rewards)
t2 = time.perf_counter()
print(returns)
print(returns_vectorized)
print(f"Correct: {torch.allclose(returns, returns_vectorized)}")
print(f"compute_returns: {t1 - t0} seconds")
print(f"compute_returns_vectorized: {t2 - t1} seconds")
print(f"compute_returns_vectorized / compute_returns: {(t2 - t1) / (t1 - t0)}")
# %%
