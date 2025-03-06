# %%
import torch

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
