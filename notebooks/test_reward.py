# %%
import numpy as np
from streaming import StreamingDataset

from hal.preprocess.transformations import add_reward_to_episode
from hal.training.config import DataConfig
from hal.training.config import TrainConfig
from hal.training.streaming_dataloader import get_dataloaders

# %%
mds_path = "/opt/projects/hal2/data/ranked/diamond/train"
ds = StreamingDataset(local=mds_path, batch_size=1, shuffle=True)
# %%
x = ds[0]
y = add_reward_to_episode(x)
# %%
y["p1_stock"]
# %%
np.where(y["p1_reward"] != 0)
# %%
y["p2_stock"]
np.where(y["p2_reward"] != 0)
# %%
y["p1_stock"][1437:1439]
# %%
len(y["p1_stock"])
# %%
len(y["p1_reward"])
# %%
train_config = TrainConfig(
    n_gpus=1,
    debug=True,
    arch="MultiToken-512-6-8_1-12",
    data=DataConfig(
        streams="cody",
        stream_stats="/opt/projects/hal2/data/top_players/Cody/stats.json",
        seq_len=256,
        gamma=0.999,
        input_preprocessing_fn="baseline_controller_fine_main_analog_shoulder",
        target_preprocessing_fn="frame_1_and_12_value",
        pred_postprocessing_fn="frame_1",
    ),
)

train, val = get_dataloaders(train_config)
# %%
