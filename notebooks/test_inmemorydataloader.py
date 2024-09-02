# %%
from training.config import DataConfig
from training.config import ReplayFilter
from training.dataset import load_filtered_parquet_as_tensordict

data_config = DataConfig(replay_filter=ReplayFilter(stage="BATTLEFIELD", ego_character="FOX"))
td = load_filtered_parquet_as_tensordict("/opt/projects/hal2/data/dev/train.parquet", data_config)

# %%
td
# %%
