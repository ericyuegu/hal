# %%
from training.config import DataConfig
from training.dataset import load_filtered_parquet_as_tensordict

data_config = DataConfig()
td = load_filtered_parquet_as_tensordict("/opt/projects/hal2/data/dev/train.parquet", data_config)

# %%
td.keys()
