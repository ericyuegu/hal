# %%
from pathlib import Path

from hal.data.stats import load_dataset_stats
from hal.preprocess.preprocess_inputs import preprocess_inputs_v0
from hal.preprocess.preprocess_targets import preprocess_targets_v0
from hal.training.config import DataConfig
from hal.training.config import ReplayFilter
from hal.training.deprecated.dataset import InMemoryTensordictDataset
from hal.training.deprecated.dataset import load_filtered_parquet_as_tensordict

data_config = DataConfig(replay_filter=ReplayFilter(stage="BATTLEFIELD", ego_character="FOX"))
td = load_filtered_parquet_as_tensordict("/opt/projects/hal2/data/dev/train.parquet", data_config)
stats_path = Path("/opt/projects/hal2/data/dev/stats.json")
stats = load_dataset_stats(stats_path)

# %%
td["p1_character"]

# %%
preprocess_inputs_v0(td[:70], 60, "p1", stats)

# %%
preprocess_targets_v0(td[:70], "p1")

# %%
dataset = InMemoryTensordictDataset(td, stats_path, data_config, DataConfig())

# %%
dataset[0]

# %%
dataset[0]["inputs"]

# %%
