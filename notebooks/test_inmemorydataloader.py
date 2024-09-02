# %%
from data.stats import load_dataset_stats
from training.config import DataConfig
from training.config import ReplayFilter
from training.dataset import load_filtered_parquet_as_tensordict
from training.zoo.preprocess.preprocess_inputs import preprocess_inputs_v0
from training.zoo.preprocess.preprocess_targets import preprocess_targets_v0

data_config = DataConfig(replay_filter=ReplayFilter(stage="BATTLEFIELD", ego_character="FOX"))
td = load_filtered_parquet_as_tensordict("/opt/projects/hal2/data/dev/train.parquet", data_config)
stats = load_dataset_stats("/opt/projects/hal2/data/dev/stats.json")

# %%
td["p1_character"]

# %%
preprocess_inputs_v0(td[:70], 60, "p1", stats)

# %%
preprocess_targets_v0(td[:70], "p1")
