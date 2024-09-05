# %%
from pathlib import Path

import pandas as pd
from tensordict import TensorDict

from hal.data.stats import load_dataset_stats
from hal.training.config import DataConfig
from hal.training.config import EmbeddingConfig
from hal.training.dataloader import create_tensordicts
from hal.training.dataset import InMemoryDataset
from hal.training.zoo.preprocess.preprocess_inputs import NUMERIC_FEATURES_V0
from hal.training.zoo.preprocess.preprocess_inputs import preprocess_inputs_v0

players = ["p1", "p2"]
cols = []
for player in players:
    for feature in NUMERIC_FEATURES_V0:
        cols.append(f"{player}_{feature}")
# %%
data_config = DataConfig(data_dir="/opt/projects/hal2/data/dev", input_len=30, target_len=5)
embed_config = EmbeddingConfig()
data_dir = Path(data_config.data_dir)
stats_path = data_dir / "stats.json"

train_td, val_td = create_tensordicts(data_config)
dataset = InMemoryDataset(train_td, stats_path, data_config, embed_config)
stats_by_feature_name = load_dataset_stats(stats_path)

# %%
raw_td = dataset.tensordict[:35]
raw_df = pd.DataFrame({key: value.numpy() for key, value in raw_td.items() if key in cols})
raw_df = raw_df.reindex(sorted(raw_df.columns), axis=1)
raw_df

# %%
processed_td = preprocess_inputs_v0(raw_td, data_config, "p1", stats_by_feature_name)
processed_df = pd.DataFrame(processed_td["gamestate"].numpy(), columns=cols)
processed_df = processed_df.reindex(sorted(processed_df.columns), axis=1)
processed_df

# %%

# %%

# %%
# %%
td = TensorDict.load(
    "/opt/projects/hal2/runs/2024-09-04_22-17-22/arch@MLPv1_local_batch_size@256_n_samples@1048576/training_samples/256"
)

# %%
td.get("inputs")

# %%
td["inputs"]["ego_action"][0, :10]

# %%
pd.DataFrame(td["inputs"]["gamestate"][0].numpy(), columns=cols)

# %%
pd.DataFrame(dataset[1]["inputs"]["gamestate"].numpy(), columns=cols)
