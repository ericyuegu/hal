# %%
import json

import numpy as np
import torch
import torch.nn as nn
from streaming import StreamingDataLoader
from streaming import StreamingDataset

# %%
ds = StreamingDataset(local="data/processed/ranked-anonymized-1/mds/val", batch_size=1)
val = StreamingDataLoader(ds, batch_size=1, num_workers=0)

# %%
stats_json_fp = "data/processed/ranked-anonymized-1/mds/stats.json"
with open(stats_json_fp) as f:
    stats_dict = json.load(f)

stats_dict


# %%
def preprocess_inputs(sample: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    # normalize values to [-1, 1]

    out = {}
    for feature_name, array in sample.items():
        # convert to single type
        arr = array.astype(np.float32)
        # Get stats for this feature
        stats = stats_dict.get(feature_name)
        if stats is not None and "min" in stats and "max" in stats:
            min_val = stats["min"]
            max_val = stats["max"]
            # avoid division by zero if min == max
            if max_val > min_val:
                arr = 2.0 * (arr - min_val) / (max_val - min_val) - 1.0
            else:
                arr = np.zeros_like(arr, dtype=np.float32)
        out[feature_name] = torch.from_numpy(arr)
    return out
