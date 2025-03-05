# %%
import numpy as np
from streaming import StreamingDataset

from hal.preprocess.transformations import add_reward_to_episode

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
