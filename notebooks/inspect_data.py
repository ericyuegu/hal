# %%
import numpy as np
from streaming import StreamingDataset

np.set_printoptions(threshold=np.inf)

# %%
mds_path = "/opt/projects/hal2/data/ranked/diamond/train"
ds = StreamingDataset(local=mds_path, batch_size=1, shuffle=True)

# %%
x = ds[0]

# %%
for k in x.keys():
    print(k)

# %%
x["p1_button_x"]
# %%
x["p1_position_y"]
# %%
x["p1_position_x"]
