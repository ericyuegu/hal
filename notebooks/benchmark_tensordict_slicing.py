# %%
import time

import numpy as np
import torch
from tensordict import TensorDict

# benchmark copying entire np array & using tensordict slicing
episode_len = 10000
seq_len = 256
np_dict = {str(i): np.random.rand(episode_len) for i in range(50)}

t0 = time.perf_counter()
random_start_idx = 100
tensordict_from_np = TensorDict({k: torch.from_numpy(v.copy()) for k, v in np_dict.items()}, batch_size=(episode_len,))
tensordict_from_np = tensordict_from_np[random_start_idx : random_start_idx + seq_len]
t1 = time.perf_counter()
full_copy_time = t1 - t0
print(f"tensordict_full_copy_slice: {full_copy_time}")

# benchmark slicing before copying
t0 = time.perf_counter()
sample_slice = {
    k: torch.from_numpy(v[random_start_idx : random_start_idx + seq_len].copy()) for k, v in np_dict.items()
}
tensordict_from_slice = TensorDict(sample_slice, batch_size=(seq_len,))
t1 = time.perf_counter()
slice_time = t1 - t0
print(f"tensordict from pre-sliced np: {slice_time}")

relative_speed = full_copy_time / slice_time
print(f"relative speed: {relative_speed}")
