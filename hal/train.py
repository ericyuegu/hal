# %%
import torch

from hal.data.dataset import MmappedParquetDataset

dataset_path = "/opt/projects/hal2/data/dev/train.parquet"
dataset = MmappedParquetDataset(input_path=dataset_path, input_len=60, target_len=10)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

# %%
for inputs, targets in dataloader:
    print(inputs)
    print(targets)
    break
# %%
