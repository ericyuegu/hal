# %%
import pyarrow as pa
from pyarrow import parquet as pq

# %%
table: pa.Table = pq.read_table("/opt/projects/hal2/Data/train.parquet")

# %%
stages = table["stage"].to_numpy()

# %%
stages
# %%
table.schema
