# %%
import numpy as np
import pyarrow as pa
from pyarrow import parquet

np.set_printoptions(threshold=np.inf)

# %%
# create dummy sequence learning dataset and save as parquet
# generate random streaks of 1s in target correlated with input dim
HIDDEN_DIM = 20
SEQ_LEN = 60
STREAK_LEN = 10


def create_sample():
    x = np.zeros((SEQ_LEN, HIDDEN_DIM))
    y = np.zeros((SEQ_LEN, HIDDEN_DIM))

    streak_start = np.random.choice(SEQ_LEN - 2, 1)[0]
    dim_idx = np.random.choice(HIDDEN_DIM, 1)[0]

    x[streak_start : streak_start + STREAK_LEN, dim_idx] = 1
    y[streak_start + 1 : streak_start + STREAK_LEN + 1, dim_idx] = 1

    return x, y


x, y = zip(*[create_sample() for _ in range(1000)])
x = np.stack(x, axis=0)
y = np.stack(y, axis=0)

# %%
# reshape first two dims
x = x.reshape(1000 * 60, 20)
y = y.reshape(1000 * 60, 20)


# %%
# make table
table = pa.Table.from_arrays([x, y], names=["x", "y"])

parquet.write_table(table, "../data/toy_dataset.parquet")
# %%
