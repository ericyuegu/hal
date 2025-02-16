# %%
import random
from pathlib import Path

import melee
import numpy as np
from streaming import StreamingDataset

from hal.data.process_replays import process_replay
from hal.data.schema import MDS_DTYPE_STR_BY_COLUMN
from hal.eval.eval_helper import mock_framedata_as_tensordict
from hal.training.config import DataConfig

# np.set_printoptions(threshold=np.inf)

# %%
ds = StreamingDataset(local="/opt/projects/hal2/data/ranked/diamond/val")
ds[0]
# %%
from hal.training.streaming_dataset import HALStreamingDataset

data_dir = "/opt/projects/hal2/data/ranked/diamond/val"
ds = HALStreamingDataset(
    local=data_dir,
    remote=None,
    batch_size=1,
    shuffle=True,
    data_config=DataConfig(data_dir=data_dir),
)
# %%
ds[0]
# %%
ds.preprocessor.trajectory_sampling_len
# %%
mock_framedata_L = mock_framedata_as_tensordict(ds.preprocessor.trajectory_sampling_len)
mock_framedata_L
# %%
# Store only a single time step to minimize copying
mock_model_inputs_ = ds.preprocessor.preprocess_inputs(mock_framedata_L, "p1")
mock_model_inputs_
# %%
for k, v in mock_model_inputs_.items():
    print(k, v)
# %%
# replay_path = Path(
#     "/opt/slippi/data/ranked-anonymized-2-151807/ranked-anonymized/master-platinum-f9770bb9a470e511f7f7c541.slp"
# )
replay_path = Path("/opt/slippi/data/ranked/ranked-anonymized-6-171694/master-master-b07692aa6f672cfaaf0b05bd.slp")
np_dict = process_replay(replay_path)

# %%
for k, v in np_dict.items():
    print(k, len(v))

# %%
np_dict["p1_nana_port"]

# %%
from streaming import MDSWriter

with MDSWriter(
    out="/tmp/test/",
    columns=MDS_DTYPE_STR_BY_COLUMN,
    exist_ok=True,
    compression="br",
) as writer:
    writer.write(np_dict)

# %%
ds = StreamingDataset(local="/tmp/test/")
ds[0]
# %%
ds = StreamingDataset(local="/opt/projects/hal2/data/ranked-3/test")
ds[10]
# %%
mds_path = "/opt/projects/hal2/data/mang0/train"
ds = StreamingDataset(local=mds_path, batch_size=1, shuffle=True)

# %%
x = ds[4623]
print(x["p1_stock"])
print(x["p1_percent"])
print(x["p2_stock"])
print(x["p2_percent"])

import random


# %%
def has_iceclimbers(replay_path: Path):
    try:
        console = melee.Console(path=str(replay_path), is_dolphin=False, allow_old_version=True)
        console.connect()
    except Exception as e:
        return None

    try:
        # Double step on first frame to match next controller state to current gamestate
        curr_gamestate = console.step()
        if curr_gamestate is None:
            return False
        players = curr_gamestate.players
        for port, player in players.items():
            # print(port, player.character)
            if player.character == melee.Character.POPO or player.character == melee.Character.NANA:
                return True
        return False
    finally:
        console.stop()


# %%
replay_dir = Path("/opt/slippi/data")
replays = list(replay_dir.glob("ranked-*/**/*.slp"))

# %%
random.shuffle(replays)
len(replays)
# %%
replays[:10]
# %%
iceclimbers_replays = []
for i, replay in enumerate(replays):
    if i > 1000:
        break
    if i % 100 == 0:
        print(i)
    if has_iceclimbers(replay):
        iceclimbers_replays.append(replay)
print(len(iceclimbers_replays))
# %%
iceclimbers_replays[:10]
# %%
console = melee.Console(path=str(iceclimbers_replays[0]), is_dolphin=False, allow_old_version=True)
console.connect()
gamestate = console.step()
p1_states = []
p1_nana_data = []
while gamestate is not None:
    p1_states.append(gamestate.players[1])
    p1_nana_data.append(gamestate.players[1].nana)
    gamestate = console.step()
console.stop()
# %%
for i, (p1_state, nana_state) in enumerate(zip(p1_states, p1_nana_data)):
    if nana_state is not None:
        nana_stock = nana_state.stock
    else:
        nana_stock = None
    print(i, p1_state.stock, nana_stock)
# %%
np_dict = process_replay(iceclimbers_replays[0])

# %%
np_dict
# %%
for k, v in np_dict.items():
    if "nana" in k:
        print(k, v)
# %%
for k in np_dict.keys():
    print(k)
# %%
import numpy as np
import numpy.ma as ma

# %%
ma.MaskedArray([1.0, 2.0, None])
# %%
# x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
x = ma.array([1.0, 2.0, 3.0], mask=[0, 1, 0], dtype=np.int32, fill_value=1 << 31)
x
# %%
x.dtype
# %%
np.dtype(x.dtype).name
# %%
x.filled()

# %%
from streaming import StreamingDataset

ds = StreamingDataset(local="/tmp/test/")
# %%
y = ma.masked_values(ds[0]["x"], 1e20)
y
# %%
