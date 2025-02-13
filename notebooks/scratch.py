# %%
import random
from pathlib import Path

import melee
from data.process_replays import process_replay
from streaming import StreamingDataset

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
replay_path = Path(
    "/opt/slippi/data/ranked-anonymized-2-151807/ranked-anonymized/master-platinum-f9770bb9a470e511f7f7c541.slp"
)
np_dict = process_replay(replay_path)
# %%
for k in np_dict.keys():
    print(k)
# %%
import numpy as np
import numpy.ma as ma

# x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
x = ma.array([1.0, 2.0, 3.0], mask=[0, 1, 0], dtype=np.float32)
x
# %%
x.dtype
# %%
np.dtype(x.dtype).name
# %%
x.filled()
# %%
from streaming import MDSWriter

with MDSWriter(
    out="/tmp/test/",
    columns={"x": "ndarray:float32"},
    exist_ok=True,
) as writer:
    writer.write({"x": x})
# %%
from streaming import StreamingDataset

ds = StreamingDataset(local="/tmp/test/")
# %%
y = ma.masked_values(ds[0]["x"], 1e20)
y
# %%
