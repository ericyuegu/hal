import random
import time
import uuid
from collections import defaultdict
from typing import Any
from typing import Dict
from typing import List

import melee
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from hal.constants import CHARACTERS
from hal.constants import STAGES


def stack_playerstate(states: List[melee.PlayerState]) -> Dict[str, Any]:
    stacked_states = defaultdict(list)
    for state in states:
        for field in state.__slots__:
            value = getattr(state, field)
            stacked_states[field].append(value)

    return stacked_states


def main() -> None:
    replay_path = "/Users/ericgu/data/ssbm-il/mang0/Game_20230614T212502.slp"

    console = melee.Console(is_dolphin=False, allow_old_version=False, path=replay_path)
    console.connect()

    gamestate: melee.GameState = console.step()

    player_states: Dict[int, List[melee.PlayerState]] = defaultdict(list)
    while gamestate is not None:
        for player_name, state in gamestate.players.items():
            player_states[player_name].append(state)
        gamestate = console.step()

    stacked_frames = {}
    for player_name, frames in player_states.items():
        stacked_frames[player_name] = stack_playerstate(frames)


def generate_random_episode(ep_len: int = 10800):
    int_fields = ("stock_1", "stock_2")
    float_fields = ("pos_x_1", "pos_y_1", "percent_1", "shield_1", "pos_x_2", "pos_y_2", "percent_2", "shield_2")
    dummy_episode = {}

    id = uuid.uuid4().int >> 64
    stage = random.choice(STAGES)
    character1 = random.choice(CHARACTERS)
    character2 = random.choice(CHARACTERS)

    dummy_episode["replay_id"] = np.array([id] * ep_len)
    dummy_episode["stage"] = [stage] * ep_len
    dummy_episode["character1"] = [character1] * ep_len
    dummy_episode["character2"] = [character2] * ep_len

    for field in float_fields:
        dummy_episode[field] = np.random.randn(ep_len)
    for field in int_fields:
        dummy_episode[field] = np.random.randint(4, size=ep_len)

    return dummy_episode


def test_pyarrow(num_eps: int = 100) -> None:
    eps = []
    for i in range(num_eps):
        ep_len = np.random.randint(3600, 21600)
        ep = generate_random_episode(ep_len)
        eps.append(ep)
        if i % 1000 == 0:
            print(f"generated {i} episodes")

    concat_dict = {}
    for key in eps[0].keys():
        concat_dict[key] = np.concatenate([ep[key] for ep in eps])

    table = pa.Table.from_pydict(concat_dict)
    pq.write_table(table, where="./test.parquet")
    print("saved parquet file, done!")


def test_filter() -> None:
    file_path = "./test.parquet"
    table = pq.read_table(file_path)
    t0 = time.perf_counter()
    expr = pc.field("character1") == "fox"
    filtered_table = table.filter(expr)
    t1 = time.perf_counter()

    print(f"Filtered in {t1 - t0:2f} sec")


if __name__ == "__main__":
    test_pyarrow(num_eps=10000)
    test_filter()
