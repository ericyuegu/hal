from typing import Dict

import attr
import melee
import torch
from tensordict import TensorDict

from hal.constants import PLAYER_1_PORT
from hal.constants import PLAYER_2_PORT
from hal.constants import Player
from hal.data.schema import NP_TYPE_BY_COLUMN


@attr.s(auto_attribs=True, slots=True)
class EpisodeStats:
    p1_damage: float = 0.0
    p2_damage: float = 0.0
    p1_stocks_lost: int = 0
    p2_stocks_lost: int = 0
    frames: int = 0
    episodes: int = 1
    _prev_p1_stock: int = 0
    _prev_p2_stock: int = 0
    _prev_p1_percent: float = 0.0
    _prev_p2_percent: float = 0.0

    def __add__(self, other: "EpisodeStats") -> "EpisodeStats":
        return EpisodeStats(
            p1_damage=self.p1_damage + other.p1_damage,
            p2_damage=self.p2_damage + other.p2_damage,
            p1_stocks_lost=self.p1_stocks_lost + other.p1_stocks_lost,
            p2_stocks_lost=self.p2_stocks_lost + other.p2_stocks_lost,
            frames=self.frames + other.frames,
            episodes=self.episodes + other.episodes,
        )

    def __radd__(self, other: "EpisodeStats") -> "EpisodeStats":
        if other == 0:
            return self
        return self.__add__(other)

    def __str__(self) -> str:
        return f"EpisodeStats({self.episodes=}, {self.p1_damage=}, {self.p2_damage=}, {self.p1_stocks_lost=}, {self.p2_stocks_lost=}, {self.frames=})"

    def update(self, gamestate: melee.GameState) -> None:
        if gamestate.menu_state not in (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH):
            return

        p1, p2 = gamestate.players[PLAYER_1_PORT], gamestate.players[PLAYER_2_PORT]
        p1_percent, p2_percent = p1.percent, p2.percent

        self.p1_damage += max(0, p1_percent - self._prev_p1_percent)
        self.p2_damage += max(0, p2_percent - self._prev_p2_percent)
        self.p1_stocks_lost += p1.stock < 4 and p1.stock < self._prev_p1_stock
        self.p2_stocks_lost += p2.stock < 4 and p2.stock < self._prev_p2_stock

        self._prev_p1_percent = p1_percent
        self._prev_p2_percent = p2_percent
        self._prev_p1_stock = p1.stock
        self._prev_p2_stock = p2.stock
        self.frames += 1

    def to_wandb_dict(self, player: Player, prefix: str = "closed_loop_eval") -> Dict[str, float]:
        # Calculate stock win rate as stocks taken / (stocks taken + stocks lost)
        stocks_taken = self.p2_stocks_lost if player == "p1" else self.p1_stocks_lost
        stocks_lost = self.p1_stocks_lost if player == "p1" else self.p2_stocks_lost
        stock_win_rate = stocks_taken / (stocks_taken + stocks_lost) if (stocks_taken + stocks_lost) > 0 else 0.0
        damage_inflicted = self.p2_damage if player == "p1" else self.p1_damage
        damage_received = self.p1_damage if player == "p1" else self.p2_damage
        return {
            f"{prefix}/episodes": self.episodes,
            f"{prefix}/damage_inflicted": damage_inflicted,
            f"{prefix}/damage_received": damage_received,
            f"{prefix}/damage_inflicted_per_episode": damage_inflicted / self.episodes,
            f"{prefix}/damage_received_per_episode": damage_received / self.episodes,
            f"{prefix}/damage_win_rate": damage_inflicted / (damage_inflicted + damage_received),
            f"{prefix}/stocks_taken": stocks_taken,
            f"{prefix}/stocks_lost": stocks_lost,
            f"{prefix}/stocks_taken_per_episode": stocks_taken / self.episodes,
            f"{prefix}/stocks_lost_per_episode": stocks_lost / self.episodes,
            f"{prefix}/stock_win_rate": stock_win_rate,
            f"{prefix}/frames": self.frames,
        }


def mock_framedata_as_tensordict(seq_len: int) -> TensorDict:
    """Mock `seq_len` frames of gamestate data."""
    return TensorDict(
        {k: torch.zeros(seq_len, dtype=dtype) for k, dtype in NP_TYPE_BY_COLUMN.items()}, batch_size=(seq_len,)
    )


def share_and_pin_memory(tensordict: TensorDict) -> TensorDict:
    """
    Move tensordict to both shared and pinned memory.

    https://github.com/pytorch/pytorch/issues/32167#issuecomment-753551842
    """
    tensordict.share_memory_()

    cudart = torch.cuda.cudart()
    if cudart is None:
        return tensordict

    for tensor in tensordict.flatten_keys().values():
        assert isinstance(tensor, torch.Tensor)
        cudart.cudaHostRegister(tensor.data_ptr(), tensor.numel() * tensor.element_size(), 0)
        assert tensor.is_shared()
        assert tensor.is_pinned()

    return tensordict
