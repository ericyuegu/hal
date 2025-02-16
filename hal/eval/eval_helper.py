from typing import Dict
from typing import List

import attr
import melee
import numpy as np
import torch
from loguru import logger
from tensordict import TensorDict

from hal.constants import INCLUDED_STAGES
from hal.constants import PLAYER_1_PORT
from hal.constants import PLAYER_2_PORT
from hal.constants import Player
from hal.data.schema import NP_TYPE_BY_COLUMN

PRIOR_STAGE_LIKELIHOODS = {
    "BATTLEFIELD": 0.3358,
    "YOSHIS_STORY": 0.2018,
    "POKEMON_STADIUM": 0.1888,
    "FOUNTAIN_OF_DREAMS": 0.1259,
    "DREAMLAND": 0.0826,
    "FINAL_DESTINATION": 0.0651,
}


PRIOR_CHARACTER_LIKELIHOODS = {
    "FOX": 0.2519,
    "FALCO": 0.1779,
    "MARTH": 0.1039,
    "SHEIK": 0.0789,
    "CPTFALCON": 0.0652,
    "JIGGLYPUFF": 0.0636,
    "PEACH": 0.0499,
    "SAMUS": 0.034,
    "LUIGI": 0.0274,
    "POPO": 0.0212,
    "GANONDORF": 0.0171,
    "PIKACHU": 0.0166,
    "DOC": 0.0161,
    "DK": 0.0159,
    "YOSHI": 0.00995,
    "NESS": 0.00855,
    "LINK": 0.00735,
    "BOWSER": 0.0059,
    "MEWTWO": 0.0058,
    "MARIO": 0.0057,
    "GAMEANDWATCH": 0.00500,
    "ROY": 0.00435,
    "YLINK": 0.0039,
    "ZELDA": 0.00245,
    "KIRBY": 0.00105,
    "PICHU": 0.0004,
}


@attr.s(auto_attribs=True, frozen=True)
class Matchup:
    stage: str
    ego_character: str
    opponent_character: str

    @classmethod
    def spacies(cls, n: int) -> List["Matchup"]:
        """Generate `n` spacies matchups"""
        matchups: List[Matchup] = []
        stages = INCLUDED_STAGES
        characters = ["FOX", "FALCO"]
        i = 0
        while len(matchups) < n:
            stage = stages[i % len(stages)]
            ego_char = characters[(i // len(stages)) % len(characters)]
            opp_char = characters[(i // len(stages) // len(characters)) % len(characters)]
            matchups.append(cls(stage=stage, ego_character=ego_char, opponent_character=opp_char))
            i += 1
        return matchups

    @classmethod
    def deterministically_generate_random_matchups(cls, n: int, seed: int = 42) -> List["Matchup"]:
        """Deterministically generate `n` random matchups."""
        rng = np.random.default_rng(seed)
        stage = rng.choice(list(PRIOR_STAGE_LIKELIHOODS.keys()), size=n, p=list(PRIOR_STAGE_LIKELIHOODS.values()))
        ego_character = rng.choice(
            list(PRIOR_CHARACTER_LIKELIHOODS.keys()), size=n, p=list(PRIOR_CHARACTER_LIKELIHOODS.values())
        )
        opponent_character = rng.choice(
            list(PRIOR_CHARACTER_LIKELIHOODS.keys()), size=n, p=list(PRIOR_CHARACTER_LIKELIHOODS.values())
        )
        matchups = []
        for stage, ego_character, opponent_character in zip(stage, ego_character, opponent_character):
            matchups.append(Matchup(stage=stage, ego_character=ego_character, opponent_character=opponent_character))
        return matchups


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
        self.p1_stocks_lost += p1.stock < self._prev_p1_stock
        self.p2_stocks_lost += p2.stock < self._prev_p2_stock

        self._prev_p1_percent = p1_percent
        self._prev_p2_percent = p2_percent
        self._prev_p1_stock = p1.stock
        self._prev_p2_stock = p2.stock
        self.frames += 1

    def to_wandb_dict(self, player: Player, prefix: str = "closed_loop_eval") -> Dict[str, float]:
        if self.episodes == 0:
            logger.warning("No closed loop episode stats recorded")
            return {}

        # Calculate stock win rate as stocks taken / (stocks taken + stocks lost)
        stocks_taken = self.p2_stocks_lost if player == "p1" else self.p1_stocks_lost
        stocks_lost = self.p1_stocks_lost if player == "p1" else self.p2_stocks_lost
        stock_win_rate = stocks_taken / (stocks_taken + stocks_lost) if (stocks_taken + stocks_lost) > 0 else 0.0
        damage_inflicted = self.p2_damage if player == "p1" else self.p1_damage
        damage_received = self.p1_damage if player == "p1" else self.p2_damage
        damage_win_rate = (
            damage_inflicted / (damage_inflicted + damage_received)
            if (damage_inflicted + damage_received) > 0
            else 0.0
        )
        return {
            f"{prefix}/episodes": self.episodes,
            f"{prefix}/damage_inflicted": damage_inflicted,
            f"{prefix}/damage_received": damage_received,
            f"{prefix}/damage_inflicted_per_episode": damage_inflicted / self.episodes,
            f"{prefix}/damage_received_per_episode": damage_received / self.episodes,
            f"{prefix}/damage_win_rate": damage_win_rate,
            f"{prefix}/stocks_taken": stocks_taken,
            f"{prefix}/stocks_lost": stocks_lost,
            f"{prefix}/stocks_taken_per_episode": stocks_taken / self.episodes,
            f"{prefix}/stocks_lost_per_episode": stocks_lost / self.episodes,
            f"{prefix}/stock_win_rate": stock_win_rate,
            f"{prefix}/frames": self.frames,
        }


def mock_framedata_as_tensordict(seq_len: int) -> TensorDict:
    """Mock `seq_len` frames of gamestate data."""
    return TensorDict({k: torch.zeros(seq_len) for k in NP_TYPE_BY_COLUMN}, batch_size=(seq_len,))


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
