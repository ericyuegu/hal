from typing import Dict

import attr
import melee
import torch
from tensordict import TensorDict

from hal.constants import INCLUDED_BUTTONS
from hal.constants import PLAYER_1_PORT
from hal.constants import PLAYER_2_PORT
from hal.constants import Player
from hal.data.schema import PYARROW_DTYPE_BY_COLUMN
from hal.training.config import EmbeddingConfig


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

    def to_wandb_dict(self, player: Player, prefix: str = "val/closed_loop") -> Dict[str, float]:
        return {
            f"{prefix}/episodes": self.episodes,
            f"{prefix}/damage_inflicted": self.p2_damage if player == "p1" else self.p1_damage,
            f"{prefix}/damage_received": self.p1_damage if player == "p1" else self.p2_damage,
            f"{prefix}/stocks_taken": self.p2_stocks_lost if player == "p1" else self.p1_stocks_lost,
            f"{prefix}/stocks_lost": self.p1_stocks_lost if player == "p1" else self.p2_stocks_lost,
            f"{prefix}/frames": self.frames,
        }


def send_controller_inputs(controller: melee.Controller, inputs: TensorDict, idx: int = -1) -> None:
    """
    Press buttons and tilt analog sticks given a dictionary of array-like values (length T for T future time steps).

    Args:
        controller_inputs (Dict[str, torch.Tensor]): Dictionary of array-like values.
        controller (melee.Controller): Controller object.
        idx (int): Index in the arrays to send.
    """
    if idx >= 0:
        assert idx < len(inputs["main_stick_x"])

    controller.tilt_analog(
        melee.Button.BUTTON_MAIN,
        inputs["main_stick_x"][idx].item(),
        inputs["main_stick_y"][idx].item(),
    )
    controller.tilt_analog(
        melee.Button.BUTTON_C,
        inputs["c_stick_x"][idx].item(),
        inputs["c_stick_y"][idx].item(),
    )

    button_idx = inputs["button"][idx].item()
    button_name = INCLUDED_BUTTONS[button_idx]
    if button_name != "NO_BUTTON":
        button = getattr(melee.Button, button_name.upper())
        controller.press_button(button)
        # logger.debug(f"Pressed {button_name}")
    controller.flush()


def mock_framedata_as_tensordict(seq_len: int) -> TensorDict:
    """Mock `seq_len` frames of gamestate data."""
    return TensorDict({k: torch.zeros(seq_len) for k in PYARROW_DTYPE_BY_COLUMN}, batch_size=(seq_len,))


def mock_preds_as_tensordict(embed_config: EmbeddingConfig) -> TensorDict:
    """Mock a single model prediction."""
    assert embed_config.num_buttons is not None
    assert embed_config.num_main_stick_clusters is not None
    assert embed_config.num_c_stick_clusters is not None
    return TensorDict(
        {
            "buttons": torch.zeros(embed_config.num_buttons),
            "main_stick": torch.zeros(embed_config.num_main_stick_clusters),
            "c_stick": torch.zeros(embed_config.num_c_stick_clusters),
        },
        batch_size=(),
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
