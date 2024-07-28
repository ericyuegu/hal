from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import torch.nn as nn

from hal.data.constants import ACTION_BY_IDX
from hal.data.constants import CHARACTER_BY_IDX
from hal.data.constants import INCLUDED_BUTTONS
from hal.data.constants import STAGE_BY_IDX
from hal.data.constants import STICK_XY_CLUSTER_CENTERS_V0
from hal.training.zoo.models.registry import Arch


# TODO: programmatically pass input size
class ResBlockLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
        super(ResBlockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.layernorm = nn.LayerNorm(hidden_size)
        self.res_head = nn.Linear(hidden_size, hidden_size)

    def forward(
        self, inputs: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = self.input_proj(inputs)
        x, (hidden, cell) = self.lstm(residual, (hidden, cell))
        x = self.layernorm(x)
        x = self.res_head(x)
        x = x + residual
        return x, hidden, cell


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1) -> None:
        super(MLP, self).__init__()
        self.c_fc = nn.Linear(input_size, 4 * hidden_size)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class ResidualRecurrentBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(input_dim)
        # TODO move lstm out into its own module w dropout
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim)

    def forward(
        self, x: torch.Tensor, hidden_in: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        residual = x
        x = self.ln_1(x)
        lstm_out, hidden_out = self.lstm(x, hidden_in)
        x = residual + self.dropout(lstm_out)

        residual = x
        x = self.ln_2(x)
        x = self.mlp(x)
        x = residual + self.dropout(x)

        return x, hidden_out


class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.stage_embed = nn.Embedding(len(STAGE_BY_IDX), 4)
        self.character_embed = nn.Embedding(len(CHARACTER_BY_IDX), 6)
        self.action_embed = nn.Embedding(len(ACTION_BY_IDX), 20)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.button_head = nn.Linear(hidden_size, len(INCLUDED_BUTTONS))
        self.main_stick_head = nn.Linear(hidden_size, len(STICK_XY_CLUSTER_CENTERS_V0))
        self.c_stick_head = nn.Linear(hidden_size, len(STICK_XY_CLUSTER_CENTERS_V0))

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        hidden: Optional[torch.Tensor] = None,
        cell: Optional[torch.Tensor] = None,
    ) -> Dict[str, Union[torch.Tensor, None]]:
        stage_embed = self.stage_embed(inputs["stage"]).squeeze(-2)
        ego_character_embed = self.character_embed(inputs["ego_character"]).squeeze(-2)
        opponent_character_embed = self.character_embed(inputs["opponent_character"]).squeeze(-2)
        ego_action_embed = self.action_embed(inputs["ego_action"]).squeeze(-2)
        opponent_action_embed = self.action_embed(inputs["opponent_action"]).squeeze(-2)
        gamestate = inputs["gamestate"]

        batch_size = stage_embed.shape[0]

        device = next(self.lstm.parameters()).device
        if hidden is None or cell is None:
            hidden, cell = (
                torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
            )

        x = torch.cat(
            [
                stage_embed,
                ego_character_embed,
                opponent_character_embed,
                ego_action_embed,
                opponent_action_embed,
                gamestate,
            ],
            dim=-1,
        )
        x, (hidden, cell) = self.lstm(x, (hidden, cell))

        assert hidden is not None
        last_hidden = hidden[-1]
        button_logits = self.button_head(last_hidden)
        main_stick_logits = self.main_stick_head(last_hidden)
        c_stick_logits = self.c_stick_head(last_hidden)

        return {
            "buttons": button_logits,
            "main_stick": main_stick_logits,
            "c_stick": c_stick_logits,
            "hidden": hidden,
            "cell": cell,
        }


Arch.register("lstm", make_net=LSTM, input_size=74, hidden_size=256, num_layers=2)
