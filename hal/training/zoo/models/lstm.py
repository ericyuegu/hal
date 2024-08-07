from typing import Iterable
from typing import Optional
from typing import Sequence
from typing import Tuple

import attr
import torch
import torch.nn as nn
from data.constants import ACTION_BY_IDX
from data.constants import CHARACTER_BY_IDX
from data.constants import STAGE_BY_IDX
from tensordict import TensorDict


@attr.s(auto_attribs=True, frozen=True)
class MLPConfig:
    n_embd: int
    dropout: float


class MLP(nn.Module):
    def __init__(self, config: MLPConfig) -> None:
        super(MLP, self).__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


@attr.s(auto_attribs=True, frozen=True)
class LSTMConfig:
    n_embd: int
    dropout: float


class LSTM(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.lstm = nn.LSTM(config.n_embd, config.n_embd, batch_first=True)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, x: torch.Tensor, hidden_in: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x, hidden_out = self.lstm(x, hidden_in)
        return self.dropout(x), hidden_out


class RecurrentResidualBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(input_dim)
        self.lstm = LSTM(input_dim, hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim)

    def forward(
        self, x: torch.Tensor, hidden_in: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        y, hidden_out = self.lstm(self.ln_1(x), hidden_in)
        # Residual
        y = x + y
        z = y + self.mlp(self.ln_2(y))
        return z, hidden_out


@attr.s(auto_attribs=True, frozen=True)
class LSTMv1Config:
    stage_embedding_dim: int
    character_embedding_dim: int
    action_embedding_dim: int
    gamestate_dim: int

    stick_embedding_dim: int
    button_embedding_dim: int

    hidden_dim: int
    num_blocks: int

    num_stages: int = len(STAGE_BY_IDX)
    num_characters: int = len(CHARACTER_BY_IDX)
    num_actions: int = len(ACTION_BY_IDX)


class LSTMv1(nn.Module):
    def __init__(self, config: LSTMv1Config) -> None:
        super().__init__()
        self.config = config

        self.input_dim = (
            config.stage_embedding_dim
            + (2 * config.character_embedding_dim)
            + (2 * config.action_embedding_dim)
            + config.gamestate_dim
        )

        self.lstm = nn.ModuleDict(
            dict(
                stage=nn.Embedding(config.num_stages, config.stage_embedding_dim),
                character=nn.Embedding(config.num_characters, config.character_embedding_dim),
                action=nn.Embedding(config.num_actions, config.action_embedding_dim),
                h=nn.ModuleList(
                    [
                        RecurrentResidualBlock(input_dim=self.input_dim, hidden_dim=config.hidden_dim)
                        for _ in range(config.num_blocks)
                    ]
                ),
            )
        )

    def forward(
        self, inputs: TensorDict, hidden_in: Optional[Iterable[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None
    ) -> Tuple[torch.Tensor, Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]]]:
        batch, seq_len = inputs.shape
        assert seq_len > 0

        stage_emb = self.lstm.stage(inputs["stage"])
        ego_character_emb = self.lstm.character(inputs["ego_character"])
        opponent_character_emb = self.lstm.character(inputs["opponent_character"])
        ego_action_emb = self.lstm.action(inputs["ego_action"])
        opponent_action_emb = self.lstm.action(inputs["opponent_action"])
        gamestate = inputs["gamestate"]
        concat_inputs = torch.cat(
            [stage_emb, ego_character_emb, opponent_character_emb, ego_action_emb, opponent_action_emb, gamestate],
            dim=-1,
        )

        if hidden_in is None:
            hidden_in = [None] * len(self.lstm.h)

        new_hidden_in = []
        for i in range(seq_len):
            x = concat_inputs[:, i].unsqueeze(1)
            for block, hidden in zip(self.lstm.h, hidden_in):
                x, new_hidden = block(x, hidden)
                new_hidden_in.append(new_hidden)

            hidden_in = new_hidden_in
            new_hidden_in = []

        # TODO fix typing
        return x, hidden_in
