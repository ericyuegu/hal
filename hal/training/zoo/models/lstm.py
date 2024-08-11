from typing import Iterable
from typing import Optional
from typing import Sequence
from typing import Tuple

import torch
import torch.nn as nn
from tensordict import TensorDict

from hal.training.config import EmbeddingConfig
from hal.training.utils import get_nembd_from_config


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float) -> None:
        super(MLP, self).__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class LSTM(nn.Module):
    def __init__(self, n_embd: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_embd, n_embd, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, hidden_in: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x, hidden_out = self.lstm(x, hidden_in)
        return self.dropout(x), hidden_out


class RecurrentResidualBlock(nn.Module):
    def __init__(self, n_embd: int, dropout: float) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.lstm = LSTM(n_embd, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(
        self, x: torch.Tensor, hidden_in: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        y, hidden_out = self.lstm(self.ln_1(x), hidden_in)
        y = x + y
        z = y + self.mlp(self.ln_2(y))
        return z, hidden_out


class LSTMv1(nn.Module):
    def __init__(self, embed_config: EmbeddingConfig, num_blocks: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_config = embed_config
        self.n_embd = get_nembd_from_config(embed_config)

        self.lstm = nn.ModuleDict(
            dict(
                stage=nn.Embedding(embed_config.num_stages, embed_config.stage_embedding_dim),
                character=nn.Embedding(embed_config.num_characters, embed_config.character_embedding_dim),
                action=nn.Embedding(embed_config.num_actions, embed_config.action_embedding_dim),
                h=nn.ModuleList(
                    [RecurrentResidualBlock(n_embd=self.n_embd, dropout=dropout) for _ in range(num_blocks)]
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
