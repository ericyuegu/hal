from typing import Iterable
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn
from tensordict import TensorDict

from hal.training.config import TrainConfig
from hal.training.models.registry import Arch
from hal.training.preprocess.registry import get_input_size_from_config


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
    def __init__(self, config: TrainConfig, n_blocks: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        embed_config = config.embedding
        assert embed_config.num_buttons is not None
        assert embed_config.num_main_stick_clusters is not None
        assert embed_config.num_c_stick_clusters is not None
        self.n_embd = get_input_size_from_config(embed_config)

        self.modules_by_name = nn.ModuleDict(
            dict(
                stage=nn.Embedding(embed_config.num_stages, embed_config.stage_embedding_dim),
                character=nn.Embedding(embed_config.num_characters, embed_config.character_embedding_dim),
                action=nn.Embedding(embed_config.num_actions, embed_config.action_embedding_dim),
                h=nn.ModuleList(
                    [RecurrentResidualBlock(n_embd=self.n_embd, dropout=dropout) for _ in range(n_blocks)]
                ),
            )
        )
        self.button_head = nn.Linear(self.n_embd, embed_config.num_buttons)
        self.main_stick_head = nn.Linear(self.n_embd, embed_config.num_main_stick_clusters)
        self.c_stick_head = nn.Linear(self.n_embd, embed_config.num_c_stick_clusters)

    def forward(
        self,
        inputs: TensorDict,
        hidden_in: Optional[Iterable[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
    ) -> Tuple[TensorDict, Iterable[Optional[Tuple[torch.Tensor, torch.Tensor]]]]:
        B, L, D = inputs["gamestate"].shape
        assert T > 0

        stage_emb = self.modules_by_name.stage(inputs["stage"]).squeeze(-2)
        ego_character_emb = self.modules_by_name.character(inputs["ego_character"]).squeeze(-2)
        opponent_character_emb = self.modules_by_name.character(inputs["opponent_character"]).squeeze(-2)
        ego_action_emb = self.modules_by_name.action(inputs["ego_action"]).squeeze(-2)
        opponent_action_emb = self.modules_by_name.action(inputs["opponent_action"]).squeeze(-2)
        gamestate = inputs["gamestate"]

        concat_inputs = torch.cat(
            [stage_emb, ego_character_emb, opponent_character_emb, ego_action_emb, opponent_action_emb, gamestate],
            dim=-1,
        )

        if hidden_in is None:
            hidden_in = [None] * len(self.modules_by_name.h)

        new_hidden_in = []
        for i in range(T):
            x = concat_inputs[:, i].unsqueeze(1)
            for block, hidden in zip(self.modules_by_name.h, hidden_in):
                x, new_hidden = block(x, hidden)
                new_hidden_in.append(new_hidden)

            hidden_in = new_hidden_in
            new_hidden_in = []

        button_output = self.button_head(x).squeeze(-2)
        main_stick_output = self.main_stick_head(x).squeeze(-2)
        c_stick_output = self.c_stick_head(x).squeeze(-2)

        return (
            TensorDict(
                {"buttons": button_output, "main_stick": main_stick_output, "c_stick": c_stick_output},
                batch_size=(B,),
            ),
            hidden_in,
        )


Arch.register("LSTMv1-2-nodropout", make_net=LSTMv1, n_blocks=2, dropout=0.0)
Arch.register("LSTMv1-2", make_net=LSTMv1, n_blocks=2)
Arch.register("LSTMv1-4", make_net=LSTMv1, n_blocks=4)
Arch.register("LSTMv1-8", make_net=LSTMv1, n_blocks=8)
Arch.register("LSTMv1-16", make_net=LSTMv1, n_blocks=16)
