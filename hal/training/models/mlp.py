import torch
import torch.nn as nn
from tensordict import TensorDict

from hal.training.config import TrainConfig
from hal.training.models.registry import Arch
from hal.training.preprocess.registry import get_input_size_from_config


class MLPBC(nn.Module):
    """
    Simple MLP that predicts next action a from past states s.
    """

    def __init__(self, config: TrainConfig, hidden_size: int, n_layer: int = 4, dropout=0.1) -> None:
        super().__init__()
        data_config = config.data
        embed_config = config.embedding
        assert embed_config.num_buttons is not None
        assert embed_config.num_main_stick_clusters is not None
        assert embed_config.num_c_stick_clusters is not None
        self.n_embd = get_input_size_from_config(embed_config)
        self.max_length = data_config.input_len

        self.modules_by_name = nn.ModuleDict(
            dict(
                stage=nn.Embedding(embed_config.num_stages, embed_config.stage_embedding_dim),
                character=nn.Embedding(embed_config.num_characters, embed_config.character_embedding_dim),
                action=nn.Embedding(embed_config.num_actions, embed_config.action_embedding_dim),
                proj_in=nn.Linear(self.max_length * self.n_embd, hidden_size),
                mlp=nn.Sequential(
                    *[
                        layer
                        for _ in range(n_layer - 1)
                        for layer in [nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_size, hidden_size)]
                    ]
                ),
            )
        )
        self.button_head = nn.Linear(hidden_size, embed_config.num_buttons)
        self.main_stick_head = nn.Linear(hidden_size, embed_config.num_main_stick_clusters)
        self.c_stick_head = nn.Linear(hidden_size, embed_config.num_c_stick_clusters)

    def forward(self, inputs: TensorDict) -> TensorDict:
        B, T, D = inputs["gamestate"].shape
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
        x = concat_inputs.view(B, -1)

        x = self.modules_by_name.proj_in(x)
        x = self.modules_by_name.mlp(x)

        button_output = self.button_head(x)
        main_stick_output = self.main_stick_head(x)
        c_stick_output = self.c_stick_head(x)

        return TensorDict(
            {"buttons": button_output, "main_stick": main_stick_output, "c_stick": c_stick_output},
            batch_size=(B,),
        )


class MLPDebug(nn.Module):
    """
    Simple MLP that predicts next action a from past states s.
    """

    def __init__(self, config: TrainConfig, hidden_size: int, n_layer: int = 4, dropout=0.1) -> None:
        super().__init__()
        data_config = config.data
        embed_config = config.embedding
        assert embed_config.num_buttons is not None
        assert embed_config.num_main_stick_clusters is not None
        assert embed_config.num_c_stick_clusters is not None

        frame_emb_dim = 200
        self.n_embd = get_input_size_from_config(embed_config) + frame_emb_dim
        self.max_length = data_config.input_len

        self.modules_by_name = nn.ModuleDict(
            dict(
                frame=nn.Embedding(28800, frame_emb_dim),
                stage=nn.Embedding(embed_config.num_stages, embed_config.stage_embedding_dim),
                character=nn.Embedding(embed_config.num_characters, embed_config.character_embedding_dim),
                action=nn.Embedding(embed_config.num_actions, embed_config.action_embedding_dim),
                proj_in=nn.Linear(self.max_length * self.n_embd, hidden_size),
                mlp=nn.Sequential(
                    *[
                        layer
                        for _ in range(n_layer - 1)
                        for layer in [nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_size, hidden_size)]
                    ]
                ),
            )
        )
        self.button_head = nn.Linear(hidden_size, embed_config.num_buttons)
        self.main_stick_head = nn.Linear(hidden_size, embed_config.num_main_stick_clusters)
        self.c_stick_head = nn.Linear(hidden_size, embed_config.num_c_stick_clusters)

    def forward(self, inputs: TensorDict) -> TensorDict:
        B, T, D = inputs["gamestate"].shape
        assert T > 0

        frame_emb = self.modules_by_name.frame(inputs["frame"]).squeeze(-2)
        stage_emb = self.modules_by_name.stage(inputs["stage"]).squeeze(-2)
        ego_character_emb = self.modules_by_name.character(inputs["ego_character"]).squeeze(-2)
        opponent_character_emb = self.modules_by_name.character(inputs["opponent_character"]).squeeze(-2)
        ego_action_emb = self.modules_by_name.action(inputs["ego_action"]).squeeze(-2)
        opponent_action_emb = self.modules_by_name.action(inputs["opponent_action"]).squeeze(-2)
        gamestate = inputs["gamestate"]

        concat_inputs = torch.cat(
            [
                frame_emb,
                stage_emb,
                ego_character_emb,
                opponent_character_emb,
                ego_action_emb,
                opponent_action_emb,
                gamestate,
            ],
            dim=-1,
        )
        x = concat_inputs.view(B, -1)

        x = self.modules_by_name.proj_in(x)
        x = self.modules_by_name.mlp(x)

        button_output = self.button_head(x)
        main_stick_output = self.main_stick_head(x)
        c_stick_output = self.c_stick_head(x)

        return TensorDict(
            {"buttons": button_output, "main_stick": main_stick_output, "c_stick": c_stick_output},
            batch_size=(B,),
        )


Arch.register("MLPv1", make_net=MLPBC, hidden_size=128, n_layer=4, dropout=0.1)
Arch.register("MLPDebug", make_net=MLPDebug, hidden_size=128, n_layer=4, dropout=0.0)
Arch.register("MLPDebug-512-8", make_net=MLPDebug, hidden_size=512, n_layer=8, dropout=0.0)
