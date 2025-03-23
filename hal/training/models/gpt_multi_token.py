"""Adapted from Karpathy's nanoGPT: https://github.com/karpathy/nanoGPT."""
import math

import attr
import torch
import torch.nn as nn
from tensordict import TensorDict

from hal.preprocess.preprocessor import Preprocessor
from hal.training.models.gpt import BaseGPT
from hal.training.models.gpt import BlockRelativePosition
from hal.training.models.gpt import GPTConfig
from hal.training.models.registry import Arch


@attr.s(auto_attribs=True, frozen=True)
class MultiTokenGPTConfig(GPTConfig):
    # Numbers of frames in the future to predict
    # e.g. (1, 2, 3) means predict 1, 2, and 3 frames in the future
    multi_token_heads: tuple[int, ...] = (1,)


class GPTMultiToken(BaseGPT):
    def __init__(self, preprocessor: Preprocessor, gpt_config: MultiTokenGPTConfig) -> None:
        super().__init__(preprocessor, gpt_config)
        # Numeric + embedded feature sizes defined programmatically in InputPreprocessConfig
        self.input_size = self.preprocessor.input_size  # G
        self.n_embd = gpt_config.n_embd  # D
        assert preprocessor.target_config.multi_token_heads is not None
        self.multi_token_heads = preprocessor.target_config.multi_token_heads
        self.num_multi_token_output_heads = len(self.multi_token_heads)

        # Categorical input embeddings
        self.emb_config = self.preprocessor.data_config
        self.stage_emb = nn.Embedding(self.emb_config.num_stages, self.emb_config.stage_embedding_dim)
        self.character_emb = nn.Embedding(self.emb_config.num_characters, self.emb_config.character_embedding_dim)
        self.action_emb = nn.Embedding(self.emb_config.num_actions, self.emb_config.action_embedding_dim)

        self.transformer = nn.ModuleDict(
            dict(
                proj_down=nn.Linear(self.input_size, gpt_config.n_embd),  # G -> D
                drop=nn.Dropout(gpt_config.dropout),
                h=nn.ModuleList([BlockRelativePosition(gpt_config) for _ in range(gpt_config.n_layer)]),
                ln_f=nn.LayerNorm(self.n_embd, bias=gpt_config.bias),
            )
        )

        # Output heads
        self.target_shapes_by_head = self.preprocessor.target_config.target_shapes_by_head
        self.shoulder_output_dim = self.target_shapes_by_head["shoulder_1"][0]
        self.c_stick_output_dim = self.target_shapes_by_head["c_stick_1"][0]
        self.main_stick_output_dim = self.target_shapes_by_head["main_stick_1"][0]
        self.button_output_dim = self.target_shapes_by_head["buttons_1"][0]

        shoulder_output_size = self.shoulder_output_dim * self.num_multi_token_output_heads
        c_stick_output_size = self.c_stick_output_dim * self.num_multi_token_output_heads
        main_stick_output_size = self.main_stick_output_dim * self.num_multi_token_output_heads
        button_output_size = self.button_output_dim * self.num_multi_token_output_heads

        # Input sizes for each head
        shoulder_input_size = self.n_embd
        c_stick_input_size = self.n_embd + shoulder_output_size
        main_stick_input_size = self.n_embd + shoulder_output_size + c_stick_output_size
        button_input_size = self.n_embd + shoulder_output_size + c_stick_output_size + main_stick_output_size

        # Put shoulder and c-stick first because they are less complex and they modify/override other inputs
        self.shoulder_head = nn.Sequential(
            nn.LayerNorm(shoulder_input_size, bias=gpt_config.bias),
            nn.Linear(shoulder_input_size, shoulder_input_size // 2),
            nn.GELU(),
            nn.Linear(shoulder_input_size // 2, shoulder_output_size),
        )

        self.c_stick_head = nn.Sequential(
            nn.LayerNorm(c_stick_input_size, bias=gpt_config.bias),
            nn.Linear(c_stick_input_size, c_stick_input_size // 2),
            nn.GELU(),
            nn.Linear(c_stick_input_size // 2, c_stick_output_size),
        )

        self.main_stick_head = nn.Sequential(
            nn.LayerNorm(main_stick_input_size, bias=gpt_config.bias),
            nn.Linear(main_stick_input_size, main_stick_input_size // 2),
            nn.GELU(),
            nn.Linear(main_stick_input_size // 2, main_stick_output_size),
        )

        self.button_head = nn.Sequential(
            nn.LayerNorm(button_input_size, bias=gpt_config.bias),
            nn.Linear(button_input_size, button_input_size // 2),
            nn.GELU(),
            nn.Linear(button_input_size // 2, button_output_size),
        )

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * gpt_config.n_layer))

    def _embed_inputs(self, inputs: TensorDict) -> torch.Tensor:
        return torch.cat(
            [
                self.stage_emb(inputs["stage"]),
                self.character_emb(inputs["ego_character"]),
                self.character_emb(inputs["opponent_character"]),
                self.action_emb(inputs["ego_action"]),
                self.action_emb(inputs["opponent_action"]),
                inputs["gamestate"],
                inputs["controller"],
            ],
            dim=-1,
        )

    def forward(self, inputs: TensorDict) -> TensorDict:
        B, L, _ = inputs["gamestate"].shape
        assert L <= self.block_size, f"Cannot forward sequence of length {L}, block size is only {self.block_size}"

        # Concatenate embeddings and numerical inputs -> project down
        combined_inputs_BLG = self._embed_inputs(inputs)
        proj_inputs_BLD = self.transformer.proj_down(combined_inputs_BLG)

        x_BLD = self.transformer.drop(proj_inputs_BLD)
        for block in self.transformer.h:
            x_BLD = block(x_BLD)
        x_BLD = self.transformer.ln_f(x_BLD)

        # Process all time steps at once for each output mode, autoregressively decode next head
        # (B,L,D) -> (B,L,N*C)
        shoulder: torch.Tensor = self.shoulder_head(x_BLD)
        c_stick: torch.Tensor = self.c_stick_head(torch.cat((x_BLD, shoulder.detach()), dim=-1))
        main_stick: torch.Tensor = self.main_stick_head(
            torch.cat((x_BLD, shoulder.detach(), c_stick.detach()), dim=-1)
        )
        button: torch.Tensor = self.button_head(
            torch.cat((x_BLD, shoulder.detach(), c_stick.detach(), main_stick.detach()), dim=-1)
        )

        shoulder = shoulder.view(B, L, self.num_multi_token_output_heads, self.shoulder_output_dim)
        c_stick = c_stick.view(B, L, self.num_multi_token_output_heads, self.c_stick_output_dim)
        main_stick = main_stick.view(B, L, self.num_multi_token_output_heads, self.main_stick_output_dim)
        button = button.view(B, L, self.num_multi_token_output_heads, self.button_output_dim)

        result = {}
        for i, offset in enumerate(self.multi_token_heads):
            result[f"shoulder_{offset}"] = shoulder[:, :, i, :]
            result[f"c_stick_{offset}"] = c_stick[:, :, i, :]
            result[f"main_stick_{offset}"] = main_stick[:, :, i, :]
            result[f"buttons_{offset}"] = button[:, :, i, :]

        return TensorDict(result, batch_size=(B, L))


class GPTMultiTokenValue(GPTMultiToken):
    def __init__(self, preprocessor: Preprocessor, gpt_config: MultiTokenGPTConfig) -> None:
        super().__init__(preprocessor, gpt_config)
        self.value_head = nn.Sequential(
            nn.LayerNorm(self.n_embd, bias=gpt_config.bias),
            nn.Linear(self.n_embd, self.n_embd // 2),
            nn.GELU(),
            nn.Linear(self.n_embd // 2, 1),
        )

    def forward(self, inputs: TensorDict) -> TensorDict:
        B, L, _ = inputs["gamestate"].shape
        assert L <= self.block_size, f"Cannot forward sequence of length {L}, block size is only {self.block_size}"

        # Concatenate embeddings and numerical inputs -> project down
        combined_inputs_BLG = self._embed_inputs(inputs)
        proj_inputs_BLD = self.transformer.proj_down(combined_inputs_BLG)

        x_BLD = self.transformer.drop(proj_inputs_BLD)
        for block in self.transformer.h:
            x_BLD = block(x_BLD)
        x_BLD = self.transformer.ln_f(x_BLD)

        # Process all time steps at once for each output mode, autoregressively decode next head
        # (B,L,D) -> (B,L,N*C)
        shoulder: torch.Tensor = self.shoulder_head(x_BLD)
        c_stick: torch.Tensor = self.c_stick_head(torch.cat((x_BLD, shoulder.detach()), dim=-1))
        main_stick: torch.Tensor = self.main_stick_head(
            torch.cat((x_BLD, shoulder.detach(), c_stick.detach()), dim=-1)
        )
        button: torch.Tensor = self.button_head(
            torch.cat((x_BLD, shoulder.detach(), c_stick.detach(), main_stick.detach()), dim=-1)
        )

        shoulder = shoulder.view(B, L, self.num_multi_token_output_heads, self.shoulder_output_dim)
        c_stick = c_stick.view(B, L, self.num_multi_token_output_heads, self.c_stick_output_dim)
        main_stick = main_stick.view(B, L, self.num_multi_token_output_heads, self.main_stick_output_dim)
        button = button.view(B, L, self.num_multi_token_output_heads, self.button_output_dim)

        result = {}
        for i, offset in enumerate(self.multi_token_heads):
            result[f"shoulder_{offset}"] = shoulder[:, :, i, :]
            result[f"c_stick_{offset}"] = c_stick[:, :, i, :]
            result[f"main_stick_{offset}"] = main_stick[:, :, i, :]
            result[f"buttons_{offset}"] = button[:, :, i, :]

        value = self.value_head(x_BLD)
        result["value"] = value

        return TensorDict(result, batch_size=(B, L))


Arch.register(
    "MultiToken-512-6-8_1-12",
    GPTMultiToken,
    gpt_config=MultiTokenGPTConfig(
        block_size=1024, n_embd=512, n_layer=6, n_head=8, dropout=0.2, multi_token_heads=(1, 12)
    ),
)

Arch.register(
    "MultiToken-512-6-8_1-12-18",
    GPTMultiToken,
    gpt_config=MultiTokenGPTConfig(
        block_size=1024, n_embd=512, n_layer=6, n_head=8, dropout=0.2, multi_token_heads=(1, 12, 18)
    ),
)

Arch.register(
    "MultiTokenValue-512-6-8_1-12",
    GPTMultiTokenValue,
    gpt_config=MultiTokenGPTConfig(
        block_size=1024,
        n_embd=512,
        n_layer=6,
        n_head=8,
        dropout=0.2,
        multi_token_heads=(1, 12),
    ),
)
Arch.register(
    "MultiTokenValue-512-6-8_1-12-dropout0.05",
    GPTMultiTokenValue,
    gpt_config=MultiTokenGPTConfig(
        block_size=1024,
        n_embd=512,
        n_layer=6,
        n_head=8,
        dropout=0.05,
        multi_token_heads=(1, 12),
    ),
)
