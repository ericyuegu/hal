import math
from typing import List
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn
from tensordict import TensorDict

from hal.preprocess.preprocessor import Preprocessor
from hal.training.models.gpt import CausalSelfAttentionRelativePosition
from hal.training.models.gpt import GPTConfig
from hal.training.models.gpt import MLP
from hal.training.models.gpt import skew
from hal.training.models.gpt_multi_token import GPTMultiToken
from hal.training.models.gpt_multi_token import MultiTokenGPTConfig
from hal.training.models.registry import Arch


class CausalSelfAttentionRelativePositionWithCache(CausalSelfAttentionRelativePosition):
    def forward(
        self,
        x: torch.Tensor,
        layer_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_kv: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, L, D = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        assert L <= self.block_size, f"Cannot forward sequence of length {L}, block size is only {self.block_size}"

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, L, self.n_head, self.hs).transpose(1, 2)  # (B, nh, L, hs)
        q = q.view(B, L, self.n_head, self.hs).transpose(1, 2)  # (B, nh, L, hs)
        v = v.view(B, L, self.n_head, self.hs).transpose(1, 2)  # (B, nh, L, hs)

        # Use cached KV if provided
        if layer_cache is not None:
            k_cache, v_cache = layer_cache
            k = torch.cat([k_cache, k], dim=2)[:, :, -L:]
            v = torch.cat([v_cache, v], dim=2)[:, :, -L:]

        # relative positional embeddings
        start = self.block_size - L
        Er_t = self.Er[start:, :].transpose(0, 1)  # (hs, L)
        QEr = q @ Er_t  # (B, nh, L, hs) x (hs, L) -> (B, nh, L, L)
        Srel = skew(QEr)  # (B, nh, L, L)

        # causal self-attention
        QK_t = q @ k.transpose(-2, -1)  # (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        scale = 1.0 / math.sqrt(k.size(-1))
        att = (QK_t + Srel) * scale
        if layer_cache is None:
            att = att.masked_fill(self.bias[:, :, :L, :L] == 0, float("-inf"))
        att = torch.nn.functional.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)
        y = y.transpose(1, 2).contiguous().view(B, L, D)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        if return_kv:
            return y, (k, v)
        return y


class BlockWithCache(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttentionRelativePositionWithCache(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        layer_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_kv: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if return_kv:
            a, kv = self.attn(self.ln_1(x), layer_cache=layer_cache, return_kv=True)
            x = x + a
            x = x + self.mlp(self.ln_2(x))
            return x, kv
        else:
            x = x + self.attn(self.ln_1(x), layer_cache=layer_cache, return_kv=False)
            x = x + self.mlp(self.ln_2(x))
            return x


class KVCache:
    def __init__(self, batch: int, config: GPTConfig, device: torch.device | str) -> None:
        self.batch = batch
        self.config = config
        self.device = device

        # Initialize KV cache
        self.n_layer = config.n_layer
        self.n_head = config.n_head
        self.block_size = config.block_size
        self.n_embd = config.n_embd
        self.head_size = self.n_embd // self.n_head

        self.kv_cache = torch.zeros(
            self.n_layer,
            2,
            self.batch,
            self.n_head,
            self.block_size,
            self.head_size,
            device=self.device,
            dtype=torch.float32,
        )
    
    def update()


class GPTMultiTokenValueWithCache(GPTMultiToken):
    def __init__(self, preprocessor: Preprocessor, gpt_config: MultiTokenGPTConfig) -> None:
        super().__init__(preprocessor, gpt_config)
        # Replace transformer blocks with cached versions
        self.transformer.h = nn.ModuleList([BlockWithCache(gpt_config) for _ in range(gpt_config.n_layer)])

        # Add value head
        self.value_head = nn.Sequential(
            nn.LayerNorm(self.n_embd, bias=gpt_config.bias),
            nn.Linear(self.n_embd, self.n_embd // 2),
            nn.GELU(),
            nn.Linear(self.n_embd // 2, 1),
        )

        # Re-init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * gpt_config.n_layer))

    def forward_with_kv_cache(
        self,
        inputs: TensorDict,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        return_kv: bool = False,
    ) -> TensorDict | Tuple[TensorDict, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with optional KV caching.

        Args:
            inputs: Input tensordict
            kv_caches: Optional list of (key, value) cache tuples for each layer
            return_kv: Whether to return updated KV caches

        Returns:
            TensorDict of model outputs, and optionally updated KV caches
        """
        B, L, _ = inputs["gamestate"].shape
        assert L <= self.block_size, f"Cannot forward sequence of length {L}, block size is only {self.block_size}"

        # Concatenate embeddings and numerical inputs -> project down
        combined_inputs_BLG = self._embed_inputs(inputs)
        proj_inputs_BLD = self.transformer.proj_down(combined_inputs_BLG)

        x_BLD = self.transformer.drop(proj_inputs_BLD)

        new_kv_caches = []
        for i, block in enumerate(self.transformer.h):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            if return_kv:
                x_BLD, kv = block(x_BLD, layer_cache=layer_cache, return_kv=True)
                new_kv_caches.append(kv)
            else:
                x_BLD = block(x_BLD, layer_cache=layer_cache, return_kv=False)

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

        result_td = TensorDict(result, batch_size=(B, L))
        if return_kv:
            return result_td, new_kv_caches
        return result_td

    def forward(self, inputs: TensorDict) -> TensorDict:
        return self.forward_with_kv_cache(inputs, kv_caches=None, return_kv=False)


Arch.register(
    "MultiTokenValueWithCache-512-6-8_1-12",
    GPTMultiTokenValueWithCache,
    gpt_config=MultiTokenGPTConfig(
        block_size=1024,
        n_embd=512,
        n_layer=6,
        n_head=8,
        dropout=0.2,
        multi_token_heads=(1, 12),
    ),
)
