"""Adapted from Karpathy's nanoGPT: https://github.com/karpathy/nanoGPT."""
import math

import attr
import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.nn import functional as F

from hal.preprocess.preprocessor import Preprocessor
from hal.training.config import TrainConfig
from hal.training.models.registry import Arch


@attr.s(auto_attribs=True, frozen=True)
class GPTConfig:
    block_size: int
    n_embd: int
    n_layer: int
    n_head: int
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x: torch.Tensor):
        B, L, D = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        q = q.view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        v = v.view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)

        # causal self-attention; Self-attend: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :L, :L] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)
        y = y.transpose(1, 2).contiguous().view(B, L, D)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class BaseGPT(nn.Module):
    def __init__(self, preprocessor: Preprocessor, gpt_config: GPTConfig) -> None:
        super().__init__()
        self.preprocessor = preprocessor
        self.gpt_config = gpt_config
        self.block_size = self.gpt_config.block_size

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, inputs: TensorDict):
        raise NotImplementedError

    def crop_block_size(self, block_size) -> None:
        # model surgery to decrease the context window if necessary
        # e.g. we may load a pretrained model checkpoint but want to use a smaller context at inference
        assert block_size <= self.block_size
        self.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float):
        """Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS."""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        L, H, Q, T = (
            self.gpt_config.n_layer,
            self.gpt_config.n_head,
            self.n_embd // self.gpt_config.n_head,
            self.block_size,
        )
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu


class GPTv1(BaseGPT):
    """
    Decoder-only transformer with learned input embeddings for characters, stages, and actions.

    Shallow output heads, absolute position embeddings.

    Outputs independent categorical predictions for buttons, main stick, and c-stick.
    """

    def __init__(self, preprocessor: Preprocessor, gpt_config: GPTConfig) -> None:
        super().__init__(preprocessor, gpt_config)
        # Numeric + embedded feature sizes defined programmatically in InputPreprocessConfig
        self.input_size = self.preprocessor.input_size  # G
        self.n_embd = gpt_config.n_embd  # D

        # categorical input embeddings
        self.emb_config = self.preprocessor.data_config
        self.stage_emb = nn.Embedding(self.emb_config.num_stages, self.emb_config.stage_embedding_dim)
        self.character_emb = nn.Embedding(self.emb_config.num_characters, self.emb_config.character_embedding_dim)
        self.action_emb = nn.Embedding(self.emb_config.num_actions, self.emb_config.action_embedding_dim)

        self.transformer = nn.ModuleDict(
            dict(
                proj_down=nn.Linear(self.input_size, gpt_config.n_embd),  # G -> D
                wpe=nn.Embedding(self.block_size, gpt_config.n_embd),
                drop=nn.Dropout(gpt_config.dropout),
                h=nn.ModuleList([Block(gpt_config) for _ in range(gpt_config.n_layer)]),
                ln_f=nn.LayerNorm(self.n_embd, bias=gpt_config.bias),
            )
        )

        # output heads
        self.target_shapes_by_head = self.preprocessor.target_config.target_shapes_by_head
        self.button_head = nn.Linear(self.n_embd, self.target_shapes_by_head["buttons"][0], bias=False)
        self.main_stick_head = nn.Linear(self.n_embd, self.target_shapes_by_head["main_stick"][0], bias=False)
        self.c_stick_head = nn.Linear(self.n_embd, self.target_shapes_by_head["c_stick"][0], bias=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * gpt_config.n_layer))

    def _embed_inputs(self, inputs: TensorDict) -> torch.Tensor:
        return torch.cat(
            [
                self.stage_emb(inputs["stage"]).squeeze(-2),
                self.character_emb(inputs["ego_character"]).squeeze(-2),
                self.character_emb(inputs["opponent_character"]).squeeze(-2),
                self.action_emb(inputs["ego_action"]).squeeze(-2),
                self.action_emb(inputs["opponent_action"]).squeeze(-2),
                inputs["gamestate"],
            ],
            dim=-1,
        )

    def forward(self, inputs: TensorDict):
        B, L, _ = inputs["gamestate"].shape
        assert L <= self.block_size, f"Cannot forward sequence of length {L}, block size is only {self.block_size}"

        # concatenate embeddings and numerical inputs -> project down
        # TODO handle Nana, call embed twice and add down proj
        combined_inputs_BLG = self._embed_inputs(inputs)
        proj_inputs_BLD = self.transformer.proj_down(combined_inputs_BLG)

        # position embeddings
        pos_L = torch.arange(0, L, dtype=torch.long, device=next(self.parameters()).device)
        pos_emb_LD = self.transformer.wpe(pos_L)

        x_BLD = self.transformer.drop(proj_inputs_BLD + pos_emb_LD)
        for block in self.transformer.h:
            x_BLD = block(x_BLD)
        x_BLD = self.transformer.ln_f(x_BLD)

        return TensorDict(
            {
                "buttons": self.button_head(x_BLD).squeeze(-2),
                "main_stick": self.main_stick_head(x_BLD).squeeze(-2),
                "c_stick": self.c_stick_head(x_BLD).squeeze(-2),
            },
            batch_size=(B, L),
        )


class GPTv1Controller(GPTv1):
    def _embed_inputs(self, inputs: TensorDict) -> torch.Tensor:
        return torch.cat(
            [
                self.stage_emb(inputs["stage"]).squeeze(-2),
                self.character_emb(inputs["ego_character"]).squeeze(-2),
                self.character_emb(inputs["opponent_character"]).squeeze(-2),
                self.action_emb(inputs["ego_action"]).squeeze(-2),
                self.action_emb(inputs["opponent_action"]).squeeze(-2),
                inputs["gamestate"],
                inputs["controller"],
            ],
            dim=-1,
        )


class GPTv2Controller(GPTv1):
    """Autoregressive shallow output heads, absolute position embeddings."""

    def __init__(self, preprocessor: Preprocessor, gpt_config: GPTConfig) -> None:
        super().__init__(preprocessor, gpt_config)

        main_stick_size = self.target_shapes_by_head["main_stick"][0]
        button_size = self.target_shapes_by_head["buttons"][0]
        c_stick_size = self.target_shapes_by_head["c_stick"][0]

        # re-define and re-initialize just the new output heads
        self.c_stick_head = nn.Linear(self.n_embd, c_stick_size, bias=False)
        self.main_stick_head = nn.Linear(self.n_embd + c_stick_size, main_stick_size, bias=False)
        self.button_head = nn.Linear(self.n_embd + main_stick_size + c_stick_size, button_size, bias=False)

        for module in [self.main_stick_head, self.button_head, self.c_stick_head]:
            self._init_weights(module)

    def _embed_inputs(self, inputs: TensorDict) -> torch.Tensor:
        return torch.cat(
            [
                self.stage_emb(inputs["stage"]).squeeze(-2),
                self.character_emb(inputs["ego_character"]).squeeze(-2),
                self.character_emb(inputs["opponent_character"]).squeeze(-2),
                self.action_emb(inputs["ego_action"]).squeeze(-2),
                self.action_emb(inputs["opponent_action"]).squeeze(-2),
                inputs["gamestate"],
                inputs["controller"],
            ],
            dim=-1,
        )

    def forward(self, inputs: TensorDict):
        B, L, _ = inputs["gamestate"].shape
        assert L <= self.block_size, f"Cannot forward sequence of length {L}, block size is only {self.block_size}"

        # concatenate embeddings and numerical inputs -> project down
        combined_inputs_BLG = self._embed_inputs(inputs)
        proj_inputs_BLD = self.transformer.proj_down(combined_inputs_BLG)

        # position embeddings
        pos_L = torch.arange(0, L, dtype=torch.long, device=next(self.parameters()).device)
        pos_emb_LD = self.transformer.wpe(pos_L)

        x_BLD = self.transformer.drop(proj_inputs_BLD + pos_emb_LD)
        for block in self.transformer.h:
            x_BLD = block(x_BLD)
        x_BLD = self.transformer.ln_f(x_BLD)

        c_stick = self.c_stick_head(x_BLD)
        main_stick = self.main_stick_head(torch.cat((x_BLD, c_stick.detach()), dim=-1))
        button = self.button_head(torch.cat((x_BLD, c_stick.detach(), main_stick.detach()), dim=-1))

        return TensorDict(
            {
                "buttons": button,
                "main_stick": main_stick,
                "c_stick": c_stick,
            },
            batch_size=(B, L),
        )


class GPTv2_1Controller(BaseGPT):
    """Autoregressive MLP output heads, absolute position embeddings."""

    def __init__(self, preprocessor: Preprocessor, gpt_config: GPTConfig) -> None:
        super().__init__(preprocessor, gpt_config)
        # Numeric + embedded feature sizes defined programmatically in InputPreprocessConfig
        self.input_size = self.preprocessor.input_size  # G
        self.n_embd = gpt_config.n_embd  # D

        # categorical input embeddings
        self.emb_config = self.preprocessor.data_config
        self.stage_emb = nn.Embedding(self.emb_config.num_stages, self.emb_config.stage_embedding_dim)
        self.character_emb = nn.Embedding(self.emb_config.num_characters, self.emb_config.character_embedding_dim)
        self.action_emb = nn.Embedding(self.emb_config.num_actions, self.emb_config.action_embedding_dim)

        self.transformer = nn.ModuleDict(
            dict(
                proj_down=nn.Linear(self.input_size, gpt_config.n_embd),  # G -> D
                wpe=nn.Embedding(self.block_size, gpt_config.n_embd),
                drop=nn.Dropout(gpt_config.dropout),
                h=nn.ModuleList([Block(gpt_config) for _ in range(gpt_config.n_layer)]),
                ln_f=nn.LayerNorm(self.n_embd, bias=gpt_config.bias),
            )
        )

        # output heads
        self.target_shapes_by_head = self.preprocessor.target_config.target_shapes_by_head
        main_stick_size = self.target_shapes_by_head["main_stick"][0]
        button_size = self.target_shapes_by_head["buttons"][0]
        c_stick_size = self.target_shapes_by_head["c_stick"][0]

        # Put c-stick first because it overrides button inputs, other heads can choose to fire if c-stick is inactive
        self.c_stick_head = nn.Sequential(
            nn.LayerNorm(self.n_embd, bias=gpt_config.bias),
            nn.Linear(self.n_embd, self.n_embd // 2),
            nn.GELU(),
            nn.Linear(self.n_embd // 2, c_stick_size),
        )

        main_stick_input_size = self.n_embd + c_stick_size
        self.main_stick_head = nn.Sequential(
            nn.LayerNorm(main_stick_input_size, bias=gpt_config.bias),
            nn.Linear(main_stick_input_size, main_stick_input_size // 2),
            nn.GELU(),
            nn.Linear(main_stick_input_size // 2, main_stick_size),
        )

        button_input_size = self.n_embd + main_stick_size + c_stick_size
        self.button_head = nn.Sequential(
            nn.LayerNorm(button_input_size, bias=gpt_config.bias),
            nn.Linear(button_input_size, button_input_size // 2),
            nn.GELU(),
            nn.Linear(button_input_size // 2, button_size),
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
                self.stage_emb(inputs["stage"]).squeeze(-2),
                self.character_emb(inputs["ego_character"]).squeeze(-2),
                self.character_emb(inputs["opponent_character"]).squeeze(-2),
                self.action_emb(inputs["ego_action"]).squeeze(-2),
                self.action_emb(inputs["opponent_action"]).squeeze(-2),
                inputs["gamestate"],
                inputs["controller"],
            ],
            dim=-1,
        )

    def forward(self, inputs: TensorDict):
        B, L, _ = inputs["gamestate"].shape
        assert L <= self.block_size, f"Cannot forward sequence of length {L}, block size is only {self.block_size}"

        # concatenate embeddings and numerical inputs -> project down
        combined_inputs_BLG = self._embed_inputs(inputs)
        proj_inputs_BLD = self.transformer.proj_down(combined_inputs_BLG)

        # position embeddings
        pos_L = torch.arange(0, L, dtype=torch.long, device=next(self.parameters()).device)
        pos_emb_LD = self.transformer.wpe(pos_L)

        x_BLD = self.transformer.drop(proj_inputs_BLD + pos_emb_LD)
        for block in self.transformer.h:
            x_BLD = block(x_BLD)
        x_BLD = self.transformer.ln_f(x_BLD)

        c_stick = self.c_stick_head(x_BLD)
        main_stick = self.main_stick_head(torch.cat((x_BLD, c_stick.detach()), dim=-1))
        button = self.button_head(torch.cat((x_BLD, c_stick.detach(), main_stick.detach()), dim=-1))

        return TensorDict(
            {
                "buttons": button,
                "main_stick": main_stick,
                "c_stick": c_stick,
            },
            batch_size=(B, L),
        )


class RelativePosition(nn.Module):
    """
    Relative Position Embeddings from Shaw et al. (2018)
    https://arxiv.org/abs/1803.02155

    Slow and memory-inefficient, materializes O(T^2) matrix.
    """

    def __init__(self, head_dim: int, max_relative_position: int) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, head_dim))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q: int, length_k: int) -> torch.Tensor:
        """Returns a tensor of shape (L, L, head_dim)"""
        # Wrap indexing in torch.no_grad() to avoid unnecessary gradient computation
        with torch.no_grad():
            range_vec_q = torch.arange(length_q)
            range_vec_k = torch.arange(length_k)
            distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
            distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
            final_mat = (distance_mat_clipped + self.max_relative_position).long()
        embeddings = self.embeddings_table[final_mat]

        return embeddings


def skew(QEr: torch.Tensor) -> torch.Tensor:
    """
    Memory-efficient "skewing" trick to avoid materializing O(T^2) `R` matrix.

    Music Transformer, Huang et al. (2018) https://arxiv.org/abs/1809.04281
    Implementation by: https://jaketae.github.io/study/relative-positional-encoding/
    """
    # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
    padded = F.pad(QEr, (1, 0))
    # padded.shape = (batch_size, num_heads, seq_len, 1 + seq_len)
    batch_size, num_heads, num_rows, num_cols = padded.shape
    reshaped = padded.reshape(batch_size, num_heads, num_cols, num_rows)
    # reshaped.size = (batch_size, num_heads, 1 + seq_len, seq_len)
    Srel = reshaped[:, :, 1:, :]
    # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
    return Srel


class CausalSelfAttentionRelativePosition(nn.Module):
    def __init__(self, config: GPTConfig, input_size: int | None = None) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config
        self.block_size = config.block_size
        self.n_embd = input_size or config.n_embd
        self.n_head = config.n_head
        self.hs = self.n_embd // config.n_head
        self.dropout = config.dropout

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)
        # relative positional embedding table of shape (L, hs)
        self.Er = nn.Parameter(torch.randn(self.block_size, self.hs))
        # regularization
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(self.block_size, self.block_size)).view(1, 1, self.block_size, self.block_size),
        )
        # disable flash attention,
        self.flash = False

    def forward(self, x: torch.Tensor):
        B, L, D = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        assert L <= self.block_size, f"Cannot forward sequence of length {L}, block size is only {self.block_size}"

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, L, self.n_head, self.hs).transpose(1, 2)  # (B, nh, L, hs)
        q = q.view(B, L, self.n_head, self.hs).transpose(1, 2)  # (B, nh, L, hs)
        v = v.view(B, L, self.n_head, self.hs).transpose(1, 2)  # (B, nh, L, hs)

        # relative positional embeddings
        start = self.block_size - L
        Er_t = self.Er[start:, :].transpose(0, 1)  # (hs, L)
        QEr = q @ Er_t  # (B, nh, L, hs) x (hs, L) -> (B, nh, L, L)
        Srel = skew(QEr)  # (B, nh, L, L)

        # causal self-attention
        QK_t = q @ k.transpose(-2, -1)  # (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        scale = 1.0 / math.sqrt(k.size(-1))
        att = (QK_t + Srel) * scale
        att = att.masked_fill(self.bias[:, :, :L, :L] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)
        y = y.transpose(1, 2).contiguous().view(B, L, D)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class BlockRelativePosition(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttentionRelativePosition(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTv3(BaseGPT):
    """
    Relative position embeddings, independent output heads.
    """

    def __init__(self, preprocessor: Preprocessor, gpt_config: GPTConfig) -> None:
        super().__init__(preprocessor, gpt_config)
        # Numeric + embedded feature sizes defined programmatically in InputPreprocessConfig
        self.input_size = self.preprocessor.input_size  # G
        self.n_embd = gpt_config.n_embd  # D

        # categorical input embeddings
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

        # output heads
        self.target_shapes_by_head = self.preprocessor.target_config.target_shapes_by_head
        self.button_head = nn.Linear(self.n_embd, self.target_shapes_by_head["buttons"][0], bias=False)
        self.main_stick_head = nn.Linear(self.n_embd, self.target_shapes_by_head["main_stick"][0], bias=False)
        self.c_stick_head = nn.Linear(self.n_embd, self.target_shapes_by_head["c_stick"][0], bias=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * gpt_config.n_layer))

    def _embed_inputs(self, inputs: TensorDict) -> torch.Tensor:
        return torch.cat(
            [
                self.stage_emb(inputs["stage"]).squeeze(-2),
                self.character_emb(inputs["ego_character"]).squeeze(-2),
                self.character_emb(inputs["opponent_character"]).squeeze(-2),
                self.action_emb(inputs["ego_action"]).squeeze(-2),
                self.action_emb(inputs["opponent_action"]).squeeze(-2),
                inputs["gamestate"],
            ],
            dim=-1,
        )

    def forward(self, inputs: TensorDict):
        B, L, _ = inputs["gamestate"].shape
        assert L <= self.block_size, f"Cannot forward sequence of length {L}, block size is only {self.block_size}"

        # concatenate embeddings and numerical inputs -> project down
        combined_inputs_BLG = self._embed_inputs(inputs)
        proj_inputs_BLD = self.transformer.proj_down(combined_inputs_BLG)

        x_BLD = self.transformer.drop(proj_inputs_BLD)
        for block in self.transformer.h:
            x_BLD = block(x_BLD)
        x_BLD = self.transformer.ln_f(x_BLD)

        return TensorDict(
            {
                "buttons": self.button_head(x_BLD).squeeze(-2),
                "main_stick": self.main_stick_head(x_BLD).squeeze(-2),
                "c_stick": self.c_stick_head(x_BLD).squeeze(-2),
            },
            batch_size=(B, L),
        )


class GPTv3Controller(GPTv3):
    def _embed_inputs(self, inputs: TensorDict) -> torch.Tensor:
        return torch.cat(
            [
                self.stage_emb(inputs["stage"]).squeeze(-2),
                self.character_emb(inputs["ego_character"]).squeeze(-2),
                self.character_emb(inputs["opponent_character"]).squeeze(-2),
                self.action_emb(inputs["ego_action"]).squeeze(-2),
                self.action_emb(inputs["opponent_action"]).squeeze(-2),
                inputs["gamestate"],
                inputs["controller"],
            ],
            dim=-1,
        )


class GPTv4Controller(BaseGPT):
    """Relative positional embeddings, autoregressive MLP output heads."""

    def __init__(self, preprocessor: Preprocessor, gpt_config: GPTConfig) -> None:
        super().__init__(preprocessor, gpt_config)
        # Numeric + embedded feature sizes defined programmatically in InputPreprocessConfig
        self.input_size = self.preprocessor.input_size  # G
        self.n_embd = gpt_config.n_embd  # D

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
        main_stick_size = self.target_shapes_by_head["main_stick"][0]
        button_size = self.target_shapes_by_head["buttons"][0]
        c_stick_size = self.target_shapes_by_head["c_stick"][0]

        # Put c-stick first because it overrides button inputs, other heads can choose to fire if c-stick is inactive
        self.c_stick_head = nn.Sequential(
            nn.LayerNorm(self.n_embd, bias=gpt_config.bias),
            nn.Linear(self.n_embd, self.n_embd // 2),
            nn.GELU(),
            nn.Linear(self.n_embd // 2, c_stick_size),
        )

        main_stick_input_size = self.n_embd + c_stick_size
        self.main_stick_head = nn.Sequential(
            nn.LayerNorm(main_stick_input_size, bias=gpt_config.bias),
            nn.Linear(main_stick_input_size, main_stick_input_size // 2),
            nn.GELU(),
            nn.Linear(main_stick_input_size // 2, main_stick_size),
        )

        button_input_size = self.n_embd + main_stick_size + c_stick_size
        self.button_head = nn.Sequential(
            nn.LayerNorm(button_input_size, bias=gpt_config.bias),
            nn.Linear(button_input_size, button_input_size // 2),
            nn.GELU(),
            nn.Linear(button_input_size // 2, button_size),
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
                self.stage_emb(inputs["stage"]).squeeze(-2),
                self.character_emb(inputs["ego_character"]).squeeze(-2),
                self.character_emb(inputs["opponent_character"]).squeeze(-2),
                self.action_emb(inputs["ego_action"]).squeeze(-2),
                self.action_emb(inputs["opponent_action"]).squeeze(-2),
                inputs["gamestate"],
                inputs["controller"],
            ],
            dim=-1,
        )

    def forward(self, inputs: TensorDict):
        B, L, _ = inputs["gamestate"].shape
        assert L <= self.block_size, f"Cannot forward sequence of length {L}, block size is only {self.block_size}"

        # concatenate embeddings and numerical inputs -> project down
        combined_inputs_BLG = self._embed_inputs(inputs)
        proj_inputs_BLD = self.transformer.proj_down(combined_inputs_BLG)

        x_BLD = self.transformer.drop(proj_inputs_BLD)
        for block in self.transformer.h:
            x_BLD = block(x_BLD)
        x_BLD = self.transformer.ln_f(x_BLD)

        c_stick = self.c_stick_head(x_BLD)
        main_stick = self.main_stick_head(torch.cat((x_BLD, c_stick.detach()), dim=-1))
        button = self.button_head(torch.cat((x_BLD, c_stick.detach(), main_stick.detach()), dim=-1))

        return TensorDict(
            {
                "buttons": button,
                "main_stick": main_stick,
                "c_stick": c_stick,
            },
            batch_size=(B, L),
        )


class GPTv5Controller(GPTv4Controller):
    def __init__(self, preprocessor: Preprocessor, gpt_config: GPTConfig) -> None:
        super().__init__(preprocessor, gpt_config)
        # Numeric + embedded feature sizes defined programmatically in InputPreprocessConfig
        self.input_size = self.preprocessor.input_size  # G
        self.n_embd = gpt_config.n_embd  # D

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
        shoulder_output_size = self.target_shapes_by_head["shoulder"][0]

        c_stick_input_size = self.n_embd + shoulder_output_size
        c_stick_output_size = self.target_shapes_by_head["c_stick"][0]

        main_stick_input_size = self.n_embd + shoulder_output_size + c_stick_output_size
        main_stick_output_size = self.target_shapes_by_head["main_stick"][0]

        button_input_size = self.n_embd + shoulder_output_size + c_stick_output_size + main_stick_output_size
        button_output_size = self.target_shapes_by_head["buttons"][0]

        # Put shoulder and c-stick first because they are less complex and they modify/override other inputs
        self.shoulder_head = nn.Sequential(
            nn.LayerNorm(self.n_embd, bias=gpt_config.bias),
            nn.Linear(self.n_embd, self.n_embd // 2),
            nn.GELU(),
            nn.Linear(self.n_embd // 2, shoulder_output_size),
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

        # Detach to avoid multiplying gradient flow through earlier heads
        shoulder: torch.Tensor = self.shoulder_head(x_BLD)
        c_stick: torch.Tensor = self.c_stick_head(torch.cat((x_BLD, shoulder.detach()), dim=-1))
        main_stick: torch.Tensor = self.main_stick_head(
            torch.cat((x_BLD, shoulder.detach(), c_stick.detach()), dim=-1)
        )
        button: torch.Tensor = self.button_head(
            torch.cat((x_BLD, shoulder.detach(), c_stick.detach(), main_stick.detach()), dim=-1)
        )

        return TensorDict(
            {
                "buttons": button,
                "main_stick": main_stick,
                "c_stick": c_stick,
                "shoulder": shoulder,
            },
            batch_size=(B, L),
        )


@attr.s(auto_attribs=True, frozen=True)
class MultiTokenGPTConfig(GPTConfig):
    n_lookahead: int = 4


class MultiTokenGPT(GPTv1):
    """Predict `n_lookahead` tokens from input sequence."""

    def __init__(self, train_config: TrainConfig, gpt_config: MultiTokenGPTConfig) -> None:
        super().__init__()
        ...

    def forward(self, inputs: TensorDict) -> None:
        ...


# Shallow output heads, absolute position embeddings
Arch.register("GPTv1-256-4-4", GPTv1, gpt_config=GPTConfig(block_size=1024, n_embd=256, n_layer=4, n_head=4))
Arch.register("GPTv1-256-8-4", GPTv1, gpt_config=GPTConfig(block_size=1024, n_embd=256, n_layer=8, n_head=4))
Arch.register(
    "GPTv1-256-8-4-dropout", GPTv1, gpt_config=GPTConfig(block_size=1024, n_embd=256, n_layer=8, n_head=4, dropout=0.1)
)
Arch.register("GPTv1-256-12-4", GPTv1, gpt_config=GPTConfig(block_size=1024, n_embd=256, n_layer=12, n_head=4))
Arch.register(
    "GPTv1-256-12-4-dropout",
    GPTv1,
    gpt_config=GPTConfig(block_size=1024, n_embd=256, n_layer=12, n_head=4, dropout=0.1),
)

Arch.register(
    "GPTv1Controller-256-4-4", GPTv1Controller, gpt_config=GPTConfig(block_size=1024, n_embd=256, n_layer=4, n_head=4)
)

# AR shallow output heads, absolute position embeddings
Arch.register(
    "GPTv2Controller-256-4-4", GPTv2Controller, gpt_config=GPTConfig(block_size=1024, n_embd=256, n_layer=4, n_head=4)
)
Arch.register(
    "GPTv2Controller-256-12-4-dropout",
    GPTv2Controller,
    gpt_config=GPTConfig(block_size=1024, n_embd=256, n_layer=12, n_head=4, dropout=0.1),
)
Arch.register(
    "GPTv2Controller-512-6-8-dropout",
    GPTv2Controller,
    gpt_config=GPTConfig(block_size=1024, n_embd=512, n_layer=6, n_head=8, dropout=0.2),
)

# AR MLP output heads, absolute position embeddings
Arch.register(
    "GPTv2_1Controller-512-6-8-dropout",
    GPTv2_1Controller,
    gpt_config=GPTConfig(block_size=1024, n_embd=512, n_layer=6, n_head=8, dropout=0.2),
)

Arch.register("GPTv3-256-4-4", GPTv3, gpt_config=GPTConfig(block_size=1024, n_embd=256, n_layer=4, n_head=4))
Arch.register(
    "GPTv3Controller-256-4-4", GPTv3Controller, gpt_config=GPTConfig(block_size=1024, n_embd=256, n_layer=4, n_head=4)
)

# AR MLP output heads, relative positional embeddings
Arch.register(
    "GPTv4Controller-256-4-4", GPTv4Controller, gpt_config=GPTConfig(block_size=1024, n_embd=256, n_layer=4, n_head=4)
)
Arch.register(
    "GPTv4Controller-256-4-4-dropout",
    GPTv4Controller,
    gpt_config=GPTConfig(block_size=1024, n_embd=256, n_layer=4, n_head=4, dropout=0.2),
)
Arch.register(
    "GPTv4Controller-256-8-4", GPTv4Controller, gpt_config=GPTConfig(block_size=1024, n_embd=256, n_layer=8, n_head=4)
)
Arch.register(
    "GPTv4Controller-256-8-4-dropout",
    GPTv4Controller,
    gpt_config=GPTConfig(block_size=1024, n_embd=256, n_layer=8, n_head=4, dropout=0.2),
)
Arch.register(
    "GPTv4Controller-384-6-8-dropout",
    GPTv4Controller,
    gpt_config=GPTConfig(block_size=1024, n_embd=384, n_layer=6, n_head=8, dropout=0.2),
)
# This OOMs with batch 256 on 24GB GPU
Arch.register(
    "GPTv4Controller-512-6-8-dropout",
    GPTv4Controller,
    gpt_config=GPTConfig(block_size=1024, n_embd=512, n_layer=6, n_head=8, dropout=0.2),
)
Arch.register(
    "GPTv4Controller-512-8-8-dropout",
    GPTv4Controller,
    gpt_config=GPTConfig(block_size=1024, n_embd=512, n_layer=8, n_head=8, dropout=0.2),
)

# AR MLP output heads + analog shoulder, relative positional embeddings
Arch.register(
    "GPTv5Controller-256-4-4-dropout",
    GPTv5Controller,
    gpt_config=GPTConfig(block_size=1024, n_embd=256, n_layer=4, n_head=4, dropout=0.2),
)
Arch.register(
    "GPTv5Controller-256-6-4-dropout",
    GPTv5Controller,
    gpt_config=GPTConfig(block_size=1024, n_embd=256, n_layer=6, n_head=4, dropout=0.2),
)
Arch.register(
    "GPTv5Controller-512-6-8-dropout",
    GPTv5Controller,
    gpt_config=GPTConfig(block_size=1024, n_embd=512, n_layer=6, n_head=8, dropout=0.2),
)
Arch.register(
    "GPTv5Controller-256-8-4-dropout",
    GPTv5Controller,
    gpt_config=GPTConfig(block_size=1024, n_embd=256, n_layer=8, n_head=4, dropout=0.2),
)
