"""Adapted from Karpathy's nanoGPT: https://github.com/karpathy/nanoGPT."""
import math

import attr
import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.nn import functional as F
from training.preprocess.preprocessor import Preprocessor

from hal.training.config import TrainConfig
from hal.training.models.registry import Arch


@attr.s(auto_attribs=True, frozen=True)
class GPTConfig:
    n_embd: int
    n_layer: int
    n_head: int
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False."""

    def __init__(self, ndim, bias) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, context_length: int, dropout: float, bias: bool) -> None:
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(context_length, context_length)).view(1, 1, context_length, context_length),
            )

    def forward(self, x: torch.Tensor):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float, bias: bool) -> None:
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, context_length: int, dropout: float, bias: bool) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.attn = CausalSelfAttention(
            n_embd=n_embd,
            n_head=n_head,
            context_length=context_length,
            dropout=dropout,
            bias=bias,
        )
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd=n_embd, dropout=dropout, bias=bias)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class BaseGPT(nn.Module):
    def __init__(self, preprocessor: Preprocessor, gpt_config: GPTConfig) -> None:
        super().__init__()
        self.preprocessor = preprocessor
        self.gpt_config = gpt_config

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
        assert block_size <= self.context_length
        self.context_length = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float):
        """Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS."""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        L, H, Q, T = (
            self.gpt_config.n_layer,
            self.gpt_config.n_head,
            self.n_embd // self.gpt_config.n_head,
            self.context_length,
        )
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.context_size else idx[:, -self.config.context_size :]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


class GPTv1(BaseGPT):
    def __init__(self, preprocessor: Preprocessor, gpt_config: GPTConfig) -> None:
        super().__init__(preprocessor, gpt_config)
        self.context_len = self.preprocessor.seq_len
        self.input_size = self.preprocessor.input_size
        self.n_embd = gpt_config.n_embd

        embed_config = self.preprocessor.embedding_config
        assert embed_config.num_buttons is not None
        assert embed_config.num_main_stick_clusters is not None
        assert embed_config.num_c_stick_clusters is not None

        self.transformer = nn.ModuleDict(
            dict(
                stage=nn.Embedding(embed_config.num_stages, embed_config.stage_embedding_dim),
                character=nn.Embedding(embed_config.num_characters, embed_config.character_embedding_dim),
                action=nn.Embedding(embed_config.num_actions, embed_config.action_embedding_dim),
                drop=nn.Dropout(gpt_config.dropout),
                # TODO get input size from preprocess config or preprocessor
                proj_down=nn.Linear(self.input_size, gpt_config.n_embd),
                wpe=nn.Embedding(self.context_len, gpt_config.n_embd),
                h=nn.ModuleList(
                    [
                        Block(
                            n_embd=gpt_config.n_embd,
                            n_head=gpt_config.n_head,
                            context_length=self.context_len,
                            dropout=gpt_config.dropout,
                            bias=gpt_config.bias,
                        )
                        for _ in range(gpt_config.n_layer)
                    ]
                ),
                ln_f=LayerNorm(self.n_embd, bias=gpt_config.bias),
            )
        )
        self.button_head = nn.Linear(self.n_embd, embed_config.num_buttons, bias=False)
        self.main_stick_head = nn.Linear(self.n_embd, embed_config.num_main_stick_clusters, bias=False)
        self.c_stick_head = nn.Linear(self.n_embd, embed_config.num_c_stick_clusters, bias=False)

        # TODO investigate weight tying
        # self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * gpt_config.n_layer))

    def forward(self, inputs: TensorDict):
        B, T, D = inputs["gamestate"].shape
        assert T <= self.context_len, f"Cannot forward sequence of length {T}, block size is only {self.context_len}"
        pos = torch.arange(0, T, dtype=torch.long, device=next(self.parameters()).device)  # shape (t)

        combined_inputs = torch.cat(
            [
                self.transformer.stage(inputs["stage"]).squeeze(-2),
                self.transformer.character(inputs["ego_character"]).squeeze(-2),
                self.transformer.character(inputs["opponent_character"]).squeeze(-2),
                self.transformer.action(inputs["ego_action"]).squeeze(-2),
                self.transformer.action(inputs["opponent_action"]).squeeze(-2),
                inputs["gamestate"],
            ],
            dim=-1,
        )
        proj_inputs = self.transformer.proj_down(combined_inputs)

        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(proj_inputs + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        return TensorDict(
            {
                "buttons": self.button_head(x).squeeze(-2),
                "main_stick": self.main_stick_head(x).squeeze(-2),
                "c_stick": self.c_stick_head(x).squeeze(-2),
            },
            batch_size=(B, T),
        )


class GPTv2(BaseGPT):
    def __init__(self, preprocessor: Preprocessor, gpt_config: GPTConfig) -> None:
        super().__init__(preprocessor, gpt_config)
        self.context_len = self.preprocessor.seq_len
        self.input_size = self.preprocessor.input_size
        self.n_embd = gpt_config.n_embd

        embed_config = self.preprocessor.embedding_config
        assert embed_config.num_buttons is not None
        assert embed_config.num_main_stick_clusters is not None
        assert embed_config.num_c_stick_clusters is not None

        self.transformer = nn.ModuleDict(
            dict(
                stage=nn.Embedding(embed_config.num_stages, embed_config.stage_embedding_dim),
                character=nn.Embedding(embed_config.num_characters, embed_config.character_embedding_dim),
                action=nn.Embedding(embed_config.num_actions, embed_config.action_embedding_dim),
                drop=nn.Dropout(gpt_config.dropout),
                # TODO get input size from preprocess config or preprocessor
                proj_down=nn.Linear(self.input_size, gpt_config.n_embd),
                wpe=nn.Embedding(self.context_len, gpt_config.n_embd),
                h=nn.ModuleList(
                    [
                        Block(
                            n_embd=gpt_config.n_embd,
                            n_head=gpt_config.n_head,
                            context_length=self.context_len,
                            dropout=gpt_config.dropout,
                            bias=gpt_config.bias,
                        )
                        for _ in range(gpt_config.n_layer)
                    ]
                ),
                ln_f=LayerNorm(self.n_embd, bias=gpt_config.bias),
            )
        )
        main_stick_size = embed_config.num_main_stick_clusters
        button_size = embed_config.num_buttons
        c_stick_size = embed_config.num_c_stick_clusters
        self.main_stick_head = nn.Linear(self.n_embd, main_stick_size, bias=False)
        self.button_head = nn.Linear(self.n_embd + main_stick_size, button_size, bias=False)
        self.c_stick_head = nn.Linear(self.n_embd + main_stick_size + button_size, c_stick_size, bias=False)

        # TODO investigate weight tying
        # self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * gpt_config.n_layer))

    def forward(self, inputs: TensorDict):
        B, T, G = inputs["gamestate"].shape
        assert T <= self.context_len, f"Cannot forward sequence of length {T}, block size is only {self.context_len}"
        pos = torch.arange(0, T, dtype=torch.long, device=next(self.parameters()).device)  # shape (t)

        combined_inputs_BTG = torch.cat(
            [
                self.transformer.stage(inputs["stage"]).squeeze(-2),
                self.transformer.character(inputs["ego_character"]).squeeze(-2),
                self.transformer.character(inputs["opponent_character"]).squeeze(-2),
                self.transformer.action(inputs["ego_action"]).squeeze(-2),
                self.transformer.action(inputs["opponent_action"]).squeeze(-2),
                inputs["gamestate"],
                inputs["controller"],
            ],
            dim=-1,
        )
        proj_inputs_BTD = self.transformer.proj_down(combined_inputs_BTG)

        pos_emb_TD = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x_BTD = self.transformer.drop(proj_inputs_BTD + pos_emb_TD)
        for block in self.transformer.h:
            x_BTD = block(x_BTD)
        x_BTD = self.transformer.ln_f(x_BTD)

        main_stick = self.main_stick_head(x_BTD)
        button = self.button_head(torch.cat((x_BTD, main_stick), dim=-1))
        c_stick = self.c_stick_head(torch.cat((x_BTD, main_stick, button), dim=-1))

        return TensorDict(
            {
                "buttons": button,
                "main_stick": main_stick,
                "c_stick": c_stick,
            },
            batch_size=(B, T),
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
        """Returns a tensor of shape (T, T, head_dim)"""
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
    def __init__(self, n_embd: int, n_head: int, context_length: int, dropout: float, bias: bool) -> None:
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.hs = n_embd // n_head  # head size
        self.context_length = context_length

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)

        # relative positional embedding table of shape (T, hs)
        self.Er = nn.Parameter(torch.randn(context_length, self.hs))

        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(context_length, context_length)).view(1, 1, context_length, context_length),
        )
        # disable flash attention,
        self.flash = False

    def forward(self, x: torch.Tensor):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        assert (
            T <= self.context_length
        ), f"Cannot forward sequence of length {T}, block size is only {self.context_length}"

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.hs).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.hs).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.hs).transpose(1, 2)  # (B, nh, T, hs)

        # relative positional embeddings
        Er_t = self.Er.T  # (hs, T)
        QEr = q @ Er_t  # (B, nh, T, hs) x (hs, T) -> (B, nh, T, T)
        Srel = skew(QEr)  # (B, nh, T, T)

        # causal self-attention
        QK_t = q @ k.transpose(-2, -1)  # (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        scale = 1.0 / math.sqrt(k.size(-1))
        att = (QK_t + Srel) * scale
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class GPTv3(BaseGPT):
    def __init__(self, preprocessor: Preprocessor, gpt_config: GPTConfig) -> None:
        super().__init__(preprocessor, gpt_config)
        ...


@attr.s(auto_attribs=True, frozen=True)
class MultiTokenGPTConfig(GPTConfig):
    n_lookahead: int = 4


class MultiTokenGPT(GPTv1):
    """Predict `n_lookahead` tokens from input sequence."""

    def __init__(self, train_config: TrainConfig, gpt_config: MultiTokenGPTConfig) -> None:
        super().__init__()
        embed_config = train_config.embedding
        assert embed_config.num_buttons is not None
        assert embed_config.num_main_stick_clusters is not None
        assert embed_config.num_c_stick_clusters is not None
        self.n_embd = get_input_size_from_config(embed_config)
        self.context_length = train_config.data.input_len

        self.train_config = train_config
        self.gpt_config = gpt_config

        self.transformer = nn.ModuleDict(
            dict(
                stage=nn.Embedding(embed_config.num_stages, embed_config.stage_embedding_dim),
                character=nn.Embedding(embed_config.num_characters, embed_config.character_embedding_dim),
                action=nn.Embedding(embed_config.num_actions, embed_config.action_embedding_dim),
                wpe=nn.Embedding(self.context_length, self.n_embd),
                drop=nn.Dropout(gpt_config.dropout),
                h=nn.ModuleList(
                    [
                        Block(
                            n_embd=self.n_embd,
                            n_head=gpt_config.n_head,
                            context_length=self.context_length,
                            dropout=gpt_config.dropout,
                            bias=gpt_config.bias,
                        )
                        for _ in range(gpt_config.n_layer)
                    ]
                ),
                ln_f=LayerNorm(self.n_embd, bias=gpt_config.bias),
            )
        )
        self.out_heads = nn.ModuleDict(
            {
                i: dict(
                    button_head=nn.Linear(self.n_embd, embed_config.num_buttons, bias=False),
                    main_stick_head=nn.Linear(self.n_embd, embed_config.num_main_stick_clusters, bias=False),
                    c_stick_head=nn.Linear(self.n_embd, embed_config.num_c_stick_clusters, bias=False),
                )
                for i in range(self.gpt_config.n_lookahead)
            }
        )

        # TODO investigate weight tying
        # self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * gpt_config.n_layer))

    def forward(self, inputs: TensorDict):
        B, T, D = inputs["gamestate"].shape
        assert (
            T <= self.context_length
        ), f"Cannot forward sequence of length {T}, block size is only {self.context_length}"
        pos = torch.arange(0, T, dtype=torch.long, device=next(self.parameters()).device)  # shape (t)

        # Embeddings
        stage_emb = self.transformer.stage(inputs["stage"]).squeeze(-2)
        ego_character_emb = self.transformer.character(inputs["ego_character"]).squeeze(-2)
        opponent_character_emb = self.transformer.character(inputs["opponent_character"]).squeeze(-2)
        ego_action_emb = self.transformer.action(inputs["ego_action"]).squeeze(-2)
        opponent_action_emb = self.transformer.action(inputs["opponent_action"]).squeeze(-2)
        gamestate = inputs["gamestate"]
        combined_inputs = torch.cat(
            [stage_emb, ego_character_emb, opponent_character_emb, ego_action_emb, opponent_action_emb, gamestate],
            dim=-1,
        )

        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(combined_inputs + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        multi_logit_dict = {}
        for i in range(self.gpt_config.n_lookahead):
            multi_logit_dict[i] = dict(
                button_logits=self.out_heads.get(i).button_head(x).squeeze(-2),
                main_stick_logits=self.out_heads.get(i).main_stick_head(x).squeeze(-2),
                c_stick_logits=self.out_heads.get(i).c_stick_head(x).squeeze(-2),
            )

        # TODO inference-time mini-optimization: only forward output heads on the very last position
        # logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim

        return TensorDict(multi_logit_dict, batch_size=(B, T))


Arch.register("GPTv1-4-4", GPTv1, gpt_config=GPTConfig(n_embd=256, n_layer=4, n_head=4))
Arch.register("GPTv1-8-4", GPTv1, gpt_config=GPTConfig(n_embd=256, n_layer=8, n_head=4))
Arch.register("GPTv1-8-4-dropout", GPTv1, gpt_config=GPTConfig(n_embd=256, n_layer=8, n_head=4, dropout=0.1))
Arch.register("GPTv1-12-4", GPTv1, gpt_config=GPTConfig(n_embd=256, n_layer=12, n_head=4))
Arch.register("GPTv1-12-4-dropout", GPTv1, gpt_config=GPTConfig(n_embd=256, n_layer=12, n_head=4, dropout=0.1))
Arch.register("GPTv1-12-512-4-dropout", GPTv1, gpt_config=GPTConfig(n_embd=512, n_layer=12, n_head=4, dropout=0.1))

Arch.register("GPTv2-4-4", GPTv2, gpt_config=GPTConfig(n_embd=256, n_layer=4, n_head=4))
Arch.register("GPTv2-12-4-dropout", GPTv2, gpt_config=GPTConfig(n_embd=256, n_layer=12, n_head=4, dropout=0.1))
