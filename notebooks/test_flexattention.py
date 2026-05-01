# %%
import math

import torch
from torch import nn

# Import flex_attention (requires PyTorch 2.1+ with FlexAttention support)
from torch.nn.attention.flex_attention import flex_attention


class FlexCausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, max_position: int) -> None:
        """
        Causal self-attention with relative positions using FlexAttention.
        embed_dim: total embedding dimension (will be split across heads)
        num_heads: number of attention heads
        max_position: maximum sequence length (defines range for relative positions)
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim: int = embed_dim
        self.num_heads: int = num_heads
        self.head_dim: int = embed_dim // num_heads
        self.max_position: int = max_position  # e.g.,  max seq length or max relative index + 1

        # Projectors for queries, keys, values, and output
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj_out = nn.Linear(embed_dim, embed_dim)

        # Relative positional embeddings for keys (for distances 0 to -max_position+1)
        # Shape: (max_position, head_dim). Index max_position-1 corresponds to distance 0.
        self.rel_key = nn.Parameter(torch.randn(max_position, self.head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (batch_size, seq_len, embed_dim)
        Returns: Tensor of shape (batch_size, seq_len, embed_dim)
        """
        B, seq_len, _ = x.shape
        if seq_len > self.max_position:
            raise ValueError(f"Sequence length {seq_len} exceeds max_position {self.max_position}")

        # Compute Q, K, V projections and reshape for multi-head attention
        q: torch.Tensor = self.query(x)  # (B, seq_len, embed_dim)
        k: torch.Tensor = self.key(x)  # (B, seq_len, embed_dim)
        v: torch.Tensor = self.value(x)  # (B, seq_len, embed_dim)
        # Reshape to (B, num_heads, seq_len, head_dim)
        q = q.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, seq_len, head_dim)
        k = k.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, seq_len, head_dim)
        v = v.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, seq_len, head_dim)

        # Prepare scaling factor
        scale: float = 1.0 / math.sqrt(self.head_dim)

        # Define score_mod function for FlexAttention
        # Inputs:
        #  - score: scalar attention score (dot product of Q and K for a specific q_idx, kv_idx)
        #  - b, h: batch index and head index (int32 scalar indices)
        #  - q_idx, kv_idx: positions of the query and key (int32 scalar indices)
        def score_mod(
            score: torch.Tensor, b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ) -> torch.Tensor:
            # Apply causal mask: if key position is ahead of query, return -inf (mask out)
            # q_idx >= kv_idx means key is not after query (allowed in causal attention)
            score = torch.where(q_idx >= kv_idx, score, torch.tensor(-float("inf"), device=score.device))
            # Add relative positional bias for allowed positions
            # Compute index in rel_key table: (kv_idx - q_idx) + (max_position - 1)
            # For kv_idx <= q_idx, (kv_idx - q_idx) is <= 0, offsetting by max_position-1 yields a valid index [0, max_position-1]
            rel_index = (kv_idx - q_idx) + (self.max_position - 1)
            # Fetch relative embedding vector of shape (head_dim,)
            rel_vec = self.rel_key[rel_index]  # using the precomputed table
            # Dot product of query vector and relative positional embedding
            # Q[b, h, q_idx] has shape (head_dim,)
            q_vec = q[b, h, q_idx]  # (head_dim,)
            rel_score = torch.dot(q_vec, rel_vec)  # scalar
            # Apply scaling to both content score and relative score
            return score * scale + rel_score * scale

        # Perform attention using FlexAttention with our score_mod.
        # This computes softmax(score_mod(Q·K^T)) and returns weighted sum: shape (B, H, seq_len, head_dim)
        attn_out: torch.Tensor = flex_attention(q, k, v, score_mod=score_mod)
        # Combine heads and project out
        attn_out = attn_out.transpose(1, 2).reshape(B, seq_len, self.embed_dim)  # (B, seq_len, embed_dim)
        return self.proj_out(attn_out)


class VanillaCausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, max_position: int) -> None:
        """
        Causal self-attention with relative positions using manual (skew trick) implementation.
        embed_dim: total embedding dimension
        num_heads: number of attention heads
        max_position: maximum sequence length (defines range for relative positions)
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim: int = embed_dim
        self.num_heads: int = num_heads
        self.head_dim: int = embed_dim // num_heads
        self.max_position: int = max_position

        # Linear projections for multi-head Q, K, V, and output
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj_out = nn.Linear(embed_dim, embed_dim)

        # Relative positional embedding table for keys (0 to -max_position+1 distances)
        # Shape: (max_position, head_dim). Index max_position-1 corresponds to 0 distance.
        self.rel_key = nn.Parameter(torch.randn(max_position, self.head_dim))

        # Causal mask as a buffer for efficiency (1 for allowed, 0 for masked positions)
        mask = torch.tril(torch.ones(max_position, max_position, dtype=torch.bool))
        # Shape of mask: (1, 1, max_position, max_position) for broadcasting
        self.register_buffer("mask", mask.unsqueeze(0).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (batch_size, seq_len, embed_dim)
        Returns: Tensor of shape (batch_size, seq_len, embed_dim)
        """
        B, seq_len, _ = x.shape
        if seq_len > self.max_position:
            raise ValueError(f"Sequence length {seq_len} exceeds max_position {self.max_position}")

        # Compute Q, K, V projections
        Q: torch.Tensor = self.query(x)  # (B, seq_len, embed_dim)
        K: torch.Tensor = self.key(x)  # (B, seq_len, embed_dim)
        V: torch.Tensor = self.value(x)  # (B, seq_len, embed_dim)
        # Reshape into (B, H, seq_len, head_dim)
        Q = Q.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, seq_len, head_dim)
        K = K.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, seq_len, head_dim)
        V = V.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, seq_len, head_dim)

        # 1. Content-based attention scores: Q @ K^T
        # K^T shape for matmul: (B, H, head_dim, seq_len)
        K_t: torch.Tensor = K.transpose(2, 3)  # (B, H, head_dim, seq_len)
        attn_content: torch.Tensor = torch.matmul(Q, K_t)  # (B, H, seq_len, seq_len)

        # 2. Relative positional scores via skew trick:
        # Extract the needed portion of rel_key for this sequence length.
        # We use the last `seq_len` entries of the table, which correspond to distances -(seq_len-1)...0.
        start_index = self.max_position - seq_len  # index in rel_key that corresponds to distance = -(seq_len-1)
        # Slice shape: (seq_len, head_dim), to be transposed for matmul
        rel_slice: torch.Tensor = self.rel_key[start_index:, :].transpose(0, 1)  # (head_dim, seq_len)
        # Compute Q * rel_slice (like Q @ E_r^T)
        QEr: torch.Tensor = torch.matmul(Q, rel_slice)  # (B, H, seq_len, seq_len)
        # Apply skewing: pad and reshape to align relative positions [oai_citation_attribution:13‡jaketae.github.io](https://jaketae.github.io/study/relative-positional-encoding/#:~:text=def%20skew%28self%2C%20QEr%29%3A%20,batch_size%2C%20num_heads%2C%20seq_len%2C%20seq_len)
        padded = nn.functional.pad(QEr, (1, 0, 0, 0))
        # After padding: shape = (B, H, seq_len, 1+seq_len)
        B_, H_, num_rows, num_cols = padded.shape  # num_rows = seq_len, num_cols = seq_len+1
        # Reshape to shift the matrix
        reshaped = padded.view(B_, H_, num_cols, num_rows)  # (B, H, seq_len+1, seq_len)
        Srel: torch.Tensor = reshaped[:, :, 1:, :]  # (B, H, seq_len, seq_len)
        # Now Srel[b,h,i,j] = Q[b,h,i] · r_{(j-i)}, which is the desired relative score

        # 3. Combine content and relative scores, then scale.
        attn_scores: torch.Tensor = attn_content + Srel  # shape (B, H, seq_len, seq_len)
        attn_scores = attn_scores * (1.0 / math.sqrt(self.head_dim))  # scale by 1/sqrt(d_head)

        # 4. Apply causal mask: disallow attending to future positions
        # Use the precomputed mask buffer (shape 1x1xmax_lenxmax_len) and slice to seq_len.
        mask = self.mask[:, :, :seq_len, :seq_len]  # (1, 1, seq_len, seq_len), 1 where allowed
        attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))  # set masked positions to -inf

        # 5. Softmax over the last dimension (keys dimension) and compute attention output
        attn_weights: torch.Tensor = torch.softmax(attn_scores, dim=-1)  # (B, H, seq_len, seq_len)
        attn_out: torch.Tensor = torch.matmul(attn_weights, V)  # (B, H, seq_len, head_dim)

        # 6. Reshape `attn_out` back to (B, seq_len, embed_dim) and apply output projection
        attn_out = attn_out.transpose(1, 2).reshape(B, seq_len, self.embed_dim)  # (B, seq_len, embed_dim)
        return self.proj_out(attn_out)


torch.manual_seed(42)

# Define dimensions for test
batch_size = 2
seq_len = 8
embed_dim = 16
num_heads = 4
max_position = 16  # max relative positions (>= seq_len)

# Initialize both attention modules
flex_attn = FlexCausalSelfAttention(embed_dim, num_heads, max_position)
vanilla_attn = VanillaCausalSelfAttention(embed_dim, num_heads, max_position)

# Copy parameters from vanilla_attn to flex_attn to ensure identical weights
for name, param in vanilla_attn.named_parameters():
    if name in dict(flex_attn.named_parameters()):
        dict(flex_attn.named_parameters())[name].data.copy_(param.data)

# Verify that all parameters (including rel_key and linear weights) are equal
for name, param in flex_attn.named_parameters():
    assert name in dict(vanilla_attn.named_parameters())
    # Check that the difference is negligible
    diff = (param.data - dict(vanilla_attn.named_parameters())[name].data).abs().max().item()
    assert diff < 1e-6, f"Parameter {name} mismatch between flex and vanilla implementations"

# Create a random input tensor
x = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)

# Forward pass through both implementations
out_flex = flex_attn(x)
out_vanilla = vanilla_attn(x)

# Compare outputs
max_diff = (out_flex - out_vanilla).abs().max().item()
print(f"Max difference between outputs: {max_diff:.6f}")
assert torch.allclose(out_flex, out_vanilla, atol=1e-6, rtol=1e-6), "Outputs differ between implementations"

# Backpropagate a simple mean squared error loss on both outputs to test gradients
target = torch.randn_like(out_flex)  # random target for loss
loss_flex = torch.nn.functional.mse_loss(out_flex, target)
loss_vanilla = torch.nn.functional.mse_loss(out_vanilla, target)

# Zero gradients
flex_attn.zero_grad()
vanilla_attn.zero_grad()
if x.grad is not None:
    x.grad.zero_()
# Use separate inputs for backward to avoid gradient interference
x_flex = x.clone().detach().requires_grad_(True)
x_van = x.clone().detach().requires_grad_(True)
out_flex2 = flex_attn(x_flex)
out_van2 = vanilla_attn(x_van)
loss_flex2 = torch.nn.functional.mse_loss(out_flex2, target.detach())
loss_van2 = torch.nn.functional.mse_loss(out_van2, target.detach())
loss_flex2.backward()
loss_van2.backward()

# Check gradient equivalence for input
assert torch.allclose(x_flex.grad, x_van.grad, atol=1e-6), "Input gradients differ"

# Check gradient equivalence for all parameters
for (name_f, param_f), (name_v, param_v) in zip(flex_attn.named_parameters(), vanilla_attn.named_parameters()):
    # The parameter names should match in our design
    assert name_f == name_v
    if param_f.grad is None or param_v.grad is None:
        raise AssertionError(f"No gradient for parameter {name_f}")
    # Compare gradients
    grad_diff = (param_f.grad - param_v.grad).abs().max().item()
    print(f"{name_f} grad max diff: {grad_diff:.6f}")
    assert torch.allclose(param_f.grad, param_v.grad, atol=1e-6, rtol=1e-6), f"Gradient mismatch for {name_f}"

print("All tests passed: both implementations are numerically equivalent and have consistent gradients.")
