"""KV-cache incremental decoding for the Melee trunk (rollout inference speedup).

The closed-loop policy replans every frame from the trailing ``L_ctx`` window. A
full forward re-embeds and re-attends all ``L_ctx`` tokens every frame (~256
token-forwards/frame). Incremental decode caches each layer's per-token K/V and
forwards only the ONE new token against the cache — a headline ~50x compute cut.

Design: append-with-eviction + periodic batched rebuild.

* ``SlotCaches`` preallocates per-slot K/V ``[n_slots, n_layers, cap, n_heads,
  head_dim]`` (cap = L_ctx) as a RING buffer: the new token overwrites the oldest
  ring slot, no shifting. Attention is permutation-invariant over keys (softmax
  over keys, each carrying its own baked RoPE), so physical order in the ring is
  irrelevant — only the validity mask matters, which keeps eviction O(1) instead
  of an O(cap) slide. It tracks per-slot ``length``, ``write_idx`` (ring cursor)
  and ``next_pos`` (the absolute RoPE position of the next token) and a
  precomputed cos/sin rope table. It owns ONLY K/V + positions; M4's collector
  owns the raw-obs history and hands rebuild its windows.
* ``step_incremental`` forwards the single new token per slot: embed, then per
  block project q/k/v, rotate q/k at each slot's absolute ``next_pos``, append
  k/v, and SDPA the 1-token query against the slot's valid cached keys (batched
  across slots with a key-padding mask over their differing lengths).
* ``rebuild`` re-seeds the caches from one batched ``forward_full_capture`` over
  the slots' trailing windows.

WHY rebuild exists — RoPE attention scores depend only on RELATIVE positions, so
before any eviction incremental decode is bit-for-bit a full re-forward (the
exact-growth contract). AFTER eviction they diverge: a full forward re-truncates
to the current ``L_ctx`` window every frame, recomputing every token's deep-layer
K/V from only the in-window tokens; the incremental cache instead keeps deep K/V
that were computed when older tokens could still attend to now-evicted history,
so they embed context beyond the current window and drift from 012's
every-frame-truncation semantics. Periodic ``rebuild`` (every
``MeleeRLConfig.refresh_every`` frames) resets the caches to the exact windowed
forward, bounding the drift at an amortised cost of ~``L_ctx/refresh_every``
token-forwards per frame instead of ``L_ctx``.
"""

import torch
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float
from jaxtyping import Int
from jaxtyping import jaxtyped
from nets_melee import PolicyValueNet
from nets_melee import apply_rotary_at
from nets_melee import rmsnorm
from torch import Tensor

from hal.training.features import Context


class SlotCaches:
    """Per-slot rolling K/V caches for incremental Melee decode.

    ``cap`` = ``L_ctx``; ``max_pos`` bounds absolute RoPE positions between rebuilds
    (a full forward resets ``next_pos`` back to the window length, so with periodic
    rebuild the position never exceeds ``L_ctx + refresh_every``). Positions past the
    table fail loud — the signal to rebuild more often. The bounds check reads
    ``_next_pos_cpu``, a plain-int mirror of ``next_pos`` kept in lockstep by
    step/rebuild/reset, so the per-frame hot path never syncs the device tensor
    (``pos.max()`` on a CUDA tensor would stall the stream every frame)."""

    def __init__(
        self, net: PolicyValueNet, n_slots: int, *, device: str | torch.device = "cpu", max_pos: int = 0
    ) -> None:
        cfg = net.cfg
        self.n_slots = n_slots
        self.n_layers = cfg.n_layers
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.cap = cfg.L_ctx
        self.device = torch.device(device)
        self.max_pos = max_pos if max_pos > 0 else 4 * self.cap
        shape = (n_slots, self.n_layers, self.cap, self.n_heads, self.head_dim)
        self.K = torch.zeros(shape, device=self.device)
        self.V = torch.zeros(shape, device=self.device)
        self.length = torch.zeros(n_slots, dtype=torch.long, device=self.device)
        self.write_idx = torch.zeros(n_slots, dtype=torch.long, device=self.device)  # ring cursor
        self.next_pos = torch.zeros(n_slots, dtype=torch.long, device=self.device)
        self._next_pos_cpu = [0] * n_slots  # host mirror of next_pos (bounds check w/o device sync)
        self._rows = torch.arange(n_slots, device=self.device)
        # Rope table, gathered per-slot in the incremental path. inv_freq is identical
        # across blocks (deterministic from head_dim), so one table serves every layer.
        inv_freq = net.blocks[0].attn.rotary.inv_freq.to(self.device)
        pos = torch.arange(self.max_pos, device=self.device, dtype=inv_freq.dtype)
        freqs = torch.outer(pos, inv_freq)  # [max_pos, half_dim]
        self.rope_cos = freqs.cos()
        self.rope_sin = freqs.sin()

    def reset_slot(self, i: int) -> None:
        """Cold-start a slot: zero its length/cursor/position (stale K/V are masked out
        until overwritten, so no need to clear the buffers)."""
        self.length[i] = 0
        self.write_idx[i] = 0
        self.next_pos[i] = 0
        self._next_pos_cpu[i] = 0

    def _rope_at(self, pos: Int[Tensor, " n"], pos_max: int) -> tuple[Tensor, Tensor]:
        """``pos_max`` is the host-side max of ``pos`` (from ``_next_pos_cpu``), so the
        fail-loud table-overrun check costs no device sync on the per-frame path."""
        if pos_max >= self.max_pos:
            raise IndexError(
                f"RoPE position {pos_max} >= table size {self.max_pos}; rebuild more often "
                f"(refresh_every) or raise max_pos"
            )
        cos = self.rope_cos[pos][:, None, None, :]  # [n, 1, 1, half]
        sin = self.rope_sin[pos][:, None, None, :]
        return cos, sin

    @jaxtyped(typechecker=beartype)
    @torch.no_grad()
    def step_incremental(
        self, net: PolicyValueNet, slot_ids: Int[Tensor, " n"] | None, ctx: Context
    ) -> Float[Tensor, "n d_model"]:
        """Forward the single new token for each stepped slot (``ctx`` carries L=1
        features per slot, in stepped order) and return the trunk hidden.

        ``slot_ids=None`` steps ALL slots and reads the caches through views (the fast
        rollout path — no gather copy); an explicit index steps a subset via gather."""
        all_slots = slot_ids is None
        ids = self._rows if all_slots else slot_ids
        # Host-side slot indices for the CPU next_pos mirror. tolist() on the subset
        # path can sync a CUDA slot_ids, but that path already gathers K/V copies;
        # the all-slots rollout hot path stays sync-free.
        ids_cpu = range(self.n_slots) if all_slots else [int(i) for i in slot_ids.tolist()]
        n = ids.shape[0]
        if ctx.batch != n:
            raise ValueError(f"ctx batch {ctx.batch} != slot count {n}")
        pos = self.next_pos[ids]  # [n] absolute position of the new token
        cos, sin = self._rope_at(pos, max(self._next_pos_cpu[i] for i in ids_cpu))

        # Per-slot ring bookkeeping (same across layers -> computed once).
        wpos = self.write_idx[ids]  # [n] ring slot the new token overwrites (oldest)
        new_len = (self.length[ids] + 1).clamp(max=self.cap)  # [n]
        # Key-padding mask over the cap ring slots: valid iff filled (< new_len).
        mask = (torch.arange(self.cap, device=self.device)[None, :] < new_len[:, None])[:, None, None, :]

        x = net.context_tokens(ctx.features)  # [n, 1, d_model]
        for li, block in enumerate(net.blocks):
            q, k, v = block.attn.qkv(rmsnorm(x))
            q = apply_rotary_at(q, cos, sin)
            k = apply_rotary_at(k, cos, sin)
            self.K[ids, li, wpos] = k[:, 0]  # in-place ring write (overwrites oldest)
            self.V[ids, li, wpos] = v[:, 0]
            k_l = self.K[:, li] if all_slots else self.K[ids, li]  # view (fast) or gather
            v_l = self.V[:, li] if all_slots else self.V[ids, li]
            y = block.attn.attend(q, k_l, v_l, mask)  # [n, 1, d_model]
            x = x + block.attn_scale * y
            x = x + block.mlp(rmsnorm(x))

        self.length[ids] = new_len
        self.next_pos[ids] = pos + 1
        self.write_idx[ids] = (wpos + 1) % self.cap
        for i in ids_cpu:
            self._next_pos_cpu[i] += 1
        return rmsnorm(x)[:, 0]

    @jaxtyped(typechecker=beartype)
    @torch.no_grad()
    def rebuild(self, net: PolicyValueNet, slot_ids: Int[Tensor, " n"], windows: Context) -> None:
        """Re-seed the slots' caches from one batched full forward over their trailing
        windows. Each slot's real (non-pad) tokens are stored at ring slots ``0..vlen-1``;
        the ring cursor is set to ``vlen % cap`` (the next token overwrites the oldest) and
        ``next_pos`` is reset to the window length so subsequent incremental tokens continue
        the same RoPE progression the forward used."""
        n = slot_ids.shape[0]
        if windows.batch != n:
            raise ValueError(f"windows batch {windows.batch} != slot count {n}")
        Lw = next(iter(windows.features.values())).shape[1]
        if Lw > self.cap:
            raise ValueError(f"rebuild window length {Lw} exceeds cap {self.cap}")
        _, kvs = net.forward_full_capture(windows)  # per layer (k, v): [n, Lw, n_heads, head_dim]
        # Host-side pad/vlen ints up front (one sync at most): the copy loop below must not
        # convert device scalars per row — that stalls the stream once per (row, layer).
        pads = windows.ctx_pad.tolist()
        vlens = [max(0, Lw - int(p)) for p in pads]
        K_all = torch.stack([k.to(self.device) for k, _ in kvs], dim=1)  # [n, n_layers, Lw, H, D]
        V_all = torch.stack([v.to(self.device) for _, v in kvs], dim=1)

        self.K[slot_ids] = 0
        self.V[slot_ids] = 0
        for r in range(n):  # one slice-assign per row across ALL layers
            vlen = vlens[r]
            pad = int(pads[r])
            self.K[slot_ids[r], :, :vlen] = K_all[r, :, pad : pad + vlen]
            self.V[slot_ids[r], :, :vlen] = V_all[r, :, pad : pad + vlen]
        valid = torch.tensor(vlens, dtype=torch.long, device=self.device)
        self.length[slot_ids] = valid
        self.write_idx[slot_ids] = valid % self.cap
        self.next_pos[slot_ids] = Lw
        for i in slot_ids.tolist():
            self._next_pos_cpu[i] = Lw
