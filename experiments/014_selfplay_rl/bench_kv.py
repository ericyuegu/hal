"""KV-cache rollout-inference benchmark (gate G4 input).

Times the two ways the closed-loop policy can produce a per-frame action for a
batch of ``n`` slots (Dolphin boots):

* full   — a fresh ``forward_full`` over the trailing ``L_ctx``=256 window every
  frame (012's inference path), then the offset-1 head + value head;
* kv      — ``SlotCaches.step_incremental`` per frame with a batched ``rebuild``
  every ``refresh_every`` frames.

Reports ms/step (one step advances all slots one frame) and slot-frames/s at slot
batch sizes {4, 8, 16}, plus a fp32 pre-eviction equivalence spot-check. Uses the
real d256/L8 012 checkpoint if present, else a random net of the same shape (timing
is weight-independent). Saves nothing.

    uv run experiments/014_selfplay_rl/bench_kv.py

NOTE: a co-resident training run contends for the GPU; treat numbers from a busy
device as provisional (see nvidia-smi at launch).
"""

import time
from pathlib import Path

import torch
from kv_cache import SlotCaches
from nets_melee import ArchConfig
from nets_melee import PolicyValueNet
from nets_melee import load_il_policy

from hal.training.features import ACTION_CHANNELS
from hal.training.features import CAT_FEATURES
from hal.training.features import FLOAT_FEATURES
from hal.training.features import Context

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REAL_CKPT = Path(
    "runs/260616-004736_012_multi_token_gpt-d256-L8-h4-Lc256-o1.5.9.13_ranked-anon-1_gpt-16k-b1024/final.pt"
)
D256_L8 = ArchConfig(
    d_model=256, n_layers=8, n_heads=4, L_ctx=256, char_vocab=32, char_dim=12, stage_vocab=32, stage_dim=4
)
_PREFIXES = ("ego", "ego_nana", "opp_nana", "opp")

SLOT_BATCHES = (4, 8, 16)
REFRESH_EVERY = 64
N_FRAMES = 256
WARMUP = 16


def _load_net() -> tuple[PolicyValueNet, ArchConfig]:
    if REAL_CKPT.is_file():
        net, cfg = load_il_policy(REAL_CKPT)
        print(f"[bench] loaded real 012 checkpoint  d{cfg.d_model}/L{cfg.n_layers}")
    else:
        net, cfg = PolicyValueNet(D256_L8), D256_L8
        print(f"[bench] real checkpoint absent; random d{cfg.d_model}/L{cfg.n_layers} net")
    return net.to(DEVICE).eval(), cfg


def _seq_features(B: int, T: int, cfg: ArchConfig, *, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    feats: dict[str, torch.Tensor] = {}
    for p in _PREFIXES:
        for f in FLOAT_FEATURES:
            feats[f"{p}_{f}"] = torch.randn(B, T, generator=g)
        for name, (vocab, _) in CAT_FEATURES.items():
            feats[f"{p}_{name}"] = torch.randint(0, vocab, (B, T), generator=g)
    for ch in ACTION_CHANNELS:
        feats[f"ego_{ch}"] = torch.randn(B, T, generator=g)
    feats["ego_character"] = torch.randint(0, cfg.char_vocab, (B, T), generator=g)
    feats["opp_character"] = torch.randint(0, cfg.char_vocab, (B, T), generator=g)
    feats["stage"] = torch.randint(0, cfg.stage_vocab, (B, T), generator=g)
    return {k: v.to(DEVICE) for k, v in feats.items()}


def _window(feats: dict[str, torch.Tensor], lo: int, hi: int, n: int) -> Context:
    return Context(
        features={k: v[:, lo:hi] for k, v in feats.items()}, ctx_pad=torch.zeros(n, device=DEVICE, dtype=torch.long)
    )


def _sync() -> None:
    if DEVICE == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def _bench_full(net: PolicyValueNet, cfg: ArchConfig, n: int) -> float:
    """ms/step for the full re-forward path (trailing L_ctx window every frame)."""
    L = cfg.L_ctx
    feats = _seq_features(n, L + N_FRAMES, cfg, seed=n)
    steps = list(range(WARMUP + N_FRAMES))
    t0 = None
    for i in steps:
        hi = L + i
        h = net.forward_full(_window(feats, hi - L, hi, n))[:, -1]
        net.policy_logits(h)
        net.values(h)
        if i == WARMUP - 1:
            _sync()
            t0 = time.perf_counter()
    _sync()
    return (time.perf_counter() - t0) / N_FRAMES * 1000.0


@torch.no_grad()
def _bench_kv(net: PolicyValueNet, cfg: ArchConfig, n: int) -> float:
    """ms/step for incremental decode with a rebuild every REFRESH_EVERY frames."""
    L = cfg.L_ctx
    feats = _seq_features(n, L + WARMUP + N_FRAMES, cfg, seed=n + 1000)
    cache = SlotCaches(net, n_slots=n, device=DEVICE, max_pos=L + REFRESH_EVERY + 8)
    slots = torch.arange(n, device=DEVICE)
    t0 = None
    for i in range(WARMUP + N_FRAMES):
        pos = L + i
        if i > 0 and i % REFRESH_EVERY == 0:  # periodic rebuild over the trailing window
            cache.rebuild(net, slots, _window(feats, pos - L, pos, n))
        h = cache.step_incremental(net, None, _window(feats, pos, pos + 1, n))
        net.policy_logits(h)
        net.values(h)
        if i == WARMUP - 1:
            _sync()
            t0 = time.perf_counter()
    _sync()
    return (time.perf_counter() - t0) / N_FRAMES * 1000.0


@torch.no_grad()
def _equivalence_spotcheck(net: PolicyValueNet, cfg: ArchConfig) -> tuple[float, float]:
    """Pre-eviction fp32 exact-growth on the real net/device: max |Δhidden|, |Δlogit|."""
    n, L = 2, cfg.L_ctx
    feats = _seq_features(n, L, cfg, seed=777)
    cache = SlotCaches(net, n_slots=n, device=DEVICE)
    dh = dl = 0.0
    for t in range(min(L, 64)):
        h_inc = cache.step_incremental(net, None, _window(feats, t, t + 1, n))
        h_full = net.forward_full(_window(feats, 0, t + 1, n))[:, -1]
        dh = max(dh, float((h_inc - h_full).abs().max()))
        dl = max(dl, float((net.policy_logits(h_inc) - net.policy_logits(h_full)).abs().max()))
    return dh, dl


def main() -> None:
    print(f"[bench] device={DEVICE}  refresh_every={REFRESH_EVERY}  frames/timing={N_FRAMES}")
    net, cfg = _load_net()

    dh, dl = _equivalence_spotcheck(net, cfg)
    print(f"[bench] fp32 pre-eviction equivalence: max|Δhidden|={dh:.2e}  max|Δlogit|={dl:.2e}")

    print(f"\n{'n_slots':>8} {'full ms/step':>14} {'kv ms/step':>12} {'speedup':>9} {'full fr/s':>11} {'kv fr/s':>11}")
    for n in SLOT_BATCHES:
        full_ms = _bench_full(net, cfg, n)
        kv_ms = _bench_kv(net, cfg, n)
        full_fps = n / full_ms * 1000.0
        kv_fps = n / kv_ms * 1000.0
        print(f"{n:>8} {full_ms:>14.3f} {kv_ms:>12.3f} {full_ms / kv_ms:>8.2f}x {full_fps:>11.1f} {kv_fps:>11.1f}")


if __name__ == "__main__":
    main()
