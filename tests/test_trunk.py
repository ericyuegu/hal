"""The shared trunk must stay identical to the frozen experiment copies, and its two attention
paths must agree.

1. Identity. Experiments 016 to 021 keep an inline copy of this stack, and ``hal.scripts.h2h``
   rebuilds their checkpoints from those files. A seeded build of ``hal.training.trunk`` therefore
   has to draw the same weights, in the same order, and give the same output at ``attn_window=0``.
2. The two paths agree. The dense bool mask is the reference. FlexAttention must match it in value,
   and its mask must match a plain double-loop statement of the three rules.
3. Sliding-window decode is exact. With a trained window, the rolling KV cache holds precisely the
   window the model saw in training, so incremental decode equals the full forward.
"""

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from hal.training.trunk import Trunk
from hal.training.trunk import TrunkConfig
from hal.training.trunk import dense_mask
from hal.training.trunk import flex_is_usable
from hal.training.trunk import flex_mask_mod

_REPO = Path(__file__).resolve().parent.parent
_EXP_DIR = _REPO / "experiments"

_GEOM = dict(d_model=64, n_layers=2, n_heads=4, L_ctx=48)

requires_flex = pytest.mark.skipif(
    not (torch.cuda.is_available() and flex_is_usable("cuda")),
    reason="FlexAttention needs a GPU that compiles it",
)


@pytest.fixture(autouse=True)
def exact_fp32_matmuls():
    """Compare the paths at full fp32. Another test in the session can turn on TF32 matmuls for the
    whole process (the experiment train loops do), and TF32 keeps only 10 mantissa bits, so the two
    kernels then differ by about 6e-4 for reasons that have nothing to do with the mask."""
    before = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(before)


def _load_experiment(filename: str):
    spec = importlib.util.spec_from_file_location(filename.split(".")[0], _EXP_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def exp020():
    return _load_experiment("020_awr.py")


def _cfg(**kwargs) -> TrunkConfig:
    return TrunkConfig(**{**_GEOM, **kwargs})


def _trunk(cfg: TrunkConfig, *, prefer_flex: bool, device: str, seed: int = 0) -> Trunk:
    torch.manual_seed(seed)
    return Trunk(cfg, prefer_flex=prefer_flex).to(device)


def test_init_draws_match_020(exp020) -> None:
    """Same seed, same weights: the shared trunk must not shift any experiment's init draws."""
    cfg020 = exp020.TrainConfig(**_GEOM, head_offsets=(1, 2), batch_size=2, max_steps=8, warmup_steps=2)
    torch.manual_seed(1234)
    frozen = nn.ModuleList([exp020.Block(cfg020) for _ in range(cfg020.n_layers)]).state_dict()
    torch.manual_seed(1234)
    shared = Trunk(_cfg()).state_dict()

    assert set(shared) == {f"blocks.{k}" for k in frozen}
    for name, value in frozen.items():
        assert torch.equal(shared[f"blocks.{name}"], value), name


def test_forward_matches_020_at_full_context(exp020) -> None:
    """At ``attn_window=0`` the dense path is 020's forward, bit for bit."""
    cfg020 = exp020.TrainConfig(**_GEOM, head_offsets=(1, 2), batch_size=2, max_steps=8, warmup_steps=2)
    frozen = nn.ModuleList([exp020.Block(cfg020) for _ in range(cfg020.n_layers)])
    trunk = Trunk(_cfg(attn_window=0), prefer_flex=False)
    trunk.blocks.load_state_dict(frozen.state_dict())

    torch.manual_seed(7)
    x = torch.randn(4, _GEOM["L_ctx"], _GEOM["d_model"])
    ctx_pad = torch.tensor([0, 1, 17, _GEOM["L_ctx"]])

    mask = exp020.GPT._attn_mask(None, ctx_pad, x.size(1), x.device)
    want = x
    for block in frozen:
        want = block(want, mask)
    want = exp020.rmsnorm(want)

    assert torch.equal(trunk(x, ctx_pad), want)


@pytest.mark.parametrize("attn_window", [0, 8, 128])
def test_dense_mask_matches_double_loop(attn_window: int) -> None:
    """The mask states three rules: causal, inside the window, and clear of the left padding. The
    diagonal is always open, so no query row is fully masked."""
    L = _GEOM["L_ctx"]
    ctx_pad = torch.tensor([0, 3, 20, L])
    mask = dense_mask(ctx_pad, L, attn_window)

    want = torch.zeros(len(ctx_pad), 1, L, L, dtype=torch.bool)
    for b, pad in enumerate(ctx_pad.tolist()):
        for q in range(L):
            for kv in range(L):
                in_window = attn_window == 0 or q - kv < attn_window
                want[b, 0, q, kv] = (kv <= q and in_window and kv >= pad) or kv == q

    assert torch.equal(mask, want)
    assert mask.any(-1).all(), "every query row must keep at least one key"


@pytest.mark.parametrize(
    "L, attn_window, pads",
    [
        (256, 128, [127, 128, 129, 255]),  # each side of the 128-wide flex block boundary
        (200, 128, [0, 127, 128, 200]),  # L is not a multiple of the flex block size
        (48, 8, [0, 1, 47, 48]),  # L is smaller than one flex block
        (64, 1, [0, 5, 63, 64]),  # window 1: the diagonal only
        (64, 4096, [0, 3, 63, 64]),  # window larger than L
    ],
)
def test_flex_mask_mod_matches_dense_mask(L: int, attn_window: int, pads: list[int]) -> None:
    """The FlexAttention rule must equal the dense reference element by element. The rule runs on any
    box, so this covers the edge cases that the CUDA-only kernel tests cannot reach."""
    ctx_pad = torch.tensor(pads)
    idx = torch.arange(L)
    b, q, kv = torch.meshgrid(torch.arange(len(pads)), idx, idx, indexing="ij")
    flex = flex_mask_mod(ctx_pad, attn_window)(b, torch.zeros_like(b), q, kv)

    assert torch.equal(flex[:, None], dense_mask(ctx_pad, L, attn_window))


@requires_flex
@pytest.mark.parametrize("attn_window", [0, 1, 8, 128])
def test_flex_matches_dense(attn_window: int) -> None:
    """The FlexAttention kernel must reproduce the dense reference under mixed padding, including a
    row with no padding and a row that is padding all through. Windows below the 128-wide flex block
    make most blocks empty, and window 1 leaves only the diagonal."""
    L = 256
    cfg = _cfg(L_ctx=L, attn_window=attn_window)
    flex = _trunk(cfg, prefer_flex=True, device="cuda")
    dense = _trunk(cfg, prefer_flex=False, device="cuda")
    dense.load_state_dict(flex.state_dict())

    torch.manual_seed(3)
    x = torch.randn(4, L, cfg.d_model, device="cuda")
    ctx_pad = torch.tensor([0, 1, 200, L], device="cuda")

    torch.testing.assert_close(flex(x, ctx_pad), dense(x, ctx_pad), rtol=1e-5, atol=1e-5)
    assert (flex.attn_path, dense.attn_path) == ("flex", "dense")


@requires_flex
def test_flex_and_dense_train_the_same() -> None:
    """Twenty steps of gradient descent must follow the same loss curve on both paths."""
    device = "cuda"
    L, attn_window = 256, 128
    cfg = _cfg(L_ctx=L, attn_window=attn_window)
    torch.manual_seed(11)
    x = torch.randn(2, L, cfg.d_model, device=device)
    y = torch.randn(2, L, cfg.d_model, device=device)
    ctx_pad = torch.tensor([0, 40], device=device)

    curves = []
    for prefer_flex in (True, False):
        trunk = _trunk(cfg, prefer_flex=prefer_flex, device=device, seed=5)
        opt = torch.optim.SGD(trunk.parameters(), lr=0.05)
        losses = []
        for _ in range(20):
            loss = (trunk(x, ctx_pad) - y).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        curves.append(torch.tensor(losses))

    torch.testing.assert_close(curves[0], curves[1], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("prefer_flex", [False, pytest.param(True, marks=requires_flex)])
def test_incremental_matches_full_forward_under_swa(prefer_flex: bool) -> None:
    """The rolling cache holds exactly the trained window, and RoPE is relative, so decoding frame
    by frame gives the same hidden state as one forward over the whole sequence."""
    device = "cuda" if prefer_flex else "cpu"
    attn_window = 16
    L = 3 * attn_window
    cfg = _cfg(L_ctx=L, attn_window=attn_window)
    trunk = _trunk(cfg, prefer_flex=prefer_flex, device=device)

    torch.manual_seed(17)
    x = torch.randn(2, L, cfg.d_model, device=device)
    ctx_pad = torch.zeros(2, dtype=torch.long, device=device)
    want = trunk(x, ctx_pad)[:, -1]

    past: list = [None] * cfg.n_layers
    for t in range(L):
        h, past = trunk.forward_incremental(x[:, t : t + 1], past)
    assert all(kv[0].size(2) == attn_window for kv in past)

    torch.testing.assert_close(h[:, -1], want, rtol=1e-5, atol=1e-5)


def test_incremental_rejects_a_chunk() -> None:
    """The cached attention has no mask inside the given chunk, so more than one token is refused."""
    trunk = _trunk(_cfg(attn_window=16), prefer_flex=False, device="cpu")
    x = torch.randn(2, 3, _GEOM["d_model"])

    with pytest.raises(ValueError, match="one token"):
        trunk.forward_incremental(x, [None] * len(trunk.blocks))


def test_incremental_rejects_decoding_past_a_full_context() -> None:
    """At full attention the cache is the whole context, so one more frame would drop history the
    full forward keeps. The trunk refuses rather than answer with a quietly wrong hidden state."""
    L = 8
    trunk = _trunk(_cfg(L_ctx=L, attn_window=0), prefer_flex=False, device="cpu")
    x = torch.randn(2, 1, _GEOM["d_model"])

    past: list = [None] * len(trunk.blocks)
    for _ in range(L):
        _, past = trunk.forward_incremental(x, past)
    with pytest.raises(ValueError, match="passed the 8-frame context"):
        trunk.forward_incremental(x, past)


@pytest.mark.skipif(flex_is_usable("cpu"), reason="the fallback needs a device without FlexAttention")
def test_require_flex_refuses_the_dense_fallback() -> None:
    """A cloud run can demand the fast kernel. On a box without it the trunk raises; it never trains
    4x slower without saying so."""
    with pytest.raises(ValueError, match="require_flex"):
        Trunk(_cfg(require_flex=True), prefer_flex=False)

    trunk = Trunk(_cfg(require_flex=True))  # CPU has no FlexAttention backward
    with pytest.raises(RuntimeError, match="require_flex"):
        trunk(torch.randn(2, _GEOM["L_ctx"], _GEOM["d_model"]), torch.zeros(2, dtype=torch.long))


def test_config_rejects_impossible_geometry() -> None:
    with pytest.raises(ValueError, match="divisible"):
        _cfg(n_heads=5)
    with pytest.raises(ValueError, match="head_dim must be even"):
        _cfg(d_model=6, n_heads=2)
    with pytest.raises(ValueError, match="attn_window"):
        _cfg(attn_window=-1)
