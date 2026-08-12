"""The shared trunk must stay identical to the frozen experiment copies, and its two attention
paths must agree.

1. Identity. Experiments 016 to 021 keep an inline copy of this stack, and ``hal.scripts.h2h``
   rebuilds their checkpoints from those files. A seeded build of ``hal.training.trunk`` therefore
   has to draw the same weights, in the same order, and give the same output at ``attn_window=0``.
2. The two paths agree. The dense bool mask is the reference. FlexAttention must match it in value,
   and its mask must match a plain double-loop statement of the three rules.
3. Sliding-window attention follows the same masking rules as full causal attention.
"""

import importlib.util
import warnings
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


@pytest.mark.parametrize("length", [128, 1024])
def test_the_rotary_table_survives_a_whole_module_cast_to_fp16(length: int) -> None:
    """The table is a lookup, not a weight. Read straight from an fp16 ``inv_freq`` — what the eval
    decode cast leaves behind — the position multiplies its 4e-4 of slack into a phase error of
    2e-2 over a 128-frame window and 3.9e-1 over 1024. Rounding the finished table costs 2.4e-4."""
    rotary = _trunk(_cfg(), prefer_flex=False, device="cpu").blocks[0].attn.rotary
    want, _ = rotary.at(length, torch.device("cpu"))

    got, _ = rotary.half().at(length, torch.device("cpu"))

    assert got.dtype == torch.float16
    assert (got.float() - want).abs().max() < 1e-3


def test_the_rotary_table_is_bit_identical_at_fp32() -> None:
    """The fp32 path must read the registered buffer, not a rebuild of it: a recomputed frequency
    can differ by an ulp across devices, and every 016-to-021 checkpoint was trained on the buffer."""
    rotary = _trunk(_cfg(), prefer_flex=False, device="cpu").blocks[0].attn.rotary
    length = 64

    cos, sin = rotary.at(length, torch.device("cpu"))

    freqs = torch.outer(torch.arange(length).float(), rotary.inv_freq)
    assert torch.equal(cos, freqs.cos()[None, :, None, :])
    assert torch.equal(sin, freqs.sin()[None, :, None, :])


def test_the_trunk_refuses_a_ctx_pad_that_would_broadcast() -> None:
    """The one shape the annotations cannot catch by failing: a ctx_pad of the wrong length does not
    raise downstream, it broadcasts one sample's cold-start prefix over the whole batch."""
    trunk = _trunk(_cfg(), prefer_flex=False, device="cpu")
    x = torch.randn(4, _GEOM["L_ctx"], _GEOM["d_model"])

    with pytest.raises(ValueError, match="ctx_pad"):
        trunk(x, torch.zeros(1, dtype=torch.long))
    with pytest.raises(ValueError, match="ctx_pad"):
        trunk(x, torch.zeros(4, 1, dtype=torch.long))
    with pytest.raises(ValueError, match="ctx_pad"):
        trunk(x[0], torch.zeros(4, dtype=torch.long))


def test_the_flex_probe_reads_the_same_under_no_grad() -> None:
    """Every eval and h2h worker runs its first forward under ``torch.no_grad()``. The probe does a
    backward, so without its own ``enable_grad`` it raises there, reads that as "no flex", and caches
    the dense path for the life of the process (measured 35% slower validation)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    flex_is_usable.cache_clear()
    outside = flex_is_usable(device)
    flex_is_usable.cache_clear()
    with torch.no_grad():
        inside = flex_is_usable(device)
    flex_is_usable.cache_clear()

    assert inside == outside
    if device == "cpu":
        assert inside is False  # no CPU backward; the verdict must be a clean False, not a raise


def test_compiled_first_forward_keeps_path_resolution_outside_dynamo() -> None:
    """The first forward resolves the attention path lazily. Dynamo must not trace through either
    the cached FlexAttention probe or Loguru's caller-frame inspection while doing so."""
    flex_is_usable.cache_clear()
    trunk = torch.compile(_trunk(_cfg(), prefer_flex=True, device="cpu"), backend="eager", dynamic=False)
    x = torch.randn(2, _GEOM["L_ctx"], _GEOM["d_model"])
    ctx_pad = torch.zeros(2, dtype=torch.long)

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        output = trunk(x, ctx_pad)
    flex_is_usable.cache_clear()

    messages = [str(warning.message) for warning in seen]
    assert output.shape == x.shape
    assert not any("functools.lru_cache" in message for message in messages)
    assert not any("sys._getframe" in message for message in messages)


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
