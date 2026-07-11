"""012 warm-start remap + FactoredCategorical + discretizer round-trip tests.

Checkpoint compatibility is the contract: ``load_il_policy`` must load a 012 state
dict with strict=True and reproduce the SOURCE model's offset-1 logits exactly.
These tests pin that with a synthetic 012-shaped checkpoint (no GPU/real data),
plus the real d256/L8 checkpoint when present.
"""

from pathlib import Path

import pytest
import torch
from nets_melee import _GROUP_VOCABS
from nets_melee import A_VOCAB
from nets_melee import N_GROUPS
from nets_melee import ArchConfig
from nets_melee import FactoredCategorical
from nets_melee import PolicyValueNet
from nets_melee import dequantize_groups
from nets_melee import load_il_policy
from nets_melee import quantize_groups

from hal.training.features import ACTION_CHANNELS
from hal.training.features import CAT_FEATURES
from hal.training.features import FLOAT_FEATURES
from hal.training.features import Context

ATOL = 1e-6
CFG = ArchConfig(d_model=32, n_layers=2, n_heads=2, L_ctx=16, char_vocab=8, char_dim=4, stage_vocab=8, stage_dim=2)
HEAD_OFFSETS = (1, 5, 9, 13)
_PREFIXES = ("ego", "ego_nana", "opp_nana", "opp")

REAL_CKPT = Path(
    "runs/260616-004736_012_multi_token_gpt-d256-L8-h4-Lc256-o1.5.9.13_ranked-anon-1_gpt-16k-b1024/final.pt"
)


def _ctx(B: int, L: int, cfg: ArchConfig, *, seed: int = 0) -> Context:
    g = torch.Generator().manual_seed(seed)
    feats: dict[str, torch.Tensor] = {}
    for p in _PREFIXES:
        for f in FLOAT_FEATURES:
            feats[f"{p}_{f}"] = torch.randn(B, L, generator=g)
        for name, (vocab, _) in CAT_FEATURES.items():
            feats[f"{p}_{name}"] = torch.randint(0, vocab, (B, L), generator=g)
    for ch in ACTION_CHANNELS:
        feats[f"ego_{ch}"] = torch.randn(B, L, generator=g)
    feats["ego_character"] = torch.randint(0, cfg.char_vocab, (B, L), generator=g)
    feats["opp_character"] = torch.randint(0, cfg.char_vocab, (B, L), generator=g)
    feats["stage"] = torch.randint(0, cfg.stage_vocab, (B, L), generator=g)
    return Context(features=feats, ctx_pad=torch.zeros(B, dtype=torch.long))


def _synthetic_012_ckpt(cfg: ArchConfig, offsets: tuple[int, ...], *, seed: int = 0) -> dict:
    """A 012-shaped checkpoint: PolicyValueNet backbone keys renamed to 012's, with a
    per-offset ``heads.i`` stack (random weights) instead of policy/value heads."""
    torch.manual_seed(seed)
    ref = PolicyValueNet(cfg)
    model: dict[str, torch.Tensor] = {}
    for k, v in ref.state_dict().items():
        if k.startswith(("policy_head.", "value_head.")):
            continue
        model[k] = v.clone()
    # One independent Linear(d_model, A_VOCAB) per offset, mirroring 012's heads ModuleList.
    for i, _ in enumerate(offsets):
        head = torch.nn.Linear(cfg.d_model, A_VOCAB)
        model[f"heads.{i}.weight"] = head.weight.detach().clone()
        model[f"heads.{i}.bias"] = head.bias.detach().clone()
    cfg_dict = {
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "L_ctx": cfg.L_ctx,
        "char_vocab": cfg.char_vocab,
        "char_dim": cfg.char_dim,
        "stage_vocab": cfg.stage_vocab,
        "stage_dim": cfg.stage_dim,
        "head_offsets": offsets,
        "eval_replicas": 16,  # a stale key from_012_cfg must ignore
    }
    return {"step": 100, "model": model, "cfg": cfg_dict, "opt": {}, "sched": {}, "wandb_id": None}


def test_warmstart_reproduces_offset1_logits(tmp_path: Path) -> None:
    # The remap must reproduce the SOURCE offset-1 head's logits exactly: policy_head ==
    # heads[index(1)] applied to the same backbone hidden.
    ckpt = _synthetic_012_ckpt(CFG, HEAD_OFFSETS, seed=1)
    path = tmp_path / "synthetic.pt"
    torch.save(ckpt, path)
    net, cfg = load_il_policy(path)
    assert cfg == CFG

    ctx = _ctx(2, CFG.L_ctx, CFG, seed=7)
    hidden = net.forward_full(ctx)
    got = net.policy_logits(hidden)

    # Reference: apply the source offset-1 head weights directly to the same hidden.
    primary = HEAD_OFFSETS.index(1)
    w = ckpt["model"][f"heads.{primary}.weight"]
    b = ckpt["model"][f"heads.{primary}.bias"]
    want = (hidden @ w.T + b).float()
    assert torch.allclose(got, want, atol=ATOL), f"remap not exact: {(got - want).abs().max()}"


def test_warmstart_value_head_zero_and_strict(tmp_path: Path) -> None:
    ckpt = _synthetic_012_ckpt(CFG, HEAD_OFFSETS, seed=2)
    path = tmp_path / "synthetic.pt"
    torch.save(ckpt, path)
    net, _ = load_il_policy(path)

    ctx = _ctx(2, CFG.L_ctx, CFG, seed=3)
    values = net.values(net.forward_full(ctx))
    assert torch.count_nonzero(values) == 0, "zero-init value head must output all zeros"

    # Strict load: a corrupted backbone key name must raise (no silent zero-init).
    bad = _synthetic_012_ckpt(CFG, HEAD_OFFSETS, seed=2)
    bad["model"]["ctx_proj.WRONG"] = bad["model"].pop("ctx_proj.weight")
    bad_path = tmp_path / "bad.pt"
    torch.save(bad, bad_path)
    with pytest.raises(RuntimeError):
        load_il_policy(bad_path)


def test_factored_categorical_decomposition() -> None:
    torch.manual_seed(0)
    logits = torch.randn(4, A_VOCAB)
    fc = FactoredCategorical(logits)

    idx = fc.sample()
    assert idx.shape == (4, N_GROUPS)
    for g, vocab in enumerate(_GROUP_VOCABS):
        assert int(idx[:, g].max()) < vocab and int(idx[:, g].min()) >= 0

    # Joint log-prob == sum of per-group class log-probs.
    per_group = torch.zeros(4)
    for g, lp in enumerate(fc._log_probs):
        per_group += lp.gather(-1, idx[:, g : g + 1]).squeeze(-1)
    assert torch.allclose(fc.log_prob(idx), per_group, atol=ATOL)

    # Joint entropy == sum of per-group entropies.
    ent_sum = sum(-(lp.exp() * lp).sum(-1) for lp in fc._log_probs)
    assert torch.allclose(fc.entropy(), ent_sum, atol=ATOL)

    # KL to self is zero.
    assert torch.allclose(fc.kl_to(fc), torch.zeros(4), atol=ATOL)
    # KL to a different dist is non-negative.
    other = FactoredCategorical(torch.randn(4, A_VOCAB))
    assert bool((fc.kl_to(other) >= -ATOL).all())


def test_discretizer_roundtrip_on_grid() -> None:
    # dequantize -> quantize is identity on grid indices (the property 012 guarantees).
    net = PolicyValueNet(CFG)
    mc, cc, tc = net.main_centers, net.c_centers, net.trig_centers
    n_trig = tc.shape[0]
    torch.manual_seed(0)
    idx = torch.stack(
        [
            torch.randint(0, _GROUP_VOCABS[0], (50,)),
            torch.randint(0, mc.shape[0], (50,)),
            torch.randint(0, cc.shape[0], (50,)),
            torch.randint(0, n_trig * n_trig, (50,)),
        ],
        dim=-1,
    )
    action = dequantize_groups(mc, cc, tc, idx)
    back = quantize_groups(mc, cc, tc, action)
    assert torch.equal(back, idx), "dequantize->quantize not identity on grid"


@pytest.mark.skipif(not REAL_CKPT.is_file(), reason="real 012 checkpoint not present")
def test_load_real_012_checkpoint() -> None:
    net, cfg = load_il_policy(REAL_CKPT)
    assert (cfg.d_model, cfg.n_layers) == (256, 8)
    n_params = sum(p.numel() for p in net.parameters())
    assert 6.0e6 < n_params < 7.0e6, f"unexpected param count {n_params}"

    ctx = _ctx(1, cfg.L_ctx, cfg, seed=0)
    logits = net.policy_logits(net.forward_full(ctx))
    assert logits.shape == (1, cfg.L_ctx, A_VOCAB)
    assert torch.isfinite(logits).all()
