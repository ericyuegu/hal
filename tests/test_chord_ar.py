"""Correctness contracts for the chord-AR action-chunk policy (experiment 006).

Mirrors ``test_classification.py``'s synthetic-batch patterns (tiny dims, CPU). The chord-
specific contracts: pack/unpack is a bijection on the quantized product space; the data-
derived vocab build and OOV projection are deterministic; the teacher-forced loss is finite
and backprops into backbone + head; the head is strictly autoregressive (logits at position
k are invariant to chord tokens at positions >= k); decode emits valid raw action vectors;
the marginal-button-prob matmul matches brute force; and the vocab round-trips through a
checkpoint to an identical decode. The experiment is loaded by path since its filename
starts with a digit."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

_EXP_PATH = Path(__file__).resolve().parent.parent / "experiments" / "006_chord_ar.py"


def _load_experiment():
    spec = importlib.util.spec_from_file_location("exp006", _EXP_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


exp = _load_experiment()


def _cfg(**overrides):
    base = dict(
        d_model=16,
        n_layers=1,
        n_heads=2,
        dim_feedforward=16,
        dropout=0.0,
        d_head=8,
        n_head_layers=1,
        head_heads=2,
        head_ff=16,
        L_ctx=6,
        L_chunk=3,
    )
    base.update(overrides)
    return exp.TrainConfig(**base)


def _features(B: int, T: int, *, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    feats: dict[str, torch.Tensor] = {}
    for prefix in ("ego", "opp"):
        for f in exp.FLOAT_FEATURES:
            feats[f"{prefix}_{f}"] = torch.randn(B, T, generator=g)
        for cat, (vocab, _) in exp.CAT_FEATURES.items():
            feats[f"{prefix}_{cat}"] = torch.randint(0, vocab, (B, T), generator=g)
        feats[f"{prefix}_character"] = torch.randint(0, 26, (B, T), generator=g)
        # controller history for BOTH players in the real action ranges (the opp block is
        # consumed only under the opp_controller roofline cheat; harmless otherwise)
        for i, ch in enumerate(exp.ACTION_CHANNELS):
            if i < 4:
                feats[f"{prefix}_{ch}"] = torch.empty(B, T).uniform_(-1, 1, generator=g)
            elif i < 6:
                feats[f"{prefix}_{ch}"] = torch.empty(B, T).uniform_(0, 1, generator=g)
            else:
                feats[f"{prefix}_{ch}"] = (torch.rand(B, T, generator=g) > 0.5).float()
    feats["stage"] = torch.randint(0, 32, (B, T), generator=g)
    return feats


def _batch(B: int, cfg, *, ctx_pad: list[int] | None = None, seed: int = 0):
    feats = _features(B, cfg.L_ctx, seed=seed)
    pad = torch.zeros(B, dtype=torch.int64) if ctx_pad is None else torch.tensor(ctx_pad, dtype=torch.int64)
    ctx = exp.Context(features=feats, ctx_pad=pad)
    g = torch.Generator().manual_seed(seed + 1)
    cont = torch.cat(
        [
            torch.empty(B, cfg.L_chunk, 4).uniform_(-1, 1, generator=g),
            torch.empty(B, cfg.L_chunk, 2).uniform_(0, 1, generator=g),
        ],
        dim=-1,
    )
    btn = (torch.rand(B, cfg.L_chunk, exp.A_DIM - 6, generator=g) > 0.5).float()
    target = torch.cat([cont, btn], dim=-1)
    return ctx, exp.TrainBatch(context=ctx, target=target)


def _vocab(cfg, *, n_batches: int = 4):
    batches = [_batch(2, cfg, seed=s)[1] for s in range(n_batches)]
    return exp.build_chord_vocab(batches, frame_budget=10_000)


def _pack(main: int, c: int, tl: int, tr: int, btn: int) -> int:
    t = lambda v: torch.tensor(v, dtype=torch.long)  # noqa: E731
    return int(exp.pack_chord(t(main), t(c), t(tl), t(tr), t(btn)))


# --- acceptance #1: chord pack/unpack bijectivity ----------------------------
def test_pack_unpack_bijective_on_random_quantized_frames():
    g = torch.Generator().manual_seed(0)
    n = 10_000
    main = torch.randint(0, exp.N_MAIN, (n,), generator=g)
    c = torch.randint(0, exp.N_C, (n,), generator=g)
    tl = torch.randint(0, exp.N_TRIG, (n,), generator=g)
    tr = torch.randint(0, exp.N_TRIG, (n,), generator=g)
    btn = torch.randint(0, exp.N_BTN, (n,), generator=g)
    chord = exp.pack_chord(main, c, tl, tr, btn)
    assert chord.min() >= 0 and chord.max() < exp.N_CHORD_SPACE
    m2, c2, tl2, tr2, b2 = exp.unpack_chord(chord)
    for a, b in ((main, m2), (c, c2), (tl, tl2), (tr, tr2), (btn, b2)):
        assert torch.equal(a, b)


def test_chord_actions_round_trip_through_quantizers():
    """chord → action vec → quantize is the identity (centers quantize to themselves)."""
    g = torch.Generator().manual_seed(1)
    chord = torch.randint(0, exp.N_CHORD_SPACE, (5_000,), generator=g)
    actions = exp.chord_to_actions(chord)
    assert actions.shape == (5_000, exp.A_DIM)
    assert torch.equal(exp.quantize_actions(actions), chord)


# --- acceptance #2: vocab build + OOV mapping determinism ---------------------
def test_vocab_build_is_deterministic():
    cfg = _cfg()
    batches = [_batch(2, cfg, seed=s)[1] for s in range(3)]
    v1 = exp.build_chord_vocab(batches, frame_budget=10_000)
    v2 = exp.build_chord_vocab(batches, frame_budget=10_000)
    assert torch.equal(v1.chord_ids, v2.chord_ids)
    assert torch.equal(v1.counts, v2.counts)
    assert (v1.counts[:-1] >= v1.counts[1:]).all()  # descending frequency


def test_vocab_excludes_left_pad_filler():
    """ctx_pad positions are zero-filled filler, not data — they must not be harvested."""
    cfg = _cfg()
    _, batch = _batch(2, cfg, ctx_pad=[cfg.L_ctx, cfg.L_ctx])  # context fully padded
    vocab = exp.build_chord_vocab([batch], frame_budget=10_000)
    assert int(vocab.counts.sum()) == 2 * cfg.L_chunk  # only the target chunk frames


def test_vocab_rejects_unsorted_counts():
    a, b = _pack(0, 0, 0, 0, 0), _pack(1, 0, 0, 0, 0)
    with pytest.raises(ValueError):
        exp.ChordVocab([a, b], [1, 100])


def test_oov_projection_prefers_same_btn_tl_tr_then_nearest_sticks():
    # vocab: idx 0 most frequent (neutral, no buttons); idx 1/2 share btn=5 with different mains
    chord_c = _pack(0, 0, 0, 0, 0)  # main (0, 0)
    chord_a = _pack(1, 0, 0, 0, 5)  # main (0.35, 0)
    chord_b = _pack(30, 0, 0, 0, 5)  # main on the rim, far from (0.35, 0)
    vocab = exp.ChordVocab([chord_c, chord_a, chord_b], [100, 50, 10])
    # tier 1: same (btn, tl, tr) → nearest main center wins
    q1 = torch.tensor([_pack(5, 0, 0, 0, 5)])  # main (0.675, 0): nearer to idx 1's main
    idx, oov = vocab.encode(q1)
    assert oov.all() and idx.item() == 1
    # tier 2: no (btn, tl, tr) match (tl=1) → same btn, nearest sticks
    idx, oov = vocab.encode(torch.tensor([_pack(5, 0, 1, 0, 5)]))
    assert oov.all() and idx.item() == 1
    # tier 3: unseen button combo → most frequent chord (idx 0)
    idx, oov = vocab.encode(torch.tensor([_pack(5, 0, 0, 0, 7)]))
    assert oov.all() and idx.item() == 0
    # in-vocab ids map to themselves with no OOV flag
    idx, oov = vocab.encode(torch.tensor([chord_a, chord_b, chord_c]))
    assert not oov.any() and idx.tolist() == [1, 2, 0]


def test_oov_projection_is_deterministic_and_cached():
    cfg = _cfg()
    q = torch.tensor([_pack(5, 0, 0, 0, 7), _pack(5, 0, 0, 0, 7)])
    v1, v2 = _vocab(cfg), _vocab(cfg)
    i1a, _ = v1.encode(q)
    i1b, _ = v1.encode(q)  # second call hits the cache
    i2, _ = v2.encode(q)
    assert torch.equal(i1a, i1b)
    assert torch.equal(i1a, i2)
    assert torch.equal(i1a[0:1], i1a[1:2])  # same id → same projection within one call


def test_vocab_state_round_trip():
    vocab = _vocab(_cfg())
    clone = exp.ChordVocab.from_state(vocab.to_state())
    assert torch.equal(clone.chord_ids, vocab.chord_ids)
    assert torch.equal(clone.counts, vocab.counts)
    assert torch.equal(clone.actions, vocab.actions)


# --- acceptance #3: teacher-forced loss runs + backward -----------------------
def test_loss_is_finite_and_backprops_to_backbone_and_head():
    cfg = _cfg()
    torch.manual_seed(0)
    vocab = _vocab(cfg)
    model = exp.ChordARPolicy(cfg, vocab)
    _, batch = _batch(2, cfg)
    nll, oov_rate = exp.action_loss(model, batch)
    assert nll.shape[-1] == cfg.L_chunk and torch.isfinite(nll).all()
    assert 0.0 <= oov_rate.item() <= 1.0
    nll.mean().backward()
    for p in (model.ctx_proj.weight, model.chord_emb.weight, model.chord_out.weight):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    enc_grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    assert enc_grads and any(g.abs().sum() > 0 for g in enc_grads)


@pytest.mark.parametrize("bad", [dict(decode="agrmax"), dict(decode_temp=0.0), dict(norm_div=0.0)])
def test_invalid_config_is_rejected(bad):
    vocab = _vocab(_cfg())
    with pytest.raises(ValueError):
        exp.ChordARPolicy(_cfg(**bad), vocab)


# --- acceptance #4: the head is strictly autoregressive -----------------------
def test_head_logits_at_k_are_invariant_to_chords_at_and_after_k():
    cfg = _cfg()
    torch.manual_seed(0)
    vocab = _vocab(cfg)
    model = exp.ChordARPolicy(cfg, vocab).eval()
    H, V = cfg.L_chunk, vocab.size
    g = torch.Generator().manual_seed(2)
    cond = torch.randn(3, cfg.d_model, generator=g)
    idx = torch.randint(0, V, (3, H), generator=g)
    with torch.no_grad():
        base = model.chord_logits(cond, idx)
        for k in range(H):
            perturbed = idx.clone()
            perturbed[:, k:] = (perturbed[:, k:] + 1) % V
            out = model.chord_logits(cond, perturbed)
            assert torch.allclose(base[:, : k + 1], out[:, : k + 1], atol=1e-6), f"suffix at {k} leaked into prefix"
            if k + 1 < H:
                assert not torch.allclose(base[:, k + 1 :], out[:, k + 1 :]), f"chord_{k} ignored by position {k + 1}"


def test_backbone_is_causal_autograd():
    cfg = _cfg()
    torch.manual_seed(0)
    model = exp.ChordARPolicy(cfg, _vocab(cfg)).eval()
    ctx, _ = _batch(1, cfg)
    x = ctx.features["ego_main_stick_x"].clone().requires_grad_(True)
    feats = dict(ctx.features)
    feats["ego_main_stick_x"] = x
    hidden = model.encode_context(exp.Context(features=feats, ctx_pad=ctx.ctx_pad))
    i = cfg.L_ctx // 2
    hidden[0, i].sum().backward()
    g = x.grad[0]
    assert torch.all(g[i + 1 :] == 0), "hidden[i] leaked from inputs at j > i"
    assert g[: i + 1].abs().sum() > 0


# --- acceptance #5: decode contract -------------------------------------------
@pytest.mark.parametrize("mode", ["argmax", "sample"])
def test_decode_returns_valid_raw_action_vector(mode):
    cfg = _cfg()
    torch.manual_seed(0)
    model = exp.ChordARPolicy(cfg, _vocab(cfg)).eval()
    ctx, _ = _batch(2, cfg)
    gen = torch.Generator().manual_seed(0)
    a = exp.decode(model, ctx, mode=mode, gen=gen)
    assert a.shape == (2, cfg.L_chunk, exp.A_DIM)
    assert not a.requires_grad
    assert a[..., 0:4].abs().max() <= 1.0  # sticks in [-1, 1]
    assert a[..., 4:6].min() >= 0.0 and a[..., 4:6].max() <= 1.0  # triggers in [0, 1]
    assert torch.isin(a[..., 6:], torch.tensor([0.0, 1.0])).all()  # buttons binary
    # every decoded frame is an in-vocab chord (decode is a gather from the vocab table)
    idx, oov = model.vocab.encode(exp.quantize_actions(a))
    assert not oov.any()


def test_decode_rejects_unknown_mode():
    cfg = _cfg()
    model = exp.ChordARPolicy(cfg, _vocab(cfg)).eval()
    ctx, _ = _batch(2, cfg)
    with pytest.raises(ValueError):
        exp.decode(model, ctx, mode="greedy")


def test_decode_argmax_is_deterministic():
    cfg = _cfg()
    torch.manual_seed(0)
    model = exp.ChordARPolicy(cfg, _vocab(cfg)).eval()
    ctx, _ = _batch(2, cfg)
    assert torch.equal(exp.decode(model, ctx, mode="argmax"), exp.decode(model, ctx, mode="argmax"))


def test_default_decode_is_sampling():
    assert exp.TrainConfig().decode == "sample"


def test_make_policy_samples_by_default_and_respects_overrides():
    cfg = _cfg()  # decode defaults to "sample"
    torch.manual_seed(0)
    model = exp.ChordARPolicy(cfg, _vocab(cfg)).eval()
    ctx, _ = _batch(3, cfg)
    pol = exp.make_policy(model, {}, cfg, device="cpu")
    a1, a2 = pol.predict_chunk(ctx, None), pol.predict_chunk(ctx, None)
    assert a1.shape == (3, cfg.L_chunk, exp.A_DIM)
    assert not np.allclose(a1, a2)
    greedy = exp.make_policy(model, {}, cfg, device="cpu", decode_mode="argmax")
    assert np.allclose(greedy.predict_chunk(ctx, None), greedy.predict_chunk(ctx, None))


# --- acceptance #6: marginal button probs vs brute force -----------------------
def test_button_marginals_match_brute_force():
    cfg = _cfg()
    vocab = _vocab(cfg)
    model = exp.ChordARPolicy(cfg, vocab)
    g = torch.Generator().manual_seed(3)
    probs = torch.softmax(torch.randn(2, cfg.L_chunk, vocab.size, generator=g), dim=-1)
    marginal = probs @ model.vocab_button_bits  # the val_metrics one-matmul path
    brute = torch.zeros(2, cfg.L_chunk, 8)
    for v in range(vocab.size):
        bits = exp.scoring.combo_to_buttons(exp.unpack_chord(vocab.chord_ids[v])[4])
        brute += probs[..., v : v + 1] * bits
    assert torch.allclose(marginal, brute, atol=1e-6)


def test_val_metrics_emits_proper_scores():
    cfg = _cfg()
    torch.manual_seed(0)
    model = exp.ChordARPolicy(cfg, _vocab(cfg))
    val_cache = [_batch(2, cfg, seed=s)[1] for s in range(3)]
    m = exp.val_metrics(model, val_cache, cfg)
    assert m["action_nll_bits_per_frame"] > 0
    assert {"buttons/logloss_bits", "buttons/brier", "buttons/multipress_rate", "chord_oov_rate"} <= set(m)
    assert 0.0 <= m["chord_oov_rate"] <= 1.0
    assert any(k.startswith("loss/horizon/") for k in m)
    assert model.training  # restored after eval


# --- acceptance #7: checkpoint round-trip (vocab travels with the weights) -----
def test_checkpoint_round_trips_vocab_and_decode(tmp_path):
    cfg = _cfg()
    torch.manual_seed(0)
    vocab = _vocab(cfg)
    model = exp.ChordARPolicy(cfg, vocab).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, exp.lr_schedule(cfg))
    path = tmp_path / "ckpt.pt"
    exp._save_checkpoint(
        path, step=7, model=model, opt=opt, sched=sched, cfg=exp.asdict(cfg), vocab=vocab, wandb_id=None
    )
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert state["step"] == 7
    m2, cfg2, vocab2 = exp._model_from_state(state, device="cpu")
    assert torch.equal(vocab2.chord_ids, vocab.chord_ids)
    assert torch.equal(vocab2.counts, vocab.counts)
    ctx, _ = _batch(2, cfg)
    assert torch.equal(exp.decode(model, ctx, mode="argmax"), exp.decode(m2, ctx, mode="argmax"))


# --- opp_controller roofline cheat ---------------------------------------------
def test_opp_controller_consumes_opponent_history_and_is_offline_only():
    cfg = _cfg(opp_controller=True)
    torch.manual_seed(0)
    vocab = _vocab(cfg)
    model = exp.ChordARPolicy(cfg, vocab)
    _, batch = _batch(2, cfg)
    nll, _ = exp.action_loss(model, batch)
    assert torch.isfinite(nll).all()
    # the cheat changes the prediction: perturbing the opp controller history moves the logits
    model.eval()
    cond1, tgt = exp._select(model, batch, multi=False)
    feats2 = {k: v.clone() for k, v in batch.context.features.items()}
    for ch in exp.ACTION_CHANNELS[:4]:
        feats2[f"opp_{ch}"] += 0.5
    batch2 = exp.TrainBatch(exp.Context(features=feats2, ctx_pad=batch.context.ctx_pad), batch.target)
    cond2, _ = exp._select(model, batch2, multi=False)
    assert not torch.allclose(cond1, cond2)
    assert "-oppc" in exp._model_tag(cfg)
    with pytest.raises(ValueError):
        exp.make_policy(model, {}, cfg, device="cpu")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
