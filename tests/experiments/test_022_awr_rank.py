"""022 forks 020 onto the shared sliding-window trunk, retunes the reward and prunes the val block.
020's own tests still pin the return, the window threading and the weights; these tests pin what the
fork changed.

1. Nothing moved that should not. With the window off, the BC arm's objective is 020's plain mean
   NLL on the same batch, and the trunk swap draws the same weights under the same seed.
2. The defaults are the tuned recipe: the reward table Eric read off the explorer, plus the
   SWA-128 long-context geometry.
3. Validation costs two trunk forwards, not five, and every surviving number is bitwise 020's.
4. The new config states are checked: a negative window, incremental decode at full attention, and
   an effective batch that does not divide into micro-batches.
5. ``final.pt`` is written before the closed-loop eval, and ``final_eval_n_matchups=0`` skips that
   eval altogether.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

from hal.data.feature_stats import FeatureStats
from hal.training.features import A_DIM
from hal.training.features import ACTION_CHANNELS
from hal.training.features import CAT_FEATURES
from hal.training.features import FLOAT_FEATURES
from hal.training.features import Context
from hal.training.features import TrainBatch

_REPO = Path(__file__).resolve().parent.parent.parent
_EXP_DIR = _REPO / "experiments"
_DEV_MDS = _REPO / "data" / "processed" / "dev" / "mds"


def _load_experiment(filename: str):
    spec = importlib.util.spec_from_file_location(filename.split(".")[0], _EXP_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


exp020 = _load_experiment("020_awr.py")
exp022 = _load_experiment("022_awr_rank.py")

_GROUPS = exp022._GROUP_NAMES

# Geometry shared by the paired 020/022 comparisons: 022 at attn_window=0 must reproduce 020.
_PAIRED = dict(d_model=64, n_layers=2, n_heads=2, L_ctx=256, head_offsets=(1, 2), warmup_steps=2, max_steps=8)
_PAIRED_AWR = dict(awr_beta=0.5, awr_weight_max=20.0)


@pytest.fixture(autouse=True)
def exact_fp32_matmuls():
    """The train loops turn on TF32 matmuls for the whole pytest process, and TF32 keeps only 10
    mantissa bits, so a bitwise comparison of two code paths passes alone and fails in the suite."""
    before = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(before)


def _stats() -> dict[str, FeatureStats]:
    keys = (*FLOAT_FEATURES, *(f"nana_{k}" for k in FLOAT_FEATURES))
    return {k: FeatureStats(mean=0.0, std=1.0, min=-1.0, max=1.0) for k in keys}


def _tiny_cfg(**kwargs):
    defaults = dict(
        d_model=64, n_layers=2, n_heads=2, L_ctx=16, head_offsets=(1, 2), batch_size=2, max_steps=8, warmup_steps=2
    )
    return exp022.TrainConfig(**{**defaults, **kwargs})


def _features(batch: int, length: int, gen: torch.Generator) -> dict[str, torch.Tensor]:
    """A synthetic observation batch with plausible random values."""
    features: dict[str, torch.Tensor] = {}
    for prefix in exp022._PLAYER_PREFIXES:
        for feat in FLOAT_FEATURES:
            features[f"{prefix}_{feat}"] = torch.randn(batch, length, generator=gen)
        for name, (vocab, _) in CAT_FEATURES.items():
            features[f"{prefix}_{name}"] = torch.randint(0, vocab, (batch, length), generator=gen)
    for channel in ACTION_CHANNELS:
        features[f"ego_{channel}"] = torch.randn(batch, length, generator=gen)
    for key in ("ego_character", "opp_character", "stage"):
        features[key] = torch.randint(0, 26, (batch, length), generator=gen)
    return features


def _batch(L_ctx: int, max_offset: int, batch: int = 4, seed: int = 0) -> tuple[TrainBatch, torch.Tensor]:
    """One ``TrainBatch`` plus the ego return column, ready to wrap in either module's ``AWRBatch``."""
    gen = torch.Generator().manual_seed(seed)
    ctx = Context(features=_features(batch, L_ctx, gen), ctx_pad=torch.tensor([0, 1] * (batch // 2)))
    target = torch.rand(batch, max_offset, A_DIM, generator=gen) * 2 - 1
    return TrainBatch(context=ctx, target=target), torch.randn(batch, L_ctx, generator=gen)


def _paired_models(seed: int = 7):
    """The same weights in both modules: 020's inline trunk and 022's shared trunk at window 0."""
    cfg020 = exp020.TrainConfig(batch_size=4, **_PAIRED, **_PAIRED_AWR)
    cfg022 = exp022.TrainConfig(batch_size=4, grad_accum_steps=1, attn_window=0, **_PAIRED, **_PAIRED_AWR)
    torch.manual_seed(seed)
    model020 = exp020.GPT(cfg020).eval()
    torch.manual_seed(seed)
    model022 = exp022.GPT(cfg022).eval()
    return (cfg020, model020), (cfg022, model022)


# --- the tuned recipe --------------------------------------------------------


def test_reward_defaults_are_the_tuned_table() -> None:
    """The values Eric read off the reward explorer; the audit puts ESS ~0.70 and ~0% on the clip."""
    cfg = exp022.TrainConfig()
    assert cfg.awr_enabled is True
    assert cfg.awr_gamma == 0.99827
    assert cfg.awr_beta == 0.8
    assert cfg.awr_weight_max == 5.0
    assert cfg.awr_damage_shaping == 0.01
    assert cfg.awr_win_reward == 0.5


def test_geometry_defaults_are_the_swa_long_context_base() -> None:
    """131,072 tokens per step, as 020 had, but four times the context under a 128-frame window."""
    cfg = exp022.TrainConfig()
    assert (cfg.L_ctx, cfg.batch_size, cfg.grad_accum_steps) == (1024, 128, 2)
    assert exp022._micro_batch(cfg) == 64
    assert cfg.batch_size * cfg.L_ctx == 131072
    assert cfg.attn_window == 128
    assert cfg.require_flex is False
    assert cfg.windows_per_replay == 4
    assert (cfg.data_root, cfg.mds_schema_version) == ("data/processed/ranked-anonymized-1/mds-v6", 6)


def test_the_loader_gets_the_micro_batch() -> None:
    kwargs = exp022._loader_kwargs(exp022.TrainConfig(), _stats())
    assert kwargs["batch_size"] == 64


def test_reward_tag_follows_the_flags() -> None:
    assert exp022._reward_tag(exp022.TrainConfig()) == "g99827-b0.8-w5-d0.01-win0.5-rank111-mc-swa128"
    assert exp022._reward_tag(exp022.TrainConfig(awr_enabled=False)) == "bc-swa128"
    assert exp022._reward_tag(exp022.TrainConfig(attn_window=0)).endswith("-swafull")
    assert "b0.3" in exp022._reward_tag(exp022.TrainConfig(awr_beta=0.3))


# --- only the trunk and the objective's weights moved ------------------------


def test_trunk_swap_keeps_020_init_draws() -> None:
    """Same seed, same weights: only the state-dict key names moved under a ``trunk.`` prefix."""
    (_, model020), (_, model022) = _paired_models()
    state020, state022 = model020.state_dict(), model022.state_dict()

    renamed = {
        k.replace("blocks.", "trunk.blocks.", 1) if k.startswith("blocks.") else k: v for k, v in state020.items()
    }
    assert set(renamed) == set(state022)
    for key, value in renamed.items():
        assert torch.equal(state022[key], value), key


def test_bc_arm_objective_equals_020_unweighted() -> None:
    """The control arm: with AWR off and the window off, 022 backpropagates 020's objective."""
    (cfg020, model020), (cfg022, model022) = _paired_models()
    inner, returns = _batch(cfg022.L_ctx, max(cfg022.head_offsets))

    parts020 = exp020.action_loss(model020, inner)
    parts022 = exp022.action_loss(model022, inner)
    obj020 = exp020.objective(parts020.nll, parts020.transition, cfg020.aux_loss_weight, cfg020.transition_loss_weight)
    obj022 = exp022.objective(parts022.nll, parts022.transition, cfg022.aux_loss_weight, cfg022.transition_loss_weight)

    assert obj022.item() == pytest.approx(obj020.item(), rel=0, abs=1e-6)
    torch.testing.assert_close(parts022.value, parts020.value, rtol=0, atol=1e-6)
    assert returns.shape == (4, cfg022.L_ctx)


def test_incremental_decode_matches_the_full_forward_under_a_window() -> None:
    """The trunk answers with the whole sequence, so the call site takes the last position. Frame by
    frame past the window, the decoded state must still equal the full forward's."""
    cfg = _tiny_cfg(L_ctx=24, attn_window=8)
    torch.manual_seed(3)
    model = exp022.GPT(cfg).eval()
    features = _features(2, cfg.L_ctx, torch.Generator().manual_seed(4))
    ctx_pad = torch.zeros(2, dtype=torch.long)

    want = model(features, ctx_pad)[:, -1]
    past: list = [None] * cfg.n_layers
    for t in range(cfg.L_ctx):
        got, past = model.forward_incremental({k: v[:, t : t + 1] for k, v in features.items()}, past)

    assert got.shape == want.shape
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_020_checkpoints_are_refused() -> None:
    """020 stores the trunk at ``blocks.*``. Loading it here would silently mismatch the stack."""
    (_, model020), (_, model022) = _paired_models()
    with pytest.raises(RuntimeError, match="016/019/020/021"):
        exp022._load_model_state(model022, model020.state_dict())


# --- validation: two forwards, and the same numbers --------------------------


def _val_cache(cfg022, n_batches: int = 2):
    inner = [_batch(cfg022.L_ctx, max(cfg022.head_offsets), seed=seed) for seed in range(n_batches)]
    batches = [b for b, _ in inner]
    cache020 = [exp020.AWRBatch(batch=b, returns=r) for b, r in inner]
    cache022 = [exp022.AWRBatch(batch=b, returns=r) for b, r in inner]
    return batches, cache020, cache022


def test_shared_hidden_states_reproduce_020_val_numbers_bitwise() -> None:
    """020 runs the trunk five times per val batch; 022 runs it twice and passes the states down.
    Every metric 022 still logs must come out with exactly the same bits."""
    (cfg020, model020), (cfg022, model022) = _paired_models()
    batches, cache020, cache022 = _val_cache(cfg022)

    want = exp020.val_metrics(model020, batches, cfg020)
    want_awr, want_weights = exp020.awr_val_metrics(model020, cache020, cfg020)

    hidden = exp022.val_hidden(model022, batches)
    got = exp022.val_metrics(model022, batches, cfg022, hidden)
    got_awr, got_weights = exp022.awr_val_metrics(model022, cache022, cfg022, hidden)

    assert set(got) <= set(want)
    for key, value in got.items():
        assert value == want[key], key
    assert got_awr == want_awr
    torch.testing.assert_close(got_weights, want_weights, rtol=0, atol=0)


def test_val_block_drops_the_uninformative_metrics() -> None:
    """The correlation study found these either uninformative or inverted against closed-loop play."""
    (_, model022) = _paired_models()[1]
    cfg022 = exp022.TrainConfig(batch_size=4, grad_accum_steps=1, attn_window=0, **_PAIRED, **_PAIRED_AWR)
    batches, _, _ = _val_cache(cfg022)
    metrics = exp022.val_metrics(model022, batches, cfg022, exp022.val_hidden(model022, batches))

    assert not [k for k in metrics if k.startswith(("recon_", "pred_persistence_", "pred_flipback_", "trans_rate_"))]
    assert "btn_multipress" not in metrics
    for group in ("main_stick", "c_stick", "triggers"):
        assert f"changeF1_{group}" not in metrics
        assert f"brier_{group}" not in metrics
    kept = {
        "loss",
        "btn_logloss",
        "btn_brier",
        "brier_buttons",
        "changeF1_buttons",
        "pred_change_rate_buttons",
        "ablate_hist_dnll",
        "ablate_hist_dnll_buttons",
        "click_trigger_invalid_mass",
        "nll_off1",
        "nll_buttons_trans",
        "nll_buttons_hold",
        *(f"nll_{group}" for group in _GROUPS),
    }
    assert kept <= set(metrics)


# --- the new configuration states --------------------------------------------


def test_validate_config_rejects_a_negative_window() -> None:
    with pytest.raises(ValueError, match="attn_window"):
        exp022.validate_config(_tiny_cfg(attn_window=-1), has_button_combo_counts=False)


def test_validate_config_rejects_incremental_decode_at_full_attention() -> None:
    """Without a window the rolling cache drops history the full forward keeps. The trunk raises
    L_ctx frames into the first match; this raises at startup."""
    with pytest.raises(ValueError, match="eval_incremental_kv"):
        exp022.validate_config(_tiny_cfg(attn_window=0, eval_incremental_kv=True), has_button_combo_counts=False)
    exp022.validate_config(_tiny_cfg(attn_window=8, eval_incremental_kv=True), has_button_combo_counts=False)


def test_validate_config_rejects_an_indivisible_effective_batch() -> None:
    with pytest.raises(ValueError, match="EFFECTIVE batch"):
        exp022.validate_config(_tiny_cfg(batch_size=6, grad_accum_steps=4), has_button_combo_counts=False)


# --- end to end on real data -------------------------------------------------


@pytest.mark.skipif(not (_DEV_MDS / "train").is_dir(), reason="local dev MDS not materialized")
@pytest.mark.parametrize("awr_enabled", [False, True], ids=["bc", "awr"])
def test_mini_train_saves_final_before_a_skipped_eval(tmp_path, monkeypatch, capsys, awr_enabled: bool) -> None:
    """Four real steps per arm over the dev MDS at a small window: the loader labels returns, the
    micro-batch/accumulation split runs, ``final.pt`` lands, and a zero matchup count means the
    closed-loop eval never starts. On CPU the trunk falls back to the dense mask — the same math."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("AWS_BUCKET", "hal-test")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://127.0.0.1:9")  # discard port: uploads fail instantly
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")

    def _no_dolphin(*args, **kwargs):
        raise AssertionError("the closed-loop eval must not run when final_eval_n_matchups is 0")

    monkeypatch.setattr(exp022, "eval_vs_cpu", _no_dolphin)

    cfg = exp022.TrainConfig(
        d_model=32,
        n_layers=2,
        n_heads=2,
        L_ctx=64,
        attn_window=16,
        head_offsets=(1, 2),
        batch_size=4,
        grad_accum_steps=2,
        max_steps=4,
        warmup_steps=1,
        val_every=2,
        val_n_batches=2,
        gradient_diagnostic_batch_size=2,
        eval_every=0,
        final_eval_n_matchups=0,
        ckpt_every=0,
        num_workers=0,
        windows_per_replay=2,
        awr_enabled=awr_enabled,
        data_root=str(_DEV_MDS),
        val_split="train",
        mds_schema_version=5,
    )
    exp022.train(cfg, _stats(), comment="pytest")

    out = capsys.readouterr().out
    assert "step 3:" in out
    assert "closed-loop eval skipped" in out
    runs = list((tmp_path / "runs").iterdir())
    assert len(runs) == 1
    assert (runs[0] / "final.pt").is_file()
