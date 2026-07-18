"""Decode-time hygiene contracts for the group-factored next-token policies (011 / 012).

Both experiments sample four independent action groups per frame (buttons 256, main-stick 65,
c-stick 9, trigger-pair 25) and dequantize to a 14-dim action vector. These tests pin the four
decode-time controls added on top of that path — button-combo support masking, per-group
temperatures, min-p nucleus filtering, and the digital-click => analog-trigger fix — plus the
invariant that with every knob at its default the sampler is bit-for-bit the pre-change one.

The experiment filenames start with a digit, so they load by path (like ``test_rtc_policy``); the
decode logic is identical across the two files, so every test runs against both via parametrization.
Models are tiny and CPU-only, with the output head overwritten by a fixed-logit stub so the sampled
group is fully controlled — no training, no real gamestate signal.
"""

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from hal.training import scoring
from hal.training.features import ACTION_CHANNELS
from hal.training.features import CAT_FEATURES
from hal.training.features import FLOAT_FEATURES
from hal.training.features import Context

_EXP_DIR = Path(__file__).resolve().parent.parent / "experiments"
_PLAYER_PREFIXES = ("ego", "ego_nana", "opp_nana", "opp")


def _load_experiment(filename: str):
    spec = importlib.util.spec_from_file_location(filename.split(".")[0], _EXP_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_EXP011 = _load_experiment("011_muon.py")
_EXP012 = _load_experiment("012_multi_token.py")
_EXPERIMENTS = [_EXP011, _EXP012]
_IDS = ["011", "012"]


class _FixedHead(torch.nn.Module):
    """Output head that ignores the backbone hidden and returns a fixed logit ``bank`` per position,
    so a test fully controls each group's logit slice. ``bank`` is ``[Bbank, A_VOCAB]`` with
    ``Bbank`` either 1 (broadcast to any batch) or the batch size. Handles both the ``[B, d_model]``
    input 012 feeds its head post-slice and the ``[B, L, d_model]`` input 011's lm_head sees."""

    def __init__(self, bank: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("bank", bank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a_vocab = self.bank.shape[-1]
        if x.ndim == 2:
            return self.bank.expand(x.shape[0], a_vocab)
        return self.bank.view(self.bank.shape[0], 1, a_vocab).expand(x.shape[0], x.shape[1], a_vocab)


def _rigged_model(exp, bank: torch.Tensor, *, L: int = 8):
    """A tiny CPU model whose head is replaced by a ``_FixedHead(bank)``, so ``decode`` samples from
    exactly ``bank``. 011 exposes ``lm_head``; 012 exposes a ``heads`` ModuleList (primary = offset 1)."""
    cfg = exp.TrainConfig(d_model=64, n_layers=1, n_heads=2, L_ctx=L)
    model = exp.GPT(cfg)
    model.eval()
    if hasattr(model, "lm_head"):
        model.lm_head = _FixedHead(bank)
    else:
        model.heads[model.primary_head_idx] = _FixedHead(bank)
    return model


def _zeros_context(B: int, L: int) -> Context:
    """A structurally valid all-zeros Context (every feature key the backbone reads). The fixed head
    ignores the hidden, so the actual values are irrelevant — only shapes/dtypes must be right."""
    features: dict[str, torch.Tensor] = {}
    for prefix in _PLAYER_PREFIXES:
        for feat in FLOAT_FEATURES:
            features[f"{prefix}_{feat}"] = torch.zeros(B, L)
        for name in CAT_FEATURES:
            features[f"{prefix}_{name}"] = torch.zeros(B, L, dtype=torch.long)
    for ch in ACTION_CHANNELS:
        features[f"ego_{ch}"] = torch.zeros(B, L)
    for key in ("ego_character", "opp_character", "stage"):
        features[key] = torch.zeros(B, L, dtype=torch.long)
    return Context(features=features, ctx_pad=torch.zeros(B, dtype=torch.long))


def _one_hot_row(exp, *, combo: int, trig: int = 0, main: int = 0, c: int = 0, hi: float = 30.0, lo: float = -30.0):
    """A ``[1, A_VOCAB]`` logit bank favoring one class per group (near-deterministic under softmax)."""
    row = torch.full((1, exp.A_VOCAB), lo)
    go = exp._GROUP_OFFSETS
    row[0, go[0] + combo] = hi
    row[0, go[1] + main] = hi
    row[0, go[2] + c] = hi
    row[0, go[3] + trig] = hi
    return row


def _combos(exp, action: torch.Tensor) -> torch.Tensor:
    """Recover the button-combo id per decoded action ``[..., A_DIM]`` (buttons are channels [6:14])."""
    return scoring.buttons_to_combo(action[..., exp._N_CONT :]).reshape(-1)


def _main_clusters(exp, model, action: torch.Tensor) -> torch.Tensor:
    """Recover the main-stick cluster id per decoded action (identity on the centers)."""
    return scoring.nearest_cluster(action[..., 0:2], model.main_centers).reshape(-1)


# --- bit-channel mapping (the click=>trigger fix's correctness anchor) --------
@pytest.mark.parametrize("exp", _EXPERIMENTS, ids=_IDS)
def test_click_trigger_channel_indices_are_pinned(exp):
    # The click=>trigger fix hinges on these: button_l is combo bit 6 / action channel 12; button_r
    # is combo bit 5 / action channel 11; the analog triggers are channels 4 (l) and 5 (r).
    assert exp._TRIGGER_L_CH == 4 and exp._TRIGGER_R_CH == 5
    assert exp._BUTTON_L_CH == 12 and exp._BUTTON_R_CH == 11
    assert ACTION_CHANNELS[12] == "button_l" and ACTION_CHANNELS[11] == "button_r"
    # a single L (resp. R) press packs to combo bit 6 (resp. 5).
    assert int(scoring.buttons_to_combo(scoring.combo_to_buttons(torch.tensor(1 << 6)))) == (1 << 6)
    assert int(scoring.buttons_to_combo(scoring.combo_to_buttons(torch.tensor(1 << 5)))) == (1 << 5)


# --- defaults preserve the exact pre-change sampler ---------------------------
@pytest.mark.parametrize("exp", _EXPERIMENTS, ids=_IDS)
def test_defaults_off_is_bit_identical_to_reference_sampler(exp):
    # With every hygiene knob at its default, decode must reproduce the pre-change semantics
    # (per-group softmax(logits/temp) then multinomial) bit-for-bit for a fixed generator.
    B, a_vocab = 64, exp.A_VOCAB
    bank = torch.randn(B, a_vocab, generator=torch.Generator().manual_seed(3))
    model = _rigged_model(exp, bank)
    ctx = _zeros_context(B, 8)

    got = exp.decode(model, ctx, gen=torch.Generator().manual_seed(11))

    ref_gen = torch.Generator().manual_seed(11)
    picks = []
    for g in range(exp.N_GROUPS):
        lo = exp._GROUP_OFFSETS[g]
        lg = bank[:, lo : lo + exp._GROUP_VOCABS[g]]
        picks.append(torch.multinomial(F.softmax(lg / 1.0, dim=-1), 1, generator=ref_gen).squeeze(-1))
    ref = exp._dequantize(model, torch.stack(picks, dim=-1))[:, None, :]
    assert torch.equal(got, ref)


# --- button-combo support masking --------------------------------------------
@pytest.mark.parametrize("exp", _EXPERIMENTS, ids=_IDS)
def test_support_mask_excludes_dead_combos_when_on(exp):
    counts = torch.tensor(scoring.BTN_COMBO_COUNTS)
    dead_idx = int((counts < 100).nonzero(as_tuple=True)[0][0])  # first combo below the 100-frame floor
    model = _rigged_model(exp, _one_hot_row(exp, combo=dead_idx))  # rig sampling straight onto it
    ctx = _zeros_context(1500, 8)

    # Masking off: the rigged dead combo is realized (independent temp-1 sampling can emit garbage).
    off = exp.decode(model, ctx, btn_support_min=0, gen=torch.Generator().manual_seed(0))
    assert bool((counts[_combos(exp, off)] < 100).any())

    # Masking on: every sampled combo clears the train-support floor; the dead favorite is gone.
    on = exp.decode(model, ctx, btn_support_min=100, gen=torch.Generator().manual_seed(0))
    assert bool((counts[_combos(exp, on)] >= 100).all())


@pytest.mark.parametrize("exp", _EXPERIMENTS, ids=_IDS)
def test_argmax_respects_support_mask(exp):
    # argmax ignores temps/min-p but must still honor the support mask: the greedy dead favorite is
    # masked to -inf, so argmax falls to a supported combo.
    counts = torch.tensor(scoring.BTN_COMBO_COUNTS)
    dead_idx = int((counts < 100).nonzero(as_tuple=True)[0][0])
    model = _rigged_model(exp, _one_hot_row(exp, combo=dead_idx))
    ctx = _zeros_context(2, 8)

    off = exp.decode(model, ctx, argmax=True, btn_support_min=0)
    assert int(_combos(exp, off)[0]) == dead_idx  # greedy picks the dead favorite when unmasked

    on = exp.decode(model, ctx, argmax=True, btn_support_min=100)
    assert bool((counts[_combos(exp, on)] >= 100).all())


# --- digital click => analog trigger = 1.0 -----------------------------------
@pytest.mark.parametrize("exp", _EXPERIMENTS, ids=_IDS)
def test_click_trigger_fix_forces_only_the_clicked_shoulder(exp):
    tl, tr, bl, br = exp._TRIGGER_L_CH, exp._TRIGGER_R_CH, exp._BUTTON_L_CH, exp._BUTTON_R_CH
    ctx = _zeros_context(2, 8)

    # Combo = digital L only (bit 6); trigger group = class 0 (both shoulders analog 0.0).
    model_l = _rigged_model(exp, _one_hot_row(exp, combo=1 << 6, trig=0))
    off = exp.decode(model_l, ctx, argmax=True, click_trigger_fix=False)
    on = exp.decode(model_l, ctx, argmax=True, click_trigger_fix=True)
    assert off[0, 0, bl] == 1.0 and off[0, 0, tl] == 0.0 and off[0, 0, tr] == 0.0  # unchanged when off
    assert on[0, 0, tl] == 1.0  # L click forces trigger_l to 1.0
    assert on[0, 0, tr] == 0.0  # the un-clicked R shoulder is untouched
    assert on[0, 0, bl] == 1.0 and on[0, 0, br] == 0.0  # button bits themselves unchanged

    # Combo = digital R only (bit 5): the fix must move trigger_r, and leave trigger_l alone.
    model_r = _rigged_model(exp, _one_hot_row(exp, combo=1 << 5, trig=0))
    on_r = exp.decode(model_r, ctx, argmax=True, click_trigger_fix=True)
    assert on_r[0, 0, br] == 1.0 and on_r[0, 0, tr] == 1.0 and on_r[0, 0, tl] == 0.0


# --- per-group temperatures ---------------------------------------------------
@pytest.mark.parametrize("exp", _EXPERIMENTS, ids=_IDS)
def test_per_group_temps_sharpen_buttons_keep_stick_diverse(exp):
    # Near-uniform button logits with one slightly-favored class; uniform main-stick logits. A cold
    # button temperature collapses buttons onto the favorite while the warm stick temperature stays
    # diverse — the whole point of decoupling the per-group temperatures.
    bank = torch.zeros(1, exp.A_VOCAB)
    bank[0, exp._GROUP_OFFSETS[0] + 0] = 1.0  # buttons: mild edge for combo 0
    model = _rigged_model(exp, bank)
    ctx = _zeros_context(2000, 8)

    action = exp.decode(model, ctx, temps=(0.01, 1.0, 1.0, 1.0), gen=torch.Generator().manual_seed(0))
    combos = _combos(exp, action)
    clusters = _main_clusters(exp, model, action)
    assert (combos == 0).float().mean() > 0.98  # buttons ~deterministic at temp 0.01
    assert int(clusters.unique().numel()) > 20  # main-stick still spread across many clusters


# --- min-p nucleus filtering --------------------------------------------------
@pytest.mark.parametrize("exp", _EXPERIMENTS, ids=_IDS)
def test_min_p_never_samples_the_tail(exp):
    # Main-stick logits: two head classes {0, 1} well above a flat tail. min-p zeroes every class
    # under min_p * p_max, so only the head is ever sampled; with min-p off the tail shows up.
    bank = torch.zeros(1, exp.A_VOCAB)
    bank[0, exp._GROUP_OFFSETS[1] + 0] = 3.0
    bank[0, exp._GROUP_OFFSETS[1] + 1] = 3.0
    model = _rigged_model(exp, bank)
    ctx = _zeros_context(1000, 8)

    filtered = _main_clusters(exp, model, exp.decode(model, ctx, min_p=0.1, gen=torch.Generator().manual_seed(0)))
    assert set(filtered.tolist()) <= {0, 1}  # tail below the min-p threshold is never drawn

    unfiltered = _main_clusters(exp, model, exp.decode(model, ctx, min_p=0.0, gen=torch.Generator().manual_seed(0)))
    assert bool((unfiltered >= 2).any())  # without min-p the flat tail is reachable


# --- fail loud on invalid arguments ------------------------------------------
@pytest.mark.parametrize("exp", _EXPERIMENTS, ids=_IDS)
def test_decode_rejects_invalid_arguments(exp):
    model = _rigged_model(exp, torch.zeros(1, exp.A_VOCAB))
    ctx = _zeros_context(2, 8)
    with pytest.raises(ValueError):
        exp.decode(model, ctx, temps=(1.0, 1.0, 1.0))  # wrong length (not one per group)
    with pytest.raises(ValueError):
        exp.decode(model, ctx, temps=(1.0, 0.0, 1.0, 1.0))  # non-positive per-group temperature
    with pytest.raises(ValueError):
        exp.decode(model, ctx, btn_support_min=-1)  # negative support floor
    with pytest.raises(ValueError):
        exp.decode(model, ctx, min_p=1.5)  # min-p outside [0, 1]
