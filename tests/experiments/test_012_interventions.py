"""Contracts for the three flag-gated 012 interventions (history dropout, transition-weighted objective,
chunked execution). Each is a delta of the multi-token experiment whose DEFAULT must reproduce prior 012
behavior bit-for-bit; these pin the on-behavior and the off-is-unchanged invariant for every one.

012 loads by path (its filename starts with a digit, like ``test_decode`` / ``test_002_flow_matching_rtc``). Models are
tiny and CPU-only; where the decoded group must be controlled the output head is overwritten by a fixed-logit
stub so no training is needed.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

from hal.training import scoring
from hal.training.features import A_DIM
from hal.training.features import ACTION_CHANNELS
from hal.training.features import CAT_FEATURES
from hal.training.features import FLOAT_FEATURES
from hal.training.features import Context
from hal.training.features import TrainBatch
from hal.training.features import stack_actions

_EXP_DIR = Path(__file__).resolve().parent.parent.parent / "experiments"


def _load_experiment(filename: str):
    spec = importlib.util.spec_from_file_location(filename.split(".")[0], _EXP_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


exp = _load_experiment("012_multi_token.py")


class _FixedHead(torch.nn.Module):
    """Output head that ignores the backbone hidden and returns a fixed ``[1, A_VOCAB]`` logit bank per
    position, so a test fully controls the sampled group. Handles both the ``[B, d_model]`` post-slice input
    the offset decode feeds its head and the ``[B, L, d_model]`` shape a full-sequence head would see."""

    def __init__(self, bank: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("bank", bank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a_vocab = self.bank.shape[-1]
        if x.ndim == 2:
            return self.bank.expand(x.shape[0], a_vocab)
        return self.bank.view(self.bank.shape[0], 1, a_vocab).expand(x.shape[0], x.shape[1], a_vocab)


def _one_hot_row(*, combo: int, trig: int = 0, main: int = 0, c: int = 0, hi: float = 30.0, lo: float = -30.0):
    """A ``[1, A_VOCAB]`` logit bank favoring one class per group (near-deterministic under softmax/argmax)."""
    row = torch.full((1, exp.A_VOCAB), lo)
    go = exp._GROUP_OFFSETS
    row[0, go[0] + combo] = hi
    row[0, go[1] + main] = hi
    row[0, go[2] + c] = hi
    row[0, go[3] + trig] = hi
    return row


def _zeros_features(B: int, L: int) -> dict[str, torch.Tensor]:
    """Every feature key the backbone reads, all zeros (structurally valid Context features)."""
    features: dict[str, torch.Tensor] = {}
    for prefix in exp._PLAYER_PREFIXES:
        for feat in FLOAT_FEATURES:
            features[f"{prefix}_{feat}"] = torch.zeros(B, L)
        for name in CAT_FEATURES:
            features[f"{prefix}_{name}"] = torch.zeros(B, L, dtype=torch.long)
    for ch in ACTION_CHANNELS:
        features[f"ego_{ch}"] = torch.zeros(B, L)
    for key in ("ego_character", "opp_character", "stage"):
        features[key] = torch.zeros(B, L, dtype=torch.long)
    return features


def _zeros_context(B: int, L: int) -> Context:
    return Context(features=_zeros_features(B, L), ctx_pad=torch.zeros(B, dtype=torch.long))


def _tiny_model(*, L: int = 8, head_offsets: tuple[int, ...] = (1, 5, 9, 13), history_dropout_p: float = 0.0):
    cfg = exp.TrainConfig(
        d_model=64, n_layers=1, n_heads=2, L_ctx=L, head_offsets=head_offsets, history_dropout_p=history_dropout_p
    )
    model = exp.GPT(cfg)
    model.eval()
    return cfg, model


def _combos(action: torch.Tensor) -> torch.Tensor:
    return scoring.buttons_to_combo(action[..., exp._N_CONT :]).reshape(-1)


# --- (1) per-sample ego-history input dropout --------------------------------
def test_history_dropout_zeros_ego_slice_only_in_train_and_leaves_targets_intact():
    B, L = 6, 8
    cfg, model = _tiny_model(L=L, history_dropout_p=1.0)  # p=1 -> every sample's ego slice dropped
    model.ctx_proj = torch.nn.Identity()  # inspect the raw pre-projection concat

    features = _zeros_features(B, L)
    for i, ch in enumerate(ACTION_CHANNELS):  # distinct nonzero ego history so a zeroing is observable
        features[f"ego_{ch}"] = torch.full((B, L), 0.25 + 0.05 * i)
    ctx = Context(features=features, ctx_pad=torch.zeros(B, dtype=torch.long))
    features_ref = {k: v.clone() for k, v in features.items()}
    ego_actions = stack_actions(features).clone()  # [B, L, A_DIM] — the un-dropped history
    assert torch.count_nonzero(ego_actions) > 0

    per_player_w = sum(model._per_player_features(features, p).shape[-1] for p in exp._PLAYER_PREFIXES)
    ego_lo, ego_hi = per_player_w, per_player_w + A_DIM

    # train + p=1: the ego-controller-history slice of the context token is zero for all samples.
    model.train()
    concat_train = model._context_tokens(features)
    assert torch.all(concat_train[..., ego_lo:ego_hi] == 0.0)
    # the SAME feature tensors targets are built from are untouched (dropout lives in input assembly only).
    for k, v in features_ref.items():
        assert torch.equal(features[k], v)
    assert torch.equal(stack_actions(ctx.features), ego_actions)

    # eval: dropout is train-only, so the ego slice reappears exactly.
    model.eval()
    concat_eval = model._context_tokens(features)
    assert torch.equal(concat_eval[..., ego_lo:ego_hi], ego_actions)

    # p=0 + train: masking disabled, slice unchanged.
    _, model0 = _tiny_model(L=L, history_dropout_p=0.0)
    model0.ctx_proj = torch.nn.Identity()
    model0.train()
    concat0 = model0._context_tokens(features)
    assert torch.equal(concat0[..., ego_lo:ego_hi], ego_actions)


def test_history_dropout_rejects_out_of_range_p():
    with pytest.raises(ValueError):
        exp.GPT(exp.TrainConfig(d_model=64, n_layers=1, n_heads=2, L_ctx=8, history_dropout_p=1.5))


# --- (2) transition-weighted objective ---------------------------------------
def test_weighted_mean_lambda1_is_the_plain_mean_exactly():
    nll = torch.tensor([2.0, 4.0, 1.0, 3.0])
    is_trans = torch.tensor([True, False, True, False])
    assert torch.equal(exp._weighted_mean(nll, is_trans, 1.0), nll.mean())


def test_weighted_mean_lambda5_matches_hand_computation():
    nll = torch.tensor([2.0, 4.0])
    is_trans = torch.tensor([True, False])
    # w = [5, 1] -> (5*2 + 1*4) / (5 + 1) = 14/6
    assert exp._weighted_mean(nll, is_trans, 5.0).item() == pytest.approx(14.0 / 6.0)


def test_objective_lambda1_reduces_to_unweighted_sum_of_means():
    nll = {
        (1, "buttons"): torch.tensor([1.0, 2.0]),
        (1, "main_stick"): torch.tensor([0.5, 1.5]),
        (5, "buttons"): torch.tensor([2.0, 2.0]),
        (5, "main_stick"): torch.tensor([1.0, 3.0]),
    }
    trans = {k: torch.tensor([True, False]) for k in nll}
    aux_weight = 0.7
    got = exp.objective(nll, trans, aux_weight, 1.0)
    want = torch.stack([(1.0 if o == 1 else aux_weight) * c.mean() for (o, _name), c in nll.items()]).sum()
    assert torch.equal(got, want)


def test_action_loss_transition_mask_aligns_to_the_target_boundary():
    # offsets=(1,): target frame for position i is i+1, a transition iff combo(i+1) != combo(i). Ego button_a
    # history [0,0,1,1] + target frame button_a=1 -> combo sequence [0,0,1,1,1]; the only boundary is frame 1->2,
    # i.e. position i=1. Other groups are constant (all zero), so their transition masks are empty.
    _, model = _tiny_model(L=4, head_offsets=(1,))
    features = _zeros_features(1, 4)
    features["ego_button_a"] = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    target = torch.zeros(1, 1, A_DIM)
    target[0, 0, ACTION_CHANNELS.index("button_a")] = 1.0
    batch = TrainBatch(context=Context(features=features, ctx_pad=torch.zeros(1, dtype=torch.long)), target=target)

    nll, trans = exp.action_loss(model, batch)
    assert set(nll) == {(1, name) for name in exp._GROUP_NAMES}
    assert trans[(1, "buttons")].tolist() == [False, True, False, False]
    for name in ("main_stick", "c_stick", "triggers"):
        assert not bool(trans[(1, name)].any())


# --- (3) chunked execution of contiguous MTP heads ---------------------------
def test_decode_chunk_places_each_offsets_combo_at_its_chunk_position():
    B, L = 3, 8
    offsets = (1, 2, 3, 4)
    combos = (1, 2, 4, 8)  # distinct single-button presses (a, b, x, y)
    cfg, model = _tiny_model(L=L, head_offsets=offsets)
    for j, o in enumerate(offsets):
        model.heads[model.head_offsets.index(o)] = _FixedHead(_one_hot_row(combo=combos[j]))
    ctx = _zeros_context(B, L)

    chunk = exp.decode_chunk(model, ctx, offsets, argmax=True)  # [B, s, A_DIM]
    assert chunk.shape == (B, len(offsets), A_DIM)
    for j in range(len(offsets)):
        assert torch.all(_combos(chunk[:, j]) == combos[j])


def test_decode_chunk_s1_is_byte_identical_to_decode():
    B = 32
    bank = torch.randn(B, exp.A_VOCAB, generator=torch.Generator().manual_seed(5))
    _, model = _tiny_model()  # offsets (1,5,9,13)
    model.heads[model.primary_head_idx] = _FixedHead(bank)
    ctx = _zeros_context(B, 8)

    out_decode = exp.decode(model, ctx, gen=torch.Generator().manual_seed(9))
    out_chunk = exp.decode_chunk(model, ctx, (1,), gen=torch.Generator().manual_seed(9))
    assert out_decode.shape == (B, 1, A_DIM)
    assert torch.equal(out_decode, out_chunk)


def test_make_policy_rejects_exec_horizon_without_the_contiguous_prefix():
    cfg, model = _tiny_model()  # offsets (1,5,9,13): no offset-2 head
    with pytest.raises(ValueError):
        exp.make_policy(model, {}, cfg, device="cpu", exec_horizon=2)


def test_make_policy_chunk_path_predicts_the_full_horizon():
    offsets = (1, 2, 3, 4)
    cfg = exp.TrainConfig(d_model=64, n_layers=1, n_heads=2, L_ctx=8, head_offsets=offsets, exec_horizon=2)
    model = exp.GPT(cfg)
    model.eval()
    policy = exp.make_policy(model, {}, cfg, device="cpu")  # exec_horizon falls back to cfg (=2)
    assert policy.s == 2 and policy.L_chunk == 2 and policy.d == 0
    out = policy.predict_chunk(_zeros_context(5, 8), None)  # numpy [B, s, A_DIM]
    assert out.shape == (5, 2, A_DIM)


def test_make_policy_default_exec_horizon_is_per_frame():
    cfg, model = _tiny_model()  # exec_horizon default 1
    policy = exp.make_policy(model, {}, cfg, device="cpu")
    assert policy.s == 1 and policy.L_chunk == 1
