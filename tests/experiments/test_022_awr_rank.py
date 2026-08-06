"""022 forks 020 onto the shared sliding-window trunk, retunes the reward and prunes the val block.
020's own tests still pin the return, the window threading and the weights; these tests pin what the
fork changed.

1. Nothing moved that should not. With the window off, the BC arm's objective is 020's plain mean
   NLL on the same batch, and the trunk swap draws the same weights under the same seed.
2. The defaults are the tuned recipe: the reward explorer's table, plus the SWA-128 long-context
   geometry.
3. Validation costs two trunk forwards per batch, not five, holds no per-frame tensor across
   batches, and still reports 020's numbers.
4. The new config states are checked: a negative window, incremental decode at full attention, and
   an effective batch that does not divide into micro-batches.
5. ``final.pt`` is written before the closed-loop eval, and ``final_eval_n_matchups=0`` skips that
   eval altogether.
6. Rank weighting: the tier is read from the window's last frame, it follows the ego coin flip, an
   UNKNOWN tier raises only when the multipliers are on, the 1:2:4 ratio survives the mean-1 rescale
   exactly, and the tier never reaches the model as a feature.
7. The IQL expectile critic: tau=0.5 is half the MSE, a true-return V leaves no residual at any tau,
   tau=0.9 puts 9x the gradient on a positive residual, the bootstrap is detached, and the last
   position (no successor state) is out of the loss.
"""

import copy
import importlib.util
import inspect
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
import torch

from hal.data.feature_stats import FeatureStats
from hal.data.schema import Rank
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
    """One ``TrainBatch`` plus the ego return column, ready to wrap in either module's ``AWRBatch``.

    The padding depends on the seed, so a val cache of several batches scores a different number of
    positions in each. A pooled mean that forgot to weight by those counts then reads wrong."""
    gen = torch.Generator().manual_seed(seed)
    pad = torch.tensor([0, 1 + 64 * seed] * (batch // 2))
    ctx = Context(features=_features(batch, L_ctx, gen), ctx_pad=pad)
    target = torch.rand(batch, max_offset, A_DIM, generator=gen) * 2 - 1
    return TrainBatch(context=ctx, target=target), torch.randn(batch, L_ctx, generator=gen)


def _rewards_from_returns(returns: torch.Tensor, gamma: float) -> torch.Tensor:
    """The reward column these returns came from: ``r_t = G_t - gamma * G_{t+1}``, the exact inverse
    of the discounted scan. A synthetic batch built this way satisfies the collate's identity
    ``returns[t] = rewards[t] + gamma * returns[t+1]``, which is what the TD critic is defined
    against. The last column has no successor, so its reward is its return (an episode end)."""
    rewards = returns.clone()
    rewards[:, :-1] -= gamma * returns[:, 1:]
    return rewards


def _awr_batch(cfg, seed: int = 0, batch: int = 4, ranks: Sequence[int] | None = None) -> exp022.AWRBatch:
    """A synthetic 022 batch: 020's inner batch plus a coherent reward/return pair and a tier."""
    inner, returns = _batch(cfg.L_ctx, max(cfg.head_offsets), batch=batch, seed=seed)
    rank = torch.tensor([int(Rank.PLATINUM)] * batch if ranks is None else list(ranks), dtype=torch.uint8)
    table = torch.tensor((1.0, *cfg.awr_rank_weights), dtype=torch.float32)
    return exp022.AWRBatch(
        batch=inner,
        returns=returns,
        rewards=_rewards_from_returns(returns, cfg.awr_gamma),
        rank=rank,
        rank_weight=table[rank.long()],
    )


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
    """The reward explorer's tuned table; the audit puts ESS ~0.70 and ~0% on the clip."""
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
    assert cfg.windows_per_replay == 2  # 32 distinct replays per micro-batch for the AWR rescale
    assert cfg.val_n_batches == 128  # x 64 windows = 020's 8,192-replay val set
    assert (cfg.data_root, cfg.mds_schema_version) == ("data/processed/ranked-anonymized-1/mds-v6", 6)


def test_head_offsets_are_contiguous_so_chunked_execution_is_possible() -> None:
    """Striding needs the heads 1..s from one forward. 012's (1,5,9,13) has no offset-2 head, so
    every execution horizon above 1 is refused with it."""
    cfg = exp022.TrainConfig()
    assert cfg.head_offsets == (1, 2, 3, 4)
    assert exp022._exec_horizon_offsets(cfg.head_offsets, 2) == (1, 2)
    with pytest.raises(ValueError, match="contiguous prefix"):
        exp022._exec_horizon_offsets((1, 5, 9, 13), 2)


def test_the_loader_gets_the_micro_batch() -> None:
    kwargs = exp022._loader_kwargs(exp022.TrainConfig(), _stats())
    assert kwargs["batch_size"] == 64


def test_reward_tag_follows_the_flags() -> None:
    """Every knob that changes the objective must be readable in the run name, and a knob that is
    off must leave no token behind."""
    assert exp022._reward_tag(exp022.TrainConfig()) == "g99827-b0.8-w5-d0.01-win0.5-swa128"
    assert exp022._reward_tag(exp022.TrainConfig(awr_enabled=False)) == "bc-swa128"
    assert exp022._reward_tag(exp022.TrainConfig(attn_window=0)).endswith("-swafull")
    assert "b0.3" in exp022._reward_tag(exp022.TrainConfig(awr_beta=0.3))
    assert exp022._reward_tag(exp022.TrainConfig(awr_value_detach_trunk=True)).endswith("-vdetach-swa128")


# --- only the trunk and the objective's weights moved ------------------------

# The reward -> return -> weight -> objective chain the fork must NOT touch. 020's tests pin the
# behavior of 020's copies, so nothing else would notice 022 drifting away from them.
_AWR_MACHINERY = (
    "stock_loss_events",
    "match_point_events",
    "damage_taken",
    "frame_reward",
    "group_nll",
    "_multi_offset_targets",
    "_offset_objective",
    "_offset_total_bits",
)

# What the rank and IQL work deliberately took OUT of the pin above, and why. Each entry names a
# behavior change the tests below pin directly; the assertion here only stops one from being
# reverted to 020's text by accident, which would silently drop the feature the reason names.
_UNPINNED_FROM_020 = {
    "replay_returns": "emits the per-frame reward beside the return, so the TD critic has its r",
    "collate_awr_batch": "stacks the reward and the ego tier, and takes the rank multipliers",
    "_attach_returns": "passes cfg.awr_rank_weights down to that collate",
    "awr_weights": "takes a rank multiplier, and an advantage of None for the rank-only BC path",
    "action_loss": "keeps V as a [B, L_ctx] grid, because the TD target reads the NEXT column",
}


@pytest.mark.parametrize("name", _AWR_MACHINERY)
def test_the_awr_machinery_is_020s_line_for_line(name: str) -> None:
    """022 changed the trunk, the geometry, the rewards, the val block, the rank weighting and the
    critic. The reward-to-return chain under all of that is still 020's, so an edit to it is a
    defect until this test says so."""
    assert inspect.getsource(getattr(exp022, name)) == inspect.getsource(getattr(exp020, name))


@pytest.mark.parametrize("name", sorted(_UNPINNED_FROM_020))
def test_the_unpinned_functions_are_the_ones_rank_and_iql_needed(name: str) -> None:
    """The other half of the pin: these five diverge from 020 on purpose (``_UNPINNED_FROM_020`` says
    what each one gained). If one ever matches 020's source again, the feature went missing."""
    assert inspect.getsource(getattr(exp022, name)) != inspect.getsource(getattr(exp020, name)), (
        f"{name} is 020's again, so it lost: {_UNPINNED_FROM_020[name]}"
    )


@pytest.mark.parametrize("gamma", [0.0, 0.5, 0.9, 0.99827, 1.0])
def test_vectorized_returns_reproduce_020s_python_scan(gamma: float) -> None:
    """The only piece of the return chain that changed: 020 scans in Python, 022 runs the same
    recurrence through ``lfilter``. Same values, on a replay-length reward with real stock events —
    otherwise the reward is unchanged and the return moved."""
    rng = np.random.default_rng(0)
    reward = rng.normal(scale=0.01, size=10_700).astype(np.float32)
    reward[[900, 3400, 5000, 9999]] = [-1.0, 1.0, -1.5, 1.5]

    got = exp022.discounted_returns(reward, gamma)
    want = exp020.discounted_returns(reward, gamma)

    assert got.dtype == want.dtype == np.float32
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("reward", [[], [0.0], [1.0, float("nan"), 2.0, 3.0]], ids=["empty", "one", "nan"])
def test_vectorized_returns_keep_020s_degenerate_cases(reward: list[float]) -> None:
    """A torn replay can arrive with no frames, and a NaN must poison the same tail in both. These
    are the shapes a C recurrence is most likely to answer differently from a Python loop."""
    x = np.asarray(reward, dtype=np.float32)

    got = exp022.discounted_returns(x, 0.9)
    want = exp020.discounted_returns(x, 0.9)

    assert got.shape == want.shape
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
    np.testing.assert_array_equal(got[~np.isnan(got)], want[~np.isnan(want)])


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


def test_the_decode_cast_leaves_the_quantization_grids_in_fp32() -> None:
    """fp16 weights are a speed choice, but the stick/trigger grids are the decode's OUTPUT scale:
    a stored value must reproduce its exact byte through the pipe, and an fp32->fp16->fp32 round trip
    moves a centre by 2e-4."""
    model = exp022.GPT(_tiny_cfg())
    before = model.main_centers.clone()

    exp022._halve_for_decode(model)

    assert next(model.parameters()).dtype == torch.float16
    for name in ("main_centers", "c_centers", "trig_centers"):
        assert getattr(model, name).dtype == torch.float32, name
    assert torch.equal(model.main_centers, before)
    assert model.button_combo_counts.dtype == torch.int64  # an integer buffer the cast must not reach


def test_the_decode_cast_moves_no_action_the_dequantizer_can_produce() -> None:
    """The named-grid check above states the rule; this one holds it for whatever grid gets added
    next. Every reachable group index must dequantize to the SAME action before and after the cast —
    that is the whole contract the fp32 grids exist for, stated on the output instead of the buffer
    list."""
    model = exp022.GPT(_tiny_cfg())
    reference = copy.deepcopy(model)
    n_trig = model.trig_centers.shape[0]
    idx = torch.stack(
        torch.meshgrid(
            torch.arange(4),
            torch.arange(model.main_centers.shape[0]),
            torch.arange(model.c_centers.shape[0]),
            torch.arange(n_trig * n_trig),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 4)

    exp022._halve_for_decode(model)

    assert torch.equal(exp022._dequantize(model, idx), exp022._dequantize(reference, idx))


def test_validation_runs_eager_while_training_stays_compiled() -> None:
    """``compile_trunk`` installs the compiled forward as an instance attribute. A compiled graph
    that then meets a validation or gradient-diagnostic shape dies inside FlexAttention with a CUDA
    illegal memory access, so ``_evaluation_mode`` — the one door every non-training forward goes
    through — takes that attribute off for the duration and puts the SAME object back, keeping the
    graph cache."""
    model = exp022.GPT(_tiny_cfg())
    eager = model.forward

    def wrapper(*args, **kwargs):  # what torch.compile leaves on the instance
        return eager(*args, **kwargs)

    model.forward = wrapper
    model.train()

    with exp022._evaluation_mode(model):
        assert model.forward is not wrapper
        assert not model.training

    assert model.forward is wrapper
    assert model.training


def test_compiling_the_bound_method_leaves_the_checkpoint_keys_alone() -> None:
    """The knob compiles the BOUND METHOD and not the module, so no trunk key gains an ``_orig_mod``
    prefix and resume still finds the weights where it left them."""
    assert exp022.TrainConfig().compile_trunk is True
    model = exp022.GPT(_tiny_cfg())
    keys = set(model.state_dict())

    model.forward = lambda *args, **kwargs: None

    assert set(model.state_dict()) == keys
    assert not any("_orig_mod" in key for key in keys)


def _incremental_policy(attn_window: int = 8, L_ctx: int = 32, exec_horizon: int = 1):
    cfg = _tiny_cfg(
        L_ctx=L_ctx, attn_window=attn_window, eval_incremental_kv=True, exec_horizon=exec_horizon, head_offsets=(1, 2)
    )
    torch.manual_seed(3)
    model = exp022.GPT(cfg).eval()
    return cfg, model, exp022.make_policy(model, _stats(), cfg, device="cpu")


def _one_frame(features: dict[str, torch.Tensor], rows: list[int], t: int, slot_ids: list[int], reset: list[bool]):
    """The ``Context`` ``RecedingHorizon`` hands an incremental decoder: ONE frame per live slot."""
    return Context(
        features={name: value[rows, t : t + 1] for name, value in features.items()},
        ctx_pad=torch.zeros(len(rows), dtype=torch.long),
        slot_ids=torch.tensor(slot_ids, dtype=torch.long),
        reset=torch.tensor(reset, dtype=torch.bool),
    )


def _closure_dict(policy, value_type: type) -> dict:
    """The per-slot ``kv_cache`` / ``hidden`` state, read out of the policy closure. Nothing else
    exposes it, and these tests are about exactly that state."""
    return next(
        cell.cell_contents
        for cell in policy.encode_frame.__wrapped__.__closure__
        if isinstance(cell.cell_contents, dict)
        and cell.cell_contents
        and all(isinstance(value, value_type) for value in cell.cell_contents.values())
    )


def _cache_lengths(policy) -> dict[int, int]:
    return {sid: entry[0][0].size(2) for sid, entry in _closure_dict(policy, list).items()}


def test_an_instant_restart_drops_that_slots_cache_and_only_that_slots() -> None:
    """The reset flag and the KV cache must move on the SAME frame, or the first frames of a new
    match attend to the last frames of the old one. A wave restarts asynchronously, so the slot that
    did not restart must keep every frame it had."""
    _, _, policy = _incremental_policy(attn_window=8)
    features = _features(2, 20, torch.Generator().manual_seed(4))
    restart_at = 12

    lengths = []
    for t in range(20):
        rows, ids = ([0], [10]) if t < 5 else ([0, 1], [10, 11])
        reset = [t == 0 or t == restart_at] if t < 5 else [t == restart_at, t == 5]
        policy.encode_frame(_one_frame(features, rows, t, ids, reset))
        lengths.append(_cache_lengths(policy))

    # Slot 10 alone: it fills, saturates at the window, and starts again from 1 on the restart frame.
    assert [x[10] for x in lengths] == [1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 1, 2, 3, 4, 5, 6, 7, 8]
    # Slot 11 joins five frames later and is untouched by the other slot's restart.
    assert [x[11] for x in lengths[5:]] == [1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8]


def test_slots_at_different_cache_lengths_decode_as_if_they_were_alone() -> None:
    """``encode_frame`` batches slots by cache LENGTH, so a wave whose matches drift apart runs
    several forwards a frame. Each slot's hidden state must be the one it would get on its own —
    including across the frame where a restart moves one slot into a different group."""
    _, _, policy = _incremental_policy(attn_window=8)
    features = _features(2, 14, torch.Generator().manual_seed(4))
    restart_at = 9

    for t in range(14):
        rows, ids = ([0], [10]) if t < 3 else ([0, 1], [10, 11])
        reset = [t == 0] if t < 3 else [t == restart_at, t == 3]
        policy.encode_frame(_one_frame(features, rows, t, ids, reset))

    got = _closure_dict(policy, torch.Tensor)

    # The same two slots, each encoded alone, from its own first frame.
    for row, sid, first in ((0, 10, restart_at), (1, 11, 3)):
        alone = _incremental_policy(attn_window=8)[2]
        for t in range(first, 14):
            alone.encode_frame(_one_frame(features, [row], t, [sid], [t == first]))
        torch.testing.assert_close(got[sid], _closure_dict(alone, torch.Tensor)[sid], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("k", [1, 2, 8, 9, 32])
def test_a_re_warming_slot_decodes_like_the_padded_full_forward_it_replaces(k: int) -> None:
    """Straight after a restart the cache holds FEWER frames than the window and no padding at all,
    where training left-padded the window and hid the prefix with ``ctx_pad``. If the two disagreed,
    every frame right after a match boundary would be decoded off-distribution. The pad frames here
    carry loud garbage, so any attention that reaches them moves the answer."""
    cfg, model, policy = _incremental_policy(attn_window=8, L_ctx=32)
    features = _features(1, cfg.L_ctx, torch.Generator().manual_seed(4))
    garbage = _features(1, cfg.L_ctx, torch.Generator().manual_seed(99))

    for t in range(k):
        policy.encode_frame(_one_frame(features, [0], t, [10], [t == 0]))
    got = _closure_dict(policy, torch.Tensor)[10]

    padded = {
        name: torch.cat([garbage[name][:, : cfg.L_ctx - k] * 50, value[:, :k]], dim=1)
        if value.is_floating_point()
        else torch.cat([garbage[name][:, : cfg.L_ctx - k], value[:, :k]], dim=1)
        for name, value in features.items()
    }
    want = model(padded, torch.tensor([cfg.L_ctx - k]))[0, -1]

    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_the_cached_path_still_refuses_a_multi_token_call() -> None:
    """The cached attention applies no mask, so two tokens in one call would see each other."""
    _, model, _ = _incremental_policy(attn_window=8)
    features = _features(1, 2, torch.Generator().manual_seed(4))

    with pytest.raises(ValueError, match="one token"):
        model.forward_incremental(features, [None] * 2)


@pytest.mark.parametrize("incremental", [True, False])
def test_the_policy_wires_the_incremental_seams_together(incremental: bool) -> None:
    """Both callbacks or neither: an incremental decoder that is not fed every frame would decode
    from a state several frames stale as soon as the execution horizon passes 1."""
    cfg = _tiny_cfg(attn_window=8, eval_incremental_kv=incremental, exec_horizon=2)
    policy = exp022.make_policy(exp022.GPT(cfg).eval(), _stats(), cfg, device="cpu")

    assert (policy.encode_frame is not None) == incremental
    assert (policy.predict_incremental is not None) == incremental
    assert policy.s == 2
    assert policy.float_dtype == torch.float32  # an fp32 model must not be handed an fp16 context


def test_020_checkpoints_are_refused() -> None:
    """020 stores the trunk at ``blocks.*``. Loading it here would silently mismatch the stack."""
    (_, model020), (_, model022) = _paired_models()
    with pytest.raises(RuntimeError, match="016/019/020/021"):
        exp022._load_model_state(model022, model020.state_dict())


# --- validation: two forwards, and the same numbers --------------------------


def _val_cache(cfg022, n_batches: int = 2):
    cache022 = [_awr_batch(cfg022, seed=seed) for seed in range(n_batches)]
    batches = [b.batch for b in cache022]
    cache020 = [exp020.AWRBatch(batch=b.batch, returns=b.returns) for b in cache022]
    return batches, cache020, cache022


# The val numbers 022 reports that 020 has no counterpart for: the TD critic's diagnostics (logged by
# both critics, so the arms compare) and the tier block.
_NEW_VAL_METRICS = frozenset(
    {"td_residual_mean", "td_expectile_loss", "rank_unknown_frac"}
    | {f"rank_{stat}_{tier.name.lower()}" for stat in ("frac", "weight_mean") for tier in exp022._RANK_TIERS}
)


@pytest.mark.parametrize(
    ("n_batches", "tolerance"),
    [(1, 0.0), (3, 1e-5)],
    ids=["one-batch-bitwise", "three-batches-pooled"],
)
def test_val_metrics_reproduce_020_numbers(n_batches: int, tolerance: float) -> None:
    """020 runs the trunk five times per val batch and concatenates a per-frame tensor for every
    metric; 022 runs it twice and folds each batch into a count-weighted mean, so a 128-batch pass
    holds no per-frame state. On one batch that is the same arithmetic, bit for bit. Several batches
    pool the same positions in a different order. A float32 sum over 65k of them carries about 1e-7
    of relative slack, and the ablation metrics subtract two nearly equal means, which turns that
    into ~1e-6 absolute — hence the tolerance. The batches pad by 1, 65 and 129 frames, so a fold
    that ignored the counts would land far outside it."""
    (cfg020, model020), (cfg022, model022) = _paired_models()
    batches, cache020, cache022 = _val_cache(cfg022, n_batches)

    want = exp020.val_metrics(model020, batches, cfg020)
    want_awr, want_weights = exp020.awr_val_metrics(model020, cache020, cfg020)
    got, got_weights = exp022.val_metrics(model022, cache022, cfg022)

    assert set(got) - _NEW_VAL_METRICS <= set(want) | set(want_awr)
    assert set(got) >= _NEW_VAL_METRICS
    for key, value in got.items():
        if key in _NEW_VAL_METRICS:
            continue
        expected = want_awr[key] if key in want_awr else want[key]
        if tolerance == 0.0:
            assert value == expected, key
        else:
            assert value == pytest.approx(expected, rel=tolerance, abs=tolerance), key
    # The weights are a property of the pooled val set (the mean-1 rescale), so they stay exact.
    torch.testing.assert_close(got_weights, want_weights, rtol=0, atol=0)


def test_val_metrics_hold_no_per_frame_state_across_batches() -> None:
    """The reason for the restructure: at 128 batches of 64 x 1024 frames, one retained per-frame
    tensor per metric is gigabytes of VRAM. Only the change masks and the weight inputs survive a
    batch, so ten batches must cost about as much retained memory as one."""
    (_, model022) = _paired_models()[1]
    cfg022 = exp022.TrainConfig(batch_size=4, grad_accum_steps=1, attn_window=0, **_PAIRED, **_PAIRED_AWR)

    accumulator = exp022._MeanAccumulator()
    accumulator.add("mean", 2.0, 3)
    accumulator.add("mean", 6.0, 1)
    accumulator.add("empty", 0.0, 0)
    assert accumulator.means() == {"mean": 3.0, "empty": 0.0}

    cache = _val_cache(cfg022, 3)[2]
    _, weights = exp022.val_metrics(model022, cache, cfg022)
    scored = sum(int((cfg022.L_ctx - b.batch.context.ctx_pad).sum()) for b in cache)
    assert weights.numel() == scored  # one weight per scored position, over every batch


def test_val_block_drops_the_uninformative_metrics() -> None:
    """The correlation study found these either uninformative or inverted against closed-loop play."""
    (_, model022) = _paired_models()[1]
    cfg022 = exp022.TrainConfig(batch_size=4, grad_accum_steps=1, attn_window=0, **_PAIRED, **_PAIRED_AWR)
    metrics, _ = exp022.val_metrics(model022, _val_cache(cfg022)[2], cfg022)

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
