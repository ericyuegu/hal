"""Contracts for the dense direct-MC critic probe."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


def _load():
    path = Path(__file__).resolve().parents[2] / "experiments" / "031_mc_critic_probe.py"
    spec = importlib.util.spec_from_file_location("test_exp031_mc", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


exp = _load()


def _codec():
    return exp._exp026.StructuredControllerCodec(embed_dim=4)


def _cfg():
    return exp.ProbeConfig(action_d_model=16, action_heads=4, action_ff_dim=32, head_hidden_dim=16)


def _chunks(batch: int = 3):
    generator = torch.Generator().manual_seed(7)
    codec = _codec()
    actions = torch.zeros(batch, 4, exp.A_DIM)
    actions[..., :4] = torch.rand(batch, 4, 4, generator=generator) * 2 - 1
    actions[..., 4:6] = torch.rand(batch, 4, 2, generator=generator)
    actions[..., 6:] = torch.randint(0, 2, (batch, 4, exp.A_DIM - 6), generator=generator).float()
    return codec, codec.quantize(actions)


def test_full_episode_returns_and_truncation_are_explicit() -> None:
    reward = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    returns, valid = exp.discounted_mc_return(reward, 0.5, terminated=True)
    np.testing.assert_allclose(returns, [2.75, 3.5, 3.0])
    assert valid.tolist() == [True, True, True]

    truncated, truncated_valid = exp.discounted_mc_return(reward, 0.5, terminated=False)
    assert np.isnan(truncated).all()
    assert not truncated_valid.any()


def test_replay_label_computes_before_windowing_and_marks_ambiguous_tail_invalid() -> None:
    sample = {
        "p1_stock": np.array([4, 4, 4], dtype=np.int32),
        "p2_stock": np.array([4, 4, 4], dtype=np.int32),
        "p1_percent": np.array([0, 0, 0], dtype=np.float32),
        "p2_percent": np.array([0, 10, 10], dtype=np.float32),
    }
    labeled = exp.label_mc_replay(sample, gamma=1.0, damage_shaping=0.1)
    assert np.isnan(labeled["p1_mc_probe_return"]).all()
    assert not labeled["p1_mc_probe_valid"].any()

    complete = exp.label_mc_replay(sample, gamma=1.0, damage_shaping=0.1, terminated=True)
    np.testing.assert_allclose(complete["p1_mc_probe_return"], [1.0, 1.0, 0.0])


def test_dense_alignment_is_s_t_to_future_action_and_g_t_plus_1() -> None:
    actions = torch.arange(2 * 7 * 4).reshape(2, 7, 4)
    returns = torch.arange(14).reshape(2, 7).float()
    valid = torch.ones(2, 7, dtype=torch.bool)
    aligned = exp.align_dense_prefixes(actions, returns, valid, torch.tensor([0, 1]), state_length=3)

    torch.testing.assert_close(aligned.chunks[:, :, 0], actions[:, 1:4])
    torch.testing.assert_close(aligned.chunks[:, :, 3], actions[:, 4:7])
    torch.testing.assert_close(aligned.target, returns[:, 1:4])
    assert aligned.valid_prefix_counts == {1: 5, 4: 5}
    assert aligned.q1_mask.tolist() == [[True, True, True], [False, True, True]]


@pytest.mark.parametrize("length,context", [(0, 8), (4, 8), (5, 8), (17, 8), (300, 128)])
def test_exhaustive_state_blocks_have_no_gaps_or_duplicates(length: int, context: int) -> None:
    blocks = exp.exhaustive_state_blocks(length, context)
    emitted = [index for start, stop in blocks for index in range(start, stop)]
    expected = list(range(max(length - 4, 0)))
    assert emitted == expected
    assert len(emitted) == len(set(emitted))
    assert all(stop - start <= context for start, stop in blocks)


def test_masked_huber_never_reads_invalid_nan_tail() -> None:
    prediction = torch.tensor([0.0, 2.0], requires_grad=True)
    target = torch.tensor([1.0, float("nan")])
    loss = exp.masked_huber(prediction, target, torch.tensor([True, False]))
    assert loss.item() == pytest.approx(0.5)
    loss.backward()
    assert prediction.grad.tolist() == [-1.0, 0.0]


def test_q1_is_causal_and_cannot_read_actions_two_through_four() -> None:
    codec, chunks = _chunks()
    critic = exp.CausalChunkCritic(codec, state_dim=8, cfg=_cfg()).eval()
    state = torch.randn(chunks.shape[0], 8)
    changed = chunks.clone()
    changed[:, 1:] = changed[:, 1:].roll(1, dims=0)
    before, after = critic(state, chunks), critic(state, changed)
    torch.testing.assert_close(before[:, 0], after[:, 0], atol=0, rtol=0)
    assert not torch.equal(before[:, 1], after[:, 1])


def test_null_control_has_exact_shape_and_ignores_chunk_content() -> None:
    codec, chunks = _chunks()
    treatment = exp.CausalChunkCritic(codec, 8, _cfg(), condition_on_action=True)
    control = exp.CausalChunkCritic(codec, 8, _cfg(), condition_on_action=False).eval()
    assert sum(p.numel() for p in treatment.parameters()) == sum(p.numel() for p in control.parameters())
    state = torch.randn(chunks.shape[0], 8)
    torch.testing.assert_close(control(state, chunks), control(state, chunks.roll(1, 0)))


def test_three_seed_members_and_ensemble_use_mean_not_minimum() -> None:
    cfg = _cfg()
    members = [exp.ProbeMember(_codec(), 8, cfg, seed) for seed in cfg.critic_seeds]
    first_weights = [next(member.value.parameters()).detach() for member in members]
    assert not torch.equal(first_weights[0], first_weights[1])
    predictions = torch.tensor([[1.0, 4.0], [2.0, 5.0], [6.0, 9.0]])
    mean, spread = exp.ensemble_statistics(predictions)
    torch.testing.assert_close(mean, torch.tensor([3.0, 6.0]))
    assert torch.all(spread > 0)


def test_action_and_weight_gates_enforce_all_requirements() -> None:
    advantage = torch.tensor([-0.2, 0.0, 0.1, 0.2])
    audit = exp.audit_advantage_weights(advantage, ["a", "a", "b", "b"])
    assert [row["beta"] for row in audit] == [0.2, 0.4, 0.8, 1.6, 3.2]
    assert all(0 < row["frame_ess"] <= 1 for row in audit)
    assert all(0 < row["replay_ess"] <= 1 for row in audit)

    passing = {
        "q1_mae": 0.1,
        "control_q1_mae": 0.2,
        "q4_mae": 0.1,
        "control_q4_mae": 0.2,
        "q1_replacement_degradation": 0.01,
        "q4_replacement_degradation": 0.01,
        "q1_replacement_lcb": 0.01,
        "q4_replacement_lcb": 0.01,
        "shuffle_lcb": 0.01,
        "sensitivity_effect_floor": 0.005,
        "shuffle_degradation": 0.01,
        "calibration_rmse": 0.1,
        "policy_sampled_spread": 0.1,
    }
    assert exp.action_conditioning_gate(
        [passing.copy() for _ in range(3)], max_calibration_rmse=0.2, max_sampled_spread=0.2
    )
    failed = [passing.copy() for _ in range(3)]
    failed[2]["shuffle_lcb"] = 0.0
    assert not exp.action_conditioning_gate(failed, max_calibration_rmse=0.2, max_sampled_spread=0.2)


def test_replacement_is_support_stratified_derangement_and_bootstrap_is_replay_blocked() -> None:
    chunks = torch.zeros(6, 4, 4, dtype=torch.long)
    chunks[:3, 0, exp._exp026.BUTTONS_G] = 1
    chunks[3:, 0, exp._exp026.BUTTONS_G] = 2
    chunks[:, 1, 0] = torch.arange(6)
    replaced = exp.support_stratified_derangement(chunks)
    torch.testing.assert_close(replaced[:, 0, exp._exp026.BUTTONS_G], chunks[:, 0, exp._exp026.BUTTONS_G])
    assert all(not torch.equal(left, right) for left, right in zip(chunks, replaced, strict=True))

    logged = torch.zeros(6)
    ablated = torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    replay_ids = ["a", "a", "b", "b", "c", "c"]
    assert exp.replay_blocked_lcb(logged, ablated, replay_ids, resamples=500, seed=2) > 0


def test_cli_exposes_single_process_build_and_run_mode() -> None:
    parser = exp.make_parser()
    args = parser.parse_args(["--checkpoint", "026.pt", "--features", "/tmp/features.pt", "--build-and-run"])
    assert args.build_and_run and not args.build_features
    assert "--build-and-run" in parser.format_help()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--checkpoint",
                "026.pt",
                "--features",
                "/tmp/features.pt",
                "--build-features",
                "--build-and-run",
            ]
        )
