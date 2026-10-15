"""Exact probability and sampler-parity tests for the latent-prefix score."""

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

from hal.inference import o50_model

_SCRIPT = Path(__file__).resolve().parents[1] / "research" / "mechinterp_20260911" / "tail_likelihood.py"
_SPEC = importlib.util.spec_from_file_location("mechinterp_tail_likelihood", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
tail = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = tail
with pytest.MonkeyPatch.context() as imports:
    imports.syspath_prepend(str(_SCRIPT.parents[2]))
    _SPEC.loader.exec_module(tail)


def _config() -> o50_model.O50Config:
    return o50_model.O50Config(
        experiment_id="050_scaled_temporal_awr_v4",
        architecture=o50_model.O50Architecture(
            d_model=16,
            n_layers=2,
            n_heads=2,
            attn_window=0,
            L_ctx=8,
            sample_chunk_length=4,
            head_offsets=(1, 2, 3, 4),
            temporal_d_model=16,
            temporal_layers=2,
            temporal_heads=2,
            temporal_ff_dim=24,
            group_head_dim=12,
            action_embed_dim=4,
            offset_embed_dim=4,
            action_vocab=1024,
            action_state_embed_dim=4,
            char_vocab=32,
            char_dim=2,
            stage_vocab=32,
            stage_dim=2,
            item_type_dim=2,
            item_state_dim=2,
            item_hidden_dim=4,
            item_dim=4,
            value_hidden_dim=4,
        ),
        prediction_frames=4,
        player_vocab_size=4,
        player_vocab_sha256="0" * 64,
        player_code_bytes=2,
        amp_dtype="float32",
        depth_alpha=0.5,
        mds_schema_version=7,
        stats_sha256="0" * 64,
    )


def test_marginal_averages_probabilities_before_logarithms() -> None:
    conditional = torch.tensor([[0.0, -math.inf, 0.0, -math.inf]])
    log_probability, ess = tail.log_mean_likelihood(conditional)
    assert log_probability.item() == pytest.approx(-math.log(2))
    assert ess.item() == pytest.approx(2)
    assert math.isinf(float(-conditional.mean()))


def test_identical_prefixes_give_exact_score_and_full_ess() -> None:
    conditional = torch.full((3, 16), math.log(0.3), dtype=torch.float64)
    log_probability, ess = tail.log_mean_likelihood(conditional)
    torch.testing.assert_close(log_probability, torch.full((3,), math.log(0.3), dtype=torch.float64))
    torch.testing.assert_close(ess, torch.full((3,), 16.0, dtype=torch.float64))


def test_marginal_is_not_expected_conditional_nll() -> None:
    probabilities = torch.tensor([[0.9, 0.1], [0.8, 0.2]], dtype=torch.float64)
    log_probability, _ = tail.log_mean_likelihood(probabilities.log())
    torch.testing.assert_close(log_probability.exp(), torch.tensor([0.5, 0.5], dtype=torch.float64))
    assert (-log_probability < -probabilities.log().mean(dim=1)).all()


def test_historical_context_masks_only_identity_without_changing_the_cohort() -> None:
    context = {"ego_player_id": torch.tensor([[2, 3], [4, 5]]), "ego_percent": torch.randn(2, 2)}
    original_identity = context["ego_player_id"].clone()
    masked = tail.historical_context(context)
    assert masked["ego_percent"] is context["ego_percent"]
    torch.testing.assert_close(masked["ego_player_id"], torch.zeros(2, 2, dtype=torch.long))
    torch.testing.assert_close(context["ego_player_id"], original_identity)


def test_particle_schedule_sorts_valid_nested_counts() -> None:
    assert tail.particle_schedule([1024, 64, 256]) == (64, 256, 1024)


@pytest.mark.parametrize("counts", ([], [1], [2, 2], [2, 3], [2, 4, 7]))
def test_invalid_particle_schedule_is_rejected(counts: list[int]) -> None:
    with pytest.raises(ValueError):
        tail.particle_schedule(counts)


@pytest.mark.parametrize(
    "weights",
    [
        torch.empty(1, 0),
        torch.zeros(2),
        torch.tensor([[math.nan]]),
        torch.tensor([[0.1]]),
        torch.tensor([[-math.inf]]),
    ],
)
def test_invalid_marginal_estimates_are_rejected(weights: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        tail.log_mean_likelihood(weights)


@pytest.mark.parametrize("argmax", [False, True])
def test_sequential_density_reconstructs_public_sampling(argmax: bool) -> None:
    torch.manual_seed(37)
    model = o50_model.O50Model(_config()).eval()
    hidden = torch.randn(3, 8, 16)
    observed = model.codec.quantize(torch.zeros(3, 14))
    uniforms = torch.rand(4, 4, 3)
    with torch.inference_mode():
        sampled = model.temporal.sample_conditioned(
            hidden,
            observed,
            torch.zeros(3, 4, 4, dtype=torch.long),
            torch.zeros(3, 4, dtype=torch.bool),
            uniforms,
            argmax=argmax,
        )
        logits = tail.sequential_teacher_logits(model, hidden, observed, sampled)
        reconstructed = torch.stack(
            [
                o50_model.sample_categorical(logits[name], argmax=argmax, uniform=uniforms[:, group].transpose(0, 1))
                for group, name in enumerate(o50_model.CONTROLLER_GROUP_NAMES)
            ],
            dim=-1,
        )
        parallel = tail.o50_probe.teacher_forced_logits(model, hidden, observed, sampled)
    torch.testing.assert_close(reconstructed, sampled, rtol=0, atol=0)
    for name in logits:
        torch.testing.assert_close(logits[name], parallel[name], rtol=2e-5, atol=2e-5)


def test_sampled_prefix_estimator_preserves_targets_and_row_order() -> None:
    torch.manual_seed(41)
    model = o50_model.O50Model(_config()).eval()
    hidden = torch.randn(2, 1, 16)
    observed = model.codec.quantize(torch.zeros(2, 14))
    targets = model.codec.quantize(torch.zeros(2, 4, 14))
    uniforms = torch.rand(4, 4, 2, 8)
    with torch.inference_mode():
        together, parity = tail.sampled_prefix_log_weights(model, hidden, observed, targets, uniforms)
        separate = [
            tail.sampled_prefix_log_weights(
                model,
                hidden[row : row + 1],
                observed[row : row + 1],
                targets[row : row + 1],
                uniforms[:, :, row : row + 1],
            )[0]
            for row in range(2)
        ]
    assert parity
    assert together.shape == (2, 8)
    assert torch.isfinite(together).all()
    torch.testing.assert_close(together, torch.cat(separate), rtol=2e-5, atol=2e-5)
