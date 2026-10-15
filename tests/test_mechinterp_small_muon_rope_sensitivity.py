"""Focused tests for the small-Muon RoPE precision sensitivity path."""

import torch

from hal.training import trunk
from research.mechinterp_20260911 import small_muon_rope_sensitivity


def test_bf16_factor_copy_matches_public_dense_trunk() -> None:
    torch.manual_seed(7)
    model = trunk.Trunk(
        trunk.TrunkConfig(
            d_model=16,
            n_layers=2,
            n_heads=2,
            L_ctx=8,
            attention_backend="dense_sdpa",
        ),
        prefer_flex=False,
    ).eval()
    tokens = torch.randn(3, 8, 16)
    padding = torch.zeros(3, dtype=torch.long)

    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        copied = small_muon_rope_sensitivity.trace_dense_trunk(
            model,
            tokens,
            padding,
            "bf16",
        )
        reference = model.forward_dense(tokens, padding)

    torch.testing.assert_close(copied.hidden, reference, rtol=0, atol=0)
    assert copied.weights.shape == (3, 2, 2, 8)
    assert copied.weights.dtype == torch.float32
    assert set(copied.projection_dtypes) == {"torch.bfloat16"}
    assert set(copied.factor_dtypes) == {"torch.bfloat16"}
    assert set(copied.rotated_dtypes) == {"torch.bfloat16"}


def test_fp32_treatment_changes_only_rotary_factor_path() -> None:
    torch.manual_seed(11)
    model = trunk.Trunk(
        trunk.TrunkConfig(
            d_model=16,
            n_layers=1,
            n_heads=2,
            L_ctx=16,
            attention_backend="dense_sdpa",
        ),
        prefer_flex=False,
    ).eval()
    tokens = torch.randn(2, 16, 16)
    padding = torch.zeros(2, dtype=torch.long)

    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        control = small_muon_rope_sensitivity.trace_dense_trunk(
            model,
            tokens,
            padding,
            "bf16",
        )
        treatment = small_muon_rope_sensitivity.trace_dense_trunk(
            model,
            tokens,
            padding,
            "fp32",
        )

    assert set(control.factor_dtypes) == {"torch.bfloat16"}
    assert set(treatment.factor_dtypes) == {"torch.float32"}
    assert set(control.projection_dtypes) == set(treatment.projection_dtypes) == {"torch.bfloat16"}
    assert set(treatment.rotated_dtypes) == {"torch.float32"}
    assert not torch.equal(control.weights, treatment.weights)
