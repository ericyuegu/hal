"""Functional parity tests for the O50 attention routing probe."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from hal.inference import o50_model

_SCRIPT = Path(__file__).resolve().parents[1] / "research" / "mechinterp_20260911" / "attention_routing.py"
_SPEC = importlib.util.spec_from_file_location("mechinterp_attention_routing", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
routing = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = routing
with pytest.MonkeyPatch.context() as imports:
    imports.syspath_prepend(str(_SCRIPT.parents[2]))
    _SPEC.loader.exec_module(routing)


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


def test_soft_routing_matches_public_dense_trunk_under_autocast() -> None:
    torch.manual_seed(53)
    model = o50_model.O50Model(_config()).eval()
    tokens = torch.randn(3, 8, 16)
    padding = torch.tensor([0, 2, 4])

    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        expected = model.trunk.forward_dense(tokens, padding)
        actual = routing.routed_trunk(model, tokens, padding, "soft")

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_protocol_identity_gates() -> None:
    routing._validate_cohort_sha256("ec61e97e332672a4e4e2fdc438e208a281a4609e10f7eded72368eccac323437")
    for run_id in ("p1fyyp1z", "tgpuo1be"):
        routing._validate_checkpoint_identity({"wandb_id": run_id, "checkpoint_step": 131_071}, run_id)

    with pytest.raises(ValueError, match="cohort differs"):
        routing._validate_cohort_sha256("0" * 64)
    with pytest.raises(ValueError, match="preselected final"):
        routing._validate_checkpoint_identity(
            {"wandb_id": "p1fyyp1z", "checkpoint_step": 131_070},
            "p1fyyp1z",
        )
    with pytest.raises(ValueError, match="preselected final"):
        routing._validate_checkpoint_identity(
            {"wandb_id": "tgpuo1be", "checkpoint_step": 131_071},
            "p1fyyp1z",
        )


@pytest.mark.parametrize("mode", ["argmax_key", "current_key"])
def test_hard_routing_changes_only_the_final_query(mode: routing.RoutingMode) -> None:
    torch.manual_seed(59)
    model = o50_model.O50Model(_config()).eval()
    tokens = torch.randn(3, 8, 16)
    padding = torch.zeros(3, dtype=torch.long)

    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        soft = routing.routed_trunk(model, tokens, padding, "soft")
        treatment = routing.routed_trunk(model, tokens, padding, mode)
        restored = routing.routed_trunk(model, tokens, padding, "soft")

    torch.testing.assert_close(treatment[:, :-1], soft[:, :-1], rtol=0, atol=0)
    assert not torch.equal(treatment[:, -1], soft[:, -1])
    torch.testing.assert_close(restored, soft, rtol=0, atol=0)
