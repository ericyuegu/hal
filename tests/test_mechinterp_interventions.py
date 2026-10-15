"""Pure-state tests for the O50 checkpoint intervention transforms."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

MODULE_PATH = Path(__file__).parents[1] / "research" / "mechinterp_20260911" / "checkpoint_interventions.py"
SPEC = importlib.util.spec_from_file_location("mechinterp_checkpoint_interventions", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load {MODULE_PATH}")
interventions = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = interventions
SPEC.loader.exec_module(interventions)


PARAMETER_SHAPES: dict[str, tuple[int, ...]] = {
    "trunk.blocks.0.attn.c_attn.weight": (6, 2),
    "trunk.blocks.0.attn.c_proj.weight": (2, 2),
    "temporal.blocks.0.qkv.weight": (6, 2),
    "temporal.blocks.0.proj.weight": (2, 2),
    "temporal.token_projection.weight": (2, 2),
    "temporal.token_projection.bias": (2,),
    "temporal.outputs.buttons.up.weight": (2, 2),
    "temporal.outputs.buttons.down.weight": (2, 2),
    "temporal.outputs.buttons.down.bias": (2,),
    "temporal.trunk_outputs.buttons.weight": (2, 2),
    "value_head.up.weight": (4, 2),
    "value_head.down.weight": (1, 2),
    "value_head.down.bias": (1,),
    "codec.class_embeddings.buttons.weight": (2, 2),
    "temporal.codec.class_embeddings.buttons.weight": (2, 2),
    "observation_encoder.weight": (2, 2),
}
PARAMETER_KEYS = frozenset(PARAMETER_SHAPES)


def _states() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    early = {name: torch.full(shape, 1.0) for name, shape in PARAMETER_SHAPES.items()}
    slump = {name: torch.full(shape, 3.0) for name, shape in PARAMETER_SHAPES.items()}
    for state in (early, slump):
        state["player_code_bytes"] = torch.tensor([1, 2, 3], dtype=torch.uint8)
        state["trunk.blocks.0.attn.rotary.inv_freq"] = torch.tensor([0.5, 1.0])
    return early, slump


def test_optimizer_roles_match_recorded_o50_contract() -> None:
    roles = interventions.parameter_roles(PARAMETER_KEYS)

    assert roles["trunk.blocks.0.attn.c_attn.weight"] == "muon"
    assert roles["temporal.blocks.0.qkv.weight"] == "muon"
    assert roles["temporal.token_projection.weight"] == "muon"
    assert roles["temporal.outputs.buttons.up.weight"] == "muon"
    assert roles["value_head.up.weight"] == "muon"
    assert roles["temporal.outputs.buttons.down.weight"] == "adamw"
    assert roles["temporal.trunk_outputs.buttons.weight"] == "adamw"
    assert roles["value_head.down.weight"] == "adamw"
    assert roles["codec.class_embeddings.buttons.weight"] == "adamw"
    assert roles["temporal.codec.class_embeddings.buttons.weight"] == "adamw"
    assert roles["observation_encoder.weight"] == "adamw"

    assert interventions.logical_splits("trunk.blocks.0.attn.c_attn.weight") == 3
    assert interventions.logical_splits("temporal.blocks.0.qkv.weight") == 3
    assert interventions.logical_splits("value_head.up.weight") == 2


def test_representation_validation_allows_width_change_but_rejects_cohort_changes() -> None:
    metadata_path = (
        Path(__file__).parents[1]
        / "research"
        / "mechinterp_20260911"
        / "artifacts"
        / "forensics"
        / "model_exports"
        / "p1fyyp1z-step-0024576.model.metadata.json"
    )
    raw_config = json.loads(metadata_path.read_bytes())["raw_cfg"]
    reference = interventions.o50_model.O50Config.from_checkpoint(
        raw_config,
        player_code_bytes=169_448,
        stats_sha256="0" * 64,
    )
    wider = dataclasses.replace(
        reference,
        architecture=dataclasses.replace(
            reference.architecture,
            d_model=reference.architecture.d_model * 2,
            n_heads=reference.architecture.n_heads * 2,
            temporal_d_model=reference.architecture.temporal_d_model * 2,
            temporal_heads=reference.architecture.temporal_heads * 2,
            group_head_dim=reference.architecture.group_head_dim * 2,
            temporal_ff_dim=reference.architecture.temporal_ff_dim * 2,
            value_hidden_dim=reference.architecture.value_hidden_dim * 2,
        ),
    )
    interventions._validate_representation(reference, wider)

    wrong_context = dataclasses.replace(
        wider,
        architecture=dataclasses.replace(wider.architecture, L_ctx=wider.architecture.L_ctx + 1),
    )
    with pytest.raises(ValueError, match="L_ctx"):
        interventions._validate_representation(reference, wrong_context)

    wrong_codec = dataclasses.replace(
        wider,
        architecture=dataclasses.replace(wider.architecture, action_vocab=wider.architecture.action_vocab + 1),
    )
    with pytest.raises(ValueError, match="action_vocab"):
        interventions._validate_representation(reference, wrong_codec)


def test_json_normalization_matches_sidecar_lists_to_checkpoint_tuples() -> None:
    raw = {"architecture": {"head_offsets": (1, 2, 3, 4)}, "source_names": ("a", "b")}
    sidecar = {"architecture": {"head_offsets": [1, 2, 3, 4]}, "source_names": ["a", "b"]}

    assert interventions._json_normalize(raw) == sidecar
    assert interventions._json_normalize(raw) != {**sidecar, "source_names": ["a", "c"]}


def test_interpolation_changes_parameters_and_preserves_buffers() -> None:
    early, slump = _states()

    result = interventions.interpolate_state(early, slump, 0.5, parameter_keys=PARAMETER_KEYS)

    assert set(result) == set(early)
    for name in PARAMETER_KEYS:
        assert torch.equal(result[name], torch.full_like(result[name], 2.0))
    assert torch.equal(result["player_code_bytes"], early["player_code_bytes"])
    assert torch.equal(result["trunk.blocks.0.attn.rotary.inv_freq"], early["trunk.blocks.0.attn.rotary.inv_freq"])
    assert torch.equal(
        interventions.interpolate_state(early, slump, 0.0, parameter_keys=PARAMETER_KEYS)["player_code_bytes"],
        early["player_code_bytes"],
    )
    assert torch.equal(
        interventions.interpolate_state(early, slump, 1.0, parameter_keys=PARAMETER_KEYS)["player_code_bytes"],
        slump["player_code_bytes"],
    )


def test_interpolation_rejects_different_buffers() -> None:
    early, slump = _states()
    slump["player_code_bytes"] = torch.tensor([9, 2, 3], dtype=torch.uint8)

    with pytest.raises(ValueError, match="non-parameter buffer differs"):
        interventions.interpolate_state(early, slump, 0.5, parameter_keys=PARAMETER_KEYS)


@pytest.mark.parametrize("partition", ["readouts", "adamw", "muon"])
def test_restore_partition_only_replaces_selected_parameters(partition: str) -> None:
    early, slump = _states()

    result = interventions.restore_partition(slump, early, partition, parameter_keys=PARAMETER_KEYS)  # type: ignore[arg-type]

    for name in PARAMETER_KEYS:
        expected_early = (
            interventions._is_final_readout(name)
            if partition == "readouts"
            else interventions.optimizer_partition(name) == partition
        )
        expected = early[name] if expected_early else slump[name]
        assert torch.equal(result[name], expected), name
    assert torch.equal(result["player_code_bytes"], slump["player_code_bytes"])


def test_restore_muon_norms_handles_fused_logical_matrices() -> None:
    early, slump = _states()
    early["trunk.blocks.0.attn.c_attn.weight"] = torch.cat(
        [torch.full((2, 2), value) for value in (1.0, 2.0, 3.0)], dim=0
    )
    slump["trunk.blocks.0.attn.c_attn.weight"] = torch.cat(
        [torch.full((2, 2), value) for value in (4.0, 5.0, 6.0)], dim=0
    )
    early["value_head.up.weight"] = torch.cat([torch.full((2, 2), value) for value in (2.0, 3.0)], dim=0)
    slump["value_head.up.weight"] = torch.cat([torch.full((2, 2), value) for value in (7.0, 8.0)], dim=0)

    result = interventions.restore_muon_norms(slump, early, parameter_keys=PARAMETER_KEYS)
    for name in ("trunk.blocks.0.attn.c_attn.weight", "value_head.up.weight"):
        splits = interventions.logical_splits(name)
        for early_chunk, result_chunk in zip(
            early[name].chunk(splits, dim=0), result[name].chunk(splits, dim=0), strict=True
        ):
            assert torch.linalg.vector_norm(result_chunk) == pytest.approx(torch.linalg.vector_norm(early_chunk))
            cosine = torch.sum(early_chunk * result_chunk) / (
                torch.linalg.vector_norm(early_chunk) * torch.linalg.vector_norm(result_chunk)
            )
            assert cosine == pytest.approx(1.0)
    assert torch.equal(result["temporal.outputs.buttons.down.weight"], slump["temporal.outputs.buttons.down.weight"])


def test_restore_attention_qk_norms_leaves_values_and_other_matrices_alone() -> None:
    early, slump = _states()
    name = "trunk.blocks.0.attn.c_attn.weight"
    early[name] = torch.cat([torch.full((2, 2), value) for value in (1.0, 2.0, 3.0)], dim=0)
    slump[name] = torch.cat([torch.full((2, 2), value) for value in (4.0, 5.0, 6.0)], dim=0)

    result = interventions.restore_attention_qk_norms(slump, early, parameter_keys=PARAMETER_KEYS)
    early_chunks = early[name].chunk(3, dim=0)
    result_chunks = result[name].chunk(3, dim=0)
    for index in (0, 1):
        assert torch.linalg.vector_norm(result_chunks[index]) == pytest.approx(
            torch.linalg.vector_norm(early_chunks[index])
        )
    assert torch.equal(result_chunks[2], slump[name].chunk(3, dim=0)[2])
    for parameter_name in PARAMETER_KEYS - {
        parameter_name for parameter_name in PARAMETER_KEYS if interventions._attention_qk_name(parameter_name)
    }:
        assert torch.equal(result[parameter_name], slump[parameter_name])


def test_qk_gauge_control_scales_only_fused_qk() -> None:
    _, slump = _states()
    name = "temporal.blocks.0.qkv.weight"
    result = interventions.qk_gauge_control(slump, parameter_keys=PARAMETER_KEYS)
    original = slump[name].chunk(3, dim=0)
    changed = result[name].chunk(3, dim=0)
    assert torch.equal(changed[0], original[0] * 2)
    assert torch.equal(changed[1], original[1] * 0.5)
    assert torch.equal(changed[2], original[2])
    assert torch.equal(result["trunk.blocks.0.attn.c_proj.weight"], slump["trunk.blocks.0.attn.c_proj.weight"])


def test_identity_and_reversal_controls_are_exact() -> None:
    early, slump = _states()
    forward = interventions.interpolate_state(early, slump, 0.25, parameter_keys=PARAMETER_KEYS)
    reverse = interventions.reverse_interpolate_state(early, slump, 0.75, parameter_keys=PARAMETER_KEYS)

    assert all(torch.equal(forward[name], reverse[name]) for name in forward)
    assert all(
        torch.equal(
            interventions.apply_intervention(spec, early, slump, parameter_keys=PARAMETER_KEYS)[name], early[name]
        )
        for spec in interventions.intervention_specs()
        if spec.name == "early_identity"
        for name in early
    )


def test_paired_output_metrics_reports_hold_change_and_bootstrap() -> None:
    baseline_logits = {
        name: torch.tensor([[[2.0, 0.0], [0.0, 2.0]]]).repeat(2, 1, 1)
        for name in ("main_stick", "c_stick", "triggers", "buttons")
    }
    treatment_logits = {name: value.flip(-1) for name, value in baseline_logits.items()}
    targets = torch.tensor([[[0], [1]]]).expand(2, 2, 4)
    observed = torch.zeros((2, 4), dtype=torch.long)
    baseline_loss = torch.ones((2, 2, 4))
    treatment_loss = torch.full((2, 2, 4), 2.0)

    result = interventions.paired_output_metrics(
        baseline_logits,
        treatment_logits,
        baseline_loss,
        treatment_loss,
        targets,
        observed,
    )

    assert result["baseline_hold_change"]["hold"]["count"] == [[2, 2, 2, 2], [0, 0, 0, 0]]
    assert result["treatment_hold_change"]["change"]["overall_mean_nll_nats"] == pytest.approx(2.0)
    assert result["paired_delta"]["nll_nats"]["mean"] == pytest.approx(1.0)
    assert result["paired_delta"]["output_js_nats"]["mean"] > 0
    assert result["paired_delta"]["argmax_change_rate"]["mean"] == pytest.approx(1.0)
    assert len(result["max_abs_logit_delta_by_group"]) == 4


def test_paired_output_metrics_handles_masked_logits() -> None:
    baseline_logits = {
        name: torch.tensor([[[float("-inf"), 1.0], [2.0, 0.0]]]).repeat(2, 1, 1)
        for name in interventions.CONTROLLER_GROUP_NAMES
    }
    treatment_logits = {name: value.clone() for name, value in baseline_logits.items()}
    targets = torch.zeros((2, 2, 4), dtype=torch.long)
    observed = torch.zeros((2, 4), dtype=torch.long)
    losses = torch.ones((2, 2, 4))

    result = interventions.paired_output_metrics(
        baseline_logits,
        treatment_logits,
        losses,
        losses,
        targets,
        observed,
    )

    assert result["paired_delta"]["output_js_nats"]["mean"] == pytest.approx(0.0, abs=1e-7)
    assert result["max_abs_logit_delta_by_group"] == [0.0] * 4


def test_attention_metrics_reports_hard_last_token() -> None:
    weights = torch.zeros((2, 1, 1, 4))
    weights[..., -1] = 1.0

    result = interventions.attention_metrics(weights)

    assert result["mean_entropy_nats"] == pytest.approx(0.0)
    assert result["mean_effective_context"] == pytest.approx(1.0)
    assert result["mean_lag"] == [[pytest.approx(0.0)]]
