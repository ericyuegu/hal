"""Parity tests for the O50 history probe."""

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
import torch

from hal.inference import o50_model

_SCRIPT = Path(__file__).resolve().parents[1] / "research" / "mechinterp_20260911" / "o50_probe.py"
_SPEC = importlib.util.spec_from_file_location("o50_probe", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
o50_probe = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = o50_probe
_SPEC.loader.exec_module(o50_probe)

_SDPA_FP32_TOLERANCE = 2e-6


def _config() -> o50_model.O50Config:
    architecture = o50_model.O50Architecture(
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
    )
    return o50_model.O50Config(
        experiment_id="050_scaled_temporal_awr_v4",
        architecture=architecture,
        prediction_frames=4,
        player_vocab_size=4,
        player_vocab_sha256="0" * 64,
        player_code_bytes=2,
        amp_dtype="float32",
        depth_alpha=0.5,
        mds_schema_version=7,
        stats_sha256="0" * 64,
    )


def test_teacher_forced_logits_reproduce_public_greedy_decode() -> None:
    torch.manual_seed(7)
    model = o50_model.O50Model(_config()).eval()
    hidden = torch.randn(3, 8, 16)
    observed = torch.stack(
        [torch.randint(vocabulary, (3,)) for vocabulary in o50_model.CONTROLLER_GROUP_VOCABS], dim=-1
    )
    forced = torch.zeros(3, 4, 4, dtype=torch.long)
    sampled = model.temporal.sample_conditioned(
        hidden,
        observed,
        forced,
        torch.zeros(3, 4, dtype=torch.bool),
        torch.full((4, 4, 3), 0.5),
        argmax=True,
    )

    logits = o50_probe.teacher_forced_logits(model, hidden, observed, sampled)
    reconstructed = torch.stack([logits[name].argmax(dim=-1) for name in o50_model.CONTROLLER_GROUP_NAMES], dim=-1)

    torch.testing.assert_close(reconstructed, sampled, rtol=0, atol=0)


def test_attention_trace_preserves_dense_trunk_hidden() -> None:
    torch.manual_seed(11)
    model = o50_model.O50Model(_config()).eval()
    tokens = torch.randn(3, 8, 16)
    padding = torch.tensor([0, 2, 7])

    traced, weights = o50_probe.trace_last_attention(model, tokens, padding)
    expected = model.trunk.forward_dense(tokens, padding)

    torch.testing.assert_close(traced, expected)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones_like(weights[..., 0]))
    assert torch.equal(weights[2, :, :, :7], torch.zeros_like(weights[2, :, :, :7]))


def test_attention_trace_uses_float32_scores_under_autocast() -> None:
    torch.manual_seed(12)
    model = o50_model.O50Model(_config()).eval()
    tokens = torch.randn(3, 8, 16)
    padding = torch.tensor([0, 2, 4])

    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        _, weights = o50_probe.trace_last_attention(model, tokens, padding)
        normalized = o50_probe.trunk.rmsnorm(tokens)
        query, key, value = model.trunk.blocks[0].attn.c_attn(normalized).split(16, dim=-1)
        shape = (3, 8, 2, 8)
        query = query.view(shape)
        key = key.view(shape)
        value = value.view(shape).transpose(1, 2)
        cosine, sine = model.trunk.blocks[0].attn.rotary(query)
        query = o50_probe.trunk.apply_rotary_emb(query, cosine, sine)
        key = o50_probe.trunk.apply_rotary_emb(key, cosine, sine)

    mask = o50_probe.trunk.dense_mask(padding, tokens.shape[1], model.trunk.attn_window)
    scores = torch.einsum("bhd,bkhd->bhk", query[:, -1].float(), key.float()) / math.sqrt(8)
    scores = scores.masked_fill(~mask[:, 0, -1, None, :], float("-inf"))
    expected_weights = torch.softmax(scores, dim=-1)
    manual_output = torch.einsum("bhk,bhkd->bhd", weights[:, 0], value.float())
    sdpa_output = torch.nn.functional.scaled_dot_product_attention(
        query[:, -1:].transpose(1, 2).float(),
        key.transpose(1, 2).float(),
        value.float(),
        attn_mask=mask[:, 0, -1, None, None, :],
    ).squeeze(2)

    assert weights.dtype == torch.float32
    torch.testing.assert_close(weights[:, 0], expected_weights, rtol=0, atol=0)
    # CPU SDPA and the explicit probability-weighted sum use different kernels.
    torch.testing.assert_close(
        manual_output,
        sdpa_output,
        rtol=_SDPA_FP32_TOLERANCE,
        atol=_SDPA_FP32_TOLERANCE,
    )


def test_head_ablation_is_effective_and_hook_is_removed() -> None:
    torch.manual_seed(13)
    model = o50_model.O50Model(_config()).eval()
    tokens = torch.randn(3, 8, 16)
    padding = torch.zeros(3, dtype=torch.long)
    baseline = model.trunk.forward_dense(tokens, padding)

    with o50_probe.zero_attention_head(model.trunk.blocks[0], 0):
        ablated = model.trunk.forward_dense(tokens, padding)
    restored = model.trunk.forward_dense(tokens, padding)

    assert not torch.equal(ablated, baseline)
    torch.testing.assert_close(restored, baseline, rtol=0, atol=0)


def test_change_report_compares_each_trajectory_with_its_own_previous_action() -> None:
    observed = torch.tensor([[0, 0, 0, 0]])
    targets = torch.tensor([[[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]]])
    actions = torch.tensor([[[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]]])

    report = o50_probe._change_report(actions, targets, observed)

    assert report["switch_rate"][0][0] == 1.0
    assert report["switch_rate"][1][1] == 0.0
    assert report["true_positive"][2][0] == 1
    assert report["false_negative"][1][1] == 1


def test_checkpoint_model_requires_exact_source_manifests(tmp_path: Path) -> None:
    config = _config()
    model = o50_model.O50Model(config)
    player_codes = b"[]"
    model.player_code_bytes.copy_(torch.tensor(list(player_codes), dtype=torch.uint8))
    raw_config = json.loads(config.to_json())
    raw_config["source_names"] = ["source"]
    raw_config["architecture"]["head_offsets"] = tuple(raw_config["architecture"]["head_offsets"])
    export = {
        "cfg": raw_config,
        "checkpoint_step": 7,
        "data_protocol": "o50-replay-ring-v8",
        "model": model.state_dict(),
        "source_checkpoint_sha256": "1" * 64,
        "source_manifest_sha256": {"source": "2" * 64},
        "source_r2_etag": "etag",
        "source_selection_sha256": "4" * 64,
        "source_uri": "s3://test/checkpoint.pt",
        "wandb_id": "run",
    }
    path = tmp_path / "model.pt"
    torch.save(export, path)
    metadata = {
        "bytes": path.stat().st_size,
        "checkpoint_step": 7,
        "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "raw_cfg": json.loads(json.dumps(raw_config)),
        "source_checkpoint_sha256": "1" * 64,
        "source_manifest_sha256": {"source": "2" * 64},
    }
    path.with_suffix(".metadata.json").write_text(json.dumps(metadata))
    stats = {"source_names": ["source"], "source_manifest_sha256": {"source": "3" * 64}}

    with pytest.raises(ValueError, match="source manifests differ"):
        o50_probe.load_checkpoint_model(
            path,
            reference_config=config,
            stats_payload=stats,
            player_codes=player_codes,
            device=torch.device("cpu"),
        )
