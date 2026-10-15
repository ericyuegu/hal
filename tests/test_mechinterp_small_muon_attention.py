import math
from typing import cast

import numpy as np
import pytest
import torch

from hal.training import trunk
from research.mechinterp_20260911 import small_muon_attention


def test_summary_averages_row_entropy_exponentials_not_the_mean_entropy() -> None:
    weights = np.asarray([[[[1.0, 0.0, 0.0], [0.1, 0.6, 0.3]]]], dtype=np.float32)
    summary = small_muon_attention.summarize_attention(weights)
    probability = weights[0, 0, 1].astype(np.float64)
    entropy = float(-(probability * np.log(probability)).sum())
    assert summary["mean_effective_frames"] == pytest.approx((1.0 + math.exp(entropy)) / 2)
    assert summary["mean_effective_frames"] != pytest.approx(math.exp(entropy / 2))
    assert summary["mean_max_weight"] == pytest.approx(0.8)
    assert summary["mean_distinct_max_positions_per_layer"] == [2.0]
    layers = cast(list[dict[str, object]], summary["layers"])
    assert layers[0]["example_key_positions"] == [0, 1]


def test_summary_does_not_average_token_positions_across_replays() -> None:
    weights = np.asarray([[[[0.8, 0.1, 0.1]]], [[[0.05, 0.05, 0.9]]]], dtype=np.float32)
    summary = small_muon_attention.summarize_attention(weights)
    assert summary["mean_max_weight"] == pytest.approx(0.85)
    layers = cast(list[dict[str, object]], summary["layers"])
    assert layers[0]["example_key_positions"] == [0]


@pytest.mark.parametrize(
    ("probabilities", "message"),
    (([0.5, 0.5], "tied maximum"), ([0.6, 0.6], "sum to one"), ([-0.1, 1.1], "nonnegative")),
)
def test_summary_rejects_ambiguous_or_invalid_probabilities(probabilities: list[float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        small_muon_attention.summarize_attention(np.asarray([[[probabilities]]], dtype=np.float32))


def test_attention_trace_matches_dense_hidden_state_and_respects_padding() -> None:
    with torch.random.fork_rng():
        torch.manual_seed(31)
        model = trunk.Trunk(trunk.TrunkConfig(d_model=16, n_layers=2, n_heads=2, L_ctx=8), prefer_flex=False)
        tokens = torch.randn(2, 8, 16)
    padding = torch.tensor([0, 3])
    with torch.inference_mode():
        hidden, weights = small_muon_attention.trace_last_attention(model, tokens, padding)
        expected = model.forward_dense(tokens, padding)
    torch.testing.assert_close(hidden, expected, rtol=0, atol=0)
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 2, 2))
    assert weights.dtype == torch.float32
    assert torch.count_nonzero(weights[1, :, :, :3]).item() == 0


def test_config_allows_only_recorded_metadata_difference() -> None:
    spec = small_muon_attention.CHECKPOINTS[0]
    config = {
        "d_model": 256,
        "n_layers": 8,
        "n_heads": 4,
        "L_ctx": 128,
        "max_steps": 16384,
        "muon_lr": 0.02,
        "observation_bundle": "base",
        "mds_schema_version": 7,
        "data_root": "data/processed/ranked-anonymized-1/mds-policy-v7",
        "experiment_id": "037_factorization_matrix_v1",
    }
    logged = {key: value for key, value in config.items() if key != "experiment_id"}
    assert len(small_muon_attention.validate_config(spec, config, logged)) == 1
    logged["muon_lr"] = 0.04
    with pytest.raises(ValueError, match="configuration difference"):
        small_muon_attention.validate_config(spec, config, logged)
