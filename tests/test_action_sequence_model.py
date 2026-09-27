"""Construction parity with the frozen 059 training model."""

import hashlib
import json
from pathlib import Path

import torch

from hal.models.action_sequence import ActionSequenceConfig
from hal.models.action_sequence import ActionSequenceTransformer

_FIXTURE = Path(__file__).parent / "fixtures" / "o59" / "model_proxy_parity.json"


def test_default_059_parameter_count() -> None:
    with torch.device("meta"):
        model = ActionSequenceTransformer(ActionSequenceConfig())
    assert sum(parameter.numel() for parameter in model.parameters()) == 246_862_205


def test_proxy_parameter_order_initialization_and_rng_match_control() -> None:
    control = json.loads(_FIXTURE.read_text())
    config = ActionSequenceConfig(
        d_model=256,
        n_layers=12,
        n_heads=4,
        temporal_d_model=256,
        temporal_layers=6,
        temporal_heads=4,
        temporal_ff_dim=1024,
        group_head_dim=256,
        value_hidden_dim=128,
    )
    torch.manual_seed(control["model_seed"])
    model = ActionSequenceTransformer(config)
    parameters = list(model.named_parameters())
    digest = hashlib.sha256()
    for name, parameter in parameters:
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().numpy().tobytes())
    assert sum(parameter.numel() for _, parameter in parameters) == control["parameter_count"]
    assert [[name, list(parameter.shape)] for name, parameter in parameters] == control["parameter_names_and_shapes"]
    assert list(model.state_dict()) == control["state_keys"]
    assert digest.hexdigest() == control["parameter_sha256"]
    assert hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest() == control["rng_after_model_sha256"]
