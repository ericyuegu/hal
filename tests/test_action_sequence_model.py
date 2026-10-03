"""Construction parity with the frozen 059 training model."""

import hashlib
import json
from pathlib import Path

import torch

from hal.models.action_sequence import ActionSequenceConfig
from hal.models.action_sequence import ActionSequenceTransformer
from hal.models.action_sequence import NonlinearActionHead
from hal.models.action_sequence import TemporalBlock

_FIXTURE = Path(__file__).parent / "fixtures" / "o59" / "model_proxy_parity.json"


def test_default_059_parameter_count() -> None:
    with torch.device("meta"):
        model = ActionSequenceTransformer(ActionSequenceConfig())
    assert sum(parameter.numel() for parameter in model.parameters()) == 246_862_205


def test_pruned81_model_uses_81_way_main_stick_heads() -> None:
    config = ActionSequenceConfig(
        d_model=32,
        n_layers=1,
        n_heads=4,
        temporal_d_model=32,
        temporal_layers=1,
        temporal_heads=4,
        temporal_ff_dim=64,
        group_head_dim=16,
        value_hidden_dim=16,
        main_stick_layout="pruned81",
    )
    with torch.device("meta"):
        model = ActionSequenceTransformer(config)

    main_stick = model.temporal.outputs["main_stick"]
    trunk_main_stick = model.temporal.trunk_outputs["main_stick"]
    assert isinstance(main_stick, NonlinearActionHead)
    assert isinstance(trunk_main_stick, NonlinearActionHead)
    assert model.codec.group_vocabs == (256, 81, 9, 25)
    assert main_stick.down.out_features == 81
    assert trunk_main_stick.down.out_features == 81


def test_zero_width_omits_return_conditioning_parameters_and_modulation() -> None:
    config = ActionSequenceConfig(
        d_model=32,
        n_layers=1,
        n_heads=4,
        temporal_d_model=32,
        temporal_layers=2,
        temporal_heads=4,
        temporal_ff_dim=64,
        group_head_dim=16,
        value_hidden_dim=16,
        return_embed_dim=0,
        return_conditioning=False,
    )
    with torch.device("meta"):
        model = ActionSequenceTransformer(config)

    conditioner = model.temporal.return_conditioner
    assert not tuple(conditioner.parameters())
    assert (
        conditioner(
            torch.empty(3, device="meta"),
            torch.zeros(3, dtype=torch.bool, device="meta"),
        )
        is None
    )

    model = ActionSequenceTransformer(config).eval()
    block = model.temporal.blocks[0]
    assert isinstance(block, TemporalBlock)
    states = torch.randn(2, 3, config.temporal_d_model)
    zero_modulation = torch.zeros(2, 1, 4, config.temporal_d_model)
    with torch.no_grad():
        torch.testing.assert_close(block(states, None), block(states, zero_modulation))


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
