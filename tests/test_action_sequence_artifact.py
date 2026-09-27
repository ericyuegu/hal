import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import nn

from hal import streams
from hal.inference.action_sequence_artifact import ActionSequenceArtifact
from hal.inference.action_sequence_artifact import build_action_sequence_model
from hal.inference.action_sequence_artifact import checkpoint_contract
from hal.inference.action_sequence_artifact import validate_checkpoint_provenance
from hal.inference.api import PolicySpec
from hal.inference.bundle import BundleDescription
from hal.inference.bundle import read_policy_manifest
from hal.inference.bundle import write_policy_bundle
from hal.models.action_sequence import ActionSequenceConfig
from hal.models.action_sequence import ActionSequenceTransformer
from hal.representation.player_identity import PlayerVocabulary

_FIXTURES = Path(__file__).parent / "fixtures" / "o59"


def _configuration() -> dict[str, object]:
    values = json.loads((_FIXTURES / "checkpoint_config.json").read_text())
    values["source_names"] = tuple(values["source_names"])
    values["architecture"]["head_offsets"] = tuple(values["architecture"]["head_offsets"])
    return values


def test_saved_configuration_becomes_a_narrow_model_configuration() -> None:
    saved = _configuration()
    contract = checkpoint_contract(saved)
    assert contract.model.d_model == 1024
    assert contract.model.n_layers == 12
    assert contract.model.L_ctx == 256
    assert contract.source_names == tuple(source.name for source in streams.POLICY_WORLD_V8_SOURCES)
    assert not hasattr(contract.model, "optimizer")
    assert not hasattr(contract.model, "return_calibration")
    assert not hasattr(contract.model, "source_names")
    assert saved == _configuration()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_artifact_builder_binds_cuda_linear_weights_once() -> None:
    vocabulary = PlayerVocabulary(())
    config = ActionSequenceConfig(
        d_model=32,
        n_layers=2,
        n_heads=4,
        L_ctx=8,
        temporal_d_model=32,
        temporal_layers=1,
        temporal_heads=4,
        temporal_ff_dim=64,
        group_head_dim=16,
        value_hidden_dim=16,
        item_hidden_dim=8,
        item_dim=5,
        player_vocab_size=vocabulary.size,
        player_vocab_sha256=vocabulary.sha256,
    )
    reference = ActionSequenceTransformer(config, vocabulary)
    artifact = ActionSequenceArtifact(
        config,
        {name: value.detach().cpu() for name, value in reference.state_dict().items()},
        {},
        vocabulary,
        "test-checkpoint",
        20.0,
        PolicySpec("test", "hal.action_sequence.test", (), (0,)),
        2,
    )
    loaded = build_action_sequence_model(artifact, device="cuda")
    linear_parameters = {
        id(parameter)
        for layer in loaded.modules()
        if isinstance(layer, nn.Linear)
        for parameter in layer.parameters(recurse=False)
    }
    assert linear_parameters
    assert all(
        parameter.dtype == (torch.bfloat16 if id(parameter) in linear_parameters else torch.float32)
        for parameter in loaded.parameters()
    )
    assert all(layer.weight.dtype == torch.float32 for layer in loaded.modules() if isinstance(layer, nn.Embedding))
    with pytest.raises(ValueError, match="CPU FP32 and CUDA BF16"):
        build_action_sequence_model(artifact, device="cuda", inference_dtype=torch.float32)


@pytest.mark.parametrize(
    "field,value",
    [
        ("experiment_id", "050"),
        ("checkpoint_format_version", 3),
        ("mds_schema_version", 6),
        ("return_conditioning", False),
        ("max_steps", 3),
        ("warmup_steps", 2),
        ("batch_size", True),
    ],
)
def test_configuration_rejects_changed_scientific_contract(field: str, value: object) -> None:
    saved = _configuration()
    saved[field] = value
    with pytest.raises(ValueError):
        checkpoint_contract(saved)


def test_configuration_rejects_reordered_sources_and_architecture_changes() -> None:
    saved = _configuration()
    saved["source_names"] = tuple(reversed(cast(tuple[str, ...], saved["source_names"])))
    with pytest.raises(ValueError, match="source list"):
        checkpoint_contract(saved)
    saved = _configuration()
    cast(dict[str, object], saved["architecture"])["head_offsets"] = (1, 2, 4)
    with pytest.raises(ValueError, match="architecture"):
        checkpoint_contract(saved)


def test_artifact_provenance_rejects_manifest_selection_and_vocabulary_changes() -> None:
    contract = checkpoint_contract(_configuration())
    provenance = json.loads((_FIXTURES / "artifacts.json").read_text())["provenance"]
    validate_checkpoint_provenance(provenance, contract)
    for field in ("source_selection_sha256", "player_vocab_sha256", "player_sidecar_sha256"):
        with pytest.raises(ValueError, match="provenance changed"):
            validate_checkpoint_provenance({**provenance, field: "0" * 64}, contract)
    changed = deepcopy(provenance)
    changed["source_manifest_sha256"][contract.source_names[0]] = "0" * 64
    with pytest.raises(ValueError, match="provenance changed"):
        validate_checkpoint_provenance(changed, contract)


def test_existing_bundle_capabilities_are_not_silently_rewritten(tmp_path: Path) -> None:
    config = tmp_path / "backend.json"
    config.write_text("{}")
    description = BundleDescription(
        policy_name="059",
        backend="o59-history-decoder",
        backend_version=1,
        required_observation_fields=(),
        supported_transport_delays=(2,),
        requires_player_code=False,
        source_sha256="1" * 64,
    )
    old = tmp_path / "old.hal"
    new = tmp_path / "new.hal"
    write_policy_bundle(old, description, {"backend.json": config})
    write_policy_bundle(
        new, replace(description, backend_version=2, supported_transport_delays=(0, 2, 3)), {"backend.json": config}
    )
    assert read_policy_manifest(old).supported_transport_delays == (2,)
    assert read_policy_manifest(old).backend_version == 1
    assert read_policy_manifest(new).supported_transport_delays == (0, 2, 3)
    assert read_policy_manifest(new).source_sha256 == read_policy_manifest(old).source_sha256
