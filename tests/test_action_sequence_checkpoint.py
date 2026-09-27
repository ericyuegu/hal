"""Artifact boundaries use real 059 tensor geometry with compact synthetic storage."""

import hashlib
import json
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import Tensor

from hal.inference.action_sequence_artifact import ACTION_SEQUENCE_BACKEND
from hal.inference.action_sequence_artifact import REQUIRED_OBSERVATION_FIELDS
from hal.inference.action_sequence_artifact import checkpoint_contract
from hal.inference.action_sequence_artifact import read_action_sequence_artifact
from hal.inference.action_sequence_artifact import read_checkpoint
from hal.inference.bundle import BundleDescription
from hal.inference.bundle import write_policy_bundle
from hal.models.action_sequence import ActionSequenceTransformer
from hal.models.attention import Rotary
from hal.models.controller_codec import DiscreteControllerCodec
from hal.representation.features import FLOAT_FEATURES
from hal.representation.features import ITEM_FLOATS
from hal.representation.player_identity import PlayerVocabulary
from hal.training.checkpoints import ResumeLineage
from hal.training.checkpoints import checkpoint_sha256
from hal.training.returns import ReturnCalibration

_FIXTURES = Path(__file__).parent / "fixtures" / "o59"


@pytest.fixture(scope="module")
def checkpoint_payload() -> dict[str, object]:
    config = json.loads((_FIXTURES / "checkpoint_config.json").read_text())
    config["source_names"] = tuple(config["source_names"])
    config["architecture"]["head_offsets"] = tuple(config["architecture"]["head_offsets"])
    vocabulary = PlayerVocabulary(("TEST#1",))
    config["player_vocab_size"] = vocabulary.size
    config["player_vocab_sha256"] = vocabulary.sha256
    contract = checkpoint_contract(config)
    with torch.device("meta"):
        model = ActionSequenceTransformer(contract.model, vocabulary)
    tensors = {
        name: torch.zeros((), dtype=reference.dtype).expand(reference.shape)
        for name, reference in model.state_dict().items()
    }
    codec_buffers = dict(DiscreteControllerCodec(contract.model.action_embed_dim).named_buffers())
    for name, reference in model.named_buffers():
        if name.startswith("codec."):
            reference = codec_buffers[name.removeprefix("codec.")]
        elif name.endswith("rotary.inv_freq"):
            reference = Rotary(reference.numel() * 2).inv_freq
        tensors[name] = reference.clone()
    provenance = json.loads((_FIXTURES / "artifacts.json").read_text())["provenance"]
    provenance["player_vocab_sha256"] = vocabulary.sha256
    calibration = ReturnCalibration(window_count=65_536)
    calibration.values = [12.0] * calibration.window_count
    calibration.valid = [True] * calibration.window_count
    calibration.replay_ids = ["synthetic-replay"] * calibration.window_count
    return {
        "cfg": config,
        "model": tensors,
        "provenance": provenance,
        "conditioning_protocol": json.loads((_FIXTURES / "conditioning_protocol.json").read_text()),
        "return_calibration": calibration.state_dict(),
        "return_masker": {
            "version": 1,
            "probability": config["return_dropout"],
            "enabled": True,
            "generator": torch.Generator().manual_seed(42).get_state(),
        },
        "step": 4096,
        "wandb_id": "supported-descendant-with-a-new-run-id",
    }


def _save_checkpoint(root: Path, payload: dict[str, object]) -> Path:
    path = root / "checkpoint.pt"
    torch.save(payload, path)
    return path


def _statistics() -> dict[str, dict[str, float]]:
    names = (*FLOAT_FEATURES, *(f"nana_{name}" for name in FLOAT_FEATURES), *(f"item_{name}" for name in ITEM_FLOATS))
    return {name: {"mean": 0.0, "std": 1.0, "min": -1.0, "max": 1.0} for name in names}


def _write_bundle(
    root: Path,
    payload: dict[str, object],
    *,
    version: int = 2,
    delays: tuple[int, ...] = (0, 2, 3),
    statistics: dict[str, dict[str, float]] | None = None,
) -> Path:
    checkpoint = _save_checkpoint(root, payload)
    identity = checkpoint_sha256(checkpoint)
    stats = _statistics() if statistics is None else statistics
    encoded = json.dumps(stats, sort_keys=True).encode()
    stats_path = root / "stats.json"
    stats_path.write_bytes(encoded)
    provenance = cast(dict[str, object], payload["provenance"])
    config = {
        "schema_version": version,
        "checkpoint_sha256": identity,
        "wandb_id": payload["wandb_id"],
        "step": payload["step"],
        "return_p90": 12.0,
        "stats_sha256": hashlib.sha256(encoded).hexdigest(),
        "source_stats_sha256": provenance["source_statistics_sha256"],
    }
    config_path = root / "backend.json"
    config_path.write_text(json.dumps(config))
    destination = root / "test.hal"
    write_policy_bundle(
        destination,
        BundleDescription(
            "synthetic 059 descendant",
            ACTION_SEQUENCE_BACKEND,
            version,
            REQUIRED_OBSERVATION_FIELDS,
            delays,
            False,
            identity,
        ),
        {"checkpoint.pt": checkpoint, "backend.json": config_path, "stats.json": stats_path},
    )
    return destination


def test_checkpoint_validation_preserves_rng_and_accepts_a_resumed_descendant(
    tmp_path: Path, checkpoint_payload: dict[str, object]
) -> None:
    provenance = dict(cast(dict[str, object], checkpoint_payload["provenance"]))
    lineage = ResumeLineage("1" * 64, cast(str, provenance["git_sha"]), "2" * 40, "3" * 64)
    provenance["git_sha"] = lineage.new_source_sha
    payload = {**checkpoint_payload, "provenance": provenance, "resume_lineage": [lineage.to_record()]}
    path = _save_checkpoint(tmp_path, payload)
    before = torch.get_rng_state()
    raw, contract, codes, p90 = read_checkpoint(path)
    assert torch.equal(torch.get_rng_state(), before)
    assert codes == ("TEST#1",)
    assert contract.model.d_model == 1024
    assert p90 == 12.0
    assert raw["resume_lineage"] == [lineage.to_record()]
    assert raw["wandb_id"] == "supported-descendant-with-a-new-run-id"


@pytest.mark.parametrize("kind", ("missing", "shape", "dtype", "nonfinite", "codebook", "rotary"))
def test_invalid_checkpoint_tensor_is_rejected(
    tmp_path: Path, checkpoint_payload: dict[str, object], kind: str
) -> None:
    tensors = dict(cast(dict[str, Tensor], checkpoint_payload["model"]))
    learned = next(name for name in tensors if name.endswith("weight"))
    if kind == "missing":
        del tensors[learned]
    elif kind == "shape":
        tensors[learned] = tensors[learned].flatten()[:1]
    elif kind == "dtype":
        tensors[learned] = tensors[learned].to(torch.float64)
    elif kind == "nonfinite":
        tensors[learned] = torch.full((), float("nan")).expand(tensors[learned].shape)
    else:
        name = "codec.main_centers" if kind == "codebook" else "trunk.blocks.0.attn.rotary.inv_freq"
        tensors[name] = tensors[name].clone() + 0.125
    with pytest.raises(ValueError, match="tensor|buffer"):
        read_checkpoint(_save_checkpoint(tmp_path, {**checkpoint_payload, "model": tensors}))


@pytest.mark.parametrize(("version", "delays"), ((1, (2,)), (2, (0, 2, 3))))
def test_artifact_read_keeps_declared_capabilities_and_identity(
    tmp_path: Path, checkpoint_payload: dict[str, object], version: int, delays: tuple[int, ...]
) -> None:
    artifact = read_action_sequence_artifact(
        _write_bundle(tmp_path, checkpoint_payload, version=version, delays=delays)
    )
    assert artifact.capability_version == version
    assert artifact.spec.supported_transport_delays == delays
    assert artifact.checkpoint_sha256 == checkpoint_sha256(tmp_path / "checkpoint.pt")
    assert artifact.vocabulary.codes == ("TEST#1",)


def test_old_artifact_cannot_declare_new_profiles(tmp_path: Path, checkpoint_payload: dict[str, object]) -> None:
    path = _write_bundle(tmp_path, checkpoint_payload, version=1)
    with pytest.raises(ValueError, match="manifest is incompatible"):
        read_action_sequence_artifact(path)


@pytest.mark.parametrize(("field", "value"), (("std", -1.0), ("min", 0.1), ("max", -0.1), ("mean", float("inf"))))
def test_invalid_bundle_statistics_are_rejected(
    tmp_path: Path, checkpoint_payload: dict[str, object], field: str, value: float
) -> None:
    stats = _statistics()
    stats["position_x"][field] = value
    path = _write_bundle(tmp_path, checkpoint_payload, statistics=stats)
    with pytest.raises(ValueError, match="statistics"):
        read_action_sequence_artifact(path)


@pytest.mark.parametrize("kind", ("missing", "extra", "empty"))
def test_bundle_statistics_require_the_current_feature_set(
    tmp_path: Path, checkpoint_payload: dict[str, object], kind: str
) -> None:
    stats = _statistics()
    if kind == "missing":
        del stats["nana_percent"]
    elif kind == "extra":
        stats["unknown_feature"] = dict(stats["position_x"])
    else:
        stats.clear()
    path = _write_bundle(tmp_path, checkpoint_payload, statistics=stats)
    with pytest.raises(ValueError, match="statistics feature names"):
        read_action_sequence_artifact(path)
