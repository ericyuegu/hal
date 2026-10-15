"""Tests for the strict small-Muon context encoder."""

from collections.abc import Mapping
from pathlib import Path

import pytest
import torch
from torch import Tensor

from hal.training import features
from hal.wire import ACTION_CHANNELS
from research.mechinterp_20260911 import small_muon_model

_SCRATCH = Path("/home/ericgu/src/hal/runs/mechinterp-20260911-scratch/small-muon-16k")
_CHECKPOINTS = (
    ("o26", _SCRATCH / "cqbbbg77-final.source.pt", 384),
    ("o37", _SCRATCH / "a117chkw-final.source.pt", 256),
    ("o37", _SCRATCH / "5wfk2esf-final.source.pt", 256),
    ("o43", _SCRATCH / "1imfy8v3-final.source.pt", 384),
)
_O37_CHECKPOINT = _SCRATCH / "a117chkw-final.source.pt"
_O43_CHECKPOINT = _SCRATCH / "1imfy8v3-final.source.pt"


def _load_checkpoint(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"required historical checkpoint is absent: {path}")
    value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(value, dict):
        raise TypeError("test checkpoint must contain a mapping")
    return value


def _checkpoint_parts(path: Path) -> tuple[dict[str, object], dict[str, Tensor]]:
    checkpoint = _load_checkpoint(path)
    raw_config = checkpoint.get("cfg")
    raw_state = checkpoint.get("model")
    if not isinstance(raw_config, Mapping) or any(not isinstance(key, str) for key in raw_config):
        raise TypeError("test checkpoint cfg must be a string-keyed mapping")
    if not isinstance(raw_state, Mapping) or any(not isinstance(key, str) for key in raw_state):
        raise TypeError("test checkpoint model must be a string-keyed mapping")
    if any(not isinstance(value, Tensor) for value in raw_state.values()):
        raise TypeError("test checkpoint model must contain only tensors")
    config = {str(key): value for key, value in raw_config.items()}
    state = {str(key): value for key, value in raw_state.items() if isinstance(value, Tensor)}
    return config, state


def _zero_features(batch: int = 1, length: int = 128) -> dict[str, Tensor]:
    values: dict[str, Tensor] = {}
    for prefix in features.BASE_PLAYER_PREFIXES:
        for name in features.FLOAT_FEATURES:
            values[f"{prefix}_{name}"] = torch.zeros(batch, length)
        for name in features.CAT_FEATURES:
            values[f"{prefix}_{name}"] = torch.zeros(batch, length, dtype=torch.long)
    for name in ACTION_CHANNELS:
        values[f"ego_{name}"] = torch.zeros(batch, length)
    values["ego_character"] = torch.zeros(batch, length, dtype=torch.long)
    values["opp_character"] = torch.zeros(batch, length, dtype=torch.long)
    values["stage"] = torch.zeros(batch, length, dtype=torch.long)
    return values


def test_required_checkpoint_absence_is_a_failure(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="required historical checkpoint"):
        _load_checkpoint(tmp_path / "missing.pt")


@pytest.mark.parametrize(("family", "checkpoint_path", "width"), _CHECKPOINTS)
def test_real_context_encoder_loads_restricted_state(
    family: small_muon_model.SmallMuonFamily,
    checkpoint_path: Path,
    width: int,
) -> None:
    config, state = _checkpoint_parts(checkpoint_path)

    model = small_muon_model.build_small_muon_model(family, config, state)
    tokens = model.context_tokens(_zero_features())

    assert tokens.shape == (1, 128, width)
    assert torch.isfinite(tokens).all()
    assert model.input_profile.player_vocabulary is None
    assert model.input_profile.data_root.endswith("ranked-anonymized-1/mds-policy-v7")
    assert model.input_profile.statistics_sha256 == (
        "031dd0e35a7201b299933ec999615f8657dd3e616e0efd3d78b156b1920e0bef"
    )
    assert model.trunk.attn_path == "unresolved"


def test_context_encoder_matches_explicit_component_order() -> None:
    config, state = _checkpoint_parts(_O37_CHECKPOINT)
    model = small_muon_model.build_small_muon_model("o37", config, state)
    values = _zero_features()

    action_indices = model.codec.quantize(features.stack_actions(values))
    cat_specs = {
        **features.CAT_FEATURES,
        "action": (model.config.action_vocabulary, model.config.action_state_embedding_dim),
    }
    parts: list[Tensor] = []
    for prefix in features.BASE_PLAYER_PREFIXES:
        player_parts = [values[f"{prefix}_{name}"][..., None] for name in features.FLOAT_FEATURES]
        player_parts.extend(
            torch.zeros_like(values[f"{prefix}_{name}"])[..., None] for name in features.FLOAT_FEATURES
        )
        for name, (vocabulary, _) in cat_specs.items():
            indices = values[f"{prefix}_{name}"].clamp(0, vocabulary - 1)
            player_parts.append(model.cat_embeds[name](indices))
        parts.append(torch.cat(player_parts, dim=-1))
    parts.extend(
        (
            model.codec.embed_frame(action_indices),
            model.char_emb(values["ego_character"]),
            model.char_emb(values["opp_character"]),
            model.stage_emb(values["stage"]),
        )
    )
    expected = model.ctx_proj(torch.cat(parts, dim=-1))

    torch.testing.assert_close(model.context_tokens(values), expected, rtol=0, atol=0)


def test_loader_rejects_missing_required_state() -> None:
    config, raw_state = _checkpoint_parts(_O37_CHECKPOINT)
    state = dict(raw_state)
    del state["ctx_proj.bias"]

    with pytest.raises(RuntimeError, match="missing=.*ctx_proj.bias"):
        small_muon_model.build_small_muon_model("o37", config, state)


def test_loader_rejects_required_tensor_dtype_change() -> None:
    config, raw_state = _checkpoint_parts(_O37_CHECKPOINT)
    state = dict(raw_state)
    state["ctx_proj.bias"] = state["ctx_proj.bias"].to(torch.float64)

    with pytest.raises(RuntimeError, match="shape/dtype"):
        small_muon_model.build_small_muon_model("o37", config, state)


def test_loader_rejects_other_representation() -> None:
    raw_config, state = _checkpoint_parts(_O37_CHECKPOINT)
    config = {**raw_config, "observation_bundle": "v6_lean"}

    with pytest.raises(ValueError, match="supported profile"):
        small_muon_model.build_small_muon_model("o37", config, state)


def test_o43_legacy_button_reducer_preserves_sequence_semantics() -> None:
    config, state = _checkpoint_parts(_O43_CHECKPOINT)
    model = small_muon_model.build_small_muon_model("o43", config, state)
    actions = torch.zeros(1, 4, len(ACTION_CHANNELS))
    actions[:, :2, ACTION_CHANNELS.index("button_a")] = 1
    actions[:, 3, ACTION_CHANNELS.index("button_b")] = 1

    classes = model.codec.quantize(actions)

    torch.testing.assert_close(classes[0, :, 0], torch.tensor([0, 0, 5, 1]))
    assert model.input_profile.action_codec == "legacy-controller-v2"


def test_o43_loader_rejects_changed_legacy_centers() -> None:
    config, raw_state = _checkpoint_parts(_O43_CHECKPOINT)
    state = dict(raw_state)
    state["codec.main_centers"] = state["codec.main_centers"].clone()
    state["codec.main_centers"][0, 0] = 0.25

    with pytest.raises(ValueError, match="legacy action codec differs"):
        small_muon_model.build_small_muon_model("o43", config, state)
