"""Strict context encoder for the small historical Muon policies.

This module reconstructs only the observation encoder, action-history codec,
and causal trunk used by the O26, O37, and O43 base-observation checkpoints. Callers
must restricted-load the checkpoint and preprocess raw schema-7 windows with
the checkpoint's own dataset statistics before they call this model.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from hal.training import features
from hal.training import scoring
from hal.training import trunk
from hal.wire import ACTION_CHANNELS

SmallMuonFamily = Literal["o26", "o37", "o43"]

_BASE_DATA_ROOT = "data/processed/ranked-anonymized-1/mds-policy-v7"
_BASE_STATS_SHA256 = "031dd0e35a7201b299933ec999615f8657dd3e616e0efd3d78b156b1920e0bef"
_CONTINUOUS_ACTIONS = 6
_GROUP_NAMES = ("buttons", "main_stick", "c_stick", "triggers")
_GROUP_VOCABS = {
    "buttons": scoring.N_BUTTON_COMBOS,
    "main_stick": scoring.STICK_CLUSTER_CENTERS_MAIN.shape[0],
    "c_stick": scoring.STICK_CLUSTER_CENTERS_C.shape[0],
    "triggers": scoring.TRIGGER_CENTERS.shape[0] ** 2,
}
_BUTTON_L_CHANNEL = ACTION_CHANNELS.index("button_l")
_BUTTON_R_CHANNEL = ACTION_CHANNELS.index("button_r")
_TRIGGER_L_CHANNEL = ACTION_CHANNELS.index("trigger_l")
_TRIGGER_R_CHANNEL = ACTION_CHANNELS.index("trigger_r")
_BUTTON_L_SEMANTIC_CHANNEL = ACTION_CHANNELS[6:].index("button_l")

# The supported O43 semantics are frozen at commit
# 22d52f127f3c56d204fbdfb330ae6434b711b07a. These are that source's signed
# equivalents of the ae29e3f legacy centers, in historical class order.
_LEGACY_MAIN_SIGNED = (
    (0.0, 0.0),
    (0.35, 0.0),
    (-0.35, 0.0),
    (0.0, 0.35),
    (0.0, -0.35),
    (0.675, 0.0),
    (-0.675, 0.0),
    (0.0, 0.675),
    (0.0, -0.675),
    (1.0, 0.0),
    (0.0, 1.0),
    (-1.0, 0.0),
    (0.0, -1.0),
    (0.95, -0.3),
    (-0.95, -0.3),
    (0.95, 0.3),
    (-0.95, 0.3),
    (0.85, -0.5),
    (0.85, 0.5),
    (-0.85, -0.5),
    (-0.85, 0.5),
    (0.7, -0.7),
    (-0.7, -0.7),
    (0.7, 0.7),
    (-0.7, 0.7),
    (0.5, 0.5),
    (-0.5, 0.5),
    (0.5, -0.5),
    (-0.5, -0.5),
    (0.5, 0.85),
    (-0.5, 0.85),
    (0.5, -0.85),
    (-0.5, -0.85),
    (0.3, -0.95),
    (0.3, 0.95),
    (-0.3, -0.95),
    (-0.3, 0.95),
)
_LEGACY_C_SIGNED = (
    (0.0, 0.0),
    (1.0, 0.0),
    (-1.0, 0.0),
    (0.0, -1.0),
    (0.0, 1.0),
    (-0.7, -0.7),
    (0.7, -0.7),
    (0.7, 0.7),
    (-0.7, 0.7),
)
_LEGACY_TRIGGER_VALUES = (0.0, 0.35, 0.6, 0.85, 1.0)
_LEGACY_GROUP_VOCABS = {
    "buttons": 6,
    "main_stick": 37,
    "c_stick": 9,
    "triggers": 5,
}

_REQUIRED_PREFIXES = (
    "cat_embeds.",
    "char_emb.",
    "codec.",
    "ctx_proj.",
    "stage_emb.",
    "trunk.",
)
_OMITTED_PREFIXES = ("temporal.", "v6_cat_embeds.", "value_head.")
_OMITTED_EXACT = frozenset({"hitstun_action"})


@dataclass(frozen=True, slots=True)
class SmallMuonInputProfile:
    """Stored representation that a caller must reproduce."""

    context_length: int
    data_root: str
    mds_schema_version: int
    observation_bundle: str
    statistics_sha256: str = _BASE_STATS_SHA256
    action_codec: str = "structured-controller-v3"
    player_vocabulary: str | None = None


@dataclass(frozen=True, slots=True)
class SmallMuonConfig:
    """Architecture fields used by the reconstructed model."""

    family: SmallMuonFamily
    d_model: int
    n_layers: int
    n_heads: int
    attention_window: int
    context_length: int
    action_vocabulary: int
    action_state_embedding_dim: int
    action_embedding_dim: int
    character_vocabulary: int
    character_embedding_dim: int
    stage_vocabulary: int
    stage_embedding_dim: int
    input_profile: SmallMuonInputProfile


def _integer(values: Mapping[str, object], name: str) -> int:
    value = values.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"checkpoint config {name!r} must be an integer")
    return value


def _string(values: Mapping[str, object], name: str) -> str:
    value = values.get(name)
    if not isinstance(value, str):
        raise TypeError(f"checkpoint config {name!r} must be a string")
    return value


def _require_equal(values: Mapping[str, object], expected: Mapping[str, object]) -> None:
    mismatches = {
        name: (want, values.get(name, "<missing>"))
        for name, want in expected.items()
        if values.get(name, "<missing>") != want
    }
    if mismatches:
        raise ValueError(f"checkpoint config differs from the supported profile: {mismatches}")


def _parse_config(family: SmallMuonFamily, values: Mapping[str, object]) -> SmallMuonConfig:
    common = {
        "L_ctx": 128,
        "action_embed_dim": 16,
        "action_state_embed_dim": 48,
        "action_vocab": 1024,
        "adam_lr": 8.5e-4,
        "attn_window": 0,
        "char_dim": 8,
        "char_vocab": 32,
        "data_root": _BASE_DATA_ROOT,
        "max_steps": 16_384,
        "mds_schema_version": 7,
        "muon_lr": 0.02,
        "observation_bundle": "base",
        "stage_dim": 4,
        "stage_vocab": 32,
        "weight_decay": 0.01,
    }
    if family == "o37":
        common.update(
            {
                "d_model": 256,
                "experiment_id": "037_factorization_matrix_v1",
                "n_heads": 4,
                "n_layers": 8,
            }
        )
        if values.get("future_conditioning") not in ("independent", "selected_ar"):
            raise ValueError("O37 future_conditioning is unsupported")
        if values.get("group_conditioning") not in ("independent", "autoregressive"):
            raise ValueError("O37 group_conditioning is unsupported")
        if values.get("actor_weighting") != "uniform":
            raise ValueError("only the O37 uniform-BC matrix is supported")
    elif family == "o26":
        common.update(
            {
                "d_model": 384,
                "decoder_arch_version": 3,
                "n_heads": 6,
                "n_layers": 8,
            }
        )
        if "codec_version" in values or "model_family" in values or "experiment_id" in values:
            raise ValueError("the O26 profile rejects later experiment identities")
    elif family == "o43":
        del common["weight_decay"]
        common.update(
            {
                "adam_weight_decay": 0.01,
                "codec_version": 2,
                "d_model": 384,
                "decoder_arch_version": 3,
                "muon_weight_decay": 0.01,
                "n_heads": 6,
                "n_layers": 8,
                "next_frame_loss_share": 0.5,
            }
        )
        if "ablation_arm" in values or "model_family" in values or "experiment_id" in values:
            raise ValueError("the O43 profile rejects later experiment identities")
    else:
        raise ValueError(f"unsupported small-Muon family {family!r}")
    _require_equal(values, common)

    data_root = _string(values, "data_root")
    context_length = _integer(values, "L_ctx")
    return SmallMuonConfig(
        family=family,
        d_model=_integer(values, "d_model"),
        n_layers=_integer(values, "n_layers"),
        n_heads=_integer(values, "n_heads"),
        attention_window=_integer(values, "attn_window"),
        context_length=context_length,
        action_vocabulary=_integer(values, "action_vocab"),
        action_state_embedding_dim=_integer(values, "action_state_embed_dim"),
        action_embedding_dim=_integer(values, "action_embed_dim"),
        character_vocabulary=_integer(values, "char_vocab"),
        character_embedding_dim=_integer(values, "char_dim"),
        stage_vocabulary=_integer(values, "stage_vocab"),
        stage_embedding_dim=_integer(values, "stage_dim"),
        input_profile=SmallMuonInputProfile(
            context_length=context_length,
            data_root=data_root,
            mds_schema_version=_integer(values, "mds_schema_version"),
            observation_bundle=_string(values, "observation_bundle"),
            action_codec="legacy-controller-v2" if family == "o43" else "structured-controller-v3",
        ),
    )


def _decoder_rmsnorm(value: Tensor) -> Tensor:
    return F.rms_norm(value, (value.shape[-1],), eps=1e-6)


class _StructuredControllerCodec(nn.Module):
    """O26/O37 physical controller codec, with checkpointed centers."""

    button_valid_for_trigger: Tensor
    c_centers: Tensor
    main_centers: Tensor
    trigger_centers: Tensor

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.class_embeddings = nn.ModuleDict(
            {name: nn.Embedding(_GROUP_VOCABS[name], embedding_dim) for name in _GROUP_NAMES}
        )
        semantic_dimensions = {
            "buttons": 8,
            "main_stick": 2,
            "c_stick": 2,
            "triggers": 2,
        }
        self.semantic_projections = nn.ModuleDict(
            {name: nn.Linear(width, embedding_dim, bias=False) for name, width in semantic_dimensions.items()}
        )
        self.register_buffer("main_centers", scoring.STICK_CLUSTER_CENTERS_MAIN.clone())
        self.register_buffer("c_centers", scoring.STICK_CLUSTER_CENTERS_C.clone())
        self.register_buffer("trigger_centers", scoring.TRIGGER_CENTERS.clone())
        self.register_buffer(
            "button_valid_for_trigger",
            torch.empty(_GROUP_VOCABS["triggers"], _GROUP_VOCABS["buttons"], dtype=torch.bool),
        )

    def _embedding(self, name: str) -> nn.Embedding:
        return cast(nn.Embedding, self.class_embeddings[name])

    def validate_stored_identity(self) -> None:
        """Reject a checkpoint whose physical action centers changed."""
        expected = {
            "main_stick": scoring.STICK_CLUSTER_CENTERS_MAIN,
            "c_stick": scoring.STICK_CLUSTER_CENTERS_C,
            "triggers": scoring.TRIGGER_CENTERS,
        }
        actual = {
            "main_stick": self.main_centers,
            "c_stick": self.c_centers,
            "triggers": self.trigger_centers,
        }
        mismatches = [name for name in expected if not torch.equal(actual[name].cpu(), expected[name].cpu())]
        if mismatches:
            raise ValueError(f"checkpoint action centers differ for groups: {mismatches}")

    def quantize(self, actions: Tensor) -> Tensor:
        if actions.shape[-1] != len(ACTION_CHANNELS):
            raise ValueError(
                f"controller actions must end in {len(ACTION_CHANNELS)} channels, got {tuple(actions.shape)}"
            )
        canonical = actions.clone()
        canonical[..., 4] = torch.where(
            canonical[..., _BUTTON_L_CHANNEL] > 0.5,
            torch.ones_like(canonical[..., 4]),
            canonical[..., 4],
        )
        canonical[..., 5] = torch.where(
            canonical[..., _BUTTON_R_CHANNEL] > 0.5,
            torch.ones_like(canonical[..., 5]),
            canonical[..., 5],
        )
        continuous = canonical[..., :_CONTINUOUS_ACTIONS]
        buttons = scoring.buttons_to_combo(canonical[..., _CONTINUOUS_ACTIONS:])
        main_stick = scoring.nearest_cluster(continuous[..., :2], self.main_centers)
        c_stick = scoring.nearest_cluster(continuous[..., 2:4], self.c_centers)
        trigger_pair = scoring.nearest_center(continuous[..., 4:6], self.trigger_centers)
        triggers = trigger_pair[..., 0] * self.trigger_centers.shape[0] + trigger_pair[..., 1]
        return torch.stack((buttons, main_stick, c_stick, triggers), dim=-1)

    def _semantic_values(self, name: str, indices: Tensor) -> Tensor:
        dtype = self._embedding(name).weight.dtype
        if name == "buttons":
            return scoring.combo_to_buttons(indices).to(dtype)
        if name == "main_stick":
            return scoring.cluster_to_xy(indices, self.main_centers)
        if name == "c_stick":
            return scoring.cluster_to_xy(indices, self.c_centers)
        if name == "triggers":
            count = self.trigger_centers.shape[0]
            return torch.stack(
                (
                    scoring.center_to_value(indices // count, self.trigger_centers),
                    scoring.center_to_value(indices % count, self.trigger_centers),
                ),
                dim=-1,
            )
        raise ValueError(f"unknown controller group {name!r}")

    def _group_embedding(self, name: str, indices: Tensor) -> Tensor:
        embedding = self._embedding(name)
        projection = cast(nn.Linear, self.semantic_projections[name])
        semantic = self._semantic_values(name, indices).to(embedding.weight.dtype)
        value = embedding(indices) + projection(semantic)
        return _decoder_rmsnorm(value)

    def embed_frame(self, indices: Tensor) -> Tensor:
        values = [self._group_embedding(name, indices[..., group]) for group, name in enumerate(_GROUP_NAMES)]
        return torch.cat(values, dim=-1)


def _legacy_button_classes(actions: Tensor) -> Tensor:
    """Apply O43's sequence-dependent ae29e3f button reducer."""
    if actions.ndim < 2 or actions.shape[-1] != len(ACTION_CHANNELS):
        raise ValueError(f"legacy button input must end in [length, {len(ACTION_CHANNELS)}]")
    button = {name: ACTION_CHANNELS.index(f"button_{name}") for name in ("a", "b", "x", "y", "z", "l", "r")}
    pressed = torch.stack(
        (
            actions[..., button["a"]] > 0.5,
            actions[..., button["b"]] > 0.5,
            (actions[..., button["x"]] > 0.5) | (actions[..., button["y"]] > 0.5),
            actions[..., button["z"]] > 0.5,
            (actions[..., button["l"]] > 0.5) | (actions[..., button["r"]] > 0.5),
        ),
        dim=-1,
    )
    length = pressed.shape[-2]
    flat = pressed.reshape(-1, length, 5)
    time = torch.arange(length, device=actions.device).expand(flat.shape[0], -1)
    previous = F.pad(flat[:, :-1], (0, 0, 1, 0))
    same = (flat == previous).all(-1)
    new = flat & ~previous
    selected = new.to(torch.int64).argmax(-1)
    none = torch.full_like(selected, _LEGACY_GROUP_VOCABS["buttons"] - 1)
    base = torch.where(new.any(-1), selected, none)
    changed = ~same | ~flat.any(-1)
    anchor = torch.where(changed, time, -1).cummax(-1).values
    output = base.gather(1, anchor)
    return output.reshape(pressed.shape[:-1])


class _LegacyStructuredControllerCodec(_StructuredControllerCodec):
    """O43 legacy semantics inside the unchanged O26-sized tables."""

    _c_quant_centers: Tensor
    _main_quant_centers: Tensor
    _trigger_quant_centers: Tensor

    def __init__(self, embedding_dim: int) -> None:
        super().__init__(embedding_dim)
        self.register_buffer(
            "_main_quant_centers",
            torch.tensor(_LEGACY_MAIN_SIGNED, dtype=torch.float64) / 2.0 + 0.5,
            persistent=False,
        )
        self.register_buffer(
            "_c_quant_centers",
            torch.tensor(_LEGACY_C_SIGNED, dtype=torch.float64) / 2.0 + 0.5,
            persistent=False,
        )
        self.register_buffer(
            "_trigger_quant_centers",
            torch.tensor(_LEGACY_TRIGGER_VALUES, dtype=torch.float64),
            persistent=False,
        )

    @staticmethod
    def _nearest_stick(values: Tensor, centers: Tensor) -> Tensor:
        points = (values.to(torch.float64) / 2.0 + 0.5).to(torch.float32).to(torch.float64)
        return (points[..., None, :] - centers).square().sum(-1).argmin(-1)

    def validate_stored_identity(self) -> None:
        expected_main = torch.zeros_like(self.main_centers)
        expected_main[: len(_LEGACY_MAIN_SIGNED)] = torch.tensor(
            _LEGACY_MAIN_SIGNED,
            dtype=expected_main.dtype,
            device=expected_main.device,
        )
        expected_c = torch.tensor(
            _LEGACY_C_SIGNED,
            dtype=self.c_centers.dtype,
            device=self.c_centers.device,
        )
        expected_trigger = torch.tensor(
            _LEGACY_TRIGGER_VALUES,
            dtype=self.trigger_centers.dtype,
            device=self.trigger_centers.device,
        )
        mismatches = []
        if not torch.equal(self.main_centers, expected_main):
            mismatches.append("main_stick")
        if not torch.equal(self.c_centers, expected_c):
            mismatches.append("c_stick")
        if not torch.equal(self.trigger_centers, expected_trigger):
            mismatches.append("triggers")
        if not self.button_valid_for_trigger.all():
            mismatches.append("button_valid_for_trigger")
        if mismatches:
            raise ValueError(f"checkpoint legacy action codec differs for fields: {mismatches}")

    def quantize(self, actions: Tensor) -> Tensor:
        if actions.shape[-1] != len(ACTION_CHANNELS):
            raise ValueError(
                f"controller actions must end in {len(ACTION_CHANNELS)} channels, got {tuple(actions.shape)}"
            )
        main = self._nearest_stick(actions[..., :2], self._main_quant_centers)
        c_stick = self._nearest_stick(actions[..., 2:4], self._c_quant_centers)
        fused = actions[..., _TRIGGER_L_CHANNEL : _TRIGGER_R_CHANNEL + 1].amax(-1)
        fused = fused.to(torch.float32).to(torch.float64)
        triggers = (fused[..., None] - self._trigger_quant_centers).square().argmin(-1)
        return torch.stack((_legacy_button_classes(actions), main, c_stick, triggers), dim=-1)

    def _semantic_values(self, name: str, indices: Tensor) -> Tensor:
        dtype = self._embedding(name).weight.dtype
        if name == "buttons":
            values = torch.zeros(*indices.shape, 8, dtype=dtype, device=indices.device)
            values[..., 0] = (indices == 0).to(dtype)
            values[..., 1] = (indices == 1).to(dtype)
            values[..., 2] = (indices == 2).to(dtype)
            values[..., 4] = (indices == 3).to(dtype)
            values[..., _BUTTON_L_SEMANTIC_CHANNEL] = (indices == 4).to(dtype)
            return values
        if name == "main_stick":
            return self.main_centers[: _LEGACY_GROUP_VOCABS[name]].to(dtype)[indices]
        if name == "c_stick":
            return self.c_centers.to(dtype)[indices]
        if name == "triggers":
            fused = self.trigger_centers[: _LEGACY_GROUP_VOCABS[name]].to(dtype)[indices]
            return torch.stack((fused, torch.zeros_like(fused)), dim=-1)
        raise ValueError(f"unknown controller group {name!r}")


class SmallMuonModel(nn.Module):
    """O26/O37/O43 base observation encoder and dense causal trunk."""

    config: SmallMuonConfig

    def __init__(self, config: SmallMuonConfig, model_state: Mapping[str, Tensor]) -> None:
        super().__init__()
        self.config = config
        self.input_profile = config.input_profile
        codec_type = _LegacyStructuredControllerCodec if config.family == "o43" else _StructuredControllerCodec
        self.codec = codec_type(config.action_embedding_dim)
        self.cat_specs = {
            **features.CAT_FEATURES,
            "action": (config.action_vocabulary, config.action_state_embedding_dim),
        }
        self.cat_embeds = nn.ModuleDict(
            {name: nn.Embedding(vocabulary, dimension) for name, (vocabulary, dimension) in self.cat_specs.items()}
        )
        self.char_emb = nn.Embedding(config.character_vocabulary, config.character_embedding_dim)
        self.stage_emb = nn.Embedding(config.stage_vocabulary, config.stage_embedding_dim)
        per_player = len(features.FLOAT_FEATURES) * 2 + sum(dimension for _, dimension in self.cat_specs.values())
        input_width = (
            len(features.BASE_PLAYER_PREFIXES) * per_player
            + len(_GROUP_NAMES) * config.action_embedding_dim
            + 2 * config.character_embedding_dim
            + config.stage_embedding_dim
        )
        self.ctx_proj = nn.Linear(input_width, config.d_model)
        self.trunk = trunk.Trunk(
            trunk.TrunkConfig(
                d_model=config.d_model,
                n_layers=config.n_layers,
                n_heads=config.n_heads,
                L_ctx=config.context_length,
                attn_window=config.attention_window,
                attention_backend="dense_sdpa",
            ),
            prefer_flex=False,
        )
        self._load_relevant_state(model_state)
        self.codec.validate_stored_identity()

    def _load_relevant_state(self, model_state: Mapping[str, Tensor]) -> None:
        if not model_state or any(not isinstance(name, str) for name in model_state):
            raise TypeError("checkpoint model state must be a non-empty string-keyed mapping")
        non_tensors = sorted(name for name, value in model_state.items() if not isinstance(value, Tensor))
        if non_tensors:
            raise TypeError(f"checkpoint model state has non-tensor values: {non_tensors}")
        unknown = sorted(
            name
            for name in model_state
            if name not in _OMITTED_EXACT
            and not name.startswith(_REQUIRED_PREFIXES)
            and not name.startswith(_OMITTED_PREFIXES)
        )
        if unknown:
            raise ValueError(f"checkpoint model state has unsupported keys: {unknown}")
        relevant = {name: value for name, value in model_state.items() if name.startswith(_REQUIRED_PREFIXES)}
        expected = self.state_dict()
        missing = sorted(expected.keys() - relevant.keys())
        unexpected = sorted(relevant.keys() - expected.keys())
        if missing or unexpected:
            raise RuntimeError(f"checkpoint state keys differ: missing={missing}, unexpected={unexpected}")
        incompatible = []
        for name in sorted(expected):
            actual = relevant[name]
            wanted = expected[name]
            if actual.shape != wanted.shape or actual.dtype != wanted.dtype:
                incompatible.append(
                    f"{name}: checkpoint shape/dtype={tuple(actual.shape)}/{actual.dtype}, "
                    f"expected={tuple(wanted.shape)}/{wanted.dtype}"
                )
        if incompatible:
            raise RuntimeError(f"checkpoint tensors differ from the model contract: {incompatible}")
        self.load_state_dict(relevant, strict=True)

    def _per_player_features(self, values: Mapping[str, Tensor], prefix: str) -> Tensor:
        reference = values[f"{prefix}_position_x"]
        parts: list[Tensor] = []
        masks: list[Tensor] = []
        for name in features.FLOAT_FEATURES:
            parts.append(values[f"{prefix}_{name}"][..., None])
            mask = values.get(f"{prefix}_{name}_mask")
            masks.append((torch.zeros_like(reference) if mask is None else mask)[..., None])
        parts.extend(masks)
        for name, (vocabulary, _) in self.cat_specs.items():
            indices = values[f"{prefix}_{name}"].clamp(0, vocabulary - 1)
            parts.append(self.cat_embeds[name](indices))
        return torch.cat(parts, dim=-1)

    def context_tokens(
        self,
        values: Mapping[str, Tensor],
        action_indices: Tensor | None = None,
    ) -> Tensor:
        """Encode one tensor per observed frame using the stored action codec."""
        if action_indices is None:
            actions = torch.stack([values[f"ego_{name}"] for name in ACTION_CHANNELS], dim=-1)
            action_indices = self.codec.quantize(actions)
        parts = [self._per_player_features(values, prefix) for prefix in features.BASE_PLAYER_PREFIXES]
        parts.append(self.codec.embed_frame(action_indices))
        parts.append(self.char_emb(values["ego_character"].clamp(0, self.config.character_vocabulary - 1)))
        parts.append(self.char_emb(values["opp_character"].clamp(0, self.config.character_vocabulary - 1)))
        parts.append(self.stage_emb(values["stage"].clamp(0, self.config.stage_vocabulary - 1)))
        tokens = self.ctx_proj(torch.cat(parts, dim=-1))
        expected = (*values["stage"].shape, self.config.d_model)
        if tokens.shape != expected or tokens.shape[1] != self.config.context_length:
            raise ValueError(
                f"encoded context has shape {tuple(tokens.shape)}, expected "
                f"[batch, {self.config.context_length}, {self.config.d_model}]"
            )
        return tokens

    def forward_dense(
        self,
        values: Mapping[str, Tensor],
        context_padding: Tensor,
    ) -> Tensor:
        """Run the public dense-SDPA correctness path."""
        return self.trunk.forward_dense(self.context_tokens(values), context_padding)


def build_small_muon_model(
    family: SmallMuonFamily,
    config: Mapping[str, object],
    model_state: Mapping[str, Tensor],
) -> SmallMuonModel:
    """Build a strict research model from an already restricted-loaded state.

    Args:
        family: Explicit historical experiment family.
        config: Raw checkpoint configuration.
        model_state: Tensor-only checkpoint model state.

    Returns:
        The base observation encoder and causal trunk.

    Raises:
        TypeError: A configuration or state field has the wrong type.
        ValueError: The checkpoint is outside the supported O26/O37/O43 profiles.
        RuntimeError: Required state keys, shapes, or dtypes differ.
    """
    return SmallMuonModel(_parse_config(family, config), model_state)
