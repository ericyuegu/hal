"""Portable export and stateful runtime for O50 policy checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
import re
import secrets
import tempfile
from collections import deque
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Final
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_model
from safetensors.torch import save_model
from torch import Tensor

from hal import streams
from hal.controller import POLICY_BUTTON_MASK
from hal.controller import ControllerAction
from hal.data.feature_stats import FeatureStats
from hal.data.schema import Rank
from hal.eval.policy_sampling import SlotGroupRng
from hal.inference.api import ObservationScalar
from hal.inference.api import Policy
from hal.inference.api import PolicyInput
from hal.inference.api import PolicyOutput
from hal.inference.api import PolicySpec
from hal.inference.api import RuntimeConfig
from hal.inference.api import validate_policy_inputs
from hal.inference.bundle import BundleDescription
from hal.inference.bundle import PolicyBundleManifest
from hal.inference.bundle import extract_policy_bundle
from hal.inference.bundle import write_policy_bundle
from hal.inference.checkpoints import resolve_checkpoint
from hal.inference.o50_model import CONTROLLER_GROUP_COUNT
from hal.inference.o50_model import CONTROLLER_GROUP_NAMES
from hal.inference.o50_model import O50_BACKEND
from hal.inference.o50_model import O50_BACKEND_VERSION
from hal.inference.o50_model import O50Config
from hal.inference.o50_model import O50Model
from hal.inference.o50_model import amp_context
from hal.training.checkpoints import checkpoint_sha256
from hal.training.ego_stats import consolidate_key
from hal.training.ego_stats import load_consolidated_mixture_stats
from hal.training.features import ACTION_CHANNELS
from hal.training.features import BASE_ITEMS_PROJECTION
from hal.training.features import BASE_PLAYER_PREFIXES
from hal.training.features import CAT_FEATURES
from hal.training.features import FLOAT_FEATURES
from hal.training.features import ITEM_COLUMNS
from hal.training.features import ITEM_FLOATS
from hal.training.features import Context
from hal.training.features import feature_kind
from hal.training.features import preprocess
from hal.training.features import stack_actions
from hal.training.physical_shard_loader import PhysicalRow
from hal.training.physical_shard_loader import RingSlotDescriptor
from hal.training.player_identity import FIRST_CONNECT_CODE_ID
from hal.training.player_identity import PlayerVocabulary
from hal.training.player_identity import decode_player_codes
from hal.training.player_identity import encode_player_codes
from hal.wire import BUTTON_BITS
from hal.wire import ITEM_SLOTS
from hal.wire import item_column

_CONFIG_MEMBER: Final[str] = "backend.json"
_STATS_SCHEMA_VERSION: Final[int] = 1
_STATS_ALGORITHM: Final[str] = "replay-weighted-consolidated-mixture-v1"
_SOURCE_COUNT: Final[int] = 44
_SHA256_PATTERN: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_SUPPORTED_DELAYS: Final[tuple[int, ...]] = (0, 2, 3)
_ACTION_FIELDS: Final[frozenset[str]] = frozenset(f"ego_{name}" for name in ACTION_CHANNELS)
_MODEL_OBSERVATION_FIELDS: Final[frozenset[str]] = BASE_ITEMS_PROJECTION.columns - _ACTION_FIELDS


def _canonical_field(name: str) -> str:
    if name.startswith("ego_"):
        return f"p1_{name[4:]}"
    if name.startswith("opp_"):
        return f"p2_{name[4:]}"
    return name


O50_REQUIRED_OBSERVATION_FIELDS: Final[tuple[str, ...]] = tuple(
    sorted(_canonical_field(name) for name in _MODEL_OBSERVATION_FIELDS)
)
_O50_STATS_FIELDS: Final[frozenset[str]] = frozenset(
    consolidate_key(name) for name in _MODEL_OBSERVATION_FIELDS if feature_kind(name, ITEM_COLUMNS) == "float"
)
_O50_BUTTON_MASK: Final[int] = sum(BUTTON_BITS[name.removeprefix("button_")] for name in ACTION_CHANNELS[6:])
if _O50_BUTTON_MASK != POLICY_BUTTON_MASK:
    raise RuntimeError("O50 action channels do not match hal.controller.v1")


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None


def _finite_number(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"O50 statistics {label} must be finite")
    return float(value)


def _stats_values(stats: Mapping[str, FeatureStats]) -> dict[str, dict[str, float]]:
    if set(stats) != _O50_STATS_FIELDS:
        raise ValueError(
            "O50 resolved statistics fields differ: "
            f"missing={sorted(_O50_STATS_FIELDS - stats.keys())}, "
            f"unexpected={sorted(stats.keys() - _O50_STATS_FIELDS)}"
        )
    output: dict[str, dict[str, float]] = {}
    for name in sorted(stats):
        value = stats[name]
        if not all(math.isfinite(number) for number in (value.mean, value.std, value.min, value.max)):
            raise ValueError(f"O50 statistic {name!r} is not finite")
        if value.std < 0 or value.min > value.max:
            raise ValueError(f"O50 statistic {name!r} has invalid bounds")
        output[name] = {"max": value.max, "mean": value.mean, "min": value.min, "std": value.std}
    return output


@dataclass(frozen=True, slots=True)
class _ResolvedStats:
    stats: Mapping[str, FeatureStats]
    source_names: tuple[str, ...]
    source_replay_weights: Mapping[str, int]
    source_manifest_sha256: Mapping[str, str]
    source_stats_sha256: Mapping[str, str]
    mds_schema_version: int


def _resolve_export_stats(checkpoint_config: Mapping[str, object]) -> _ResolvedStats:
    raw_names = checkpoint_config.get("source_names")
    if not isinstance(raw_names, (list, tuple)) or any(not isinstance(name, str) for name in raw_names):
        raise ValueError("O50 checkpoint source_names must be a sequence of strings")
    source_names = cast(tuple[str, ...], tuple(raw_names))
    expected_names = tuple(source.name for source in streams.POLICY_WORLD_V8_SOURCES)
    if source_names != expected_names or len(source_names) != _SOURCE_COUNT:
        raise ValueError("O50 checkpoint does not name the frozen 44 policy-world-v8 sources in order")
    raw_mds_schema_version = checkpoint_config.get("mds_schema_version")
    if raw_mds_schema_version != 7 or isinstance(raw_mds_schema_version, bool):
        raise ValueError(f"O50 checkpoint needs mds_schema_version=7, got {raw_mds_schema_version!r}")
    mds_schema_version = cast(int, raw_mds_schema_version)

    paths = tuple(streams.ensure_stats(source.local_root / "stats.json") for source in streams.POLICY_WORLD_V8_SOURCES)
    weights = {name: streams.POLICY_WORLD_V8_TRAIN_REPLAYS[name] for name in source_names}
    stats = load_consolidated_mixture_stats(
        paths,
        [float(weights[name]) for name in source_names],
        expected_mds_schema_version=mds_schema_version,
    )
    _stats_values(stats)
    return _ResolvedStats(
        stats=stats,
        source_names=source_names,
        source_replay_weights=weights,
        source_manifest_sha256={name: streams.POLICY_WORLD_V8_TRAIN_MANIFEST_SHA256[name] for name in source_names},
        source_stats_sha256={name: _sha256_file(path) for name, path in zip(source_names, paths, strict=True)},
        mds_schema_version=mds_schema_version,
    )


def _encode_stats(resolved: _ResolvedStats) -> tuple[bytes, str]:
    finalized = _stats_values(resolved.stats)
    content: dict[str, object] = {
        "algorithm": _STATS_ALGORITHM,
        "finalized": finalized,
        "mds_schema_version": resolved.mds_schema_version,
        "schema_version": _STATS_SCHEMA_VERSION,
        "source_count": len(resolved.source_names),
        "source_manifest_sha256": dict(resolved.source_manifest_sha256),
        "source_names": list(resolved.source_names),
        "source_replay_weights": dict(resolved.source_replay_weights),
        "source_stats_sha256": dict(resolved.source_stats_sha256),
    }
    finalized_sha256 = _sha256_bytes(_canonical_json(finalized))
    content_sha256 = _sha256_bytes(_canonical_json(content))
    return _canonical_json(
        {
            **content,
            "content_sha256": content_sha256,
            "finalized_sha256": finalized_sha256,
        }
    ), content_sha256


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"O50 statistics {label} must be an object")
    return cast(Mapping[str, object], value)


def _decode_stats(encoded: bytes, config: O50Config) -> dict[str, FeatureStats]:
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("O50 statistics are not valid JSON") from error
    raw = _mapping(value, "root")
    expected = {
        "algorithm",
        "content_sha256",
        "finalized",
        "finalized_sha256",
        "mds_schema_version",
        "schema_version",
        "source_count",
        "source_manifest_sha256",
        "source_names",
        "source_replay_weights",
        "source_stats_sha256",
    }
    if set(raw) != expected:
        raise ValueError(
            f"O50 statistics fields differ: missing={sorted(expected - raw.keys())}, "
            f"unexpected={sorted(raw.keys() - expected)}"
        )
    if (
        raw["algorithm"] != _STATS_ALGORITHM
        or raw["schema_version"] != _STATS_SCHEMA_VERSION
        or isinstance(raw["schema_version"], bool)
    ):
        raise ValueError("O50 statistics algorithm or schema is unsupported")
    if raw["mds_schema_version"] != config.mds_schema_version or isinstance(raw["mds_schema_version"], bool):
        raise ValueError("O50 statistics MDS schema does not match the backend config")
    names_value = raw["source_names"]
    if not isinstance(names_value, list) or any(not isinstance(name, str) or not name for name in names_value):
        raise ValueError("O50 statistics source_names must be non-empty strings")
    source_names = cast(list[str], names_value)
    if len(source_names) != _SOURCE_COUNT or len(set(source_names)) != len(source_names):
        raise ValueError(f"O50 statistics must identify exactly {_SOURCE_COUNT} unique sources")
    if raw["source_count"] != len(source_names) or isinstance(raw["source_count"], bool):
        raise ValueError("O50 statistics source_count does not match source_names")
    expected_sources = set(source_names)
    weights = _mapping(raw["source_replay_weights"], "source_replay_weights")
    manifests = _mapping(raw["source_manifest_sha256"], "source_manifest_sha256")
    stats_hashes = _mapping(raw["source_stats_sha256"], "source_stats_sha256")
    if set(weights) != expected_sources or set(manifests) != expected_sources or set(stats_hashes) != expected_sources:
        raise ValueError("O50 statistics provenance maps do not match source_names")
    if any(not isinstance(weight, int) or isinstance(weight, bool) or weight <= 0 for weight in weights.values()):
        raise ValueError("O50 statistics source replay weights must be positive integers")
    if any(not _is_sha256(digest) for digest in (*manifests.values(), *stats_hashes.values())):
        raise ValueError("O50 statistics provenance contains an invalid SHA-256")

    finalized_raw = _mapping(raw["finalized"], "finalized")
    if set(finalized_raw) != _O50_STATS_FIELDS:
        raise ValueError("O50 finalized statistics have the wrong feature set")
    finalized_payload: dict[str, dict[str, float]] = {}
    stats: dict[str, FeatureStats] = {}
    for name in sorted(finalized_raw):
        block = _mapping(finalized_raw[name], name)
        if set(block) != {"max", "mean", "min", "std"}:
            raise ValueError(f"O50 statistic {name!r} has the wrong fields")
        maximum = _finite_number(block["max"], f"{name}.max")
        mean = _finite_number(block["mean"], f"{name}.mean")
        minimum = _finite_number(block["min"], f"{name}.min")
        std = _finite_number(block["std"], f"{name}.std")
        if std < 0 or minimum > maximum:
            raise ValueError(f"O50 statistic {name!r} has invalid bounds")
        finalized_payload[name] = {"max": maximum, "mean": mean, "min": minimum, "std": std}
        stats[name] = FeatureStats(mean=mean, std=std, min=minimum, max=maximum)

    if not _is_sha256(raw["finalized_sha256"]) or raw["finalized_sha256"] != _sha256_bytes(
        _canonical_json(finalized_payload)
    ):
        raise ValueError("O50 finalized statistics SHA-256 does not match their values")
    content = {name: raw[name] for name in expected - {"content_sha256", "finalized_sha256"}}
    content_sha256 = _sha256_bytes(_canonical_json(content))
    if raw["content_sha256"] != content_sha256 or config.stats_sha256 != content_sha256:
        raise ValueError("O50 statistics content SHA-256 does not match the backend config")
    return stats


def _safe_load_checkpoint(path: Path) -> Mapping[str, object]:
    allowed_types = (PhysicalRow, RingSlotDescriptor)
    allowed_names = {f"{value.__module__}.{value.__qualname__}" for value in allowed_types}
    unsafe = set(torch.serialization.get_unsafe_globals_in_checkpoint(path))
    unexpected = unsafe - allowed_names
    if unexpected:
        raise ValueError(f"O50 checkpoint uses unapproved Python globals: {sorted(unexpected)}")
    with torch.serialization.safe_globals(list(allowed_types)):
        raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, Mapping) or any(not isinstance(key, str) for key in raw):
        raise ValueError("O50 checkpoint root must be an object with string keys")
    return cast(Mapping[str, object], raw)


def export_o50_policy(
    checkpoint_source: str | Path,
    destination: str | Path,
    *,
    cache_root: str | Path = Path("runs"),
) -> PolicyBundleManifest:
    """Export one raw O50 checkpoint as a strict, portable policy bundle."""
    checkpoint = resolve_checkpoint(str(checkpoint_source), cache_root=cache_root)
    raw = _safe_load_checkpoint(checkpoint)
    checkpoint_config = _mapping(raw.get("cfg"), "checkpoint config")
    model_state = _mapping(raw.get("model"), "checkpoint model state")
    encoded_codes = model_state.get("player_code_bytes")
    if not isinstance(encoded_codes, Tensor) or encoded_codes.dtype != torch.uint8 or encoded_codes.ndim != 1:
        raise ValueError("O50 checkpoint has no one-dimensional uint8 player_code_bytes buffer")
    player_codes = encoded_codes.detach().cpu().contiguous().numpy().tobytes()
    codes = decode_player_codes(player_codes)
    if encode_player_codes(codes) != player_codes:
        raise ValueError("O50 checkpoint player vocabulary is not canonically encoded")

    resolved_stats = _resolve_export_stats(checkpoint_config)
    stats_json, stats_sha256 = _encode_stats(resolved_stats)
    config = O50Config.from_checkpoint(
        checkpoint_config,
        player_code_bytes=len(player_codes),
        stats_sha256=stats_sha256,
    )
    if config.player_vocab_sha256 != _sha256_bytes(player_codes):
        raise ValueError("O50 checkpoint player vocabulary SHA-256 does not match its bytes")
    if config.player_vocab_size != FIRST_CONNECT_CODE_ID + len(codes):
        raise ValueError("O50 checkpoint player vocabulary size does not match its codes")

    model = O50Model(config)
    tensor_state = cast(Mapping[str, Tensor], model_state)
    model.load_state_dict(tensor_state, strict=True)
    model.eval()
    with tempfile.TemporaryDirectory(prefix="hal-o50-export-") as temporary:
        root = Path(temporary)
        config_path = root / _CONFIG_MEMBER
        weights_path = root / config.weights_member
        stats_path = root / config.stats_member
        players_path = root / config.player_codes_member
        config_path.write_bytes(config.to_json())
        stats_path.write_bytes(stats_json)
        players_path.write_bytes(player_codes)
        save_model(
            model,
            str(weights_path),
            metadata={"backend": O50_BACKEND, "backend_version": str(O50_BACKEND_VERSION)},
        )
        validation_model = O50Model(config)
        load_model(validation_model, weights_path, strict=True, device="cpu")
        description = BundleDescription(
            policy_name=f"O50 {checkpoint.stem}",
            backend=O50_BACKEND,
            backend_version=O50_BACKEND_VERSION,
            required_observation_fields=O50_REQUIRED_OBSERVATION_FIELDS,
            supported_transport_delays=_SUPPORTED_DELAYS,
            requires_player_code=True,
            source_sha256=checkpoint_sha256(checkpoint),
            backend_config=_CONFIG_MEMBER,
        )
        return write_policy_bundle(
            destination,
            description,
            {
                _CONFIG_MEMBER: config_path,
                config.player_codes_member: players_path,
                config.stats_member: stats_path,
                config.weights_member: weights_path,
            },
        )


def _validate_manifest(manifest: PolicyBundleManifest) -> None:
    expected = (
        manifest.backend == O50_BACKEND,
        manifest.backend_version == O50_BACKEND_VERSION,
        manifest.action_schema == "hal.controller.v1",
        manifest.observation_schema == "hal.flat.numeric.v1",
        manifest.required_observation_fields == O50_REQUIRED_OBSERVATION_FIELDS,
        manifest.supported_transport_delays == _SUPPORTED_DELAYS,
        manifest.requires_player_code,
        manifest.backend_config == _CONFIG_MEMBER,
    )
    if not all(expected):
        raise ValueError("policy bundle manifest does not match the O50 runtime contract")


def _action_vector(action: ControllerAction) -> np.ndarray:
    unknown = action.buttons & ~POLICY_BUTTON_MASK
    if unknown:
        raise ValueError(f"O50 controller action has unsupported button bits 0x{unknown:04x}")
    buttons = [float(bool(action.buttons & BUTTON_BITS[name.removeprefix("button_")])) for name in ACTION_CHANNELS[6:]]
    return np.asarray(
        (action.main_x, action.main_y, action.c_x, action.c_y, action.trigger_l, action.trigger_r, *buttons),
        dtype=np.float32,
    )


def _controller_action(values: np.ndarray) -> ControllerAction:
    if values.shape != (len(ACTION_CHANNELS),):
        raise ValueError(f"O50 decoded action has shape {values.shape}")
    buttons = 0
    for value, name in zip(values[6:], ACTION_CHANNELS[6:], strict=True):
        if value > 0.5:
            buttons |= BUTTON_BITS[name.removeprefix("button_")]
    return ControllerAction(
        main_x=float(np.clip(values[0], -1.0, 1.0)),
        main_y=float(np.clip(values[1], -1.0, 1.0)),
        c_x=float(np.clip(values[2], -1.0, 1.0)),
        c_y=float(np.clip(values[3], -1.0, 1.0)),
        trigger_l=float(np.clip(values[4], 0.0, 1.0)),
        trigger_r=float(np.clip(values[5], 0.0, 1.0)),
        buttons=buttons,
    )


def _relative_name(name: str, controlled_port: int) -> str:
    if name.startswith("p1_"):
        return f"{'ego' if controlled_port == 1 else 'opp'}_{name[3:]}"
    if name.startswith("p2_"):
        return f"{'ego' if controlled_port == 2 else 'opp'}_{name[3:]}"
    return name


def _relative_observation(item: PolicyInput) -> dict[str, ObservationScalar]:
    output: dict[str, ObservationScalar] = {}
    for name in O50_REQUIRED_OBSERVATION_FIELDS:
        value = item.observation[name]
        if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, (bool, np.bool_)):
            raise ValueError(f"O50 observation field {name!r} must be numeric")
        number = float(value)
        if math.isinf(number):
            raise ValueError(f"O50 observation field {name!r} must not be infinite")
        output[_relative_name(name, item.controlled_port)] = (
            int(value) if isinstance(value, (int, np.integer)) else number
        )
    action = _action_vector(item.applied_action)
    output.update({f"ego_{name}": float(value) for name, value in zip(ACTION_CHANNELS, action, strict=True)})
    if set(output) != BASE_ITEMS_PROJECTION.columns:
        raise RuntimeError("O50 flat observation adapter does not match its feature projection")
    return output


@dataclass(slots=True)
class _StreamState:
    history: deque[dict[str, ObservationScalar]] = field(default_factory=deque)
    queued: deque[ControllerAction] = field(default_factory=deque)
    last_frame_id: int | None = None
    controlled_port: int | None = None
    player_id: int | None = None
    reset_pending: bool = True

    def reset(self) -> None:
        self.history.clear()
        self.queued.clear()
        self.last_frame_id = None
        self.controlled_port = None
        self.player_id = None
        self.reset_pending = True


TrunkCall = Callable[[dict[str, Tensor], Tensor, Tensor], Tensor]
DecoderCall = Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]


class O50Policy:
    """Stateful O50 adapter over flat canonical observations."""

    def __init__(
        self,
        model: O50Model,
        config: O50Config,
        stats: Mapping[str, FeatureStats],
        codes: tuple[str, ...],
        *,
        name: str,
        device: torch.device,
        seed: int | None,
        compiled: bool,
    ) -> None:
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise ValueError("O50 sampling seed must be an integer or None")
        if not isinstance(compiled, bool):
            raise ValueError("O50 compiled must be a boolean")
        self._model = model
        self._config = config
        self._stats = dict(stats)
        self._player_vocabulary = PlayerVocabulary(codes)
        self._code_to_id = {code: FIRST_CONNECT_CODE_ID + index for index, code in enumerate(codes)}
        self._device = device
        self._seed = secrets.randbits(64) if seed is None else seed
        self._compiled_requested = compiled
        self._compiled = False
        self._runtime: RuntimeConfig | None = None
        self._replan_interval = 0
        self._states: dict[int, _StreamState] = {}
        self._rng: SlotGroupRng | None = None
        self._trunk: TrunkCall | None = None
        self._decoder: DecoderCall | None = None
        self._spec = PolicySpec(
            name=name,
            backend=O50_BACKEND,
            required_observation_fields=O50_REQUIRED_OBSERVATION_FIELDS,
            supported_transport_delays=_SUPPORTED_DELAYS,
            requires_player_identity=True,
        )

    @property
    def spec(self) -> PolicySpec:
        return self._spec

    def prepare(self, config: RuntimeConfig) -> None:
        if self._runtime is not None:
            raise RuntimeError("O50 policy is already prepared")
        delay = config.transport_delay_frames
        if delay not in _SUPPORTED_DELAYS:
            raise ValueError(f"O50 does not support transport delay {delay}")
        if delay in (2, 3):
            expected = self._config.prediction_frames - delay
            replan = expected if config.replan_interval_frames is None else config.replan_interval_frames
            if replan != expected:
                raise ValueError(f"O50 delay {delay} requires replan interval {expected}, got {replan}")
        else:
            replan = 1 if config.replan_interval_frames is None else config.replan_interval_frames
            if replan not in (1, 2):
                raise ValueError("O50 delay 0 supports replan interval 1 or 2")
        self._runtime = config
        self._replan_interval = replan
        self._states.clear()
        self._rng = SlotGroupRng(self._seed, CONTROLLER_GROUP_NAMES)
        self._compiled = self._compiled_requested and self._device.type == "cuda"
        trunk: TrunkCall = self._model.forward_dense

        def decode_tail(hidden: Tensor, observed: Tensor, forced: Tensor, uniforms: Tensor) -> Tensor:
            return self._model.temporal.sample_conditioned(hidden, observed, forced, uniforms)

        decoder: DecoderCall = decode_tail

        if self._compiled:
            trunk = cast(
                TrunkCall,
                torch.compile(trunk, dynamic=False, fullgraph=True, mode="reduce-overhead"),
            )
            decoder = cast(DecoderCall, torch.compile(decoder, dynamic=False, mode="reduce-overhead"))
        self._trunk = trunk
        self._decoder = decoder
        try:
            self._prewarm()
        except BaseException:
            self._runtime = None
            self._trunk = None
            self._decoder = None
            self._rng = None
            raise

    def _synthetic_features(self, rows: int) -> dict[str, Tensor]:
        length = self._config.architecture.L_ctx
        features: dict[str, Tensor] = {}
        for prefix in BASE_PLAYER_PREFIXES:
            for name in FLOAT_FEATURES:
                features[f"{prefix}_{name}"] = torch.zeros(rows, length, device=self._device)
                features[f"{prefix}_{name}_mask"] = torch.zeros(rows, length, device=self._device)
            for name in CAT_FEATURES:
                features[f"{prefix}_{name}"] = torch.zeros(rows, length, dtype=torch.long, device=self._device)
        for name in ACTION_CHANNELS:
            features[f"ego_{name}"] = torch.zeros(rows, length, device=self._device)
        features["ego_character"] = torch.zeros(rows, length, dtype=torch.long, device=self._device)
        features["opp_character"] = torch.zeros(rows, length, dtype=torch.long, device=self._device)
        features["stage"] = torch.zeros(rows, length, dtype=torch.long, device=self._device)
        features["ego_player_id"] = torch.zeros(rows, length, dtype=torch.long, device=self._device)
        for slot in range(ITEM_SLOTS):
            for name in ITEM_COLUMNS.cats:
                features[item_column(slot, name)] = torch.zeros(rows, length, dtype=torch.long, device=self._device)
            for name in ITEM_FLOATS:
                column = item_column(slot, name)
                features[column] = torch.zeros(rows, length, device=self._device)
                features[f"{column}_mask"] = torch.zeros(rows, length, device=self._device)
        return {name: features[name] for name in sorted(features)}

    @torch.inference_mode()
    def _prewarm(self) -> None:
        runtime = self._require_runtime()
        rows = runtime.max_batch_size
        delay = runtime.transport_delay_frames
        features = self._synthetic_features(rows)
        padding = torch.zeros(rows, dtype=torch.long, device=self._device)
        neutral = torch.zeros(rows, delay, len(ACTION_CHANNELS), device=self._device)
        forced = self._model.codec.quantize(neutral)
        uniforms = torch.full(
            (self._config.prediction_frames - delay, CONTROLLER_GROUP_COUNT, rows),
            0.5,
            device=self._device,
        )
        repeats = 2 if self._compiled else 1
        for _ in range(repeats):
            self._decode(features, padding, forced, uniforms)
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

    def _require_runtime(self) -> RuntimeConfig:
        if self._runtime is None:
            raise RuntimeError("O50 policy must be prepared before step")
        return self._runtime

    @torch.inference_mode()
    def _decode(
        self,
        features: dict[str, Tensor],
        context_padding: Tensor,
        forced: Tensor,
        uniforms: Tensor,
    ) -> Tensor:
        if self._trunk is None or self._decoder is None:
            raise RuntimeError("O50 inference functions are not prepared")
        if self._compiled:
            torch.compiler.cudagraph_mark_step_begin()
        observed = self._model.codec.quantize(stack_actions(features))
        with amp_context(self._config, self._device):
            hidden = self._trunk(features, context_padding, observed)
            indices = self._decoder(hidden, observed[:, -1], forced, uniforms)
        return self._model.codec.dequantize(indices)

    def _player_id(self, identity: str | None) -> int:
        if identity is None:
            raise ValueError("O50 requires a player identity")
        try:
            rank = Rank[identity]
        except KeyError:
            pass
        else:
            return self._player_vocabulary.id_for_rank(rank)
        try:
            return self._code_to_id[identity]
        except KeyError as error:
            raise KeyError(f"player identity {identity!r} is absent from the O50 training vocabulary") from error

    def _ingest(self, item: PolicyInput) -> _StreamState:
        if not isinstance(item.frame_id, int) or isinstance(item.frame_id, bool):
            raise ValueError(f"stream {item.stream_id} frame_id must be an integer")
        player_id = self._player_id(item.player_identity)
        is_new = item.stream_id not in self._states
        if is_new and not item.reset:
            raise ValueError(f"new O50 stream {item.stream_id} must start with reset=True")
        state = self._states.setdefault(item.stream_id, _StreamState())
        discontinuity = state.last_frame_id is not None and item.frame_id != state.last_frame_id + 1
        if item.reset or discontinuity:
            state.reset()
        if state.controlled_port is not None and state.controlled_port != item.controlled_port:
            raise ValueError(f"stream {item.stream_id} changed controlled port without a reset")
        if state.player_id is not None and state.player_id != player_id:
            raise ValueError(f"stream {item.stream_id} changed player identity without a reset")
        state.controlled_port = item.controlled_port
        state.player_id = player_id
        state.last_frame_id = item.frame_id
        state.history.append(_relative_observation(item))
        while len(state.history) > self._config.architecture.L_ctx:
            state.history.popleft()
        return state

    def _context(self, due: Sequence[tuple[PolicyInput, _StreamState]]) -> Context:
        length = self._config.architecture.L_ctx
        rows = len(due)
        columns: dict[str, np.ndarray] = {}
        for name in sorted(BASE_ITEMS_PROJECTION.columns):
            first = due[0][1].history[0][name]
            integer = isinstance(first, int)
            for _item, state in due:
                if any(isinstance(frame[name], int) != integer for frame in state.history):
                    raise ValueError(f"O50 observation field {name!r} changed numeric type within a batch")
            columns[name] = np.zeros((rows, length), dtype=np.int32 if integer else np.float32)
        pads = np.empty(rows, dtype=np.int64)
        player_ids = np.empty(rows, dtype=np.int64)
        for row, (_item, state) in enumerate(due):
            history = tuple(state.history)
            pad = length - len(history)
            pads[row] = pad
            if state.player_id is None:
                raise RuntimeError("O50 stream has no resolved player ID")
            player_ids[row] = state.player_id
            for name in columns:
                columns[name][row, pad:] = [frame[name] for frame in history]
        features = preprocess(columns, self._stats, extra=ITEM_COLUMNS, projection=BASE_ITEMS_PROJECTION)
        for prefix in BASE_PLAYER_PREFIXES:
            for name in FLOAT_FEATURES:
                key = f"{prefix}_{name}"
                features.setdefault(f"{key}_mask", torch.zeros_like(features[key]))
        for slot in range(ITEM_SLOTS):
            for name in ITEM_FLOATS:
                key = item_column(slot, name)
                features.setdefault(f"{key}_mask", torch.zeros_like(features[key]))
        features["ego_player_id"] = torch.from_numpy(np.repeat(player_ids[:, None], length, axis=1))
        context = Context(
            features={name: features[name] for name in sorted(features)},
            ctx_pad=torch.from_numpy(pads),
            slot_ids=torch.tensor([item.stream_id for item, _state in due], dtype=torch.long),
            reset=torch.tensor([state.reset_pending for _item, state in due], dtype=torch.bool),
        )
        return context.to(self._device)

    def _pad_context(self, context: Context, rows: int) -> tuple[dict[str, Tensor], Tensor]:
        real_rows = context.ctx_pad.shape[0]
        if real_rows > rows:
            raise ValueError(f"O50 batch {real_rows} exceeds prepared batch {rows}")
        missing = rows - real_rows
        if not missing:
            return context.features, context.ctx_pad
        features = {
            name: torch.cat((value, torch.zeros((missing, *value.shape[1:]), dtype=value.dtype, device=value.device)))
            for name, value in context.features.items()
        }
        padding = torch.cat(
            (
                context.ctx_pad,
                torch.full(
                    (missing,),
                    self._config.architecture.L_ctx,
                    dtype=torch.long,
                    device=self._device,
                ),
            )
        )
        return features, padding

    @torch.inference_mode()
    def _plan(self, due: Sequence[tuple[PolicyInput, _StreamState]]) -> None:
        runtime = self._require_runtime()
        context = self._context(due)
        real_rows = len(due)
        features, padding = self._pad_context(context, runtime.max_batch_size)
        forced_actions = (
            np.stack([np.stack([_action_vector(action) for action in item.pending_actions]) for item, _state in due])
            if runtime.transport_delay_frames
            else np.empty((real_rows, 0, len(ACTION_CHANNELS)), dtype=np.float32)
        )
        forced = self._model.codec.quantize(torch.from_numpy(forced_actions).to(self._device))
        if real_rows < runtime.max_batch_size:
            forced = torch.cat(
                (
                    forced,
                    torch.zeros(
                        runtime.max_batch_size - real_rows,
                        runtime.transport_delay_frames,
                        CONTROLLER_GROUP_COUNT,
                        dtype=torch.long,
                        device=self._device,
                    ),
                )
            )
        if self._rng is None:
            raise RuntimeError("O50 random streams are not prepared")
        self._rng.begin(context)
        draws = [
            torch.stack([self._rng.uniforms(name) for name in CONTROLLER_GROUP_NAMES])
            for _ in range(self._config.prediction_frames - runtime.transport_delay_frames)
        ]
        uniforms = torch.stack(draws)
        uniforms = F.pad(uniforms, (0, runtime.max_batch_size - real_rows), value=0.5)
        planned = self._decode(features, padding, forced, uniforms)[:real_rows, : self._replan_interval]
        values = planned.float().cpu().numpy()
        for row, (_item, state) in enumerate(due):
            state.queued.extend(_controller_action(action) for action in values[row])
            state.reset_pending = False

    def step(self, inputs: Sequence[PolicyInput]) -> Sequence[PolicyOutput]:
        runtime = self._require_runtime()
        validate_policy_inputs(self._spec, runtime, inputs)
        states = [(item, self._ingest(item)) for item in inputs]
        due = [(item, state) for item, state in states if not state.queued]
        if due:
            self._plan(due)
        outputs = []
        for item, state in states:
            if not state.queued:
                raise RuntimeError(f"O50 stream {item.stream_id} has no planned action")
            outputs.append(PolicyOutput(stream_id=item.stream_id, action=state.queued.popleft()))
        return outputs


def load_o50_policy(
    bundle_path: str | Path,
    *,
    device: str,
    seed: int | None,
    compiled: bool = True,
) -> Policy:
    """Load and validate a portable O50 bundle without importing an experiment."""
    target = torch.device(device)
    with extract_policy_bundle(bundle_path) as (manifest, root):
        _validate_manifest(manifest)
        config = O50Config.from_json((root / manifest.backend_config).read_bytes())
        member_names = {member.name for member in manifest.members}
        expected_members = {
            manifest.backend_config,
            config.weights_member,
            config.stats_member,
            config.player_codes_member,
        }
        if member_names != expected_members:
            raise ValueError("O50 bundle members do not match its backend config")
        stats = _decode_stats((root / config.stats_member).read_bytes(), config)
        player_codes = (root / config.player_codes_member).read_bytes()
        if len(player_codes) != config.player_code_bytes or _sha256_bytes(player_codes) != config.player_vocab_sha256:
            raise ValueError("O50 player vocabulary bytes do not match the backend config")
        codes = decode_player_codes(player_codes)
        if encode_player_codes(codes) != player_codes:
            raise ValueError("O50 player vocabulary is not canonically encoded")
        if config.player_vocab_size != FIRST_CONNECT_CODE_ID + len(codes):
            raise ValueError("O50 player vocabulary size does not match its codes")
        model = O50Model(config).to(target)
        load_model(model, root / config.weights_member, strict=True, device=str(target))
        embedded_codes = model.player_code_bytes.detach().cpu().contiguous().numpy().tobytes()
        if embedded_codes != player_codes:
            raise ValueError("O50 weights and player vocabulary member disagree")
        model.eval()
    return O50Policy(
        model,
        config,
        stats,
        codes,
        name=manifest.policy_name,
        device=target,
        seed=seed,
        compiled=compiled,
    )
