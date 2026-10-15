"""Evaluate bounded O50 checkpoint interventions against the CPU prior.

Planning is the default. Pass ``--execute`` only inside an environment that
already contains the verified checkpoints, HAL fixtures, and a CUDA device.
The runner never uploads checkpoints, replays, or results.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import sys
import tempfile
import time
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from typing import Literal
from typing import Protocol
from typing import cast

import numpy as np
import torch
from torch import Tensor

from hal import fixtures
from hal import streams
from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.data import feature_stats
from hal.eval import cross_stage
from hal.eval import harness
from hal.eval import policy as eval_policy
from hal.inference import api as inference_api
from hal.inference import bundle as inference_bundle
from hal.inference import o50
from hal.inference import o50_model
from hal.inference import transport as inference_transport
from hal.sim import process_vec
from hal.training import ego_stats
from hal.training import player_identity
from research.mechinterp_20260911 import checkpoint_interventions

_EXPECTED_RUN_ID: Final[str] = "p1fyyp1z"
_EXPECTED_EXPERIMENT_ID: Final[str] = "050_scaled_temporal_awr_v4"
_EXPECTED_DATA_PROTOCOL: Final[str] = "o50-replay-ring-v8"
_EARLY_STEP: Final[int] = 24_576
_SLUMP_STEP: Final[int] = 57_344
_EXPECTED_SOURCE_SELECTION_SHA256: Final[str] = "2593361352b92e705be3fbeae1b4e9bb1a3c9f1787cd713014a7a95b7df62477"
_ADAPTER_IDENTITY: Final[str] = "PLATINUM"
_BASE_GIT_SHA: Final[str] = "52da5b2de1d402b86404fc4d24e382409649e525"


@dataclass(frozen=True, slots=True)
class GameplayProtocol:
    """A complete gameplay protocol; different labels are not pooled."""

    label: str
    boots: int
    max_frames: int
    max_parallel: int = 32
    cpu_level: int = 9
    seed: int = 0
    transport_delay: int = 2
    replan_interval: int = 2
    context_length: int = 256
    start_retries: int = 0


@dataclass(frozen=True, slots=True)
class CudaExecution:
    """An explicit inference implementation; it is not a gameplay treatment."""

    target: Literal["local_cuda", "modal_h100"]
    compiled: bool

    def __post_init__(self) -> None:
        if self.target not in ("local_cuda", "modal_h100"):
            raise ValueError(f"unknown CUDA target {self.target!r}")
        if not isinstance(self.compiled, bool):
            raise ValueError("compiled must be a boolean")


PROTOCOLS: Final[dict[str, GameplayProtocol]] = {
    "historical96": GameplayProtocol(
        label="o50-historical-prior-96boot-f7200-v1",
        boots=96,
        max_frames=7_200,
    ),
    "pilot32": GameplayProtocol(
        label="o50-intervention-pilot-32boot-f7200-v1",
        boots=32,
        max_frames=7_200,
    ),
    "short32": GameplayProtocol(
        label="o50-intervention-short-32boot-f1800-v1",
        boots=32,
        max_frames=1_800,
    ),
}


@dataclass(frozen=True, slots=True)
class CheckpointExport:
    """A validated model-only export and its immutable identity."""

    path: Path
    boundary_step: int
    export_sha256: str
    config: Mapping[str, object]
    model: Mapping[str, Tensor]
    source_checkpoint_sha256: str
    source_r2_etag: str
    source_manifest_sha256: Mapping[str, str]
    source_selection_sha256: str
    source_uri: str
    wandb_id: str


@dataclass(frozen=True, slots=True)
class RuntimeAssets:
    """Public O50 runtime inputs resolved from the checkpoint and hosted stats."""

    config: o50_model.O50Config
    stats: Mapping[str, feature_stats.FeatureStats]
    player_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TreatmentSpec:
    """One weight or deployment-semantic treatment against the slump control."""

    name: str
    decoder_semantics: Literal["historical_latent_prefix", "portable_committed_prefix"]
    weight_intervention: checkpoint_interventions.InterventionSpec | None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _string_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object with string keys")
    return cast(Mapping[str, object], value)


def _tensor_mapping(value: object, label: str) -> Mapping[str, Tensor]:
    raw = _string_mapping(value, label)
    if not raw or any(not isinstance(item, Tensor) for item in raw.values()):
        raise ValueError(f"{label} must contain only tensors")
    return cast(Mapping[str, Tensor], raw)


def load_checkpoint_export(path: Path, *, expected_boundary_step: int) -> CheckpointExport:
    """Load one safe model export and verify its adjacent metadata contract."""
    metadata_path = path.with_suffix(".metadata.json")
    metadata = _string_mapping(json.loads(metadata_path.read_bytes()), "checkpoint metadata")
    expected_bytes = metadata.get("bytes")
    expected_sha256 = metadata.get("output_sha256")
    actual_sha256 = _sha256_file(path)
    if path.stat().st_size != expected_bytes or actual_sha256 != expected_sha256:
        raise ValueError(f"checkpoint export failed size/SHA-256 validation: {path}")
    raw = _string_mapping(torch.load(path, map_location="cpu", weights_only=True), "checkpoint export")
    checkpoint_step = raw.get("checkpoint_step")
    if checkpoint_step != expected_boundary_step - 1:
        raise ValueError(f"checkpoint boundary {expected_boundary_step} stores step {checkpoint_step!r}")
    if metadata.get("run_id") != _EXPECTED_RUN_ID or metadata.get("boundary_step") != expected_boundary_step:
        raise ValueError(f"checkpoint is not from W&B run {_EXPECTED_RUN_ID}")
    config = _string_mapping(raw.get("cfg"), "checkpoint config")
    if config.get("experiment_id") != _EXPECTED_EXPERIMENT_ID or config.get("mds_schema_version") != 7:
        raise ValueError("checkpoint experiment or MDS schema identity differs")
    if raw.get("data_protocol") != _EXPECTED_DATA_PROTOCOL:
        raise ValueError("checkpoint data protocol differs from O50 production")
    manifests_raw = _string_mapping(raw.get("source_manifest_sha256"), "source manifests")
    if any(not isinstance(value, str) for value in manifests_raw.values()):
        raise ValueError("source-manifest hashes must be strings")
    manifests = cast(Mapping[str, str], manifests_raw)
    source_selection = raw.get("source_selection_sha256")
    if not isinstance(source_selection, str) or source_selection != _EXPECTED_SOURCE_SELECTION_SHA256:
        raise ValueError("checkpoint source-selection identity differs from O50 production")
    source_sha256 = raw.get("source_checkpoint_sha256")
    source_etag = raw.get("source_r2_etag")
    if not isinstance(source_sha256, str) or not isinstance(source_etag, str):
        raise ValueError("checkpoint export has no source checkpoint SHA-256 or R2 ETag")
    source_uri = raw.get("source_uri")
    wandb_id = raw.get("wandb_id")
    if not isinstance(source_uri, str) or not isinstance(wandb_id, str) or wandb_id != _EXPECTED_RUN_ID:
        raise ValueError("checkpoint export has no exact source URI or W&B identity")
    if (
        metadata.get("source_checkpoint_sha256") != source_sha256
        or metadata.get("source_r2_etag") != source_etag
        or _canonical_json(metadata.get("raw_cfg")) != _canonical_json(config)
    ):
        raise ValueError("checkpoint export disagrees with its source metadata")
    return CheckpointExport(
        path=path,
        boundary_step=expected_boundary_step,
        export_sha256=actual_sha256,
        config=config,
        model=_tensor_mapping(raw.get("model"), "model state"),
        source_checkpoint_sha256=source_sha256,
        source_r2_etag=source_etag,
        source_manifest_sha256=manifests,
        source_selection_sha256=source_selection,
        source_uri=source_uri,
        wandb_id=wandb_id,
    )


def validate_checkpoint_pair(early: CheckpointExport, slump: CheckpointExport) -> None:
    """Reject a pair that differs in anything except learned tensors and step."""
    if early.boundary_step != _EARLY_STEP or slump.boundary_step != _SLUMP_STEP:
        raise ValueError("gameplay intervention requires Muon boundaries 24,576 and 57,344")
    if early.config != slump.config:
        raise ValueError("early and slump checkpoint configs differ")
    if early.source_manifest_sha256 != slump.source_manifest_sha256:
        raise ValueError("early and slump source manifests differ")
    validate_checkpoint_source_identity(early)
    validate_checkpoint_source_identity(slump)


def validate_checkpoint_source_identity(checkpoint: CheckpointExport) -> None:
    """Reject a checkpoint whose ordered v8 sources or manifests differ."""
    expected_names = tuple(source.name for source in streams.POLICY_WORLD_V8_SOURCES)
    source_names = checkpoint.config.get("source_names")
    if not isinstance(source_names, (list, tuple)) or tuple(source_names) != expected_names:
        raise ValueError("checkpoint source order differs from the public policy-world-v8 registry")
    if dict(checkpoint.source_manifest_sha256) != streams.POLICY_WORLD_V8_TRAIN_MANIFEST_SHA256:
        raise ValueError("checkpoint source manifests differ from the public policy-world-v8 registry")


def masked_identity_features(features: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Copy a feature mapping and force the historical masked player ID zero."""
    identity = features.get("ego_player_id")
    if identity is None:
        raise KeyError("O50 context is missing ego_player_id")
    output = dict(features)
    output["ego_player_id"] = torch.zeros_like(identity)
    return output


class MaskedIdentityO50Model(o50_model.O50Model):
    """O50 model whose public context-token path preserves historical ID zero."""

    def context_tokens(self, features: dict[str, Tensor], action_indices: Tensor) -> Tensor:
        return super().context_tokens(masked_identity_features(features), action_indices)


class _ConditionedDecoder(Protocol):
    def sample_conditioned(
        self,
        hidden: Tensor,
        observed: Tensor,
        forced: Tensor,
        force_mask: Tensor,
        uniforms: Tensor,
        *,
        argmax: bool = False,
        sample_start: int = 0,
    ) -> Tensor: ...


class HistoricalLatentPrefixDecoder:
    """Adapt the public decoder to return the frozen executable suffix first."""

    def __init__(self, decoder: _ConditionedDecoder, latent_frames: int) -> None:
        self._decoder = decoder
        self._latent_frames = latent_frames

    def sample_conditioned(
        self,
        hidden: Tensor,
        observed: Tensor,
        forced: Tensor,
        force_mask: Tensor,
        uniforms: Tensor,
        *,
        argmax: bool = False,
        sample_start: int = 0,
    ) -> Tensor:
        """Sample all offsets, then rotate the frozen +3/+4 suffix forward."""
        del force_mask, sample_start
        sampled = self._decoder.sample_conditioned(
            hidden,
            observed,
            forced,
            torch.zeros(forced.shape[:2], dtype=torch.bool, device=forced.device),
            uniforms,
            argmax=argmax,
            sample_start=0,
        )
        return sampled.roll(-self._latent_frames, dims=1)


class HistoricalLatentPrefixModel:
    """Public O50 model facade with the frozen latent-prefix decode contract."""

    def __init__(self, model: MaskedIdentityO50Model, latent_frames: int) -> None:
        self._model = model
        self.codec = model.codec
        self.temporal = HistoricalLatentPrefixDecoder(model.temporal, latent_frames)

    def forward_dense(
        self,
        features: dict[str, Tensor],
        context_padding: Tensor,
        action_indices: Tensor,
    ) -> Tensor:
        return self._model.forward_dense(features, context_padding, action_indices)


class HistoricalLatentPrefixPolicy:
    """Compose public O50 inference with the frozen discard-prefix controller."""

    def __init__(
        self,
        model: MaskedIdentityO50Model,
        assets: RuntimeAssets,
        *,
        name: str,
        device: torch.device,
        seed: int,
        compiled: bool,
    ) -> None:
        self._outer_runtime: inference_api.RuntimeConfig | None = None
        facade = HistoricalLatentPrefixModel(model, latent_frames=2)
        self._inner = o50.O50Policy(
            cast(o50_model.O50Model, facade),
            assets.config,
            assets.stats,
            assets.player_codes,
            name=name,
            device=device,
            seed=seed,
            compiled=compiled,
        )
        self._spec = inference_api.PolicySpec(
            name=name,
            backend="hal.o50.historical-latent-prefix.v1",
            required_observation_fields=self._inner.spec.required_observation_fields,
            supported_transport_delays=(2,),
            requires_player_identity=True,
        )

    @property
    def spec(self) -> inference_api.PolicySpec:
        return self._spec

    def prepare(self, config: inference_api.RuntimeConfig) -> None:
        if self._outer_runtime is not None:
            raise RuntimeError("historical latent-prefix policy is already prepared")
        if config.transport_delays != (2,) or config.replan_interval_frames != 2:
            raise ValueError("historical latent-prefix policy requires delay 2 and replan 2")
        self._inner.prepare(
            inference_api.RuntimeConfig(
                max_batch_size=config.max_batch_size,
                transport_delays=(0,),
                replan_interval_frames=2,
            )
        )
        self._outer_runtime = config

    def step(self, inputs: Sequence[inference_api.PolicyInput]) -> Sequence[inference_api.PolicyOutput]:
        if self._outer_runtime is None:
            raise RuntimeError("historical latent-prefix policy must be prepared before step")
        inference_api.validate_policy_inputs(self.spec, self._outer_runtime, inputs)
        inner_inputs = [
            inference_api.PolicyInput(
                stream_id=item.stream_id,
                frame_id=item.frame_id,
                controlled_port=item.controlled_port,
                observation=item.observation,
                applied_action=item.applied_action,
                pending_actions=(),
                player_identity=item.player_identity,
                reset=item.reset,
            )
            for item in inputs
        ]
        return list(self._inner.step(inner_inputs))


def _runtime_assets(checkpoint: CheckpointExport, scratch: Path) -> RuntimeAssets:
    """Resolve exact config and statistics through public HAL APIs."""
    bundle_path = scratch / "identity-reference.halpolicy"
    manifest = o50.export_o50_policy(checkpoint.path, bundle_path)
    with inference_bundle.extract_policy_bundle(bundle_path) as (extracted, root):
        if extracted != manifest:
            raise ValueError("O50 bundle manifest changed during extraction")
        config = o50_model.O50Config.from_json((root / manifest.backend_config).read_bytes())

    source_names = tuple(cast(list[str] | tuple[str, ...], checkpoint.config["source_names"]))
    stats_paths = tuple(
        streams.ensure_stats(source.local_root / "stats.json") for source in streams.POLICY_WORLD_V8_SOURCES
    )
    weights = [float(streams.POLICY_WORLD_V8_TRAIN_REPLAYS[name]) for name in source_names]
    stats = ego_stats.load_consolidated_mixture_stats(
        stats_paths,
        weights,
        expected_mds_schema_version=config.mds_schema_version,
    )
    encoded_codes = checkpoint.model.get("player_code_bytes")
    if not isinstance(encoded_codes, Tensor) or encoded_codes.dtype != torch.uint8:
        raise ValueError("checkpoint has no uint8 player-code buffer")
    codes = player_identity.decode_player_codes(encoded_codes.detach().cpu().contiguous().numpy().tobytes())
    if config.architecture.L_ctx != 256:
        raise ValueError(f"historical O50 context length must be 256, got {config.architecture.L_ctx}")
    return RuntimeAssets(config=config, stats=stats, player_codes=codes)


def _policy_factory(
    state: Mapping[str, Tensor],
    assets: RuntimeAssets,
    protocol: GameplayProtocol,
    *,
    name: str,
    device: torch.device,
    compiled: bool,
    decoder_semantics: Literal["historical_latent_prefix", "portable_committed_prefix"],
) -> Callable[[], eval_policy.PolicyBatchAdapter]:
    model = MaskedIdentityO50Model(assets.config)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    runtime = inference_api.RuntimeConfig(
        max_batch_size=protocol.max_parallel,
        transport_delays=(protocol.transport_delay,),
        replan_interval_frames=protocol.replan_interval,
    )
    policy_index = itertools.count()

    def factory() -> eval_policy.PolicyBatchAdapter:
        seed = protocol.seed + next(policy_index)
        if decoder_semantics == "historical_latent_prefix":
            policy: inference_api.Policy = HistoricalLatentPrefixPolicy(
                model,
                assets,
                name=name,
                device=device,
                seed=seed,
                compiled=compiled,
            )
        else:
            policy = o50.O50Policy(
                model,
                assets.config,
                assets.stats,
                assets.player_codes,
                name=name,
                device=device,
                seed=seed,
                compiled=compiled,
            )
        policy.prepare(runtime)
        return eval_policy.PolicyBatchAdapter(
            policy,
            runtime,
            player_identity=_ADAPTER_IDENTITY,
        )

    return factory


def _schedule(protocol: GameplayProtocol) -> list[dict[str, str]]:
    return [
        {"ego": ego.name, "opponent": opponent.name}
        for ego, opponent in cross_stage.matchups_for_vs_cpu(protocol.boots)
    ]


def protocol_record(protocol: GameplayProtocol) -> dict[str, object]:
    """Return the complete, hash-bound protocol without starting an emulator."""
    schedule = _schedule(protocol)
    encoded = json.dumps(schedule, separators=(",", ":"), sort_keys=True).encode()
    return {
        **asdict(protocol),
        "adapter_identity_placeholder": _ADAPTER_IDENTITY,
        "effective_player_id": 0,
        "instant_match_restart": True,
        "schedule": schedule,
        "schedule_sha256": hashlib.sha256(encoded).hexdigest(),
        "comparison": "character-blocked boot bootstrap; separate sessions are not common RNG",
    }


def _condition_result(
    name: str,
    state: Mapping[str, Tensor],
    assets: RuntimeAssets,
    protocol: GameplayProtocol,
    *,
    device: torch.device,
    execution: CudaExecution,
    replay_dir: Path,
    decoder_semantics: Literal["historical_latent_prefix", "portable_committed_prefix"],
) -> dict[str, object]:
    started = time.perf_counter()
    telemetry = process_vec.ProcessVecTelemetry()
    torch.cuda.reset_peak_memory_stats(device)
    results, rows = cross_stage.sweep_vs_cpu_prior_with_rows(
        _policy_factory(
            state,
            assets,
            protocol,
            name=name,
            device=device,
            compiled=execution.compiled,
            decoder_semantics=decoder_semantics,
        ),
        session_cfg=harness.default_session_cfg(replay_dir, instant_match_restart=True),
        n_matchups=protocol.boots,
        max_parallel=protocol.max_parallel,
        cpu_level=protocol.cpu_level,
        max_frames=protocol.max_frames,
        start_retries=protocol.start_retries,
        process_telemetry=telemetry,
    )
    torch.cuda.synchronize(device)
    metrics = cross_stage.vs_cpu_metrics(results, seed=protocol.seed)
    complete = (
        metrics.get("scheduled_boots") == protocol.boots
        and metrics.get("completed_boots") == protocol.boots
        and metrics.get("crashed") == 0
    )
    record: dict[str, object] = {
        "condition": name,
        "complete": complete,
        "decoder_semantics": decoder_semantics,
        "execution": execution_record(execution, device),
        "metrics": metrics,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device),
        "process_metrics": telemetry.metrics(),
        "replays": artifact_inventory(replay_dir),
        "rows": [row.as_dict() for row in rows],
        "wall_seconds": time.perf_counter() - started,
    }
    return record


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, sort_keys=True) + "\n")
    temporary.replace(path)


def artifact_inventory(root: Path) -> dict[str, object]:
    """Hash every regular file below one condition's replay directory."""
    files = [path for path in sorted(root.rglob("*")) if path.is_file()]
    return {
        "files": [
            {
                "bytes": path.stat().st_size,
                "path": str(path.resolve()),
                "relative_path": str(path.relative_to(root)),
                "sha256": _sha256_file(path),
            }
            for path in files
        ],
        "root": str(root.resolve()),
        "total_bytes": sum(path.stat().st_size for path in files),
    }


def prepare_replay_root(root: Path) -> None:
    """Create an empty replay root and reject results from a previous run."""
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"replay directory must be absent or empty: {root}")
    root.mkdir(parents=True, exist_ok=True)


def guard_result_output(path: Path) -> None:
    """Reject overwriting an existing nonempty result artifact."""
    if path.is_dir():
        raise ValueError(f"result output must be a file path: {path}")
    if path.exists() and path.stat().st_size:
        raise ValueError(f"result output already contains data: {path}")


def gameplay_fixture_record() -> dict[str, object]:
    """Resolve and record the exact hosted ISO and emulator fixture identities."""
    iso_path = fixtures.ensure(fixtures.ISO)
    emulator_root = fixtures.ensure(fixtures.DOLPHIN_EXIAI)
    emulator_path = emulator_root / "squashfs-root" / "AppRun"
    if not emulator_path.is_file():
        raise FileNotFoundError(f"emulator entrypoint is absent: {emulator_path}")
    return {
        "emulator": {
            "archive_bytes": fixtures.DOLPHIN_EXIAI.size_bytes,
            "archive_sha256": fixtures.DOLPHIN_EXIAI.sha256,
            "entrypoint": str(emulator_path.resolve()),
            "root": str(emulator_root.resolve()),
            "url": fixtures.DOLPHIN_EXIAI.url,
        },
        "iso": {
            "bytes": fixtures.ISO.size_bytes,
            "path": str(iso_path.resolve()),
            "r2_key": fixtures.ISO.r2_key,
            "sha256": fixtures.ISO.sha256,
        },
    }


def source_record() -> dict[str, object]:
    """Hash the research runner and imported HAL policy/evaluation boundaries."""
    modules = {
        "checkpoint_interventions": checkpoint_interventions,
        "hal.eval.cross_stage": cross_stage,
        "hal.eval.harness": harness,
        "hal.eval.policy": eval_policy,
        "hal.inference.api": inference_api,
        "hal.inference.o50": o50,
        "hal.inference.o50_model": o50_model,
        "hal.inference.transport": inference_transport,
    }
    files = {"historical_gameplay": Path(__file__).resolve()}
    for name, module in modules.items():
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            raise RuntimeError(f"module {name!r} has no source file")
        files[name] = Path(module_file).resolve()
    return {
        "base_git_sha": _BASE_GIT_SHA,
        "files": {name: {"path": str(path), "sha256": _sha256_file(path)} for name, path in sorted(files.items())},
        "platform": platform.platform(),
        "python": sys.version,
    }


def _rows_by_boot(rows: Sequence[cross_stage.MatchRow]) -> dict[int, list[cross_stage.MatchRow]]:
    grouped: dict[int, list[cross_stage.MatchRow]] = {}
    for row in rows:
        grouped.setdefault(row.boot_index, []).append(row)
    return grouped


def _net_boot_values(rows: Sequence[cross_stage.MatchRow]) -> tuple[float, float, float]:
    active = [row for row in rows if row.active_frames > 0]
    active_minutes = sum(row.active_frames for row in active) / cross_stage.FRAMES_PER_MINUTE
    if active_minutes == 0:
        return 0.0, 0.0, 0.0
    net_stocks = float(sum(row.stocks_taken - row.stocks_lost for row in active))
    net_damage = sum(row.damage_dealt - row.damage_taken for row in active)
    return active_minutes, net_stocks, net_damage


def blocked_net_deltas(
    rows_a: Sequence[cross_stage.MatchRow],
    rows_b: Sequence[cross_stage.MatchRow],
    *,
    expected_boots: int,
    bootstrap_resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, object]:
    """Directly resample joint per-boot net rates for one blocked comparison."""
    if expected_boots < 1 or bootstrap_resamples < 1:
        raise ValueError("expected_boots and bootstrap_resamples must be positive")
    by_boot_a = _rows_by_boot(rows_a)
    by_boot_b = _rows_by_boot(rows_b)
    expected_ids = set(range(expected_boots))
    if set(by_boot_a) != expected_ids or set(by_boot_b) != expected_ids:
        raise ValueError("both conditions must contain every expected boot exactly")
    for boot_index in sorted(expected_ids):
        matchup_a = {(row.ego_character, row.opp_character) for row in by_boot_a[boot_index]}
        matchup_b = {(row.ego_character, row.opp_character) for row in by_boot_b[boot_index]}
        if len(matchup_a) != 1 or matchup_a != matchup_b:
            raise ValueError(f"boot {boot_index} has a mismatched character block")

    values_a = np.array(
        [_net_boot_values(by_boot_a[index]) for index in range(expected_boots)],
        dtype=np.float64,
    )
    values_b = np.array(
        [_net_boot_values(by_boot_b[index]) for index in range(expected_boots)],
        dtype=np.float64,
    )
    rated = (values_a[:, 0] > 0) & (values_b[:, 0] > 0)
    rated_a = values_a[rated]
    rated_b = values_b[rated]
    if not len(rated_a):
        raise ValueError("the comparison has no jointly rated boots")
    indices = np.random.default_rng(seed).integers(
        0,
        len(rated_a),
        size=(bootstrap_resamples, len(rated_a)),
    )

    estimands: dict[str, dict[str, dict[str, float]]] = {
        "equal_boot": {},
        "pooled_ratio": {},
    }
    for name, column in (("net_stock_per_min", 1), ("net_damage_per_min", 2)):
        per_boot_delta = rated_a[:, column] / rated_a[:, 0] - rated_b[:, column] / rated_b[:, 0]
        equal_samples = per_boot_delta[indices].mean(axis=1)
        pooled_point = rated_a[:, column].sum() / rated_a[:, 0].sum() - rated_b[:, column].sum() / rated_b[:, 0].sum()
        pooled_samples = rated_a[indices, column].sum(axis=1) / rated_a[indices, 0].sum(axis=1) - rated_b[
            indices, column
        ].sum(axis=1) / rated_b[indices, 0].sum(axis=1)
        equal_lo, equal_hi = np.percentile(equal_samples, (2.5, 97.5))
        pooled_lo, pooled_hi = np.percentile(pooled_samples, (2.5, 97.5))
        estimands["equal_boot"][name] = {
            "ci_hi": float(equal_hi),
            "ci_lo": float(equal_lo),
            "point": float(per_boot_delta.mean()),
        }
        estimands["pooled_ratio"][name] = {
            "ci_hi": float(pooled_hi),
            "ci_lo": float(pooled_lo),
            "point": float(pooled_point),
        }
    return {
        "active_time": {
            "a_max_boot_minutes": float(values_a[:, 0].max()),
            "a_min_boot_minutes": float(values_a[:, 0].min()),
            "a_total_minutes": float(values_a[:, 0].sum()),
            "b_max_boot_minutes": float(values_b[:, 0].max()),
            "b_min_boot_minutes": float(values_b[:, 0].min()),
            "b_total_minutes": float(values_b[:, 0].sum()),
            "per_boot_minutes_equal": bool(np.array_equal(values_a[:, 0], values_b[:, 0])),
        },
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": seed,
        "coverage": {
            "excluded_not_jointly_active": int((~rated).sum()),
            "matched_scheduled_boots": expected_boots,
            "rated_boots": int(rated.sum()),
            "zero_active_a": int((values_a[:, 0] == 0).sum()),
            "zero_active_b": int((values_b[:, 0] == 0).sum()),
        },
        "estimands": estimands,
    }


def write_gameplay_net_analysis(result_path: Path, output_path: Path) -> None:
    """Write direct net bootstraps beside, without modifying, a raw gameplay result."""
    raw = _string_mapping(json.loads(result_path.read_bytes()), "gameplay result")
    if raw.get("status") != "complete":
        raise ValueError("gameplay result must be complete")
    protocol = _string_mapping(raw.get("protocol"), "gameplay protocol")
    boots = protocol.get("boots")
    if not isinstance(boots, int):
        raise ValueError("gameplay protocol has no integer boot count")
    raw_conditions = _string_mapping(raw.get("conditions"), "gameplay conditions")

    def rows(name: str) -> list[cross_stage.MatchRow]:
        condition = _string_mapping(raw_conditions.get(name), f"condition {name}")
        raw_rows = condition.get("rows")
        if not isinstance(raw_rows, list):
            raise ValueError(f"condition {name!r} has no row list")
        return [cross_stage.MatchRow.from_dict(cast(Mapping[str, int | float], row)) for row in raw_rows]

    early = rows("early_24576")
    slump = rows("slump_57344")
    portable = rows("portable_committed_prefix")
    payload = {
        "comparisons": {
            "portable_minus_slump": blocked_net_deltas(portable, slump, expected_boots=boots),
            "slump_minus_early": blocked_net_deltas(slump, early, expected_boots=boots),
        },
        "protocol_label": protocol.get("label"),
        "raw_result_path": str(result_path.resolve()),
        "raw_result_sha256": _sha256_file(result_path),
        "source_sha256": _sha256_file(Path(__file__)),
    }
    _atomic_json(output_path, payload)


def _factorial_boot_values(rows: Sequence[cross_stage.MatchRow]) -> tuple[float, float, float, float, float]:
    active = [row for row in rows if row.active_frames > 0]
    active_minutes = sum(row.active_frames for row in active) / cross_stage.FRAMES_PER_MINUTE
    if active_minutes == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    return (
        active_minutes,
        float(sum(row.stocks_taken for row in active)),
        float(sum(row.stocks_lost for row in active)),
        sum(row.damage_dealt for row in active),
        sum(row.damage_taken for row in active),
    )


def blocked_factorial_interaction(
    early_historical: Sequence[cross_stage.MatchRow],
    early_portable: Sequence[cross_stage.MatchRow],
    slump_historical: Sequence[cross_stage.MatchRow],
    slump_portable: Sequence[cross_stage.MatchRow],
    *,
    expected_boots: int,
    bootstrap_resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, object]:
    """Estimate the portable-effect change from early to slump by joint boot."""
    if expected_boots < 1 or bootstrap_resamples < 1:
        raise ValueError("expected_boots and bootstrap_resamples must be positive")
    rows_by_cell = {
        "early_historical": _rows_by_boot(early_historical),
        "early_portable": _rows_by_boot(early_portable),
        "slump_historical": _rows_by_boot(slump_historical),
        "slump_portable": _rows_by_boot(slump_portable),
    }
    expected_ids = set(range(expected_boots))
    for name, by_boot in rows_by_cell.items():
        if set(by_boot) != expected_ids:
            raise ValueError(f"condition {name!r} must contain every expected boot")
    for boot_index in sorted(expected_ids):
        matchups = [
            {(row.ego_character, row.opp_character) for row in by_boot[boot_index]}
            for by_boot in rows_by_cell.values()
        ]
        if len(matchups[0]) != 1 or any(matchup != matchups[0] for matchup in matchups[1:]):
            raise ValueError(f"boot {boot_index} has a mismatched character block")

    values = {
        name: np.array(
            [_factorial_boot_values(by_boot[index]) for index in range(expected_boots)],
            dtype=np.float64,
        )
        for name, by_boot in rows_by_cell.items()
    }
    rated = np.logical_and.reduce([cell[:, 0] > 0 for cell in values.values()])
    rated_values = {name: cell[rated] for name, cell in values.items()}
    rated_boots = int(rated.sum())
    if rated_boots == 0:
        raise ValueError("the factorial analysis has no jointly rated boots")
    indices = np.random.default_rng(seed).integers(
        0,
        rated_boots,
        size=(bootstrap_resamples, rated_boots),
    )

    def outcome(cell: np.ndarray, column: int, subtract_column: int | None) -> np.ndarray:
        numerator = cell[..., column]
        if subtract_column is not None:
            numerator = numerator - cell[..., subtract_column]
        return numerator

    metrics = (
        ("stocks_taken_per_min", 1, None),
        ("stocks_lost_per_min", 2, None),
        ("damage_dealt_per_min", 3, None),
        ("damage_taken_per_min", 4, None),
        ("net_stock_per_min", 1, 2),
        ("net_damage_per_min", 3, 4),
    )
    estimands: dict[str, dict[str, dict[str, float]]] = {"equal_boot": {}, "pooled_ratio": {}}
    cell_points: dict[str, dict[str, dict[str, float]]] = {"equal_boot": {}, "pooled_ratio": {}}
    for metric, column, subtract_column in metrics:
        per_boot_rates = {
            name: outcome(cell, column, subtract_column) / cell[:, 0] for name, cell in rated_values.items()
        }
        per_boot_interaction = (
            per_boot_rates["slump_portable"]
            - per_boot_rates["slump_historical"]
            - per_boot_rates["early_portable"]
            + per_boot_rates["early_historical"]
        )
        equal_samples = per_boot_interaction[indices].mean(axis=1)

        pooled_rates = {
            name: float(outcome(cell, column, subtract_column).sum() / cell[:, 0].sum())
            for name, cell in rated_values.items()
        }
        pooled_point = (
            pooled_rates["slump_portable"]
            - pooled_rates["slump_historical"]
            - pooled_rates["early_portable"]
            + pooled_rates["early_historical"]
        )
        pooled_samples_by_cell = {}
        for name, cell in rated_values.items():
            sampled = cell[indices]
            pooled_samples_by_cell[name] = outcome(sampled, column, subtract_column).sum(axis=1) / sampled[..., 0].sum(
                axis=1
            )
        pooled_samples = (
            pooled_samples_by_cell["slump_portable"]
            - pooled_samples_by_cell["slump_historical"]
            - pooled_samples_by_cell["early_portable"]
            + pooled_samples_by_cell["early_historical"]
        )
        equal_lo, equal_hi = np.percentile(equal_samples, (2.5, 97.5))
        pooled_lo, pooled_hi = np.percentile(pooled_samples, (2.5, 97.5))
        estimands["equal_boot"][metric] = {
            "ci_hi": float(equal_hi),
            "ci_lo": float(equal_lo),
            "point": float(per_boot_interaction.mean()),
        }
        estimands["pooled_ratio"][metric] = {
            "ci_hi": float(pooled_hi),
            "ci_lo": float(pooled_lo),
            "point": pooled_point,
        }
        cell_points["equal_boot"][metric] = {name: float(rates.mean()) for name, rates in per_boot_rates.items()}
        cell_points["pooled_ratio"][metric] = pooled_rates

    return {
        "active_time": {
            name: {
                "max_boot_minutes": float(cell[:, 0].max()),
                "min_boot_minutes": float(cell[:, 0].min()),
                "total_minutes": float(cell[:, 0].sum()),
            }
            for name, cell in values.items()
        },
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": seed,
        "cell_points": cell_points,
        "contrast": "(slump_portable - slump_historical) - (early_portable - early_historical)",
        "coverage": {
            "excluded_not_jointly_active": int((~rated).sum()),
            "matched_scheduled_boots": expected_boots,
            "rated_boots": rated_boots,
            "zero_active_by_cell": {name: int((cell[:, 0] == 0).sum()) for name, cell in values.items()},
        },
        "estimands": estimands,
        "interval_status": "nominal_post_hoc",
    }


def write_gameplay_factorial_analysis(
    three_condition_result_path: Path,
    early_portable_result_path: Path,
    output_path: Path,
) -> None:
    """Write the post-hoc checkpoint-by-decoder interaction without changing raw results."""
    raw_three = _string_mapping(json.loads(three_condition_result_path.read_bytes()), "three-condition result")
    raw_early_portable = _string_mapping(
        json.loads(early_portable_result_path.read_bytes()),
        "early-portable result",
    )
    if raw_three.get("status") != "complete" or raw_early_portable.get("status") != "complete":
        raise ValueError("both gameplay results must be complete")
    protocol = _string_mapping(raw_three.get("protocol"), "three-condition protocol")
    portable_protocol = _string_mapping(raw_early_portable.get("protocol"), "early-portable protocol")
    if protocol != portable_protocol:
        raise ValueError("factorial cells use different gameplay protocols")
    boots = protocol.get("boots")
    if not isinstance(boots, int):
        raise ValueError("gameplay protocol has no integer boot count")
    raw_conditions = _string_mapping(raw_three.get("conditions"), "gameplay conditions")

    def rows(raw_condition: object, name: str) -> list[cross_stage.MatchRow]:
        condition = _string_mapping(raw_condition, f"condition {name}")
        raw_rows = condition.get("rows")
        if not isinstance(raw_rows, list):
            raise ValueError(f"condition {name!r} has no row list")
        return [cross_stage.MatchRow.from_dict(cast(Mapping[str, int | float], row)) for row in raw_rows]

    early_portable_condition = _string_mapping(raw_early_portable.get("condition"), "early-portable condition")
    if early_portable_condition.get("condition") != "early_24576_portable_committed_prefix":
        raise ValueError("unexpected early-portable condition identity")
    interaction = blocked_factorial_interaction(
        rows(raw_conditions.get("early_24576"), "early_24576"),
        rows(early_portable_condition, "early_24576_portable_committed_prefix"),
        rows(raw_conditions.get("slump_57344"), "slump_57344"),
        rows(raw_conditions.get("portable_committed_prefix"), "portable_committed_prefix"),
        expected_boots=boots,
    )
    payload = {
        "analysis_label": "post_hoc_checkpoint_by_decoder_factorial",
        "interaction": interaction,
        "protocol_label": protocol.get("label"),
        "raw_results": {
            "early_portable": {
                "path": str(early_portable_result_path.resolve()),
                "sha256": _sha256_file(early_portable_result_path),
            },
            "three_condition": {
                "path": str(three_condition_result_path.resolve()),
                "sha256": _sha256_file(three_condition_result_path),
            },
        },
        "source_sha256": _sha256_file(Path(__file__)),
    }
    _atomic_json(output_path, payload)


def treatment_spec(name: str) -> TreatmentSpec:
    """Resolve one public weight transform or the committed-prefix comparator."""
    if name == "portable_committed_prefix":
        return TreatmentSpec(
            name=name,
            decoder_semantics="portable_committed_prefix",
            weight_intervention=None,
        )
    matches = [spec for spec in checkpoint_interventions.intervention_specs() if spec.name == name]
    if len(matches) != 1:
        raise ValueError(f"unknown intervention {name!r}")
    spec = matches[0]
    if spec.kind.startswith("identity_") or spec.alpha is not None:
        raise ValueError("gameplay does not accept identity or interpolation controls")
    return TreatmentSpec(
        name=spec.name,
        decoder_semantics="historical_latent_prefix",
        weight_intervention=spec,
    )


def execution_record(execution: CudaExecution, device: torch.device) -> dict[str, object]:
    """Validate and record the selected CUDA implementation."""
    if device.type != "cuda":
        raise ValueError("the d1024 gameplay runner requires an explicit CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device_index = torch.cuda.current_device() if device.index is None else device.index
    properties = torch.cuda.get_device_properties(device_index)
    device_name = properties.name
    if execution.target == "modal_h100" and "H100" not in device_name:
        raise ValueError(f"modal_h100 requires an H100, got {device_name!r}")
    return {
        "compiled": execution.compiled,
        "cuda_capability": list(torch.cuda.get_device_capability(device_index)),
        "cuda_runtime": torch.version.cuda,
        "device_index": device_index,
        "device_name": device_name,
        "device_total_memory_bytes": properties.total_memory,
        "target": execution.target,
        "torch_version": torch.__version__,
    }


def _benchmark_inputs(batch_size: int, frame_id: int) -> list[inference_api.PolicyInput]:
    observation = {name: 0 for name in o50.O50_REQUIRED_OBSERVATION_FIELDS}
    pending = (NEUTRAL_CONTROLLER_ACTION, NEUTRAL_CONTROLLER_ACTION)
    return [
        inference_api.PolicyInput(
            stream_id=stream_id,
            frame_id=frame_id,
            controlled_port=1,
            observation=observation,
            applied_action=NEUTRAL_CONTROLLER_ACTION,
            pending_actions=pending,
            player_identity=_ADAPTER_IDENTITY,
            reset=frame_id == 0,
        )
        for stream_id in range(batch_size)
    ]


def benchmark_full_context(
    checkpoint_path: Path,
    output: Path,
    *,
    batch_sizes: Sequence[int],
    device: torch.device,
    execution: CudaExecution,
    measured_plan_calls: int = 10,
) -> None:
    """Measure historical-policy replans after 256 public context steps."""
    if measured_plan_calls < 1:
        raise ValueError("measured_plan_calls must be positive")
    if not batch_sizes or any(size < 1 for size in batch_sizes):
        raise ValueError("batch_sizes must contain positive integers")
    execution_identity = execution_record(execution, device)
    checkpoint = load_checkpoint_export(checkpoint_path, expected_boundary_step=_EARLY_STEP)
    with tempfile.TemporaryDirectory(prefix="hal-gameplay-benchmark-") as temporary:
        assets = _runtime_assets(checkpoint, Path(temporary))
    records: dict[str, object] = {}
    for batch_size in batch_sizes:
        model = MaskedIdentityO50Model(assets.config)
        model.load_state_dict(checkpoint.model, strict=True)
        model.to(device).eval()
        policy = HistoricalLatentPrefixPolicy(
            model,
            assets,
            name=f"benchmark-b{batch_size}",
            device=device,
            seed=0,
            compiled=execution.compiled,
        )
        prepare_started = time.perf_counter()
        policy.prepare(
            inference_api.RuntimeConfig(
                max_batch_size=batch_size,
                transport_delays=(2,),
                replan_interval_frames=2,
            )
        )
        prepare_seconds = time.perf_counter() - prepare_started
        torch.cuda.reset_peak_memory_stats(device)
        prime_started = time.perf_counter()
        for frame_id in range(assets.config.architecture.L_ctx):
            policy.step(_benchmark_inputs(batch_size, frame_id))
        context_prime_seconds = time.perf_counter() - prime_started
        latencies_ms = []
        frame_id = assets.config.architecture.L_ctx
        for _ in range(measured_plan_calls):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            policy.step(_benchmark_inputs(batch_size, frame_id))
            torch.cuda.synchronize(device)
            latencies_ms.append((time.perf_counter() - started) * 1_000)
            policy.step(_benchmark_inputs(batch_size, frame_id + 1))
            frame_id += 2
        records[str(batch_size)] = {
            "batch_size": batch_size,
            "context_prime_seconds": context_prime_seconds,
            "context_length": assets.config.architecture.L_ctx,
            "latency_max_ms": max(latencies_ms),
            "latency_mean_ms": float(np.mean(latencies_ms)),
            "latency_p50_ms": float(np.percentile(latencies_ms, 50)),
            "latency_p95_ms": float(np.percentile(latencies_ms, 95)),
            "measured_plan_calls": measured_plan_calls,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(device),
            "prepare_seconds": prepare_seconds,
            "replan_interval_frames": 2,
        }
        del policy, model
        torch.cuda.empty_cache()
        _atomic_json(
            output,
            {
                "batches": records,
                "checkpoint_export_sha256": checkpoint.export_sha256,
                "checkpoint_source_sha256": checkpoint.source_checkpoint_sha256,
                "execution": execution_identity,
                "status": "incomplete" if len(records) < len(batch_sizes) else "complete",
            },
        )


def run_early_portable_control(
    checkpoint_path: Path,
    output: Path,
    replay_root: Path,
    *,
    device: torch.device,
) -> None:
    """Run only the pre-registered early committed-prefix post-hoc control."""
    guard_result_output(output)
    prepare_replay_root(replay_root)
    execution = CudaExecution(target="local_cuda", compiled=False)
    execution_identity = execution_record(execution, device)
    fixture_identity = gameplay_fixture_record()
    code_identity = source_record()
    protocol = PROTOCOLS["short32"]
    checkpoint = load_checkpoint_export(checkpoint_path, expected_boundary_step=_EARLY_STEP)
    validate_checkpoint_source_identity(checkpoint)
    with tempfile.TemporaryDirectory(prefix="hal-early-portable-control-") as temporary:
        assets = _runtime_assets(checkpoint, Path(temporary))
    condition = _condition_result(
        "early_24576_portable_committed_prefix",
        checkpoint.model,
        assets,
        protocol,
        device=device,
        execution=execution,
        replay_dir=replay_root,
        decoder_semantics="portable_committed_prefix",
    )
    payload = {
        "checkpoint_identity": {
            "boundary_step": checkpoint.boundary_step,
            "config": checkpoint.config,
            "config_sha256": hashlib.sha256(_canonical_json(checkpoint.config).encode()).hexdigest(),
            "export_sha256": checkpoint.export_sha256,
            "source_checkpoint_sha256": checkpoint.source_checkpoint_sha256,
            "source_manifest_sha256": checkpoint.source_manifest_sha256,
            "source_r2_etag": checkpoint.source_r2_etag,
            "source_selection_sha256": checkpoint.source_selection_sha256,
            "source_uri": checkpoint.source_uri,
            "wandb_id": checkpoint.wandb_id,
        },
        "condition": condition,
        "execution": execution_identity,
        "fixtures": fixture_identity,
        "protocol": protocol_record(protocol),
        "replay_root": str(replay_root.resolve()),
        "selection": {
            "label": "post_hoc_factorial_completion",
            "question": "Is the committed-prefix stock benefit slump-specific or general?",
            "selected_after": "slump portable-minus-historical short32 result",
        },
        "source": code_identity,
        "status": "complete" if condition["complete"] is True else "incomplete",
    }
    _atomic_json(output, payload)
    if condition["complete"] is not True:
        metrics = cast(Mapping[str, object], condition["metrics"])
        raise RuntimeError(
            "early portable control failed zero-crash completeness: "
            f"scheduled={metrics.get('scheduled_boots')!r}, "
            f"completed={metrics.get('completed_boots')!r}, "
            f"crashed={metrics.get('crashed')!r}"
        )


def run_gameplay(
    early_path: Path,
    slump_path: Path,
    intervention_name: str,
    protocol: GameplayProtocol,
    output: Path,
    replay_root: Path,
    *,
    device: torch.device,
    execution: CudaExecution,
) -> None:
    """Run early, slump, and one intervention as separate blocked conditions."""
    guard_result_output(output)
    execution_identity = execution_record(execution, device)
    fixture_identity = gameplay_fixture_record()
    code_identity = source_record()
    if protocol.context_length != 256 or protocol.start_retries != 0:
        raise ValueError("historical context length and zero-retry policy are invariant")
    early = load_checkpoint_export(early_path, expected_boundary_step=_EARLY_STEP)
    slump = load_checkpoint_export(slump_path, expected_boundary_step=_SLUMP_STEP)
    validate_checkpoint_pair(early, slump)
    prepare_replay_root(replay_root)
    treatment_specification = treatment_spec(intervention_name)
    with tempfile.TemporaryDirectory(prefix="hal-gameplay-intervention-") as temporary:
        assets = _runtime_assets(early, Path(temporary))
    parameter_model = MaskedIdentityO50Model(assets.config)
    parameter_keys = tuple(name for name, _parameter in parameter_model.named_parameters())
    if treatment_specification.weight_intervention is None:
        treatment = checkpoint_interventions.identity_state(slump.model)
    else:
        treatment = checkpoint_interventions.apply_intervention(
            treatment_specification.weight_intervention,
            early.model,
            slump.model,
            parameter_keys=parameter_keys,
        )
    del parameter_model

    condition_states = (
        ("early_24576", early.model, "historical_latent_prefix"),
        ("slump_57344", slump.model, "historical_latent_prefix"),
        (treatment_specification.name, treatment, treatment_specification.decoder_semantics),
    )
    records: dict[str, dict[str, object]] = {}
    for name, state, decoder_semantics in condition_states:
        condition_replays = replay_root / name
        records[name] = _condition_result(
            name,
            state,
            assets,
            protocol,
            device=device,
            execution=execution,
            replay_dir=condition_replays,
            decoder_semantics=decoder_semantics,
        )
        _atomic_json(
            output,
            {
                "checkpoint_identity": {
                    "config": early.config,
                    "config_sha256": hashlib.sha256(_canonical_json(early.config).encode()).hexdigest(),
                    "early_export_sha256": early.export_sha256,
                    "early_source_r2_etag": early.source_r2_etag,
                    "early_source_sha256": early.source_checkpoint_sha256,
                    "early_source_uri": early.source_uri,
                    "slump_export_sha256": slump.export_sha256,
                    "slump_source_r2_etag": slump.source_r2_etag,
                    "slump_source_sha256": slump.source_checkpoint_sha256,
                    "slump_source_uri": slump.source_uri,
                    "source_manifest_sha256": early.source_manifest_sha256,
                    "source_selection_sha256": early.source_selection_sha256,
                    "wandb_id": early.wandb_id,
                },
                "conditions": records,
                "execution": execution_identity,
                "fixtures": fixture_identity,
                "treatment": asdict(treatment_specification),
                "protocol": protocol_record(protocol),
                "replay_root": str(replay_root.resolve()),
                "source": code_identity,
                "status": "incomplete" if len(records) < len(condition_states) else "complete",
            },
        )
        if records[name]["complete"] is not True:
            metrics = cast(Mapping[str, object], records[name]["metrics"])
            raise RuntimeError(
                f"{name} failed zero-crash completeness: scheduled={metrics.get('scheduled_boots')!r}, "
                f"completed={metrics.get('completed_boots')!r}, "
                f"crashed={metrics.get('crashed')!r}"
            )

    rows: dict[str, list[cross_stage.MatchRow]] = {}
    for name, record in records.items():
        raw_rows = record["rows"]
        if not isinstance(raw_rows, list):
            raise TypeError(f"condition {name!r} rows must be a list")
        rows[name] = [cross_stage.MatchRow.from_dict(cast(Mapping[str, int | float], row)) for row in raw_rows]
    comparisons = {
        "slump_minus_early": cross_stage.matched_vs_cpu_deltas(
            rows["slump_57344"], rows["early_24576"], seed=protocol.seed
        ),
        "treatment_minus_slump": cross_stage.matched_vs_cpu_deltas(
            rows[treatment_specification.name], rows["slump_57344"], seed=protocol.seed
        ),
    }
    payload = json.loads(output.read_bytes())
    payload["blocked_comparisons"] = comparisons
    _atomic_json(output, payload)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--early", type=Path, required=True)
    parser.add_argument("--slump", type=Path, required=True)
    parser.add_argument("--intervention", required=True)
    parser.add_argument("--protocol", choices=sorted(PROTOCOLS), default="short32")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--cuda-target", choices=("local_cuda", "modal_h100"), required=True)
    compilation = parser.add_mutually_exclusive_group(required=True)
    compilation.add_argument("--compiled", action="store_true", dest="compiled")
    compilation.add_argument("--eager", action="store_false", dest="compiled")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-historical96", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    protocol = PROTOCOLS[args.protocol]
    execution = CudaExecution(target=args.cuda_target, compiled=args.compiled)
    plan = {
        "early": str(args.early),
        "execution": asdict(execution),
        "treatment": asdict(treatment_spec(args.intervention)),
        "protocol": protocol_record(protocol),
        "replay_root": str(args.replay_root),
        "slump": str(args.slump),
    }
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    if args.protocol == "historical96" and not args.allow_historical96:
        raise SystemExit("historical96 requires --allow-historical96")
    run_gameplay(
        args.early,
        args.slump,
        args.intervention,
        protocol,
        args.output,
        args.replay_root,
        device=torch.device("cuda"),
        execution=execution,
    )


if __name__ == "__main__":
    main()
