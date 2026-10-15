"""Measure the paired O50 Muon learning-rate control at step 57,344."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import subprocess
import tarfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from hal.inference import o50_model
from hal.training import features
from research.mechinterp_20260911 import attention_detail
from research.mechinterp_20260911 import o50_probe

FIXED_COHORT_SHA256 = attention_detail.FIXED_COHORT_SHA256
SOURCE_SELECTION_SHA256 = "2593361352b92e705be3fbeae1b4e9bb1a3c9f1787cd713014a7a95b7df62477"
EXPECTED_WINDOWS = attention_detail.EXPECTED_WINDOWS
EXPECTED_CONTEXT_LENGTH = attention_detail.EXPECTED_CONTEXT_LENGTH
EXPECTED_LAYERS = attention_detail.EXPECTED_LAYERS
EXPECTED_HEADS = attention_detail.EXPECTED_HEADS
EXPECTED_BATCH_SIZE = 8
EXPECTED_MODEL_DIMENSION = 1024

FloatArray = NDArray[np.float64]
IntegerArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class CheckpointSpec:
    """One strictly identified model-only checkpoint export."""

    run_id: str
    checkpoint_step: int
    experiment_id: str
    model_filename: str
    model_export_sha256: str
    source_checkpoint_sha256: str
    source_selection_sha256: str

    @property
    def boundary_step(self) -> int:
        """Return the human-facing one-based checkpoint boundary."""
        return self.checkpoint_step + 1

    @property
    def label(self) -> str:
        """Return the stable label used by the attention artifacts."""
        return f"{self.run_id} step {self.boundary_step:,}"


CHECKPOINT_SPECS = (
    CheckpointSpec(
        run_id="p1fyyp1z",
        checkpoint_step=57343,
        experiment_id="050_scaled_temporal_awr_v4",
        model_filename="p1fyyp1z-step-0057344.model.pt",
        model_export_sha256="fda4dd85a1f60c06c3c19e507834071e54cb000b483bdac40315482b69215466",
        source_checkpoint_sha256="495b4b57f09c867d6ddef4bb3d5c2a74931a2a43906dfde8e753bde44c62deca",
        source_selection_sha256=SOURCE_SELECTION_SHA256,
    ),
    CheckpointSpec(
        run_id="7dgv85x8",
        checkpoint_step=57343,
        experiment_id="050_scaled_temporal_awr_v5",
        model_filename="7dgv85x8-step-0057344.model.pt",
        model_export_sha256="2ad0c0f8015c75235def1e8d591a7350145e64a389fca5af63b39b53e6516be2",
        source_checkpoint_sha256="89c941d64f6407eed34bc6b9ad3ccc1466aace67d48f0ef45978bcb7575a4b08",
        source_selection_sha256=SOURCE_SELECTION_SHA256,
    ),
)


@dataclass(frozen=True, slots=True)
class CohortPackage:
    """Validated fixed-cohort tensors and provenance metadata."""

    context_features: dict[str, Tensor]
    reference_config: o50_model.O50Config
    stats_payload: Mapping[str, object]
    player_codes: bytes | None
    target: Tensor
    data: dict[str, object]
    package_sha256: str
    data_json_sha256: str


@dataclass(frozen=True, slots=True)
class CheckpointMeasurement:
    """Attention result for one control checkpoint."""

    spec: CheckpointSpec
    provenance: dict[str, object]
    summary: attention_detail.AttentionSummary
    example_weights: NDArray[np.float32]
    example_rows: dict[str, FloatArray | IntegerArray]
    trace_max_abs_error: float
    probability_sum_max_abs_error: float
    metadata_sha256: str


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a mapping with string keys")
    return cast(Mapping[str, object], value)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return _sha256_bytes(encoded)


def _bytes_tensor(value: object, label: str) -> bytes:
    if not isinstance(value, Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
        raise TypeError(f"{label} must be a one-dimensional uint8 tensor")
    return value.detach().cpu().contiguous().numpy().tobytes()


def validated_source_manifests(stats_payload: Mapping[str, object]) -> dict[str, str]:
    if stats_payload.get("schema_version") != 1 or stats_payload.get("mds_schema_version") != 7:
        raise ValueError("fixed cohort statistics do not use the validated O50 schema")
    if stats_payload.get("source_count") != 44:
        raise ValueError("fixed cohort statistics do not contain the full 44-source mixture")
    source_names = stats_payload.get("source_names")
    if (
        not isinstance(source_names, list)
        or len(source_names) != 44
        or any(not isinstance(name, str) or not name.endswith("-policy-world-v8") for name in source_names)
        or len(set(source_names)) != 44
    ):
        raise ValueError("fixed cohort source names do not identify the full v8 mixture")
    source_names = cast(list[str], source_names)
    source_manifests = _mapping(stats_payload.get("source_manifest_sha256"), "source_manifest_sha256")
    if set(source_manifests) != set(source_names):
        raise ValueError("fixed cohort source manifests do not match source names")
    validated: dict[str, str] = {}
    for name in source_names:
        value = source_manifests[name]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"source manifest SHA is invalid for {name}")
        validated[name] = value
    return validated


def load_fixed_cohort(path: Path) -> CohortPackage:
    """Load and validate the exact fixed rank-identity cohort."""
    package_bytes = path.read_bytes()
    package_sha256 = _sha256_bytes(package_bytes)
    if package_sha256 != FIXED_COHORT_SHA256:
        raise ValueError(f"fixed cohort package hash differs: {package_sha256}")
    raw = _mapping(torch.load(path, map_location="cpu", weights_only=True), "cohort package")
    expected_fields = {
        "bundle_reference_json",
        "context_features",
        "data_json",
        "player_codes",
        "reference_config",
        "stats_payload",
        "target",
    }
    if set(raw) != expected_fields:
        raise ValueError(f"cohort fields differ: expected={sorted(expected_fields)}, found={sorted(raw)}")

    context_raw = _mapping(raw["context_features"], "cohort context_features")
    context_features: dict[str, Tensor] = {}
    for name, value in context_raw.items():
        if not isinstance(value, Tensor) or value.ndim != 2:
            raise TypeError(f"cohort context feature {name!r} must be rank 2")
        if tuple(value.shape) != (EXPECTED_WINDOWS, EXPECTED_CONTEXT_LENGTH):
            raise ValueError(f"cohort context feature {name!r} has shape {tuple(value.shape)}")
        context_features[name] = value

    target = raw["target"]
    if not isinstance(target, Tensor) or target.ndim != 3 or target.shape[0] != EXPECTED_WINDOWS:
        raise ValueError("cohort target must have one row per fixed-cohort replay")

    data_json = raw["data_json"]
    if not isinstance(data_json, str):
        raise TypeError("cohort data_json must be a string")
    data_bytes = data_json.encode()
    data_mapping = _mapping(json.loads(data_bytes), "cohort data_json")
    windows_raw = data_mapping.get("windows")
    if not isinstance(windows_raw, list) or len(windows_raw) != EXPECTED_WINDOWS:
        raise ValueError("cohort data_json must retain exactly 256 window records")
    windows: list[dict[str, object]] = []
    for index, raw_window in enumerate(windows_raw):
        window = dict(_mapping(raw_window, f"cohort window {index}"))
        if window.get("identity_source") != "rank":
            raise ValueError(f"cohort window {index} is not the recorded rank identity input")
        player_id = window.get("ego_player_id")
        if not isinstance(player_id, int) or isinstance(player_id, bool):
            raise ValueError(f"cohort window {index} has no integer ego_player_id")
        windows.append(window)

    player_ids = context_features.get("ego_player_id")
    if player_ids is None:
        raise ValueError("cohort context has no ego_player_id feature")
    player_id_rows = player_ids.detach().cpu().numpy()
    for index, window in enumerate(windows):
        if not np.all(player_id_rows[index] == player_id_rows[index, 0]):
            raise ValueError(f"cohort player identity changes within window {index}")
        if int(player_id_rows[index, 0]) != cast(int, window["ego_player_id"]):
            raise ValueError(f"cohort rank-derived player identity differs at window {index}")

    stats_payload = _mapping(
        json.loads(_bytes_tensor(raw["stats_payload"], "stats_payload")),
        "stats_payload",
    )
    validated_source_manifests(stats_payload)
    bundle_reference_json = raw["bundle_reference_json"]
    if not isinstance(bundle_reference_json, str):
        raise TypeError("bundle_reference_json must be a string")
    bundle_reference = _mapping(json.loads(bundle_reference_json), "bundle_reference_json")
    if bundle_reference.get("stats_content_sha256") != stats_payload.get("content_sha256"):
        raise ValueError("cohort bundle reference does not match its statistics payload")
    reference_config = o50_model.O50Config.from_json(_bytes_tensor(raw["reference_config"], "reference_config"))
    if reference_config.architecture.L_ctx != EXPECTED_CONTEXT_LENGTH:
        raise ValueError("cohort reference context length differs from the fixed protocol")
    player_codes_bytes = _bytes_tensor(raw["player_codes"], "player_codes")
    player_codes = player_codes_bytes or None
    data = dict(data_mapping)
    data["windows"] = windows
    return CohortPackage(
        context_features=context_features,
        reference_config=reference_config,
        stats_payload=stats_payload,
        player_codes=player_codes,
        target=target,
        data=data,
        package_sha256=package_sha256,
        data_json_sha256=_sha256_bytes(data_bytes),
    )


def validate_checkpoint_metadata(
    path: Path,
    spec: CheckpointSpec,
    source_manifests: Mapping[str, str],
) -> tuple[dict[str, object], str]:
    """Validate one export sidecar before loading its model tensors."""
    metadata_path = path.with_suffix(".metadata.json")
    metadata = _mapping(json.loads(metadata_path.read_bytes()), str(metadata_path))
    checks = {
        "run_id": spec.run_id,
        "checkpoint_step": spec.checkpoint_step,
        "boundary_step": spec.boundary_step,
        "experiment_id": spec.experiment_id,
        "source_checkpoint_sha256": spec.source_checkpoint_sha256,
        "source_selection_sha256": spec.source_selection_sha256,
        "output_sha256": spec.model_export_sha256,
    }
    for name, expected in checks.items():
        if metadata.get(name) != expected:
            raise ValueError(f"{metadata_path}: {name} differs from the requested checkpoint")
    if metadata.get("bytes") != path.stat().st_size:
        raise ValueError(f"{metadata_path}: model byte count differs from the checkpoint")
    if _mapping(metadata.get("source_manifest_sha256"), "metadata source manifests") != dict(source_manifests):
        raise ValueError(f"{metadata_path}: source manifests differ from the fixed cohort identity")
    raw_config = _mapping(metadata.get("raw_cfg"), "metadata raw_cfg")
    if raw_config.get("experiment_id") != spec.experiment_id:
        raise ValueError(f"{metadata_path}: raw config experiment differs from checkpoint identity")
    if raw_config.get("source_names") != list(source_manifests):
        raise ValueError(f"{metadata_path}: raw config sources differ from checkpoint identity")
    if _sha256_file(path) != spec.model_export_sha256:
        raise ValueError(f"{path}: model export bytes differ from the requested checkpoint")
    return dict(metadata), _sha256_file(metadata_path)


def _validate_control_geometry(config: o50_model.O50Config, label: str) -> None:
    architecture = config.architecture
    expected = {
        "d_model": architecture.d_model == EXPECTED_MODEL_DIMENSION,
        "n_layers": architecture.n_layers == EXPECTED_LAYERS,
        "n_heads": architecture.n_heads == EXPECTED_HEADS,
        "L_ctx": architecture.L_ctx == EXPECTED_CONTEXT_LENGTH,
    }
    if not all(expected.values()):
        raise ValueError(f"{label}: checkpoint geometry is not the predeclared O50 control geometry: {expected}")


def _example_rows(weights: NDArray[np.float32]) -> dict[str, FloatArray | IntegerArray]:
    summary = attention_detail.summarize_attention(weights[None, ...])
    return {
        "entropy_nats": summary.per_head.mean_entropy_nats,
        "effective_context": summary.per_head.mean_exp_entropy,
        "row_max_weight": summary.per_head.mean_row_max_weight,
        "weighted_lag": summary.per_head.mean_weighted_lag,
        "current_frame_weight": summary.per_head.current_frame_weight,
        "argmax_lag": summary.per_head.most_frequent_argmax_lag,
    }


def summary_to_dict(summary: attention_detail.AttentionSummary) -> dict[str, object]:
    """Serialize one attention summary using the existing interactive schema."""
    per_head = summary.per_head
    global_metrics = summary.global_metrics
    return {
        "shape": list(summary.shape),
        "lag_definition": "lag 0 is current key; lag key_count-1 is oldest key",
        "per_head": {
            "mean_entropy_nats": per_head.mean_entropy_nats.tolist(),
            "mean_exp_entropy": per_head.mean_exp_entropy.tolist(),
            "mean_row_max_weight": per_head.mean_row_max_weight.tolist(),
            "mean_weighted_lag": per_head.mean_weighted_lag.tolist(),
            "current_frame_weight": per_head.current_frame_weight.tolist(),
            "most_frequent_argmax_lag": per_head.most_frequent_argmax_lag.tolist(),
            "argmax_lag_frequency": per_head.argmax_lag_frequency.tolist(),
        },
        "global": {
            "mean_entropy_nats": global_metrics.mean_entropy_nats,
            "mean_exp_entropy": global_metrics.mean_exp_entropy,
            "mean_row_max_weight": global_metrics.mean_row_max_weight,
            "mean_weighted_lag": global_metrics.mean_weighted_lag,
            "current_frame_weight": global_metrics.current_frame_weight,
            "most_frequent_argmax_lag": global_metrics.most_frequent_argmax_lag,
            "argmax_lag_frequency": global_metrics.argmax_lag_frequency,
            "peak_weight_quantiles": global_metrics.peak_weight_quantiles,
            "mean_attention_by_lag": global_metrics.mean_attention_by_lag.tolist(),
            "argmax_lag_histogram": global_metrics.argmax_lag_histogram.tolist(),
            "nearest_global_mean_lag": global_metrics.nearest_global_mean_lag,
            "mean_weight_at_nearest_global_mean_lag": global_metrics.mean_weight_at_nearest_global_mean_lag,
        },
    }


def validate_parent_parity(
    summary: attention_detail.AttentionSummary,
    example_weights: NDArray[np.float32],
    example_rows: Mapping[str, FloatArray | IntegerArray],
    detail_path: Path,
    example_path: Path,
) -> None:
    """Require exact equality with the existing p1 step-57,344 measurement."""
    raw = _mapping(json.loads(detail_path.read_bytes()), str(detail_path))
    checkpoints = raw.get("checkpoints")
    if not isinstance(checkpoints, list):
        raise ValueError("existing attention detail has no checkpoint list")
    matching: list[Mapping[str, object]] = []
    for checkpoint_raw in checkpoints:
        if not isinstance(checkpoint_raw, Mapping):
            continue
        checkpoint = _mapping(checkpoint_raw, "existing checkpoint")
        if (
            checkpoint.get("run_id") == CHECKPOINT_SPECS[0].run_id
            and checkpoint.get("checkpoint_step") == CHECKPOINT_SPECS[0].checkpoint_step
        ):
            matching.append(checkpoint)
    if len(matching) != 1:
        raise ValueError("existing attention detail does not contain exactly one parent checkpoint")
    expected_summary = _mapping(matching[0].get("summary"), "existing parent summary")
    actual_summary = summary_to_dict(summary)
    if expected_summary != actual_summary:
        raise ValueError("parent attention summary differs exactly from the existing measurement")

    expected_fields = {
        "weights",
        "entropy_nats",
        "effective_context",
        "row_max_weight",
        "weighted_lag",
        "current_frame_weight",
        "argmax_lag",
    }
    with np.load(example_path, allow_pickle=False) as archive:
        expected_keys = {f"checkpoint_01__{field}" for field in expected_fields}
        if not expected_keys <= set(archive.files):
            raise ValueError("existing attention example does not contain the parent checkpoint entry")
        actual_fields: dict[str, np.ndarray] = {"weights": example_weights}
        actual_fields.update(example_rows)
        for field, actual in actual_fields.items():
            expected = np.asarray(archive[f"checkpoint_01__{field}"])
            if not np.array_equal(expected, actual):
                raise ValueError(f"parent attention example field {field} differs exactly")


def _configure_precision() -> dict[str, object]:
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    precision: dict[str, object] = {
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "qk_score_dtype": "torch.float32",
        "softmax_output_dtype": "torch.float32",
        "score_and_softmax_autocast_enabled": False,
        "upstream_qk_projection_dtype": "torch.bfloat16",
    }
    if (
        precision["cuda_matmul_allow_tf32"]
        or precision["cudnn_allow_tf32"]
        or precision["float32_matmul_precision"] != "highest"
    ):
        raise RuntimeError(f"failed to configure FP32 precision: {precision}")
    return precision


def _measure_checkpoint(
    spec: CheckpointSpec,
    *,
    scratch: Path,
    cohort: CohortPackage,
    device: torch.device,
    batch_size: int,
) -> CheckpointMeasurement:
    checkpoint_path = scratch / spec.model_filename
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing checkpoint {checkpoint_path}")
    source_manifests = validated_source_manifests(cohort.stats_payload)
    _, metadata_sha256 = validate_checkpoint_metadata(checkpoint_path, spec, source_manifests)
    model, config, provenance = o50_probe.load_checkpoint_model(
        checkpoint_path,
        reference_config=cohort.reference_config,
        stats_payload=cohort.stats_payload,
        player_codes=cohort.player_codes,
        device=device,
    )
    if provenance.get("wandb_id") != spec.run_id or provenance.get("checkpoint_step") != spec.checkpoint_step:
        raise ValueError(f"loaded checkpoint provenance differs for {spec.label}")
    if config.experiment_id != spec.experiment_id:
        raise ValueError(f"loaded checkpoint config differs for {spec.label}")
    _validate_control_geometry(config, spec.label)
    provenance = dict(provenance)
    provenance.update(
        {
            "experiment_id": spec.experiment_id,
            "resolved_config": json.loads(config.to_json()),
            "architecture_source": "checkpoint config; cohort reference supplies stats and vocabulary",
            "cohort_reference_architecture": json.loads(cohort.reference_config.to_json())["architecture"],
            "source_manifest_sha256": source_manifests,
            "metadata_sha256": metadata_sha256,
        }
    )

    attention_chunks: list[NDArray[np.float32]] = []
    trace_max_error = 0.0
    probability_sum_max_error = 0.0
    attention_dtypes: set[str] = set()
    try:
        for start in range(0, EXPECTED_WINDOWS, batch_size):
            stop = min(start + batch_size, EXPECTED_WINDOWS)
            context = {name: value[start:stop].to(device) for name, value in cohort.context_features.items()}
            context_actions = model.codec.quantize(features.stack_actions(context))
            padding = torch.zeros(stop - start, dtype=torch.long, device=device)
            with torch.inference_mode(), o50_model.amp_context(config, device):
                tokens = model.context_tokens(context, context_actions)
                traced_hidden, attention = o50_probe.trace_last_attention(model, tokens, padding)
                reference_hidden = model.forward_dense(context, padding, context_actions)
            if attention.dtype != torch.float32:
                raise RuntimeError(f"{spec.label}: runtime attention dtype is {attention.dtype}, not torch.float32")
            attention_dtypes.add(str(attention.dtype))
            if not torch.equal(traced_hidden, reference_hidden):
                trace_max_error = float((traced_hidden.float() - reference_hidden.float()).abs().max().item())
                raise RuntimeError(f"{spec.label}: traced trunk differs by {trace_max_error}")
            if not torch.isfinite(attention).all():
                raise RuntimeError(f"{spec.label}: attention contains a non-finite probability")
            probability_error = float((attention.sum(dim=-1) - 1.0).abs().max().item())
            probability_sum_max_error = max(probability_sum_max_error, probability_error)
            if probability_error > 1e-5:
                raise RuntimeError(f"{spec.label}: attention probabilities do not sum to one")
            if attention.shape[1:] != (EXPECTED_LAYERS, EXPECTED_HEADS, EXPECTED_CONTEXT_LENGTH):
                raise ValueError(f"{spec.label}: unexpected attention shape {tuple(attention.shape)}")
            attention_chunks.append(np.asarray(attention.detach().cpu(), dtype=np.float32))
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if attention_dtypes != {"torch.float32"}:
        raise RuntimeError(f"{spec.label}: attention dtype records are {sorted(attention_dtypes)}")
    weights = np.concatenate(attention_chunks, axis=0)
    summary = attention_detail.summarize_attention(weights)
    example_weights = np.ascontiguousarray(weights[0], dtype=np.float32)
    return CheckpointMeasurement(
        spec=spec,
        provenance=provenance,
        summary=summary,
        example_weights=example_weights,
        example_rows=_example_rows(example_weights),
        trace_max_abs_error=trace_max_error,
        probability_sum_max_abs_error=probability_sum_max_error,
        metadata_sha256=metadata_sha256,
    )


def validate_control_config_pair(measurements: tuple[CheckpointMeasurement, ...]) -> None:
    """Require equal inference configurations apart from experiment identity."""
    if len(measurements) != 2:
        raise ValueError("the Muon control requires exactly two measurements")
    configs = []
    for measurement in measurements:
        config = dict(_mapping(measurement.provenance.get("resolved_config"), "resolved_config"))
        config.pop("experiment_id", None)
        configs.append(config)
    if configs[0] != configs[1]:
        raise ValueError("parent and half-rate control inference configurations differ")


def _write_source_archive(output: Path) -> tuple[str, dict[str, str], str]:
    archive_path = output / "attention_muon_control_source.tar"
    if archive_path.exists():
        raise FileExistsError(f"refusing to overwrite {archive_path}")
    source_paths = {
        "attention_muon_control": Path(__file__),
        "attention_detail": Path(attention_detail.__file__),
        "o50_probe": Path(o50_probe.__file__),
    }
    source_hashes = {name: _sha256_file(path) for name, path in source_paths.items()}
    plan_path = Path(__file__).with_name("attention_detail_plan.md")
    if plan_path.is_file():
        source_paths["attention_detail_plan"] = plan_path
        source_hashes["attention_detail_plan"] = _sha256_file(plan_path)
    control_plan_path = Path(__file__).with_name("attention_muon_control_plan.md")
    if not control_plan_path.is_file():
        raise FileNotFoundError(f"required control plan is missing: {control_plan_path}")
    source_paths["attention_muon_control_plan"] = control_plan_path
    source_hashes["attention_muon_control_plan"] = _sha256_file(control_plan_path)
    with tarfile.open(archive_path, mode="w") as archive:
        for name, path in source_paths.items():
            archive.add(path, arcname=f"research/mechinterp_20260911/{name}{path.suffix}")
    return _sha256_file(archive_path), source_hashes, str(archive_path)


def _write_example_npz(output: Path, measurements: tuple[CheckpointMeasurement, ...]) -> dict[str, object]:
    path = output / "attention_example0.npz"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    arrays: dict[str, NDArray[np.float32] | FloatArray | IntegerArray] = {}
    checkpoint_keys: dict[str, str] = {}
    fields = (
        "weights",
        "entropy_nats",
        "effective_context",
        "row_max_weight",
        "weighted_lag",
        "current_frame_weight",
        "argmax_lag",
    )
    for index, measurement in enumerate(measurements):
        key = f"checkpoint_{index:02d}"
        checkpoint_keys[key] = (
            f"{measurement.spec.run_id}:boundary_step={measurement.spec.boundary_step}"
            f" (checkpoint_step={measurement.spec.checkpoint_step})"
        )
        arrays[f"{key}__weights"] = measurement.example_weights
        for field in fields[1:]:
            arrays[f"{key}__{field}"] = measurement.example_rows[field]
    np.savez_compressed(path, **arrays)
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "checkpoint_keys": checkpoint_keys,
        "example_index": 0,
        "weights_dtype": "float32",
        "weights_shape": [EXPECTED_LAYERS, EXPECTED_HEADS, EXPECTED_CONTEXT_LENGTH],
        "weights_order": "[layer, head, key], key 255=current/lag0 and key 0=oldest/lag255",
        "row_fields": list(fields[1:]),
    }


def _environment(device: torch.device, precision: Mapping[str, object]) -> dict[str, object]:
    environment: dict[str, object] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "precision": dict(precision),
    }
    environment["sha256"] = _json_sha256(environment)
    return environment


def _research_git_sha() -> str:
    repository = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    git_sha = completed.stdout.strip()
    if len(git_sha) != 40 or any(character not in "0123456789abcdef" for character in git_sha):
        raise ValueError(f"git rev-parse returned an invalid SHA: {git_sha!r}")
    return git_sha


def _result(
    *,
    cohort_path: Path,
    cohort: CohortPackage,
    measurements: tuple[CheckpointMeasurement, ...],
    precision: Mapping[str, object],
    source_archive_sha256: str,
    source_hashes: Mapping[str, str],
    example_manifest: Mapping[str, object],
    device: torch.device,
    source_archive_path: str,
    research_git_sha: str,
    parent_detail_sha256: str,
    parent_example_sha256: str,
) -> dict[str, object]:
    data = dict(cohort.data)
    data.update(
        {
            "package_path": str(cohort_path),
            "package_sha256": cohort.package_sha256,
            "data_json_sha256": cohort.data_json_sha256,
            "context_feature_names": sorted(cohort.context_features),
            "target_shape": list(cohort.target.shape),
            "target_sha256": _sha256_bytes(cohort.target.detach().cpu().contiguous().numpy().tobytes()),
            "fixed_cohort_sha256": FIXED_COHORT_SHA256,
        }
    )
    checkpoints: list[dict[str, object]] = []
    for index, measurement in enumerate(measurements):
        checkpoints.append(
            {
                "run_id": measurement.spec.run_id,
                "checkpoint_step": measurement.spec.checkpoint_step,
                "boundary_step": measurement.spec.boundary_step,
                "label": measurement.spec.label,
                "model_filename": measurement.spec.model_filename,
                "provenance": measurement.provenance,
                "summary": summary_to_dict(measurement.summary),
                "controls": {
                    "attention_dtype": "torch.float32",
                    "traced_trunk_max_abs_error": measurement.trace_max_abs_error,
                    "attention_probability_sum_max_abs_error": measurement.probability_sum_max_abs_error,
                    "parent_exact_parity": index == 0,
                },
            }
        )
    return {
        "schema_version": 1,
        "protocol": {
            "name": "O50 detailed last-query attention",
            "control": "p1fyyp1z versus 7dgv85x8 half-Muon-LR continuation",
            "windows": EXPECTED_WINDOWS,
            "batch_size": EXPECTED_BATCH_SIZE,
            "example_index": 0,
            "lag_definition": "lag 0 is current key; lag 255 is oldest key",
            "attention_shape": [EXPECTED_WINDOWS, EXPECTED_LAYERS, EXPECTED_HEADS, EXPECTED_CONTEXT_LENGTH],
            "no_training": True,
            "player_rank_inputs_retained": True,
        },
        "cohort": data,
        "resolved_config_reference": json.loads(cohort.reference_config.to_json()),
        "precision": dict(precision),
        "source": {
            "attention_muon_control_sha256": source_hashes["attention_muon_control"],
            "attention_detail_sha256": source_hashes["attention_detail"],
            "o50_probe_sha256": source_hashes["o50_probe"],
            "files": dict(source_hashes),
            "source_archive_sha256": source_archive_sha256,
            "source_archive_path": source_archive_path,
            "research_git_sha": research_git_sha,
        },
        "environment": _environment(device, precision),
        "parent_exact_parity": {
            "detail_sha256": parent_detail_sha256,
            "example_npz_sha256": parent_example_sha256,
            "comparison": "exact Python/NumPy equality; no tolerance",
        },
        "example_npz": dict(example_manifest),
        "checkpoints": checkpoints,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_scratch = Path("/home/ericgu/src/hal/runs/mechinterp-20260911-scratch")
    default_artifacts = Path(__file__).with_name("artifacts")
    parser.add_argument("--cohort", type=Path, default=default_scratch / "o50-cohort-256.pt")
    parser.add_argument("--scratch", type=Path, default=default_scratch)
    parser.add_argument(
        "--old-detail",
        type=Path,
        default=default_artifacts / "attention_detail" / "attention_detail.json",
    )
    parser.add_argument(
        "--old-example",
        type=Path,
        default=default_artifacts / "attention_detail" / "attention_example0.npz",
    )
    parser.add_argument("--output", type=Path, default=default_artifacts / "attention_muon_control")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=EXPECTED_BATCH_SIZE)
    args = parser.parse_args()
    if args.batch_size != EXPECTED_BATCH_SIZE:
        raise ValueError(f"the frozen control protocol requires batch size {EXPECTED_BATCH_SIZE}")
    owned_outputs = (
        args.output / "attention_detail.json",
        args.output / "attention_example0.npz",
        args.output / "attention_muon_control_source.tar",
    )
    if any(path.exists() for path in owned_outputs):
        raise FileExistsError(f"refusing to overwrite a control artifact in {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    precision = _configure_precision()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    cohort = load_fixed_cohort(args.cohort)
    research_git_sha = _research_git_sha()
    source_archive_sha256, source_hashes, source_archive_path = _write_source_archive(args.output)
    measurements_list: list[CheckpointMeasurement] = []
    for spec in CHECKPOINT_SPECS:
        print(f"Starting {spec.label}", flush=True)
        measurements_list.append(
            _measure_checkpoint(
                spec,
                scratch=args.scratch,
                cohort=cohort,
                device=device,
                batch_size=args.batch_size,
            )
        )
        print(f"Finished {spec.label}", flush=True)
    measurements = tuple(measurements_list)
    validate_control_config_pair(measurements)
    validate_parent_parity(
        measurements[0].summary,
        measurements[0].example_weights,
        measurements[0].example_rows,
        args.old_detail,
        args.old_example,
    )
    example_manifest = _write_example_npz(args.output, measurements)
    result = _result(
        cohort_path=args.cohort,
        cohort=cohort,
        measurements=measurements,
        precision=precision,
        source_archive_sha256=source_archive_sha256,
        source_hashes=source_hashes,
        example_manifest=example_manifest,
        device=device,
        source_archive_path=source_archive_path,
        research_git_sha=research_git_sha,
        parent_detail_sha256=_sha256_file(args.old_detail),
        parent_example_sha256=_sha256_file(args.old_example),
    )
    result_path = args.output / "attention_detail.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite {result_path}")
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"Saved {result_path}; {len(measurements)} checkpoints", flush=True)


if __name__ == "__main__":
    main()
