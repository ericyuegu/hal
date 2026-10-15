"""Recompute O50 last-query attention with explicit FP32 scores."""

import argparse
import hashlib
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from hal.inference import o50_model
from hal.training import features
from research.mechinterp_20260911 import o50_probe

_EXPECTED_COHORT_FIELDS = {
    "bundle_reference_json",
    "context_features",
    "data_json",
    "player_codes",
    "reference_config",
    "stats_payload",
    "target",
}
_LEGACY_PROBE_SHA256 = "bae7012998b0ac1ddbe701690b0ec4b0fd9138dd15648b6252d0a8a24a85c977"
_FIXED_COHORT_SHA256 = "ec61e97e332672a4e4e2fdc438e208a281a4609e10f7eded72368eccac323437"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a mapping with string keys")
    return cast(Mapping[str, object], value)


def _tensor_bytes(value: object, label: str) -> bytes:
    if not isinstance(value, Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
        raise TypeError(f"{label} must be a one-dimensional uint8 tensor")
    return value.numpy().tobytes()


def _metric_delta(
    corrected: Mapping[str, object],
    legacy: Mapping[str, object],
) -> dict[str, list[list[float]]]:
    output = {}
    for name, corrected_rows in corrected.items():
        legacy_rows = legacy.get(name)
        if not isinstance(corrected_rows, list) or not isinstance(legacy_rows, list):
            raise ValueError(f"attention metric {name!r} is not a matrix")
        corrected_tensor = torch.tensor(corrected_rows, dtype=torch.float64)
        legacy_tensor = torch.tensor(legacy_rows, dtype=torch.float64)
        if corrected_tensor.shape != legacy_tensor.shape:
            raise ValueError(f"attention metric {name!r} changed shape")
        output[name] = (corrected_tensor - legacy_tensor).tolist()
    return output


def run_reaudit(
    cohort_path: Path,
    checkpoint_path: Path,
    legacy_result_path: Path,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    """Return corrected attention summaries for one validated checkpoint."""
    started = time.monotonic()
    source_hashes = {
        "attention_reaudit_sha256": _sha256(Path(__file__)),
        "corrected_o50_probe_sha256": _sha256(Path(o50_probe.__file__)),
    }
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    matmul_precision = torch.get_float32_matmul_precision()
    if matmul_precision != "highest":
        raise ValueError(f"attention re-audit requires highest FP32 matmul precision, got {matmul_precision!r}")
    package = _mapping(torch.load(cohort_path, map_location="cpu", weights_only=True), "cohort package")
    if set(package) != _EXPECTED_COHORT_FIELDS:
        raise ValueError("cohort package fields differ from the fixed protocol")
    reference_config = o50_model.O50Config.from_json(_tensor_bytes(package["reference_config"], "reference config"))
    stats = _mapping(json.loads(_tensor_bytes(package["stats_payload"], "statistics")), "statistics")
    player_codes = _tensor_bytes(package["player_codes"], "player codes") or None
    context_raw = _mapping(package["context_features"], "context features")
    if any(not isinstance(value, Tensor) for value in context_raw.values()):
        raise TypeError("context feature values must be tensors")
    contexts = cast(Mapping[str, Tensor], context_raw)
    windows = next(iter(contexts.values())).shape[0]
    if windows != 256 or any(
        value.shape[:2] != (windows, reference_config.architecture.L_ctx) for value in contexts.values()
    ):
        raise ValueError("attention re-audit requires the fixed 256-replay cohort")

    legacy_result = _mapping(json.loads(legacy_result_path.read_text()), "legacy functional result")
    cohort_sha256 = _sha256(cohort_path)
    if cohort_sha256 != _FIXED_COHORT_SHA256:
        raise ValueError("attention re-audit cohort differs from the fixed 256-replay cohort")
    legacy_cohort = _mapping(legacy_result.get("cohort_package"), "legacy cohort provenance")
    if legacy_cohort.get("sha256") != cohort_sha256 or legacy_cohort.get("windows") != windows:
        raise ValueError("legacy result used a different cohort")
    legacy_probe = _mapping(legacy_result.get("probe"), "legacy probe provenance")
    legacy_attention = _mapping(legacy_result.get("attention"), "legacy attention summary")

    model, config, checkpoint = o50_probe.load_checkpoint_model(
        checkpoint_path,
        reference_config=reference_config,
        stats_payload=stats,
        player_codes=player_codes,
        device=device,
    )
    legacy_checkpoint = _mapping(legacy_result.get("checkpoint"), "legacy checkpoint provenance")
    if legacy_checkpoint.get("model_export_sha256") != checkpoint["model_export_sha256"]:
        raise ValueError("legacy result used a different model export")

    attention_chunks = []
    trace_max_error = 0.0
    probability_sum_max_error = 0.0
    for start in range(0, windows, batch_size):
        stop = min(start + batch_size, windows)
        context = {name: value[start:stop].to(device) for name, value in contexts.items()}
        context_actions = model.codec.quantize(features.stack_actions(context))
        padding = torch.zeros(stop - start, dtype=torch.long, device=device)
        with torch.inference_mode(), o50_model.amp_context(config, device):
            tokens = model.context_tokens(context, context_actions)
            traced_hidden, attention = o50_probe.trace_last_attention(model, tokens, padding)
            reference_hidden = model.forward_dense(context, padding, context_actions)
        if attention.dtype != torch.float32:
            raise RuntimeError(f"corrected attention has dtype {attention.dtype}, expected float32")
        trace_max_error = max(trace_max_error, float((traced_hidden - reference_hidden).abs().max()))
        probability_sum_max_error = max(
            probability_sum_max_error,
            float((attention.sum(dim=-1) - 1).abs().max()),
        )
        attention_chunks.append(attention.cpu())

    corrected_attention = o50_probe.attention_metrics(torch.cat(attention_chunks))
    source_snapshot = (
        Path(__file__).with_name("artifacts") / "source_snapshots" / (f"o50_probe-{_LEGACY_PROBE_SHA256[:16]}.py")
    )
    if _sha256(source_snapshot) != _LEGACY_PROBE_SHA256:
        raise ValueError("legacy source snapshot hash differs")
    return {
        "schema_version": 1,
        "attention": corrected_attention,
        "corrected_minus_legacy": _metric_delta(corrected_attention, legacy_attention),
        "checkpoint": checkpoint,
        "cohort": {"path": str(cohort_path), "sha256": cohort_sha256, "windows": windows},
        "controls": {
            "attention_probability_sum_max_abs_error": probability_sum_max_error,
            "traced_trunk_max_abs_error": trace_max_error,
        },
        "device": {
            "batch_size": batch_size,
            "name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
        },
        "elapsed_seconds": time.monotonic() - started,
        "legacy_functional_artifact": {
            "path": str(legacy_result_path),
            "sha256": _sha256(legacy_result_path),
            "probe_sha256": legacy_probe.get("sha256"),
            "attention_score_precision": "autocast permitted BF16 QK scores and softmax",
        },
        "precision": {
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "float32_matmul_precision": matmul_precision,
            "qk_score_dtype": "torch.float32",
            "score_and_softmax_autocast_enabled": False,
            "softmax_output_dtype": "torch.float32",
            "upstream_qk_projection_dtype": str(config.amp_dtype),
        },
        "probe": {
            **source_hashes,
            "legacy_o50_probe_snapshot": str(source_snapshot),
            "legacy_o50_probe_sha256": _LEGACY_PROBE_SHA256,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--legacy-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 64:
        raise ValueError("batch size must be in [1, 64]")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    torch.set_float32_matmul_precision("highest")
    result = run_reaudit(
        args.cohort,
        args.checkpoint,
        args.legacy_result,
        device=torch.device(args.device),
        batch_size=args.batch_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False, sort_keys=True) + "\n")
    print(f"Saved {args.output}; {result['elapsed_seconds']:.1f} seconds", flush=True)


if __name__ == "__main__":
    main()
