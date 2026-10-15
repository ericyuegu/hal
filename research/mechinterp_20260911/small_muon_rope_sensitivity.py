"""Measure small-Muon attention sensitivity to RoPE factor precision.

The BF16-factor control reproduces the frozen small-model measurement. The
FP32-factor treatment restores O26's training-time RoPE arithmetic. It is a
counterfactual precision treatment for O37 and O43, whose exact source already
requested BF16 factors under AMP.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor

from hal.training import features
from hal.training import trunk
from research.mechinterp_20260911 import small_muon_inputs
from research.mechinterp_20260911 import small_muon_model

_INPUT_SHA256 = "7149b3ccbc81798ad2877e2d996cf14e51f4160f1c1a241685f6f8cca943ea33"
_FROZEN_ATTENTION_SHA256 = "5a6768a41f8799d2ff283e6f863fc8b16840126716ac71c27a648a8bff2fbd65"
_FROZEN_WEIGHTS_SHA256 = "c384dbb42481665efc9bd209a2966f68b40daee5b2aed1f0096f468d42075741"
_SEED = 20260911
_BATCH_SIZE = 16
_LAYERS = 8
_CONTEXT_LENGTH = 128
_REPLAY_COUNT = 256
_CONTROL_HIDDEN_TOLERANCE = 1e-5
_FROZEN_WEIGHT_TOLERANCE = 1e-6

FactorPrecision = Literal["bf16", "fp32"]
FloatArray = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class CheckpointSpec:
    run_id: str
    family: small_muon_model.SmallMuonFamily
    sha256: str
    heads: int
    fp32_factors_match_training: bool
    historical_trunk_sha256: str


_CHECKPOINTS = (
    CheckpointSpec(
        "a117chkw",
        "o37",
        "665a21857debb092cd80cdd3bf60257c9a732df7608fa2d513c34d91ac2bf88e",
        4,
        False,
        "9ebf58a59860981f71548de42be4ac0f68756df951e8438cea394d564eccada5",
    ),
    CheckpointSpec(
        "5wfk2esf",
        "o37",
        "0d62a529df3a08a20c170cca325cecf5c3de489e12f15b02496e66d93c01bc5c",
        4,
        False,
        "9ebf58a59860981f71548de42be4ac0f68756df951e8438cea394d564eccada5",
    ),
    CheckpointSpec(
        "cqbbbg77",
        "o26",
        "22333d1d61d6b648c757f0f1f3e887925fbb12a08fdffd1cb4ae72d6d6f2ef88",
        6,
        True,
        "13f082e5eda3f5c9d6e4e45d2cf970c4c76e4f315bb866550a4773d3b63322ed",
    ),
    CheckpointSpec(
        "1imfy8v3",
        "o43",
        "07111f0854914f0017ae210cd79a290b7d7968b62bcae32aef486e4913f64712",
        6,
        False,
        "2c1cc069d38f9d9ee8f657e10b8a4debf9fc978a90ae0e637bdb4469706bb5c7",
    ),
)


@dataclass(frozen=True, slots=True)
class TraceResult:
    hidden: Tensor
    weights: Tensor
    projection_dtypes: tuple[str, ...]
    factor_dtypes: tuple[str, ...]
    rotated_dtypes: tuple[str, ...]
    attention_output_dtypes: tuple[str, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rotary_factors(
    rotary: trunk.Rotary,
    length: int,
    device: torch.device,
    output_dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """Rebuild factors from FP32 geometry without touching module caches."""
    if rotary.inv_freq.dtype != torch.float32:
        raise ValueError("The sensitivity protocol requires stored FP32 inverse frequencies")
    frequencies = torch.outer(
        torch.arange(length, device=device, dtype=torch.float32),
        rotary.inv_freq,
    )
    return (
        frequencies.cos().to(output_dtype)[None, :, None, :],
        frequencies.sin().to(output_dtype)[None, :, None, :],
    )


def trace_dense_trunk(
    model_trunk: trunk.Trunk,
    tokens: Tensor,
    padding: Tensor,
    factor_precision: FactorPrecision,
) -> TraceResult:
    """Run the dense trunk and retain every last-query attention row.

    This function must run inside BF16 autocast. The only branch between the
    two protocols is whether the finished FP32 sine/cosine table is cast to
    the BF16 Q/K projection dtype before rotary multiplication.
    """
    state = tokens
    length = tokens.shape[1]
    mask = trunk.dense_mask(padding, length, model_trunk.attn_window)
    rows = []
    projection_dtypes = []
    factor_dtypes = []
    rotated_dtypes = []
    attention_output_dtypes = []
    for raw_block in model_trunk.blocks:
        block = cast(trunk.Block, raw_block)
        normalized = trunk.rmsnorm(state)
        query, key, value = block.attn.c_attn(normalized).split(block.attn.d_model, dim=-1)
        if query.dtype != torch.bfloat16 or key.dtype != query.dtype or value.dtype != query.dtype:
            raise ValueError(
                "The sensitivity protocol requires BF16 Q/K/V projections; "
                f"got {query.dtype}, {key.dtype}, {value.dtype}"
            )
        shape = (state.shape[0], length, block.attn.n_heads, block.attn.head_dim)
        query, key, value = query.view(shape), key.view(shape), value.view(shape)
        factor_dtype = query.dtype if factor_precision == "bf16" else torch.float32
        cosine, sine = _rotary_factors(block.attn.rotary, length, query.device, factor_dtype)
        query = trunk.apply_rotary_emb(query, cosine, sine)
        key = trunk.apply_rotary_emb(key, cosine, sine)
        projection_dtypes.append(str(value.dtype))
        factor_dtypes.append(str(cosine.dtype))
        rotated_dtypes.append(str(query.dtype))
        with torch.autocast(device_type=query.device.type, enabled=False):
            scores = torch.einsum(
                "bhd,bkhd->bhk",
                query[:, -1].float(),
                key.float(),
            ) / math.sqrt(block.attn.head_dim)
            scores = scores.masked_fill(~mask[:, 0, -1, None, :], float("-inf"))
            rows.append(F.softmax(scores, dim=-1))
        attended = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=mask,
        )
        attention_output_dtypes.append(str(attended.dtype))
        attended = attended.transpose(1, 2).contiguous().view_as(state)
        state = state + block.attn_scale * block.attn.c_proj(attended)
        state = state + block.mlp_scale * block.mlp(trunk.rmsnorm(state))
    return TraceResult(
        hidden=trunk.rmsnorm(state),
        weights=torch.stack(rows, dim=1),
        projection_dtypes=tuple(projection_dtypes),
        factor_dtypes=tuple(factor_dtypes),
        rotated_dtypes=tuple(rotated_dtypes),
        attention_output_dtypes=tuple(attention_output_dtypes),
    )


def _row_metrics(weights: FloatArray) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    values = weights.astype(np.float64)
    maxima = values.max(axis=-1)
    entropy = -(values * np.log(np.maximum(values, 1e-30))).sum(axis=-1)
    return maxima, entropy, np.exp(entropy)


def _summarize(weights: FloatArray) -> dict[str, float]:
    maxima, entropy, effective = _row_metrics(weights)
    return {
        "mean_max_weight": float(maxima.mean()),
        "mean_entropy_nats": float(entropy.mean()),
        "mean_effective_frames": float(effective.mean()),
    }


def _compare(control: FloatArray, treatment: FloatArray) -> dict[str, object]:
    control_max, control_entropy, control_effective = _row_metrics(control)
    treatment_max, treatment_entropy, treatment_effective = _row_metrics(treatment)
    control_summary = _summarize(control)
    treatment_summary = _summarize(treatment)
    return {
        "bf16_factor_control": control_summary,
        "fp32_factor_treatment": treatment_summary,
        "treatment_minus_control": {name: treatment_summary[name] - control_summary[name] for name in control_summary},
        "maximum_absolute_changes": {
            "probability": float(np.abs(treatment.astype(np.float64) - control).max()),
            "row_max_weight": float(np.abs(treatment_max - control_max).max()),
            "row_entropy_nats": float(np.abs(treatment_entropy - control_entropy).max()),
            "row_effective_frames": float(np.abs(treatment_effective - control_effective).max()),
        },
        "fraction_rows_with_changed_peak_position": float(
            (treatment.argmax(axis=-1) != control.argmax(axis=-1)).mean()
        ),
    }


def _load_checkpoint(path: Path, spec: CheckpointSpec) -> tuple[Mapping[str, object], Mapping[str, Tensor]]:
    if _sha256_file(path) != spec.sha256:
        raise ValueError(f"{spec.run_id}: checkpoint hash differs")
    if torch.serialization.get_unsafe_globals_in_checkpoint(path):
        raise ValueError(f"{spec.run_id}: checkpoint needs an unapproved pickle global")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != {
        "step",
        "model",
        "opt",
        "sched",
        "cfg",
        "wandb_id",
    }:
        raise ValueError(f"{spec.run_id}: unexpected checkpoint fields")
    if checkpoint["step"] != 16384 or checkpoint["wandb_id"] != spec.run_id:
        raise ValueError(f"{spec.run_id}: checkpoint identity differs")
    config = checkpoint["cfg"]
    model_state = checkpoint["model"]
    if not isinstance(config, Mapping) or not isinstance(model_state, Mapping):
        raise TypeError(f"{spec.run_id}: checkpoint config and model must be mappings")
    if any(not isinstance(name, str) for name in config):
        raise TypeError(f"{spec.run_id}: checkpoint config keys must be strings")
    if any(not isinstance(name, str) or not isinstance(value, Tensor) for name, value in model_state.items()):
        raise TypeError(f"{spec.run_id}: checkpoint model must be tensor-only with string keys")
    return config, model_state


def _measure_checkpoint(
    spec: CheckpointSpec,
    checkpoint_root: Path,
    context: Mapping[str, Tensor],
    frozen_weights: FloatArray,
    device: torch.device,
) -> tuple[dict[str, object], FloatArray, FloatArray]:
    checkpoint_path = checkpoint_root / f"{spec.run_id}-final.source.pt"
    config, model_state = _load_checkpoint(checkpoint_path, spec)
    model = small_muon_model.build_small_muon_model(spec.family, config, model_state).to(device).eval()
    control_chunks = []
    treatment_chunks = []
    control_hidden_error = 0.0
    treatment_hidden_change = 0.0
    dtype_records: dict[str, list[str]] | None = None
    for start in range(0, _REPLAY_COUNT, _BATCH_SIZE):
        inputs = {name: value[start : start + _BATCH_SIZE].to(device) for name, value in context.items()}
        padding = torch.zeros(_BATCH_SIZE, dtype=torch.long, device=device)
        action_indices = model.codec.quantize(features.stack_actions(inputs))
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            tokens = model.context_tokens(inputs, action_indices)
            control = trace_dense_trunk(model.trunk, tokens, padding, "bf16")
            reference = model.trunk.forward_dense(tokens, padding)
            treatment = trace_dense_trunk(model.trunk, tokens, padding, "fp32")
        error = float((control.hidden.float() - reference.float()).abs().max().item())
        change = float((treatment.hidden.float() - control.hidden.float()).abs().max().item())
        control_hidden_error = max(control_hidden_error, error)
        treatment_hidden_change = max(treatment_hidden_change, change)
        if not math.isfinite(error) or error > _CONTROL_HIDDEN_TOLERANCE:
            raise RuntimeError(f"{spec.run_id}: copied BF16 control differs from current dense path by {error}")
        expected_shape = (_BATCH_SIZE, _LAYERS, spec.heads, _CONTEXT_LENGTH)
        if control.weights.dtype != torch.float32 or tuple(control.weights.shape) != expected_shape:
            raise ValueError(f"{spec.run_id}: unexpected control attention dtype or shape")
        if treatment.weights.dtype != torch.float32 or tuple(treatment.weights.shape) != expected_shape:
            raise ValueError(f"{spec.run_id}: unexpected treatment attention dtype or shape")
        current_dtypes = {
            "projection": list(control.projection_dtypes),
            "bf16_factor": list(control.factor_dtypes),
            "bf16_rotated_qk": list(control.rotated_dtypes),
            "bf16_attention_output": list(control.attention_output_dtypes),
            "fp32_factor": list(treatment.factor_dtypes),
            "fp32_rotated_qk": list(treatment.rotated_dtypes),
            "fp32_attention_output": list(treatment.attention_output_dtypes),
        }
        if dtype_records is not None and current_dtypes != dtype_records:
            raise RuntimeError(f"{spec.run_id}: observed dtypes changed between batches")
        dtype_records = current_dtypes
        control_chunks.append(control.weights.cpu().numpy())
        treatment_chunks.append(treatment.weights.cpu().numpy())
    control_weights = np.concatenate(control_chunks, axis=0)
    treatment_weights = np.concatenate(treatment_chunks, axis=0)
    frozen_error = float(np.abs(control_weights.astype(np.float64) - frozen_weights).max())
    if frozen_error > _FROZEN_WEIGHT_TOLERANCE:
        raise RuntimeError(f"{spec.run_id}: BF16 control differs from frozen weights by {frozen_error}")
    result = {
        "run_id": spec.run_id,
        "family": spec.family,
        "checkpoint_sha256": spec.sha256,
        "historical_trunk_sha256": spec.historical_trunk_sha256,
        "fp32_factor_treatment_interpretation": (
            "original O26 training protocol restoration"
            if spec.fp32_factors_match_training
            else "counterfactual precision treatment; training used BF16 factors"
        ),
        "copied_control_vs_current_dense_hidden_max_abs": control_hidden_error,
        "copied_control_vs_frozen_probability_max_abs": frozen_error,
        "fp32_treatment_vs_control_hidden_max_abs": treatment_hidden_change,
        "observed_dtypes_by_layer": dtype_records,
        **_compare(control_weights, treatment_weights),
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps(result, sort_keys=True), flush=True)
    return result, control_weights, treatment_weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--frozen-attention-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    input_path = args.input_root / "inputs.pt"
    frozen_json_path = args.frozen_attention_root / "attention.json"
    frozen_weights_path = args.frozen_attention_root / "last-query-attention.npz"
    expected_hashes = {
        input_path: _INPUT_SHA256,
        frozen_json_path: _FROZEN_ATTENTION_SHA256,
        frozen_weights_path: _FROZEN_WEIGHTS_SHA256,
    }
    for path, expected_hash in expected_hashes.items():
        if _sha256_file(path) != expected_hash:
            raise ValueError(f"Frozen input differs: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("This protocol requires the local CUDA GPU; no CPU fallback")
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(_SEED)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    package = torch.load(input_path, map_location="cpu", weights_only=True)
    if not isinstance(package, Mapping) or set(package) != {"context_features", "metadata_json"}:
        raise ValueError("Unexpected prepared input package fields")
    context = package["context_features"]
    if not isinstance(context, Mapping) or any(
        not isinstance(name, str) or not isinstance(value, Tensor) for name, value in context.items()
    ):
        raise TypeError("Prepared context must be a string-to-tensor mapping")
    metadata = json.loads(package["metadata_json"])
    if small_muon_inputs.hash_context(context) != metadata["prepared_context_sha256"]:
        raise ValueError("The prepared context tensor hash differs")
    results = []
    arrays = {}
    with np.load(frozen_weights_path, allow_pickle=False) as frozen:
        if set(frozen.files) != {spec.run_id for spec in _CHECKPOINTS}:
            raise ValueError("Frozen attention checkpoint identities differ")
        for spec in _CHECKPOINTS:
            result, control, treatment = _measure_checkpoint(
                spec,
                args.checkpoint_root,
                context,
                frozen[spec.run_id],
                torch.device("cuda"),
            )
            results.append(result)
            arrays[f"{spec.run_id}__bf16_factor_control"] = control
            arrays[f"{spec.run_id}__fp32_factor_treatment"] = treatment
    array_path = output / "all-rows.npz"
    with array_path.open("xb") as target:
        np.savez_compressed(target, **arrays)
    study = Path(__file__).resolve().parent
    repository = study.parents[1]
    source_paths = (
        Path(__file__).resolve(),
        study / "small_muon_model.py",
        study / "small_muon_inputs.py",
        repository / "hal/training/trunk.py",
    )
    metric_names = (
        "probability",
        "row_max_weight",
        "row_entropy_nats",
        "row_effective_frames",
    )
    payload = {
        "schema_version": 1,
        "treatment": {
            "control": "finished RoPE sine/cosine factors cast to BF16 projection dtype",
            "treatment": "finished RoPE sine/cosine factors retained in FP32",
            "interpretation": {
                "o26": "original training protocol restoration",
                "o37": "counterfactual precision treatment; original training used BF16 factors",
                "o43": "counterfactual precision treatment; original training used BF16 factors",
            },
            "invariants": (
                "same four checkpoints, 256 inputs, seed, B16 batching, dense SDPA, BF16 QKV "
                "projections, and FP32 reported QK scores/softmax"
            ),
        },
        "models": results,
        "maximum_absolute_changes_across_models": {
            name: max(cast(dict[str, float], result["maximum_absolute_changes"])[name] for result in results)
            for name in metric_names
        },
        "inputs": {
            "package_sha256": _INPUT_SHA256,
            "context_sha256": metadata["prepared_context_sha256"],
            "frozen_attention_json_sha256": _FROZEN_ATTENTION_SHA256,
            "frozen_attention_weights_sha256": _FROZEN_WEIGHTS_SHA256,
        },
        "source": {
            "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "files": {str(path.relative_to(repository)): _sha256_file(path) for path in source_paths},
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "gpu": torch.cuda.get_device_name(),
            "seed": _SEED,
            "batch_size": _BATCH_SIZE,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "tf32": False,
            "mode": "eval, inference_mode, eager dense SDPA",
        },
        "control_validation": {
            "copied_dense_hidden_tolerance": _CONTROL_HIDDEN_TOLERANCE,
            "frozen_probability_tolerance": _FROZEN_WEIGHT_TOLERANCE,
        },
        "all_rows": {
            "path": array_path.name,
            "sha256": _sha256_file(array_path),
            "dtype": "float32",
        },
    }
    result_path = output / "sensitivity.json"
    with result_path.open("x") as target:
        json.dump(payload, target, indent=2, sort_keys=True, allow_nan=False)
        target.write("\n")
    manifest = {
        "schema_version": 1,
        "sensitivity_json_sha256": _sha256_file(result_path),
        "all_rows_sha256": _sha256_file(array_path),
        "measurement_source_sha256": _sha256_file(Path(__file__).resolve()),
    }
    with (output / "manifest.json").open("x") as target:
        json.dump(manifest, target, indent=2, sort_keys=True)
        target.write("\n")
    print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
