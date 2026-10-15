"""Measure last-query attention in four completed, smaller Muon policies."""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import subprocess
import tarfile
from collections.abc import Mapping
from dataclasses import asdict
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

INPUT_SHA256 = "7149b3ccbc81798ad2877e2d996cf14e51f4160f1c1a241685f6f8cca943ea33"
SEED = 20260911
BATCH_SIZE = 16
HISTORICAL_SOURCES = {
    "o26": (
        "experiments/026_temporal_mtp.py",
        "ddc3538dde22abc3787fa7d006274f825a975476710c6a72999dbf6ccc2cfbe6",
        "3a257f88b2e17f8bbf2d0ee465352650ecb9c96b",
    ),
    "o37": (
        "experiments/037_factorization_matrix.py",
        "3cbf78ef60755b2e9a52a9ad1ed2aac70df410ba50193823aea311a0eda3e111",
        "f0f4ef29a019df0d85037537c21d6947e0e5a8b3",
    ),
    "o43": (
        "experiments/043_legacy_codec.py",
        "3109a088ba18c5ca03a5c1ff0e2dfdb6b01e5c1aec8ae0976d475f7897107420",
        "22d52f127f3c56d204fbdfb330ae6434b711b07a",
    ),
}


@dataclass(frozen=True, slots=True)
class CheckpointSpec:
    run_id: str
    family: Literal["o26", "o37", "o43"]
    label: str
    sha256: str
    parameters: int
    width: int
    heads: int


CHECKPOINTS = (
    CheckpointSpec(
        "a117chkw",
        "o37",
        "7.1M · O37 D1",
        "665a21857debb092cd80cdd3bf60257c9a732df7608fa2d513c34d91ac2bf88e",
        7147504,
        256,
        4,
    ),
    CheckpointSpec(
        "5wfk2esf",
        "o37",
        "7.1M · O37 D3",
        "0d62a529df3a08a20c170cca325cecf5c3de489e12f15b02496e66d93c01bc5c",
        7147504,
        256,
        4,
    ),
    CheckpointSpec(
        "cqbbbg77",
        "o26",
        "15.1M · O26",
        "22333d1d61d6b648c757f0f1f3e887925fbb12a08fdffd1cb4ae72d6d6f2ef88",
        15053039,
        384,
        6,
    ),
    CheckpointSpec(
        "1imfy8v3",
        "o43",
        "15.1M · O43",
        "07111f0854914f0017ae210cd79a290b7d7968b62bcae32aef486e4913f64712",
        15053039,
        384,
        6,
    ),
)


def validate_config(
    spec: CheckpointSpec, config: Mapping[str, object], wandb_config: Mapping[str, object]
) -> list[str]:
    """Allow only the two independently checked metadata serialization differences."""
    differences = []
    for key in sorted(set(config) | set(wandb_config)):
        if key in config and key in wandb_config and config[key] == wandb_config[key]:
            continue
        if (
            spec.family == "o37"
            and key == "experiment_id"
            and key not in wandb_config
            and config[key] == "037_factorization_matrix_v1"
        ):
            differences.append("Checkpoint adds experiment_id=037_factorization_matrix_v1")
            continue
        actual = config.get(key)
        logged = wandb_config.get(key)
        if (
            spec.family == "o43"
            and key == "lr_floor_ratio"
            and isinstance(actual, float)
            and math.nextafter(actual, math.inf) == logged
        ):
            differences.append("W&B lr_floor_ratio is one float64 ULP above the checkpoint value")
            continue
        raise ValueError(f"{spec.run_id}: unresolved configuration difference for {key}")
    for key, expected in {
        "d_model": spec.width,
        "n_layers": 8,
        "n_heads": spec.heads,
        "L_ctx": 128,
        "max_steps": 16384,
        "muon_lr": 0.02,
        "observation_bundle": "base",
        "mds_schema_version": 7,
        "data_root": "data/processed/ranked-anonymized-1/mds-policy-v7",
    }.items():
        if config.get(key) != expected:
            raise ValueError(f"{spec.run_id}: unexpected {key}={config.get(key)!r}")
    return differences


def trace_last_attention(model_trunk: trunk.Trunk, tokens: Tensor, padding: Tensor) -> tuple[Tensor, Tensor]:
    """Keep the normal forward path; compute each head's last-query scores in FP32."""
    state = tokens
    length = tokens.shape[1]
    mask = trunk.dense_mask(padding, length, model_trunk.attn_window)
    rows = []
    for raw_block in model_trunk.blocks:
        block = cast(trunk.Block, raw_block)
        query, key, _ = block.attn.c_attn(trunk.rmsnorm(state)).split(block.attn.d_model, dim=-1)
        shape = (state.shape[0], length, block.attn.n_heads, block.attn.head_dim)
        query, key = query.view(shape), key.view(shape)
        cosine, sine = block.attn.rotary(query)
        query = trunk.apply_rotary_emb(query, cosine, sine)
        key = trunk.apply_rotary_emb(key, cosine, sine)
        with torch.autocast(device_type=query.device.type, enabled=False):
            scores = torch.einsum("bhd,bkhd->bhk", query[:, -1].float(), key.float()) / math.sqrt(block.attn.head_dim)
            scores = scores.masked_fill(~mask[:, 0, -1, None, :], float("-inf"))
            rows.append(F.softmax(scores, dim=-1))
        state = block(state, mask, padding)
    return trunk.rmsnorm(state), torch.stack(rows, dim=1)


def summarize_attention(weights_rlhk: NDArray[np.float32]) -> dict[str, object]:
    """Average row statistics after measuring each replay/layer/head distribution."""
    if weights_rlhk.ndim != 4 or min(weights_rlhk.shape) < 1 or weights_rlhk.dtype != np.float32:
        raise ValueError("Expected float32 [replay, layer, head, key] probabilities")
    values = weights_rlhk.astype(np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Attention probabilities must be finite and nonnegative")
    if not np.allclose(values.sum(axis=-1), 1.0, atol=1e-5, rtol=0):
        raise ValueError("Attention probabilities must sum to one over key positions")
    maxima = values.max(axis=-1)
    if ((values == maxima[..., None]).sum(axis=-1) != 1).any():
        raise ValueError("An exact tied maximum makes the selected token ambiguous")
    entropy = -(values * np.log(np.maximum(values, 1e-30))).sum(axis=-1)
    effective = np.exp(entropy)
    positions = values.argmax(axis=-1)
    replay_count, layer_count, head_count, key_count = values.shape
    distinct = np.empty((replay_count, layer_count), dtype=np.int64)
    for replay in range(replay_count):
        for layer in range(layer_count):
            distinct[replay, layer] = np.unique(positions[replay, layer]).size
    layers = []
    for layer in range(layer_count):
        layers.append(
            {
                "layer": layer,
                "mean_max_weight": maxima[:, layer].mean(axis=0).tolist(),
                "mean_entropy_nats": entropy[:, layer].mean(axis=0).tolist(),
                "mean_effective_frames": effective[:, layer].mean(axis=0).tolist(),
                "mean_distinct_max_positions": float(distinct[:, layer].mean()),
                "example_key_positions": positions[0, layer].tolist(),
                "example_max_weights": maxima[0, layer].tolist(),
                "example_entropy_nats": entropy[0, layer].tolist(),
                "example_effective_frames": effective[0, layer].tolist(),
                "example_distinct_positions": int(distinct[0, layer]),
                "example_weights": weights_rlhk[0, layer].tolist(),
            }
        )
    return {
        "shape": list(values.shape),
        "mean_max_weight": float(maxima.mean()),
        "mean_entropy_nats": float(entropy.mean()),
        "mean_effective_frames": float(effective.mean()),
        "mean_current_token_weight": float(values[..., -1].mean()),
        "fraction_rows_peak_at_current_token": float((positions == key_count - 1).mean()),
        "fraction_rows_peak_at_least_90_percent": float((maxima >= 0.9).mean()),
        "max_weight_quantiles": dict(
            zip(("p10", "p50", "p90"), np.quantile(maxima, (0.1, 0.5, 0.9)).tolist(), strict=True)
        ),
        "mean_distinct_max_positions_per_layer": distinct.mean(axis=0).tolist(),
        "heads": head_count,
        "layers": layers,
    }


def measure_checkpoint(
    spec: CheckpointSpec, checkpoint_root: Path, inventory_root: Path, context: dict[str, Tensor], device: torch.device
) -> tuple[dict[str, object], NDArray[np.float32]]:
    path = checkpoint_root / f"{spec.run_id}-final.source.pt"
    if small_muon_inputs.sha256_file(path) != spec.sha256:
        raise ValueError(f"{spec.run_id}: checkpoint hash differs")
    if torch.serialization.get_unsafe_globals_in_checkpoint(path):
        raise ValueError(f"{spec.run_id}: checkpoint needs an unapproved pickle global")
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if set(state) != {"step", "model", "opt", "sched", "cfg", "wandb_id"}:
        raise ValueError(f"{spec.run_id}: unexpected checkpoint fields")
    if state["step"] != 16384 or state["wandb_id"] != spec.run_id:
        raise ValueError(f"{spec.run_id}: checkpoint identity differs")
    config = json.loads(json.dumps(state["cfg"]))
    inventory_path = inventory_root / spec.run_id / "run.json"
    inventory = json.loads(inventory_path.read_text())
    differences = validate_config(spec, config, inventory["config"])
    model = small_muon_model.build_small_muon_model(spec.family, config, state["model"]).to(device).eval()
    chunks = []
    max_error = 0.0
    for start in range(0, 256, BATCH_SIZE):
        inputs = {name: value[start : start + BATCH_SIZE].to(device) for name, value in context.items()}
        padding = torch.zeros(BATCH_SIZE, dtype=torch.long, device=device)
        action_indices = model.codec.quantize(features.stack_actions(inputs))
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            tokens = model.context_tokens(inputs, action_indices)
            traced, attention = trace_last_attention(model.trunk, tokens, padding)
            reference = model.trunk.forward_dense(tokens, padding)
        error = float((traced.float() - reference.float()).abs().max().item())
        max_error = max(max_error, error)
        if not math.isfinite(error) or error > 1e-5 or not torch.isfinite(reference).all():
            raise RuntimeError(f"{spec.run_id}: traced hidden state failed parity, max error {error}")
        if attention.dtype != torch.float32 or tuple(attention.shape) != (BATCH_SIZE, 8, spec.heads, 128):
            raise ValueError(f"{spec.run_id}: unexpected attention dtype or shape")
        chunks.append(attention.cpu().numpy())
    weights = np.concatenate(chunks, axis=0)
    result = {
        "run_id": spec.run_id,
        "label": spec.label,
        "family": spec.family,
        "parameters": spec.parameters,
        "width": spec.width,
        "checkpoint_step": 16384,
        "checkpoint_sha256": spec.sha256,
        "checkpoint_path": str(path.resolve()),
        "resolved_config": config,
        "input_profile": asdict(model.input_profile),
        "wandb_snapshot_sha256": small_muon_inputs.sha256_file(inventory_path),
        "config_metadata_differences": differences,
        "trace_max_abs_error": max_error,
        **summarize_attention(weights),
    }
    del model, state
    gc.collect()
    torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "run_id",
                    "mean_max_weight",
                    "mean_entropy_nats",
                    "mean_effective_frames",
                    "trace_max_abs_error",
                )
            }
        ),
        flush=True,
    )
    return result, weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--inventory-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--historical-source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    input_path = args.input_root / "inputs.pt"
    if small_muon_inputs.sha256_file(input_path) != INPUT_SHA256:
        raise ValueError("The prepared input package hash differs")
    if not torch.cuda.is_available():
        raise RuntimeError("This protocol requires the local CUDA GPU; no CPU fallback")
    historical_paths = {}
    for family, (relative_path, expected_hash, _) in HISTORICAL_SOURCES.items():
        historical_path = args.historical_source_root / family / relative_path
        if small_muon_inputs.sha256_file(historical_path) != expected_hash:
            raise ValueError(f"{family}: historical source hash differs")
        historical_paths[family] = historical_path
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=False)
    study = Path(__file__).resolve().parent
    repository = study.parents[1]
    source_names = (
        "small_muon_attention.py",
        "small_muon_inputs.py",
        "small_muon_model.py",
        "small_muon_inventory.py",
        "small_muon_attention_plan.md",
    )
    source_paths = [study / name for name in source_names]
    source_paths.extend(sorted((repository / "tests").glob("test_mechinterp_small_muon_*.py")))
    source_paths.extend(
        repository / name
        for name in (
            "hal/training/trunk.py",
            "hal/training/scoring.py",
            "hal/training/features.py",
            "hal/training/dataloader.py",
            "hal/training/ego_stats.py",
            "hal/data/mds.py",
            "hal/data/feature_stats.py",
            "hal/wire.py",
        )
    )
    with tarfile.open(output / "measurement-source.tar", "x") as archive:
        for source_path in source_paths:
            archive.add(source_path, arcname=str(source_path.relative_to(repository)))
        for family, source_path in historical_paths.items():
            archive.add(source_path, arcname=f"historical_source/{family}/{HISTORICAL_SOURCES[family][0]}")
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    package = torch.load(input_path, map_location="cpu", weights_only=True)
    if set(package) != {"context_features", "metadata_json"}:
        raise ValueError("Unexpected prepared input package fields")
    context = package["context_features"]
    metadata = json.loads(package["metadata_json"])
    if small_muon_inputs.hash_context(context) != metadata["prepared_context_sha256"]:
        raise ValueError("The prepared context tensor hash differs")
    models = []
    arrays = {}
    for spec in CHECKPOINTS:
        result, weights = measure_checkpoint(
            spec, args.checkpoint_root, args.inventory_root, context, torch.device("cuda")
        )
        models.append(result)
        arrays[spec.run_id] = weights
    array_path = output / "last-query-attention.npz"
    with array_path.open("xb") as array_file:
        np.savez_compressed(array_file, **arrays)
    payload = {
        "schema_version": 1,
        "scope": {
            "replays": 256,
            "context_length": 128,
            "query_token_position": 127,
            "example_index": 0,
            "averaging": "Mean of row statistics over replay, layer, and head; example arrays are not averaged",
            "example": metadata["windows"][0],
        },
        "models": models,
        "inputs": {
            "package_sha256": INPUT_SHA256,
            "metadata_sha256": small_muon_inputs.sha256_file(args.input_root / "inputs.json"),
            "context_sha256": metadata["prepared_context_sha256"],
            "stats_sha256": metadata["stats_sha256"],
        },
        "source": {
            "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "archive_sha256": small_muon_inputs.sha256_file(output / "measurement-source.tar"),
            "files": {str(path.relative_to(repository)): small_muon_inputs.sha256_file(path) for path in source_paths},
            "historical_sources": {
                family: {"path": path, "sha256": sha256, "matching_git_sha": git_sha}
                for family, (path, sha256, git_sha) in HISTORICAL_SOURCES.items()
            },
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "gpu": torch.cuda.get_device_name(),
            "seed": SEED,
            "batch_size": BATCH_SIZE,
        },
        "precision": {
            "projection": "BF16 autocast",
            "qk_scores_and_softmax": "FP32",
            "statistics": "float64",
            "tf32": False,
            "mode": "eval, inference_mode",
            "trace_parity_absolute_tolerance": 1e-5,
        },
        "probabilities": {
            "path": array_path.name,
            "sha256": small_muon_inputs.sha256_file(array_path),
            "dtype": "float32",
        },
    }
    with (output / "attention.json").open("x") as target:
        json.dump(payload, target, indent=2, sort_keys=True, allow_nan=False)
        target.write("\n")


if __name__ == "__main__":
    main()
