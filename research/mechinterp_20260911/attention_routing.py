"""Test whether O50's soft attention is causally close to hard routing."""

import argparse
import hashlib
import json
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Literal
from typing import cast

import numpy as np
import torch
from torch import Tensor

from hal.inference import o50_model
from hal.training import features
from hal.training import trunk
from research.mechinterp_20260911 import o50_probe
from research.mechinterp_20260911 import tail_likelihood

RoutingMode = Literal["soft", "argmax_key", "current_key"]
ROUTING_MODES: tuple[RoutingMode, ...] = ("soft", "argmax_key", "current_key")
SEED = 20_260_914
_FINAL_CHECKPOINT_NAMES = {
    "p1fyyp1z-step-0131072.model.pt",
    "tgpuo1be-step-0131072.model.pt",
}
_FINAL_CHECKPOINT_RUNS = {
    "p1fyyp1z-step-0131072.model.pt": "p1fyyp1z",
    "tgpuo1be-step-0131072.model.pt": "tgpuo1be",
}
_FINAL_CHECKPOINT_STEP = 131_071
_FIXED_COHORT_SHA256 = "ec61e97e332672a4e4e2fdc438e208a281a4609e10f7eded72368eccac323437"


def routed_trunk(
    model: o50_model.O50Model,
    tokens: Tensor,
    context_padding: Tensor,
    mode: RoutingMode,
) -> Tensor:
    """Run the trunk, replacing only each layer's final-query routing."""
    if mode not in ROUTING_MODES:
        raise ValueError(f"unknown routing mode {mode!r}")
    state = tokens
    batch, length, _ = state.shape
    mask = trunk.dense_mask(context_padding, length, model.trunk.attn_window)
    for raw_block in model.trunk.blocks:
        block = cast(trunk.Block, raw_block)
        normalized = trunk.rmsnorm(state)
        attention_update = block.attn(normalized, mask, context_padding)
        if mode == "soft":
            state = state + block.attn_scale * attention_update
            state = state + block.mlp_scale * block.mlp(trunk.rmsnorm(state))
            continue

        query, key, value = block.attn.c_attn(normalized).split(block.attn.d_model, dim=-1)
        shape = (batch, length, block.attn.n_heads, block.attn.head_dim)
        query = query.view(shape)
        key = key.view(shape)
        value = value.view(shape).transpose(1, 2)
        cosine, sine = block.attn.rotary(query)
        query = trunk.apply_rotary_emb(query, cosine, sine).transpose(1, 2)
        key = trunk.apply_rotary_emb(key, cosine, sine).transpose(1, 2)
        if mode == "current_key":
            selected = value[:, :, -1]
        else:
            with torch.autocast(device_type=query.device.type, enabled=False):
                scores = torch.einsum(
                    "bhd,bhkd->bhk",
                    query[:, :, -1].float(),
                    key.float(),
                ) / math.sqrt(block.attn.head_dim)
                scores = scores.masked_fill(~mask[:, 0, -1, None, :], float("-inf"))
                selected_indices = scores.argmax(dim=-1)
            selected = torch.gather(
                value,
                2,
                selected_indices[:, :, None, None].expand(-1, -1, 1, block.attn.head_dim),
            ).squeeze(2)
        attention_update = attention_update.clone()
        attention_update[:, -1] = block.attn.c_proj(selected.reshape(batch, block.attn.d_model))
        state = state + block.attn_scale * attention_update
        state = state + block.mlp_scale * block.mlp(trunk.rmsnorm(state))
    return trunk.rmsnorm(state)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a mapping with string keys")
    return cast(Mapping[str, object], value)


def _tensor_bytes(value: object, label: str) -> bytes:
    if not isinstance(value, Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
        raise TypeError(f"{label} must be a one-dimensional uint8 tensor")
    return value.numpy().tobytes()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_cohort_sha256(actual: str) -> None:
    if actual != _FIXED_COHORT_SHA256:
        raise ValueError("routing probe cohort differs from the fixed 256-replay cohort")


def _validate_checkpoint_identity(checkpoint: Mapping[str, object], expected_run_id: str) -> None:
    run_id = checkpoint.get("wandb_id")
    if run_id != expected_run_id or checkpoint.get("checkpoint_step") != _FINAL_CHECKPOINT_STEP:
        raise ValueError("routing probe checkpoint is not a preselected final Muon or AdamW export")


def _paired_summary(delta: Tensor) -> dict[str, object]:
    values = delta.double().cpu().numpy()
    generator = np.random.default_rng(SEED)
    indices = generator.integers(0, len(values), size=(20_000, len(values)))
    means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "bootstrap_95": [float(value) for value in np.quantile(means, (0.025, 0.975))],
        "positive_replays": int((values > 0).sum()),
    }


def _greedy_actions(model: o50_model.O50Model, hidden: Tensor, observed: Tensor) -> Tensor:
    rows = hidden.shape[0]
    forced = torch.zeros(rows, 4, o50_model.CONTROLLER_GROUP_COUNT, dtype=torch.long, device=hidden.device)
    force_mask = torch.zeros(rows, 4, dtype=torch.bool, device=hidden.device)
    uniforms = torch.full((4, o50_model.CONTROLLER_GROUP_COUNT, rows), 0.5, device=hidden.device)
    return model.temporal.sample_conditioned(
        hidden,
        observed,
        forced,
        force_mask,
        uniforms,
        argmax=True,
    )


def _treatment_report(
    baseline_loss: Tensor,
    baseline_logits: Mapping[str, Tensor],
    baseline_greedy: Tensor,
    treatment_loss: Tensor,
    treatment_logits: Mapping[str, Tensor],
    treatment_greedy: Tensor,
) -> dict[str, object]:
    baseline_bits = baseline_loss.sum(dim=-1).mean(dim=-1) / math.log(2)
    treatment_bits = treatment_loss.sum(dim=-1).mean(dim=-1) / math.log(2)
    delta_bits = treatment_bits - baseline_bits
    baseline_tail_bits = baseline_loss[:, 2:].sum(dim=-1).mean(dim=-1) / math.log(2)
    treatment_tail_bits = treatment_loss[:, 2:].sum(dim=-1).mean(dim=-1) / math.log(2)
    tail_delta_bits = treatment_tail_bits - baseline_tail_bits
    js = o50_probe.group_js_divergence(baseline_logits, treatment_logits)
    greedy_changes = (treatment_greedy != baseline_greedy).float()
    return {
        "joint_nll_bits_per_predicted_frame": float(treatment_bits.mean()),
        "paired_joint_nll_delta_bits_per_predicted_frame": _paired_summary(delta_bits),
        "per_replay_joint_nll_delta_bits_per_predicted_frame": delta_bits.tolist(),
        "tail_given_expert_prefix_bits_per_executed_frame": float(treatment_tail_bits.mean()),
        "paired_tail_nll_delta_bits_per_executed_frame": _paired_summary(tail_delta_bits),
        "per_replay_tail_nll_delta_bits_per_executed_frame": tail_delta_bits.tolist(),
        "output_js_nats": {
            "group_mean": js.mean(dim=0).tolist(),
            "mean": float(js.mean()),
            "per_replay_group": js.tolist(),
        },
        "greedy_change_rate": {
            "group_horizon": greedy_changes.mean(dim=0).tolist(),
            "mean": float(greedy_changes.mean()),
            "per_replay": greedy_changes.mean(dim=(1, 2)).tolist(),
        },
    }


def run_checkpoint(
    package: Mapping[str, object],
    checkpoint_path: Path,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    reference_config = o50_model.O50Config.from_json(_tensor_bytes(package["reference_config"], "reference config"))
    stats = _mapping(json.loads(_tensor_bytes(package["stats_payload"], "statistics")), "statistics")
    player_codes = _tensor_bytes(package["player_codes"], "player codes") or None
    context_raw = _mapping(package["context_features"], "context features")
    if any(not isinstance(value, Tensor) for value in context_raw.values()):
        raise TypeError("context feature values must be tensors")
    contexts = cast(Mapping[str, Tensor], context_raw)
    targets = package["target"]
    if not isinstance(targets, Tensor) or targets.shape != (256, 4, 14):
        raise ValueError("routing probe requires the fixed 256-replay cohort")
    model, config, checkpoint = o50_probe.load_checkpoint_model(
        checkpoint_path,
        reference_config=reference_config,
        stats_payload=stats,
        player_codes=player_codes,
        device=device,
    )
    expected_run_id = _FINAL_CHECKPOINT_RUNS.get(checkpoint_path.name)
    if expected_run_id is None:
        raise ValueError("routing probe checkpoint filename is not preselected")
    _validate_checkpoint_identity(checkpoint, expected_run_id)

    loss_chunks: dict[RoutingMode, list[Tensor]] = {mode: [] for mode in ROUTING_MODES}
    logit_chunks: dict[RoutingMode, dict[str, list[Tensor]]] = {
        mode: {name: [] for name in o50_model.CONTROLLER_GROUP_NAMES} for mode in ROUTING_MODES
    }
    greedy_chunks: dict[RoutingMode, list[Tensor]] = {mode: [] for mode in ROUTING_MODES}
    soft_parity_max_error = 0.0
    for start in range(0, 256, batch_size):
        stop = min(start + batch_size, 256)
        context = {name: value[start:stop].to(device) for name, value in contexts.items()}
        context["ego_player_id"] = torch.zeros_like(context["ego_player_id"])
        context_actions = model.codec.quantize(features.stack_actions(context))
        target = model.codec.quantize(targets[start:stop].to(device))
        observed = context_actions[:, -1]
        padding = torch.zeros(stop - start, dtype=torch.long, device=device)
        with torch.inference_mode(), o50_model.amp_context(config, device):
            tokens = model.context_tokens(context, context_actions)
            reference_soft = model.forward_dense(context, padding, context_actions)
            for mode in ROUTING_MODES:
                hidden = routed_trunk(model, tokens, padding, mode)
                if mode == "soft":
                    soft_parity_max_error = max(
                        soft_parity_max_error,
                        float((hidden - reference_soft).abs().max()),
                    )
                logits = tail_likelihood.sequential_teacher_logits(model, hidden, observed, target)
                loss = o50_probe.horizon_group_nll(logits, target)
                greedy = _greedy_actions(model, hidden, observed)
                loss_chunks[mode].append(loss.cpu())
                greedy_chunks[mode].append(greedy.cpu())
                for name in o50_model.CONTROLLER_GROUP_NAMES:
                    logit_chunks[mode][name].append(logits[name].float().cpu())
    if soft_parity_max_error != 0:
        raise RuntimeError(f"functional soft trunk differs from forward_dense by {soft_parity_max_error}")

    losses = {mode: torch.cat(chunks) for mode, chunks in loss_chunks.items()}
    logits = {
        mode: {name: torch.cat(chunks) for name, chunks in groups.items()} for mode, groups in logit_chunks.items()
    }
    greedy = {mode: torch.cat(chunks) for mode, chunks in greedy_chunks.items()}
    baseline_bits = losses["soft"].sum(dim=-1).mean(dim=-1) / math.log(2)
    baseline_tail_bits = losses["soft"][:, 2:].sum(dim=-1).mean(dim=-1) / math.log(2)
    return {
        "checkpoint": checkpoint,
        "baseline": {
            "joint_nll_bits_per_predicted_frame": float(baseline_bits.mean()),
            "per_replay_joint_nll_bits_per_predicted_frame": baseline_bits.tolist(),
            "tail_given_expert_prefix_bits_per_executed_frame": float(baseline_tail_bits.mean()),
            "per_replay_tail_given_expert_prefix_bits_per_executed_frame": baseline_tail_bits.tolist(),
        },
        "treatments": {
            mode: _treatment_report(
                losses["soft"],
                logits["soft"],
                greedy["soft"],
                losses[mode],
                logits[mode],
                greedy[mode],
            )
            for mode in ("argmax_key", "current_key")
        },
        "controls": {"soft_reconstruction_max_abs_error": soft_parity_max_error},
    }


def projection_shape_control(
    package: Mapping[str, object],
    checkpoint_path: Path,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    """Compare short and public-shape projections on one fixed batch."""
    reference_config = o50_model.O50Config.from_json(_tensor_bytes(package["reference_config"], "reference config"))
    stats = _mapping(json.loads(_tensor_bytes(package["stats_payload"], "statistics")), "statistics")
    player_codes = _tensor_bytes(package["player_codes"], "player codes") or None
    context_raw = _mapping(package["context_features"], "context features")
    if any(not isinstance(value, Tensor) for value in context_raw.values()):
        raise TypeError("context feature values must be tensors")
    contexts = cast(Mapping[str, Tensor], context_raw)
    model, config, checkpoint = o50_probe.load_checkpoint_model(
        checkpoint_path,
        reference_config=reference_config,
        stats_payload=stats,
        player_codes=player_codes,
        device=device,
    )
    expected_run_id = _FINAL_CHECKPOINT_RUNS.get(checkpoint_path.name)
    if expected_run_id is None:
        raise ValueError("projection control checkpoint filename is not preselected")
    _validate_checkpoint_identity(checkpoint, expected_run_id)

    context = {name: value[:batch_size].to(device) for name, value in contexts.items()}
    context["ego_player_id"] = torch.zeros_like(context["ego_player_id"])
    context_actions = model.codec.quantize(features.stack_actions(context))
    padding = torch.zeros(batch_size, dtype=torch.long, device=device)
    errors: dict[str, list[float]] = {mode: [] for mode in ("argmax_key", "current_key")}
    with torch.inference_mode(), o50_model.amp_context(config, device):
        state = model.context_tokens(context, context_actions)
        length = state.shape[1]
        mask = trunk.dense_mask(padding, length, model.trunk.attn_window)
        for raw_block in model.trunk.blocks:
            block = cast(trunk.Block, raw_block)
            normalized = trunk.rmsnorm(state)
            query, key, value = block.attn.c_attn(normalized).split(block.attn.d_model, dim=-1)
            shape = (batch_size, length, block.attn.n_heads, block.attn.head_dim)
            query = query.view(shape)
            key = key.view(shape)
            value = value.view(shape).transpose(1, 2)
            cosine, sine = block.attn.rotary(query)
            query = trunk.apply_rotary_emb(query, cosine, sine).transpose(1, 2)
            key = trunk.apply_rotary_emb(key, cosine, sine).transpose(1, 2)
            with torch.autocast(device_type=query.device.type, enabled=False):
                scores = torch.einsum(
                    "bhd,bhkd->bhk",
                    query[:, :, -1].float(),
                    key.float(),
                ) / math.sqrt(block.attn.head_dim)
                scores = scores.masked_fill(~mask[:, 0, -1, None, :], float("-inf"))
                selected_indices = scores.argmax(dim=-1)
            selections = {
                "argmax_key": torch.gather(
                    value,
                    2,
                    selected_indices[:, :, None, None].expand(-1, -1, 1, block.attn.head_dim),
                ).squeeze(2),
                "current_key": value[:, :, -1],
            }
            for mode, selected in selections.items():
                selected_flat = selected.reshape(batch_size, block.attn.d_model)
                short = block.attn.c_proj(selected_flat)
                full_input = value.new_zeros(batch_size, length, block.attn.d_model)
                full_input[:, -1] = selected_flat
                shape_matched = block.attn.c_proj(full_input)[:, -1]
                errors[mode].append(float((short - shape_matched).abs().max()))
            state = block(state, mask, padding)
    return {
        "checkpoint": checkpoint,
        "batch_rows": batch_size,
        "per_layer_max_abs_error": errors,
        "max_abs_error": {mode: max(values) for mode, values in errors.items()},
        "comparison": "c_proj([B,D]) versus final row of c_proj([B,L,D]) for identical selected BF16 values",
    }


def run_projection_control(
    cohort_path: Path,
    checkpoint_paths: tuple[Path, Path],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    """Run the fixed-batch projection-shape numerical control."""
    started = time.monotonic()
    source_hashes = {
        "o50_probe_sha256": _sha256(Path(o50_probe.__file__)),
        "routing_probe_sha256": _sha256(Path(__file__)),
    }
    package = _mapping(torch.load(cohort_path, map_location="cpu", weights_only=True), "cohort package")
    cohort_sha256 = _sha256(cohort_path)
    _validate_cohort_sha256(cohort_sha256)
    return {
        "schema_version": 1,
        "kind": "routing_projection_shape_numerical_control",
        "cohort": {"path": str(cohort_path), "sha256": cohort_sha256},
        "identity": "ego_player_id masked to zero",
        "checkpoints": [
            projection_shape_control(package, checkpoint, device=device, batch_size=batch_size)
            for checkpoint in checkpoint_paths
        ],
        "device": {
            "batch_size": batch_size,
            "name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
        },
        "elapsed_seconds": time.monotonic() - started,
        "source": source_hashes,
    }


def run_probe(
    cohort_path: Path,
    checkpoint_paths: tuple[Path, Path],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    """Run the fixed two-checkpoint routing comparison."""
    started = time.monotonic()
    source_hashes = {
        "o50_probe_sha256": _sha256(Path(o50_probe.__file__)),
        "routing_probe_sha256": _sha256(Path(__file__)),
        "tail_likelihood_sha256": _sha256(Path(tail_likelihood.__file__)),
    }
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    package = _mapping(torch.load(cohort_path, map_location="cpu", weights_only=True), "cohort package")
    expected = {
        "bundle_reference_json",
        "context_features",
        "data_json",
        "player_codes",
        "reference_config",
        "stats_payload",
        "target",
    }
    if set(package) != expected:
        raise ValueError("cohort package fields differ from the fixed protocol")
    cohort_sha256 = _sha256(cohort_path)
    _validate_cohort_sha256(cohort_sha256)
    data = _mapping(json.loads(cast(str, package["data_json"])), "cohort data")
    return {
        "schema_version": 1,
        "protocol": {
            "name": "o50-final-query-hard-routing-v1",
            "modes": list(ROUTING_MODES),
            "identity": "ego_player_id masked to zero",
            "scope": "only the final query is rerouted at every trunk layer; earlier queries remain soft",
            "selection": "preselected final Muon and final AdamW checkpoints; not a gameplay-selection metric",
            "seed": SEED,
        },
        "cohort": {
            "path": str(cohort_path),
            "sha256": cohort_sha256,
            "data_sha256": hashlib.sha256(cast(str, package["data_json"]).encode()).hexdigest(),
            "windows": len(cast(list[object], data["windows"])),
        },
        "checkpoints": [
            run_checkpoint(package, checkpoint, device=device, batch_size=batch_size)
            for checkpoint in checkpoint_paths
        ],
        "device": {
            "batch_size": batch_size,
            "name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
        },
        "elapsed_seconds": time.monotonic() - started,
        "source": source_hashes,
        "limitations": [
            "The probe tests routing sensitivity on replay contexts, not the cause of training instability or return.",
            "Hard routing is an off-manifold activation intervention and recomputes downstream keys and values.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--projection-control", action="store_true")
    args = parser.parse_args()
    if len(args.checkpoint) != 2:
        raise ValueError("routing probe requires exactly two checkpoints")
    if {path.name for path in args.checkpoint} != _FINAL_CHECKPOINT_NAMES:
        raise ValueError("routing probe is restricted to the preselected final Muon and AdamW checkpoints")
    if not 1 <= args.batch_size <= 32:
        raise ValueError("batch size must be in [1, 32]")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    torch.set_float32_matmul_precision("highest")
    runner = run_projection_control if args.projection_control else run_probe
    result = runner(args.cohort, tuple(args.checkpoint), device=torch.device(args.device), batch_size=args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False, sort_keys=True) + "\n")
    print(f"Saved {args.output}; {result['elapsed_seconds']:.1f} seconds", flush=True)


if __name__ == "__main__":
    main()
