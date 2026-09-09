# %% [markdown]
# # O50 checkpoint forensics
#
# This notebook compares the 205M O50 reference run with its half-Muon-LR
# continuation. It reads immutable checkpoints and W&B history, performs
# no-step held-out probes, and writes a separate, immutable R2 analysis artifact.
# It never initializes or writes a W&B run.

# %%
from __future__ import annotations

import functools
import gc
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any
from typing import Final
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tyro
import wandb
from botocore.exceptions import ClientError
from streaming.base.util import clean_stale_shared_memory
from torch import Tensor
from torch import nn

from hal import r2
from hal import streams
from hal.scripts.h2h import import_experiment
from hal.training import returns as returns_lib
from hal.training.checkpoints import checkpoint_sha256
from hal.training.dataloader import make_loader
from hal.training.features import ITEM_PLAYER_COLUMNS
from hal.training.features import ITEM_PLAYER_PROJECTION
from hal.training.features import AWRBatch
from hal.training.features import TrainBatch
from hal.training.muon import muon_update
from hal.training.player_identity import ReplayPlayerLookup
from hal.training.trunk import apply_rotary_emb

_SCHEMA_VERSION: Final[int] = 1
_EXPERIMENT_PATH: Final[str] = "experiments/050_scaled_temporal_awr.py"
_EXPERIMENT_SHA256: Final[str] = "72d2166ac00fb78e4c6c080b0b15bd5d470e15f86c14335f490444347e4544be"
_REFERENCE_WANDB_ID: Final[str] = "p1fyyp1z"
_TREATMENT_RUN: Final[str] = "o50-p1fyyp1z-u24576-muon-half"
_TREATMENT_WANDB_ID: Final[str] = "7dgv85x8"
_FORK_UPDATE: Final[int] = 24_576
_CHECKPOINT_EVERY: Final[int] = 8_192
_NOT_FOUND: Final[frozenset[str]] = frozenset({"404", "NoSuchKey"})


@dataclass(frozen=True, slots=True)
class Args:
    """Inputs for the immutable checkpoint analysis."""

    analysis_id: str
    """Unique artifact name below analysis/o50-checkpoint-forensics/v1/."""
    through_update: int = 65_536
    probe_batches: int = 64
    probe_batch_size: int = 128
    activation_probe_batches: int = 8
    head_probe_batches: int = 8
    sentinel_parameter_count: int = 12
    cache_limit: str = "64gb"
    cache_dir: str = "/tmp/o50-checkpoint-forensics-cache"
    output_dir: str = "/tmp/o50-checkpoint-forensics-output"
    reference_run: str = ""
    """Optional reference R2 run name. Empty derives it from the treatment lineage."""
    treatment_run: str = _TREATMENT_RUN
    reference_wandb_id: str = _REFERENCE_WANDB_ID
    treatment_wandb_id: str = _TREATMENT_WANDB_ID
    wandb_entity: str = "ericyuegu"
    wandb_project: str = "hal"
    coordinate_sample_size: int = 100_000
    upload: bool = False

    @property
    def artifact_prefix(self) -> str:
        return f"analysis/o50-checkpoint-forensics/v{_SCHEMA_VERSION}/{self.analysis_id}"


@dataclass(frozen=True, slots=True)
class CheckpointRef:
    arm: str
    run_name: str
    update: int
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class UpdateResult:
    delta: Tensor
    direction: Tensor
    next_first_moment: Tensor
    next_second_moment: Tensor | None


@dataclass(frozen=True, slots=True)
class ProbeReplayLabels:
    """Combine lazy returns with frame-length player identity labels."""

    returns: Any
    players: ReplayPlayerLookup

    def __call__(self, compact: Mapping[str, object]) -> dict[str, np.ndarray]:
        labels = self.returns(compact)
        labels.update(self.players(compact))
        return labels


class ScalarAccumulator:
    """Aggregate finite scalars without retaining per-batch tensors."""

    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.maximum = -math.inf

    def add(self, value: float) -> None:
        if not math.isfinite(value):
            raise FloatingPointError(f"probe produced a non-finite scalar: {value}")
        self.count += 1
        self.total += value
        self.maximum = max(self.maximum, value)

    def row(self) -> dict[str, float | int]:
        if not self.count:
            raise RuntimeError("cannot summarize an empty accumulator")
        return {"batches": self.count, "mean": self.total / self.count, "max": self.maximum}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_args(args: Args) -> None:
    if not args.analysis_id or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in args.analysis_id
    ):
        raise ValueError("analysis_id must contain only lowercase letters, digits, hyphens, and underscores")
    if args.through_update < _FORK_UPDATE or args.through_update % _CHECKPOINT_EVERY:
        raise ValueError(f"through_update must be a multiple of {_CHECKPOINT_EVERY} at or after {_FORK_UPDATE}")
    if args.probe_batches < 1 or args.probe_batch_size < 1:
        raise ValueError("probe_batches and probe_batch_size must be positive")
    if not 1 <= args.activation_probe_batches <= args.probe_batches:
        raise ValueError("activation_probe_batches must be between 1 and probe_batches")
    if not 1 <= args.head_probe_batches <= args.probe_batches:
        raise ValueError("head_probe_batches must be between 1 and probe_batches")
    if args.sentinel_parameter_count < 1:
        raise ValueError("sentinel_parameter_count must be positive")
    if args.coordinate_sample_size < 1:
        raise ValueError("coordinate_sample_size must be positive")


def _experiment() -> ModuleType:
    source = Path(_EXPERIMENT_PATH)
    actual = _sha256(source)
    if actual != _EXPERIMENT_SHA256:
        raise RuntimeError(f"O50 source SHA-256 {actual} != frozen {_EXPERIMENT_SHA256}")
    return import_experiment(_EXPERIMENT_PATH)


def checkpoint_updates(through_update: int) -> tuple[int, ...]:
    """Return every common immutable checkpoint from the fork through the limit."""
    if through_update < _FORK_UPDATE or through_update % _CHECKPOINT_EVERY:
        raise ValueError("checkpoint limit is not an O50 milestone")
    return tuple(range(_FORK_UPDATE, through_update + 1, _CHECKPOINT_EVERY))


def _download_checkpoint(run_name: str, update: int, root: Path) -> CheckpointRef:
    destination = root / run_name / f"step-{update:07d}.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    key = f"runs/{run_name}/checkpoints/step-{update:07d}.pt"
    if not destination.is_file():
        temporary = destination.with_suffix(".pt.tmp")
        temporary.unlink(missing_ok=True)
        r2.client().download_file(r2.bucket(), key, str(temporary))
        temporary.replace(destination)
    digest = checkpoint_sha256(destination)
    return CheckpointRef("", run_name, update, destination, digest)


def download_checkpoints(args: Args) -> tuple[list[CheckpointRef], str]:
    """Download only requested immutable checkpoints, never a latest pointer."""
    cache = Path(args.cache_dir) / "checkpoints"
    treatment_fork = _download_checkpoint(args.treatment_run, _FORK_UPDATE, cache)
    fork_state = _load_raw(treatment_fork)
    treatment_cfg = fork_state["cfg"]
    reference_run = treatment_cfg.get("parent_run_name")
    if not isinstance(reference_run, str) or not reference_run:
        raise ValueError("treatment fork checkpoint has no parent R2 run name")
    if args.reference_run and args.reference_run != reference_run:
        raise ValueError(f"configured reference run {args.reference_run!r} != child lineage {reference_run!r}")
    if fork_state.get("wandb_id") != args.treatment_wandb_id:
        raise ValueError("treatment fork checkpoint has the wrong W&B ID")
    if treatment_cfg.get("parent_wandb_id") != args.reference_wandb_id:
        raise ValueError("treatment fork checkpoint has the wrong parent W&B ID")

    refs: list[CheckpointRef] = []
    for arm, run_name in (("reference", reference_run), ("half_muon", args.treatment_run)):
        for update in checkpoint_updates(args.through_update):
            downloaded = (
                treatment_fork
                if arm == "half_muon" and update == _FORK_UPDATE
                else _download_checkpoint(run_name, update, cache)
            )
            refs.append(CheckpointRef(arm, run_name, update, downloaded.path, downloaded.sha256))
            print(f"[checkpoint] {arm} {update:,} {downloaded.sha256[:12]}", flush=True)
    return refs, reference_run


def _load_raw(ref: CheckpointRef, device: torch.device | str = "cpu") -> dict[str, Any]:
    state = torch.load(ref.path, map_location=device, weights_only=False)
    actual = int(state["step"]) + 1
    if actual != ref.update:
        raise ValueError(f"{ref.path} contains update {actual}, expected {ref.update}")
    return cast(dict[str, Any], state)


def _load_model_optimizer(
    experiment: ModuleType,
    ref: CheckpointRef,
    device: torch.device,
) -> tuple[nn.Module, Any, Any, dict[str, Any]]:
    state = _load_raw(ref, device)
    cfg = experiment.config_from_state(state["cfg"])
    experiment.validate_config(cfg)
    encoded = state["model"].get("player_code_bytes")
    if not isinstance(encoded, Tensor) or not encoded.numel():
        raise ValueError(f"{ref.path} has no embedded player vocabulary")
    vocabulary = experiment.PlayerVocabulary(experiment.decode_player_codes(encoded.detach().cpu().numpy().tobytes()))
    model = experiment.GPT(cfg, vocabulary).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    optimizer = experiment.make_optimizer(model, cfg)
    optimizer.load_state_dict(state["opt"])
    return model, optimizer, cfg, state


def _recursive_equal(left: Any, right: Any, path: str) -> None:
    if isinstance(left, Tensor) and isinstance(right, Tensor):
        if left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left.cpu(), right.cpu()):
            raise ValueError(f"fork mismatch at {path}")
        return
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            raise ValueError(f"fork mapping keys differ at {path}")
        for key in left:
            _recursive_equal(left[key], right[key], f"{path}/{key}")
        return
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            raise ValueError(f"fork sequence lengths differ at {path}")
        for index, (left_value, right_value) in enumerate(zip(left, right, strict=True)):
            _recursive_equal(left_value, right_value, f"{path}/{index}")
        return
    if left != right:
        raise ValueError(f"fork values differ at {path}: {left!r} != {right!r}")


def validate_fork(reference: CheckpointRef, treatment: CheckpointRef) -> dict[str, Any]:
    """Prove the child starts from the same weights and optimizer moments."""
    if reference.update != _FORK_UPDATE or treatment.update != _FORK_UPDATE:
        raise ValueError("fork validation needs both 24,576 checkpoints")
    parent = _load_raw(reference)
    child = _load_raw(treatment)
    _recursive_equal(parent["model"], child["model"], "model")
    _recursive_equal(parent["opt"]["state"], child["opt"]["state"], "optimizer/state")
    parent_groups = parent["opt"]["param_groups"]
    child_groups = child["opt"]["param_groups"]
    if len(parent_groups) != len(child_groups):
        raise ValueError("fork optimizer group counts differ")
    muon_ratios: list[float] = []
    for index, (parent_group, child_group) in enumerate(zip(parent_groups, child_groups, strict=True)):
        allowed = {"lr", "initial_lr"}
        _recursive_equal(
            {key: value for key, value in parent_group.items() if key not in allowed},
            {key: value for key, value in child_group.items() if key not in allowed},
            f"optimizer/group/{index}",
        )
        ratio = float(child_group["lr"]) / float(parent_group["lr"])
        if parent_group["use_muon"]:
            muon_ratios.append(ratio)
            if not math.isclose(ratio, 0.5, rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"Muon group {index} has child/parent LR ratio {ratio}")
        elif not math.isclose(ratio, 1.0, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"AdamW group {index} changed LR at the fork")
    return {"model_equal": True, "optimizer_state_equal": True, "muon_lr_ratios": muon_ratios}


def _rms(tensor: Tensor) -> float:
    return float(tensor.detach().float().square().mean().sqrt())


def _sample_flat(tensor: Tensor, limit: int) -> Tensor:
    flat = tensor.detach().float().reshape(-1)
    if flat.numel() <= limit:
        return flat
    stride = math.ceil(flat.numel() / limit)
    return flat[::stride][:limit]


def _percentiles(values: Tensor) -> dict[str, float]:
    finite = values[torch.isfinite(values)]
    if not finite.numel():
        return {"p99": math.inf, "p999": math.inf, "max": math.inf, "nonfinite_fraction": 1.0}
    quantiles = torch.quantile(finite, torch.tensor((0.99, 0.999), device=finite.device))
    return {
        "p99": float(quantiles[0]),
        "p999": float(quantiles[1]),
        "max": float(finite.max()),
        "nonfinite_fraction": 1.0 - finite.numel() / values.numel(),
    }


def _step_number(state: Mapping[str, Any]) -> int:
    value = state["step"]
    if isinstance(value, Tensor):
        return int(value.item())
    return int(value)


@torch.no_grad()
def hypothetical_update(
    parameter: Tensor,
    gradient: Tensor,
    state: Mapping[str, Any],
    group: Mapping[str, Any],
    *,
    reset_state: bool,
) -> UpdateResult:
    """Return the exact next O50 optimizer update without mutating inputs."""
    input_tensors = (parameter, gradient, *(value for value in state.values() if isinstance(value, Tensor)))
    before_versions = tuple(value._version for value in input_tensors)
    lr = float(group["lr"])
    weight_decay = float(group["weight_decay"])
    if group["use_muon"]:
        momentum = torch.zeros_like(parameter) if reset_state else cast(Tensor, state["momentum_buffer"]).clone()
        next_momentum = momentum.clone()
        grad_copy = gradient.clone()
        direction = muon_update(
            grad_copy,
            next_momentum,
            beta=float(group["momentum"]),
            ns_steps=5,
            nesterov=True,
            muon_scale_clamp_min_one=bool(group["muon_scale_clamp_min_one"]),
            logical_splits=int(group["logical_splits"]),
        ).reshape_as(parameter)
        second = None
    else:
        if group.get("update_clip_threshold") is not None:
            raise ValueError("O50 checkpoint unexpectedly enables per-tensor Adam update clipping")
        beta1, beta2 = cast(tuple[float, float], group["betas"])
        first = torch.zeros_like(parameter) if reset_state else cast(Tensor, state["exp_avg"]).clone()
        second = torch.zeros_like(parameter) if reset_state else cast(Tensor, state["exp_avg_sq"]).clone()
        first.lerp_(gradient, 1 - beta1)
        second.lerp_(gradient.square(), 1 - beta2)
        step = 1 if reset_state else _step_number(state) + 1
        direction = (first / (1 - beta1**step)) / ((second / (1 - beta2**step)).sqrt() + float(group["eps"]))
        next_momentum = first
    delta = -lr * (direction + weight_decay * parameter)
    if before_versions != tuple(value._version for value in input_tensors):
        raise RuntimeError("hypothetical update mutated a parameter, gradient, or optimizer-state input")
    return UpdateResult(delta, direction, next_momentum, second)


@torch.no_grad()
def _stored_direction(
    parameter: Tensor,
    state: Mapping[str, Any],
    group: Mapping[str, Any],
) -> Tensor:
    if group["use_muon"]:
        momentum = cast(Tensor, state["momentum_buffer"]).clone()
        return muon_update(
            momentum.clone(),
            momentum.clone(),
            beta=float(group["momentum"]),
            ns_steps=5,
            nesterov=True,
            muon_scale_clamp_min_one=bool(group["muon_scale_clamp_min_one"]),
            logical_splits=int(group["logical_splits"]),
        ).reshape_as(parameter)
    beta1, beta2 = cast(tuple[float, float], group["betas"])
    step = _step_number(state)
    first = cast(Tensor, state["exp_avg"]).float() / (1 - beta1**step)
    second = cast(Tensor, state["exp_avg_sq"]).float() / (1 - beta2**step)
    return first / (second.sqrt() + float(group["eps"]))


def _parameter_bindings(model: nn.Module, optimizer: Any) -> dict[int, tuple[str, nn.Parameter, Mapping[str, Any]]]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    bindings: dict[int, tuple[str, nn.Parameter, Mapping[str, Any]]] = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            name = names.get(id(parameter))
            if name is None:
                raise RuntimeError("optimizer contains an unnamed model parameter")
            bindings[id(parameter)] = (name, parameter, group)
    if len(bindings) != len(names):
        raise RuntimeError("optimizer does not cover every model parameter exactly once")
    return bindings


def _subsystem_names(experiment: ModuleType, model: nn.Module) -> dict[int, str]:
    return {
        id(parameter): subsystem
        for subsystem, parameters in experiment.parameter_subsystems(model).items()
        for parameter in parameters
    }


@torch.no_grad()
def checkpoint_parameter_rows(
    experiment: ModuleType,
    ref: CheckpointRef,
    device: torch.device,
    sample_size: int,
) -> list[dict[str, Any]]:
    """Measure weight and optimizer-state scale for every parameter."""
    model, optimizer, _cfg, _state = _load_model_optimizer(experiment, ref, device)
    del _state
    bindings = _parameter_bindings(model, optimizer)
    subsystems = _subsystem_names(experiment, model)
    rows: list[dict[str, Any]] = []
    for _parameter_id, (name, parameter, group) in bindings.items():
        state = optimizer.state[parameter]
        direction = _stored_direction(parameter, state, group)
        weight_rms = _rms(parameter)
        direction_rms = _rms(direction)
        lr = float(group["lr"])
        delta = -lr * (direction + float(group["weight_decay"]) * parameter)
        raw = _sample_flat(direction, sample_size).abs() / _sample_flat(parameter, sample_size).abs()
        floor = max(weight_rms * 1e-3, torch.finfo(torch.float32).tiny)
        stabilized = _sample_flat(direction, sample_size).abs() / _sample_flat(parameter, sample_size).abs().clamp_min(
            floor
        )
        raw_stats = _percentiles(raw)
        stabilized_stats = _percentiles(stabilized)
        row: dict[str, Any] = {
            "schema_version": _SCHEMA_VERSION,
            "arm": ref.arm,
            "run_name": ref.run_name,
            "update": ref.update,
            "parameter": name,
            "subsystem": subsystems[id(parameter)],
            "optimizer": "muon" if group["use_muon"] else "adamw",
            "elements": parameter.numel(),
            "weight_rms": weight_rms,
            "first_moment_rms": _rms(state["momentum_buffer"] if group["use_muon"] else state["exp_avg"]),
            "second_moment_rms": math.nan if group["use_muon"] else _rms(state["exp_avg_sq"]),
            "direction_rms": direction_rms,
            "direction_weight_rms_ratio": direction_rms / max(weight_rms, torch.finfo(torch.float32).tiny),
            "lr": lr,
            "weight_decay": float(group["weight_decay"]),
            "implied_update_rms": _rms(delta),
            "implied_update_weight_rms_ratio": _rms(delta) / max(weight_rms, torch.finfo(torch.float32).tiny),
            "relative_coordinate_zero_weight_fraction": float(
                (_sample_flat(parameter, sample_size) == 0).float().mean()
            ),
        }
        row.update({f"relative_coordinate_raw_{key}": value for key, value in raw_stats.items()})
        row.update({f"relative_coordinate_stabilized_{key}": value for key, value in stabilized_stats.items()})
        rows.append(row)
    del optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


@torch.no_grad()
def displacement_rows(
    left: CheckpointRef,
    right: CheckpointRef,
    *,
    comparison: str,
    parameter_names: frozenset[str],
) -> list[dict[str, Any]]:
    left_state = _load_raw(left)["model"]
    right_state = _load_raw(right)["model"]
    if set(left_state) != set(right_state):
        raise ValueError(f"model state keys differ for {comparison}")
    rows: list[dict[str, Any]] = []
    for name in sorted(left_state):
        if name not in parameter_names:
            continue
        left_tensor = left_state[name]
        right_tensor = right_state[name]
        if not isinstance(left_tensor, Tensor) or not left_tensor.is_floating_point():
            continue
        delta = right_tensor.float() - left_tensor.float()
        denominator = _rms(left_tensor)
        rows.append(
            {
                "schema_version": _SCHEMA_VERSION,
                "comparison": comparison,
                "arm": right.arm,
                "from_update": left.update,
                "to_update": right.update,
                "parameter": name,
                "displacement_rms": _rms(delta),
                "displacement_weight_rms_ratio": _rms(delta) / max(denominator, torch.finfo(torch.float32).tiny),
            }
        )
    return rows


def select_turning_updates(
    rows: Iterable[Mapping[str, Any]],
    *,
    through_update: int,
    fork_update: int = _FORK_UPDATE,
) -> tuple[int, ...]:
    """Select fork, worst gameplay LCB, and latest common checkpoint."""
    values: dict[int, float] = {}
    for row in rows:
        update = int(row["update"])
        value = row.get("net_stock_lcb")
        if fork_update <= update <= through_update and value is not None and math.isfinite(float(value)):
            numeric = float(value)
            if update in values and not math.isclose(values[update], numeric, rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"conflicting eval LCB values at update {update}")
            values[update] = numeric
    if not values:
        raise RuntimeError("W&B history has no closed-loop LCB in the checkpoint interval")
    worst = min(values, key=lambda update: values[update])
    return tuple(sorted({fork_update, worst, through_update}))


def sentinel_parameter_names(
    rows: Iterable[Mapping[str, Any]],
    *,
    arm: str,
    update: int,
    top_count: int,
) -> frozenset[str]:
    """Select state outliers and the full button-output path for batch probes."""
    matching = [row for row in rows if row["arm"] == arm and int(row["update"]) == update]
    if not matching:
        raise ValueError(f"no checkpoint parameter rows for {arm} at {update}")
    ranked = sorted(
        matching,
        key=lambda row: float(row["implied_update_weight_rms_ratio"]),
        reverse=True,
    )
    selected = {str(row["parameter"]) for row in ranked[:top_count]}
    button_fragments = (
        "temporal.group_condition.buttons.",
        "temporal.outputs.buttons.",
        "temporal.trunk_outputs.buttons.",
    )
    selected.update(
        str(row["parameter"])
        for row in matching
        if any(fragment in str(row["parameter"]) for fragment in button_fragments)
    )
    return frozenset(selected)


def _wandb_history(args: Args, wandb_id: str, arm: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    api = wandb.Api()
    run = api.run(f"{args.wandb_entity}/{args.wandb_project}/{wandb_id}")
    eval_rows: dict[int, dict[str, Any]] = {}
    eval_keys = ["global_step", "eval/checkpoint_step", "eval/net_stock_per_min", "eval/net_stock_lcb"]
    for raw in run.scan_history(keys=eval_keys, page_size=1000):
        checkpoint_step = raw.get("eval/checkpoint_step")
        if checkpoint_step is None:
            continue
        update = int(checkpoint_step)
        values = {
            "schema_version": _SCHEMA_VERSION,
            "arm": arm,
            "wandb_id": wandb_id,
            "update": update,
            "net_stock_per_min": raw.get("eval/net_stock_per_min"),
            "net_stock_lcb": raw.get("eval/net_stock_lcb"),
        }
        existing = eval_rows.get(update)
        if existing is not None:
            for key in ("net_stock_per_min", "net_stock_lcb"):
                if (
                    existing[key] is not None
                    and values[key] is not None
                    and not math.isclose(float(existing[key]), float(values[key]), rel_tol=0, abs_tol=1e-12)
                ):
                    raise ValueError(f"W&B run {wandb_id} has conflicting {key} at {update}")
        eval_rows[update] = {**(existing or {}), **{key: value for key, value in values.items() if value is not None}}

    update_rows: list[dict[str, Any]] = []
    if arm == "half_muon":
        prefixes = [
            f"diagnostics/optimizer/{subsystem}/{optimizer_name}"
            for subsystem in ("trunk", "temporal_decoder", "group_heads", "value_head", "other")
            for optimizer_name in ("muon", "adamw", "all")
        ]
        diagnostic_keys = [
            "global_step",
            *(f"{prefix}/{metric}" for prefix in prefixes for metric in ("update_rms", "update_parameter_rms_ratio")),
        ]
        for raw in run.scan_history(keys=diagnostic_keys, page_size=1000):
            if raw.get("global_step") is None:
                continue
            values = {key: raw.get(key) for key in diagnostic_keys[1:]}
            if not any(value is not None for value in values.values()):
                continue
            update_rows.append(
                {
                    "schema_version": _SCHEMA_VERSION,
                    "arm": arm,
                    "wandb_id": wandb_id,
                    "update": int(raw["global_step"]),
                    **values,
                }
            )
    return [eval_rows[key] for key in sorted(eval_rows)], update_rows


def _attach_awr_labels(
    windows: list[dict[str, Any]],
    batch: TrainBatch,
    *,
    context_length: int,
    return_column: str,
    valid_column: str,
) -> AWRBatch:
    next_frames = slice(1, context_length + 1)
    returns = np.stack([window[return_column] for window in windows])[:, next_frames]
    eligible = np.stack([window[valid_column] for window in windows])[:, next_frames]
    return AWRBatch(
        batch=batch,
        returns=torch.from_numpy(np.ascontiguousarray(returns)),
        eligible=torch.from_numpy(np.ascontiguousarray(eligible)).bool(),
    )


def make_lazy_probe_loader(
    experiment: ModuleType,
    cfg: Any,
    stats: Mapping[str, Any],
    args: Args,
) -> Iterable[AWRBatch]:
    """Build the bounded streaming validation path, not the training reservoir."""
    sidecar_path = Path(args.cache_dir) / "identity" / Path(cfg.player_sidecar_local).name
    sidecar_cfg = replace(cfg, player_sidecar_local=str(sidecar_path))
    sidecar = experiment.load_identity_sidecar(sidecar_cfg)
    player_lookup = ReplayPlayerLookup(sidecar.by_replay)
    labels = ProbeReplayLabels(
        returns=returns_lib.PolicyReturnLabels(
            player_lookup=player_lookup,
            gamma=cfg.awr.gamma,
            damage_shaping=cfg.awr.damage_shaping,
            win_reward=cfg.awr.win_reward,
            stock_value=cfg.awr.stock_value,
            suffix=cfg.awr.return_suffix,
        ),
        players=player_lookup,
    )
    projection = experiment.FeatureProjection(
        columns=ITEM_PLAYER_PROJECTION.columns | {cfg.awr.ego_return_column, cfg.awr.ego_return_valid_column},
        derive_spatial=ITEM_PLAYER_PROJECTION.derive_spatial,
    )
    lazy_sources = tuple(
        streams.StreamSource(
            name=source.name,
            remote=source.remote,
            local=Path(args.cache_dir).resolve() / "streams" / source.name,
        )
        for source in (streams.BY_NAME[name] for name in cfg.source_names)
    )
    return cast(
        Iterable[AWRBatch],
        make_loader(
            data_root=None,
            split=cfg.val_split,
            stats=cast(dict[str, Any], stats),
            L_ctx=cfg.arch.L_ctx,
            L_chunk=cfg.arch.sample_chunk_length,
            batch_size=args.probe_batch_size,
            seed=cfg.seed,
            sources=lazy_sources,
            cache_limit=args.cache_limit,
            shuffle_block_size=8192,
            shuffle_seed=cfg.seed,
            num_workers=0,
            predownload=max(args.probe_batch_size, 1024),
            schema_version=cfg.mds_schema_version,
            extra=ITEM_PLAYER_COLUMNS,
            projection=projection,
            replay_format="policy-world",
            replay_labels=labels,
            require_full_context=True,
            shuffle=True,
            batch_transform=functools.partial(
                _attach_awr_labels,
                context_length=cfg.arch.L_ctx,
                return_column=cfg.awr.ego_return_column,
                valid_column=cfg.awr.ego_return_valid_column,
            ),
        ),
    )


def _batch_identity(batch: AWRBatch) -> bytes:
    digest = hashlib.sha256()
    replay_ids = batch.batch.replay_ids
    if replay_ids is not None:
        digest.update(json.dumps(replay_ids, separators=(",", ":")).encode())
    for name in sorted(batch.batch.context.features):
        value = batch.batch.context.features[name]
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    digest.update(batch.batch.context.ctx_pad.numpy().tobytes())
    digest.update(batch.batch.target.numpy().tobytes())
    digest.update(batch.returns.numpy().tobytes())
    digest.update(batch.eligible.numpy().tobytes())
    return digest.digest()


def iter_probe_batches(loader: Iterable[AWRBatch], count: int) -> Iterable[AWRBatch]:
    """Yield a fixed count across deterministic lazy validation epochs."""
    produced = 0
    while produced < count:
        epoch_count = 0
        for batch in loader:
            yield batch
            produced += 1
            epoch_count += 1
            if produced == count:
                return
        if epoch_count == 0:
            raise RuntimeError("validation loader yielded no batches")


def _attention_head_means(qkv: Tensor, n_heads: int, mask: Tensor, rotary: Any) -> Tensor:
    batch, length, fused_width = qkv.shape
    d_model = fused_width // 3
    head_dim = d_model // n_heads
    query, key, _value = qkv.split(d_model, dim=-1)
    query = query.view(batch, length, n_heads, head_dim)
    key = key.view(batch, length, n_heads, head_dim)
    cosine, sine = rotary(query)
    query = apply_rotary_emb(query, cosine, sine).transpose(1, 2).float()
    key = apply_rotary_emb(key, cosine, sine).transpose(1, 2).float()
    scores = (query @ key.transpose(-2, -1)) * (head_dim**-0.5)
    keep = mask.expand(batch, 1, length, length)
    log_probabilities = F.log_softmax(scores.masked_fill(~keep, -torch.inf), dim=-1)
    probabilities = log_probabilities.exp()
    entropy = -(probabilities * log_probabilities.masked_fill(~keep, 0)).sum(dim=-1)
    legal_counts = keep.sum(dim=-1).expand(batch, n_heads, length)
    valid = legal_counts > 1
    normalized = entropy / legal_counts.clamp_min(2).log()
    denominator = valid.sum(dim=(0, 2)).clamp_min(1)
    return normalized.masked_fill(~valid, 0).sum(dim=(0, 2)) / denominator


@torch.no_grad()
def fixed_batch_rows(
    experiment: ModuleType,
    ref: CheckpointRef,
    fixed_batch: TrainBatch,
    baseline: Mapping[str, Tensor],
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model, _optimizer, cfg, _state = _load_model_optimizer(experiment, ref, device)
    del _state
    batch = fixed_batch.to(device)
    trunk_outputs: list[Tensor] = []
    temporal_outputs: list[Tensor] = []
    handles = []
    for block in model.trunk.blocks:
        handles.append(
            block.attn.c_attn.register_forward_hook(lambda _m, _i, output: trunk_outputs.append(output.detach()))
        )
    for block in model.temporal.blocks:
        handles.append(
            block.qkv.register_forward_hook(lambda _m, _i, output: temporal_outputs.append(output.detach()))
        )
    try:
        history, targets, valid = experiment.prepared_targets(model, batch)
        with experiment.amp_context(cfg, device):
            hidden = model.forward_dense(batch.context.features, batch.context.ctx_pad)
            hidden = hidden[:, cfg.arch.direct_loss_start :]
            logits = model.temporal.teacher_forced_logits_by_group(hidden, history, targets)
    finally:
        for handle in handles:
            handle.remove()

    policy_rows: list[dict[str, Any]] = []
    for group_index, name in enumerate(experiment.CONTROLLER_GROUP_NAMES):
        current_log = F.log_softmax(logits[name].float(), dim=-1)
        reference_log = baseline[name].to(device).float()
        if current_log.shape != reference_log.shape or not torch.equal(
            torch.isfinite(current_log), torch.isfinite(reference_log)
        ):
            raise ValueError(f"fixed policy support changed for {name}")
        legal = torch.isfinite(current_log)
        legal_count = legal.sum(dim=-1, keepdim=True)
        centered = (
            logits[name].float() - logits[name].float().masked_fill(~legal, 0).sum(dim=-1, keepdim=True) / legal_count
        )
        probabilities = current_log.exp().masked_fill(~legal, 0)
        entropy = -(probabilities * current_log.masked_fill(~legal, 0)).sum(dim=-1)
        target = targets[..., group_index]
        target_logit = logits[name].float().gather(-1, target[..., None]).squeeze(-1)
        competitor = logits[name].float().scatter(-1, target[..., None], -torch.inf).amax(dim=-1)
        kl = torch.where(legal, reference_log.exp() * (reference_log - current_log), 0).sum(dim=-1)
        for horizon_index, offset in enumerate(model.head_offsets):
            selected_rows = valid
            selected_logits = centered[:, :, horizon_index][legal[:, :, horizon_index]]
            selected_entropy = entropy[:, :, horizon_index][selected_rows]
            selected_margin = (target_logit - competitor)[:, :, horizon_index][selected_rows]
            selected_kl = kl[:, :, horizon_index][selected_rows]
            policy_rows.append(
                {
                    "schema_version": _SCHEMA_VERSION,
                    "arm": ref.arm,
                    "update": ref.update,
                    "group": name,
                    "offset": offset,
                    "centered_logit_rms": _rms(selected_logits),
                    "centered_logit_abs_p99": float(torch.quantile(selected_logits.abs(), 0.99)),
                    "centered_logit_abs_p999": float(torch.quantile(selected_logits.abs(), 0.999)),
                    "centered_logit_abs_max": float(selected_logits.abs().max()),
                    "entropy_nats": float(selected_entropy.mean()),
                    "target_margin_mean": float(selected_margin.mean()),
                    "target_margin_p01": float(torch.quantile(selected_margin, 0.01)),
                    "kl_from_fork_nats": float(selected_kl.mean()),
                }
            )

    attention_rows: list[dict[str, Any]] = []
    trunk_mask = experiment.dense_mask(batch.context.ctx_pad, cfg.arch.L_ctx, cfg.arch.attn_window)
    for layer, (qkv, block) in enumerate(zip(trunk_outputs, model.trunk.blocks, strict=True)):
        means = _attention_head_means(qkv, cfg.arch.n_heads, trunk_mask, block.attn.rotary)
        for head, value in enumerate(means):
            attention_rows.append(
                {
                    "schema_version": _SCHEMA_VERSION,
                    "arm": ref.arm,
                    "update": ref.update,
                    "stack": "trunk",
                    "layer": layer,
                    "head": head,
                    "normalized_entropy": float(value),
                }
            )
    temporal_mask = torch.ones(
        (1, 1, len(cfg.arch.head_offsets), len(cfg.arch.head_offsets)), dtype=torch.bool, device=device
    ).tril()
    for layer, (qkv, block) in enumerate(zip(temporal_outputs, model.temporal.blocks, strict=True)):
        means = _attention_head_means(qkv, cfg.arch.temporal_heads, temporal_mask, block.rotary)
        for head, value in enumerate(means):
            attention_rows.append(
                {
                    "schema_version": _SCHEMA_VERSION,
                    "arm": ref.arm,
                    "update": ref.update,
                    "stack": "temporal",
                    "layer": layer,
                    "head": head,
                    "normalized_entropy": float(value),
                }
            )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return policy_rows, attention_rows


class ActivationRecorder:
    """Collect activation RMS and maximum values for named leaf modules."""

    def __init__(self, model: nn.Module) -> None:
        self.enabled = True
        self.squares: dict[str, Tensor] = {}
        self.counts: dict[str, int] = defaultdict(int)
        self.maxima: dict[str, Tensor] = {}
        self.handles = []
        for name, module in model.named_modules():
            if name and isinstance(module, (nn.Linear, nn.Embedding)):
                self.handles.append(module.register_forward_hook(functools.partial(self._capture, name)))

    def _capture(self, name: str, _module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        if not self.enabled or not isinstance(output, Tensor):
            return
        value = output.detach().float()
        square = value.square().sum()
        maximum = value.abs().max()
        self.squares[name] = square if name not in self.squares else self.squares[name] + square
        self.counts[name] += value.numel()
        self.maxima[name] = maximum if name not in self.maxima else torch.maximum(self.maxima[name], maximum)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _probe_objective(
    experiment: ModuleType,
    model: nn.Module,
    batch: AWRBatch,
    cfg: Any,
    update: int,
) -> tuple[Tensor, Tensor, dict[str, Tensor], Tensor, Tensor, Tensor, dict[str, Tensor]]:
    history, targets, valid = experiment.prepared_targets(model, batch)
    with experiment.amp_context(cfg, batch.target.device):
        hidden = model.forward_dense(batch.context.features, batch.context.ctx_pad)
        hidden = hidden[:, cfg.arch.direct_loss_start :]
        logits = model.temporal.teacher_forced_logits_by_group(hidden, history, targets)
        dense_nll = model.temporal.nll_from_logits(logits, targets)
    value = model.value_head(experiment.decoder_rmsnorm(hidden).detach().float()).squeeze(-1)
    value_loss, advantage, _stats = experiment.value_objective(
        value,
        batch.returns[:, cfg.arch.direct_loss_start :],
        batch.eligible[:, cfg.arch.direct_loss_start :],
        beta=cfg.awr.beta,
        valid=valid,
    )
    weights, _weight_stats = experiment.advantage_weights(
        advantage,
        batch.eligible[:, cfg.arch.direct_loss_start :],
        beta=cfg.awr.beta,
        weight_max=cfg.awr.weight_max,
        active=update >= cfg.awr.start_update,
        valid=valid,
    )
    valid_prefixes = int(valid.sum())
    _near, _far, policy_loss = experiment.temporal_objective_parts(
        dense_nll,
        weights,
        valid_prefixes=valid_prefixes,
        aux_loss_weight=cfg.awr.auxiliary_loss_weight,
        valid=valid,
    )
    return policy_loss + cfg.awr.value_loss_weight * value_loss, hidden, logits, dense_nll, weights, valid, targets


def _group_objective(experiment: ModuleType, nll: Tensor, weights: Tensor, valid: Tensor, cfg: Any) -> Tensor:
    selected = torch.where(valid[..., None], nll.float(), 0)
    near_count = experiment.AWRCalibration.near_offsets
    valid_prefixes = valid.sum().clamp_min(1)
    near = (selected[..., :near_count] * weights[..., None]).sum() / (valid_prefixes * near_count)
    far = selected[..., near_count:].sum() / (valid_prefixes * (nll.shape[-1] - near_count))
    return (near + cfg.awr.auxiliary_loss_weight * far) / (1 + cfg.awr.auxiliary_loss_weight)


def _gradient_scale(parameters: tuple[nn.Parameter, ...], clip: float) -> tuple[float, float]:
    squares = torch.zeros((), device=next(iter(parameters)).device)
    for parameter in parameters:
        if parameter.grad is not None:
            squares += parameter.grad.detach().float().square().sum()
    norm = float(squares.sqrt())
    return norm, min(1.0, clip / (norm + 1e-6))


def probe_checkpoint(
    experiment: ModuleType,
    ref: CheckpointRef,
    args: Args,
    device: torch.device,
    expected_batch_sha256: str | None,
    sentinel_names: frozenset[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], str]:
    """Run bounded held-out forward/backward probes without optimizer.step().

    Every batch contributes gradients and subsystem statistics. Exact hypothetical
    updates use the mean clipped gradient once per parameter. A small sentinel set
    also receives exact per-batch updates to expose data-triggered variance.
    """
    model, optimizer, cfg, _state = _load_model_optimizer(experiment, ref, device)
    del _state
    stats = experiment.load_stats(cfg)
    bindings = _parameter_bindings(model, optimizer)
    unknown_sentinels = sentinel_names - {name for name, _parameter, _group in bindings.values()}
    if unknown_sentinels:
        raise ValueError(f"sentinel parameters are absent from the checkpoint: {sorted(unknown_sentinels)}")
    subsystems = _subsystem_names(experiment, model)
    recorder = ActivationRecorder(model)
    parameter_accumulators: dict[tuple[str, str], ScalarAccumulator] = {}
    mean_gradients: dict[int, Tensor] = {}
    subsystem_accumulators: dict[tuple[str, str], ScalarAccumulator] = {}
    head_accumulators: dict[tuple[str, str], ScalarAccumulator] = {}
    batch_digest = hashlib.sha256()
    parameters = tuple(model.parameters())
    parameter_versions = {id(parameter): parameter._version for parameter in parameters}
    optimizer_versions = {
        id(value): value._version
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, Tensor)
    }
    loader = make_lazy_probe_loader(experiment, cfg, stats, args)
    processed_batches = 0
    try:
        for batch_index, cpu_batch in enumerate(iter_probe_batches(loader, args.probe_batches)):
            if not isinstance(cpu_batch, AWRBatch):
                raise TypeError(f"lazy probe loader yielded {type(cpu_batch).__name__}, expected AWRBatch")
            batch_digest.update(_batch_identity(cpu_batch))
            batch = cpu_batch.to(device)
            recorder.enabled = batch_index < args.activation_probe_batches
            model.zero_grad(set_to_none=True)
            loss, hidden, logits, dense_nll, weights, valid, _targets = _probe_objective(
                experiment, model, batch, cfg, ref.update
            )
            for group_index, name in enumerate(experiment.CONTROLLER_GROUP_NAMES):
                group_loss = _group_objective(experiment, dense_nll[..., group_index], weights, valid, cfg)
                head_accumulators.setdefault((name, "loss"), ScalarAccumulator()).add(float(group_loss.detach()))
                if batch_index < args.head_probe_batches:
                    hidden_gradient, logit_gradient = torch.autograd.grad(
                        group_loss, (hidden, logits[name]), retain_graph=True
                    )
                    hidden_norm = _rms(hidden_gradient)
                    logit_norm = _rms(logit_gradient)
                    for metric, value in (
                        ("hidden_gradient_rms", hidden_norm),
                        ("logit_gradient_rms", logit_norm),
                        ("local_amplification", hidden_norm / max(logit_norm, torch.finfo(torch.float32).tiny)),
                    ):
                        head_accumulators.setdefault((name, metric), ScalarAccumulator()).add(value)
                for horizon_index, offset in enumerate(model.head_offsets):
                    horizon_nll = dense_nll[..., horizon_index, group_index].float()[valid]
                    horizon_weights = weights[valid] if horizon_index < experiment.AWRCalibration.near_offsets else 1.0
                    head_accumulators.setdefault((name, f"nll_offset_{offset}"), ScalarAccumulator()).add(
                        float(horizon_nll.detach().mean())
                    )
                    weighted = horizon_nll * horizon_weights
                    head_accumulators.setdefault((name, f"weighted_nll_offset_{offset}"), ScalarAccumulator()).add(
                        float(weighted.detach().mean())
                    )
            loss.backward()
            global_norm, clip_scale = _gradient_scale(parameters, cfg.grad_clip)
            subsystem_accumulators.setdefault(("total", "gradient_norm"), ScalarAccumulator()).add(global_norm)
            subsystem_accumulators.setdefault(("total", "clip_scale"), ScalarAccumulator()).add(clip_scale)

            per_subsystem_squares: dict[str, Tensor] = {}
            per_subsystem_count: dict[str, int] = defaultdict(int)
            per_subsystem_max: dict[str, Tensor] = {}
            for _parameter_id, (name, parameter, group) in bindings.items():
                gradient = parameter.grad
                if gradient is None:
                    raise RuntimeError(f"{name} has no gradient on probe batch {batch_index}")
                raw_gradient = gradient.detach()
                clipped_gradient = raw_gradient * clip_scale
                if id(parameter) not in mean_gradients:
                    mean_gradients[id(parameter)] = torch.zeros_like(raw_gradient, dtype=torch.float32)
                mean_gradients[id(parameter)].add_(clipped_gradient.float())
                optimizer_state = optimizer.state[parameter]
                if name in sentinel_names:
                    for state_name, reset in (("stored", False), ("reset", True)):
                        result = hypothetical_update(
                            parameter.detach(), clipped_gradient, optimizer_state, group, reset_state=reset
                        )
                        prefix = f"batch_postclip_{state_name}"
                        rho = _rms(result.delta) / max(_rms(parameter), torch.finfo(torch.float32).tiny)
                        parameter_accumulators.setdefault((name, f"{prefix}_rho"), ScalarAccumulator()).add(rho)
                        parameter_accumulators.setdefault((name, f"{prefix}_update_rms"), ScalarAccumulator()).add(
                            _rms(result.delta)
                        )
                subsystem = subsystems[id(parameter)]
                square = raw_gradient.float().square().sum()
                maximum = raw_gradient.abs().max()
                per_subsystem_squares[subsystem] = (
                    square if subsystem not in per_subsystem_squares else per_subsystem_squares[subsystem] + square
                )
                per_subsystem_count[subsystem] += raw_gradient.numel()
                per_subsystem_max[subsystem] = (
                    maximum
                    if subsystem not in per_subsystem_max
                    else torch.maximum(per_subsystem_max[subsystem], maximum)
                )
            for subsystem in per_subsystem_count:
                raw_rms = float((per_subsystem_squares[subsystem] / per_subsystem_count[subsystem]).sqrt())
                raw_max = float(per_subsystem_max[subsystem])
                for metric, value in (
                    ("gradient_rms", raw_rms),
                    ("gradient_abs_max", raw_max),
                    ("postclip_gradient_rms", raw_rms * clip_scale),
                    ("postclip_gradient_abs_max", raw_max * clip_scale),
                ):
                    subsystem_accumulators.setdefault((subsystem, metric), ScalarAccumulator()).add(value)
            print(
                f"[probe] {ref.arm} {ref.update:,} batch {batch_index + 1}/{args.probe_batches} "
                f"loss={float(loss.detach()):.5f} grad={global_norm:.4f}",
                flush=True,
            )
            processed_batches += 1
    finally:
        recorder.close()
        del loader
        gc.collect()
        clean_stale_shared_memory()

    if processed_batches != args.probe_batches:
        raise RuntimeError(f"validation yielded {processed_batches} batches, expected {args.probe_batches}")

    for _parameter_id, (name, parameter, group) in bindings.items():
        mean_gradient = mean_gradients[id(parameter)] / processed_batches
        optimizer_state = optimizer.state[parameter]
        for state_name, reset in (("stored", False), ("reset", True)):
            result = hypothetical_update(parameter.detach(), mean_gradient, optimizer_state, group, reset_state=reset)
            prefix = f"mean_postclip_{state_name}"
            rho = _rms(result.delta) / max(_rms(parameter), torch.finfo(torch.float32).tiny)
            parameter_accumulators.setdefault((name, f"{prefix}_rho"), ScalarAccumulator()).add(rho)
            parameter_accumulators.setdefault((name, f"{prefix}_update_rms"), ScalarAccumulator()).add(
                _rms(result.delta)
            )
        if not group["use_muon"]:
            second = cast(Tensor, optimizer_state["exp_avg_sq"])
            beta2 = float(group["betas"][1])
            correction = 1 - beta2 ** _step_number(optimizer_state)
            ratio = mean_gradient.square() / (second.float() / correction + float(group["eps"]) ** 2)
            parameter_accumulators.setdefault(
                (name, "mean_postclip_gradient_squared_over_v_mean"), ScalarAccumulator()
            ).add(float(ratio.mean()))
            first = cast(Tensor, optimizer_state["exp_avg"]).float()
            cosine = F.cosine_similarity(mean_gradient.reshape(1, -1), first.reshape(1, -1), dim=1)
            parameter_accumulators.setdefault(
                (name, "mean_postclip_gradient_momentum_cosine"), ScalarAccumulator()
            ).add(float(cosine))

    if parameter_versions != {id(parameter): parameter._version for parameter in parameters}:
        raise RuntimeError("no-step probe mutated model parameters")
    current_optimizer_versions = {
        id(value): value._version
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, Tensor)
    }
    if optimizer_versions != current_optimizer_versions:
        raise RuntimeError("no-step probe mutated optimizer state")

    actual_batch_sha256 = batch_digest.hexdigest()
    if expected_batch_sha256 is not None and actual_batch_sha256 != expected_batch_sha256:
        raise ValueError(
            f"held-out batch stream changed: {actual_batch_sha256} != first checkpoint {expected_batch_sha256}"
        )
    common = {"schema_version": _SCHEMA_VERSION, "arm": ref.arm, "update": ref.update}
    parameter_rows = [
        {**common, "parameter": name, "metric": metric, **accumulator.row()}
        for (name, metric), accumulator in sorted(parameter_accumulators.items())
    ]
    subsystem_rows = [
        {**common, "subsystem": subsystem, "metric": metric, **accumulator.row()}
        for (subsystem, metric), accumulator in sorted(subsystem_accumulators.items())
    ]
    head_rows = [
        {**common, "group": group, "metric": metric, **accumulator.row()}
        for (group, metric), accumulator in sorted(head_accumulators.items())
    ]
    activation_rows = [
        {
            **common,
            "module": name,
            "activation_rms": float((recorder.squares[name] / recorder.counts[name]).sqrt()),
            "activation_abs_max": float(recorder.maxima[name]),
            "elements": recorder.counts[name],
        }
        for name in sorted(recorder.counts)
    ]
    del mean_gradients, bindings, parameters, recorder, optimizer, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return parameter_rows, subsystem_rows, head_rows, activation_rows, actual_batch_sha256


def _write_table(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty table {path.name}")
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_plots(output: Path, eval_rows: list[dict[str, Any]], parameter_rows: list[dict[str, Any]]) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    frame = pd.DataFrame(eval_rows)
    for arm, values in frame.groupby("arm"):
        selected = values.sort_values("update")
        axis.plot(selected["update"], selected["net_stock_lcb"], marker="o", label=arm)
    axis.axvline(_FORK_UPDATE, color="black", linestyle="--", linewidth=1)
    axis.axhline(0, color="gray", linewidth=0.8)
    axis.set(xlabel="Update", ylabel="Net stocks/min LCB", title="Closed-loop gameplay")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "closed_loop_lcb.png", dpi=180)
    plt.close(figure)

    latest = pd.DataFrame(parameter_rows)
    latest = latest[latest["update"] == latest["update"].max()]
    top = latest.nlargest(20, "implied_update_weight_rms_ratio").sort_values("implied_update_weight_rms_ratio")
    figure, axis = plt.subplots(figsize=(9, 7))
    axis.barh(top["arm"] + ": " + top["parameter"], top["implied_update_weight_rms_ratio"])
    axis.set(xscale="log", xlabel="Implied update RMS / weight RMS", title="Largest state-implied relative updates")
    figure.tight_layout()
    figure.savefig(output / "largest_implied_updates.png", dpi=180)
    plt.close(figure)


def _write_report(
    output: Path,
    args: Args,
    eval_rows: list[dict[str, Any]],
    checkpoint_rows: list[dict[str, Any]],
    probe_rows: list[dict[str, Any]],
    head_rows: list[dict[str, Any]],
    selected: Mapping[str, tuple[int, ...]],
) -> None:
    checkpoint = pd.DataFrame(checkpoint_rows)
    latest = checkpoint[checkpoint["update"] == args.through_update].nlargest(15, "implied_update_weight_rms_ratio")
    probe = pd.DataFrame(probe_rows)
    stored = probe[probe["metric"] == "mean_postclip_stored_rho"].rename(columns={"mean": "stored"})
    reset = probe[probe["metric"] == "mean_postclip_reset_rho"].rename(columns={"mean": "reset"})
    joined = stored.merge(reset[["arm", "update", "parameter", "reset"]], on=["arm", "update", "parameter"])
    joined["stored_reset_ratio"] = joined["stored"] / joined["reset"].clip(lower=np.finfo(np.float32).tiny)
    contaminated = joined.nlargest(15, "stored_reset_ratio")
    heads = pd.DataFrame(head_rows)
    amplification = heads[heads["metric"] == "local_amplification"].sort_values("mean", ascending=False)
    eval_frame = pd.DataFrame(eval_rows).sort_values(["arm", "update"])

    lines = [
        "# O50 checkpoint forensics",
        "",
        f"Compared `{args.reference_run}` and `{args.treatment_run}` through update {args.through_update:,}.",
        "The analysis did not take an optimizer step and did not write to W&B.",
        "",
        "## Selected no-step probes",
        "",
        *[f"- {arm}: {', '.join(f'{value:,}' for value in updates)}" for arm, updates in selected.items()],
        "",
        "The selection uses the worst closed-loop net-stock LCB in each arm, plus the fork and latest common checkpoint.",
        "",
        "## Closed-loop history",
        "",
        eval_frame.to_markdown(index=False),
        "",
        "## Largest state-implied relative updates at the latest checkpoint",
        "",
        latest[["arm", "parameter", "optimizer", "implied_update_weight_rms_ratio"]].to_markdown(index=False),
        "",
        "## Strongest stored-state versus reset-state signals",
        "",
        contaminated[["arm", "update", "parameter", "stored", "reset", "stored_reset_ratio"]].to_markdown(index=False),
        "",
        "A large stored/reset ratio supports optimizer-state contamination. Large values in both columns support a weight- or data-driven local instability instead.",
        "",
        "## Highest action-head local amplification",
        "",
        amplification[["arm", "update", "group", "mean", "max"]].head(20).to_markdown(index=False),
        "",
        "The Parquet tables contain per-parameter trajectories, coordinate tails, checkpoint displacement, fixed-policy KL, centered legal logits, per-head attention entropy, activations, subsystem gradients, mean-gradient update probes, and batch-level sentinel probes.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def _manifest(output: Path) -> dict[str, Any]:
    files = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name not in {"manifest.json", "complete.json"}:
            files.append({"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)})
    return {"schema_version": _SCHEMA_VERSION, "files": files}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _remote_complete(args: Args) -> dict[str, Any] | None:
    client = r2.client()
    key = f"{args.artifact_prefix}/complete.json"
    try:
        response = client.get_object(Bucket=r2.bucket(), Key=key)
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") in _NOT_FOUND:
            return None
        raise
    return cast(dict[str, Any], json.loads(response["Body"].read()))


def _upload(output: Path, args: Args, completion: Mapping[str, Any]) -> None:
    client = r2.client()
    bucket = r2.bucket()
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "complete.json":
            client.upload_file(str(path), bucket, f"{args.artifact_prefix}/{path.name}")
            print(f"[upload] {path.name}", flush=True)
    client.put_object(
        Bucket=bucket,
        Key=f"{args.artifact_prefix}/complete.json",
        Body=(json.dumps(completion, indent=2, sort_keys=True) + "\n").encode(),
        ContentType="application/json",
    )


def run(args: Args) -> Path:
    """Run the complete read-only analysis and return its local artifact path."""
    _validate_args(args)
    if args.upload:
        existing = _remote_complete(args)
        if existing is not None:
            if existing.get("git_sha") != _git_sha() or existing.get("through_update") != args.through_update:
                raise RuntimeError(f"artifact prefix {args.artifact_prefix} already contains a different analysis")
            print(f"[complete] existing r2://{r2.bucket()}/{args.artifact_prefix}/complete.json", flush=True)
            return Path(args.output_dir)

    started = time.time()
    output = Path(args.output_dir) / args.analysis_id
    output.mkdir(parents=True, exist_ok=False)
    experiment = _experiment()
    if not torch.cuda.is_available():
        raise RuntimeError("the full O50 analysis requires CUDA")
    device = torch.device("cuda")
    refs, reference_run = download_checkpoints(args)
    args = replace(args, reference_run=reference_run)
    by_key = {(ref.arm, ref.update): ref for ref in refs}
    fork_validation = validate_fork(by_key[("reference", _FORK_UPDATE)], by_key[("half_muon", _FORK_UPDATE)])

    eval_rows: list[dict[str, Any]] = []
    online_rows: list[dict[str, Any]] = []
    for arm, wandb_id in (("reference", args.reference_wandb_id), ("half_muon", args.treatment_wandb_id)):
        arm_eval, arm_online = _wandb_history(args, wandb_id, arm)
        eval_rows.extend(arm_eval)
        online_rows.extend(arm_online)
    selected = {
        arm: select_turning_updates(
            (row for row in eval_rows if row["arm"] == arm), through_update=args.through_update
        )
        for arm in ("reference", "half_muon")
    }

    checkpoint_rows: list[dict[str, Any]] = []
    for ref in refs:
        checkpoint_rows.extend(checkpoint_parameter_rows(experiment, ref, device, args.coordinate_sample_size))
    parameter_names = frozenset(str(row["parameter"]) for row in checkpoint_rows)

    displacement: list[dict[str, Any]] = []
    for arm in ("reference", "half_muon"):
        arm_refs = sorted((ref for ref in refs if ref.arm == arm), key=lambda ref: ref.update)
        for left, right in zip(arm_refs, arm_refs[1:]):
            displacement.extend(
                displacement_rows(
                    left,
                    right,
                    comparison="successive_checkpoint",
                    parameter_names=parameter_names,
                )
            )
    divergence: list[dict[str, Any]] = []
    for update in checkpoint_updates(args.through_update):
        divergence.extend(
            displacement_rows(
                by_key[("reference", update)],
                by_key[("half_muon", update)],
                comparison="half_muon_minus_reference",
                parameter_names=parameter_names,
            )
        )

    child_fork = _load_raw(by_key[("half_muon", _FORK_UPDATE)])
    fixed_state = child_fork.get("fixed_diagnostics")
    if not isinstance(fixed_state, Mapping):
        raise RuntimeError("treatment fork checkpoint has no fixed diagnostic state")
    tracker = experiment.FixedDiagnosticTracker.from_state(dict(fixed_state))
    for update in checkpoint_updates(args.through_update):
        child_state = _load_raw(by_key[("half_muon", update)])
        child_fixed = child_state.get("fixed_diagnostics")
        if not isinstance(child_fixed, Mapping) or child_fixed.get("batch_sha256") != tracker.batch_sha256:
            raise ValueError(f"treatment fixed diagnostic batch changed at update {update}")
    del child_fork, child_state
    fixed_batch = tracker.batch
    baseline = tracker.baseline_log_probabilities
    fixed_policy: list[dict[str, Any]] = []
    attention: list[dict[str, Any]] = []
    for ref in refs:
        policy_values, attention_values = fixed_batch_rows(experiment, ref, fixed_batch, baseline, device)
        fixed_policy.extend(policy_values)
        attention.extend(attention_values)

    probe_parameters: list[dict[str, Any]] = []
    probe_subsystems: list[dict[str, Any]] = []
    probe_heads: list[dict[str, Any]] = []
    activations: list[dict[str, Any]] = []
    batch_sha256: str | None = None
    probe_sentinels: dict[str, list[str]] = {}
    probe_updates = sorted({update for values in selected.values() for update in values})
    for arm in ("reference", "half_muon"):
        for update in probe_updates:
            sentinels = sentinel_parameter_names(
                checkpoint_rows,
                arm=arm,
                update=update,
                top_count=args.sentinel_parameter_count,
            )
            probe_sentinels[f"{arm}:{update}"] = sorted(sentinels)
            result = probe_checkpoint(
                experiment,
                by_key[(arm, update)],
                args,
                device,
                expected_batch_sha256=batch_sha256,
                sentinel_names=sentinels,
            )
            parameter_values, subsystem_values, head_values, activation_values, actual_hash = result
            batch_sha256 = actual_hash if batch_sha256 is None else batch_sha256
            probe_parameters.extend(parameter_values)
            probe_subsystems.extend(subsystem_values)
            probe_heads.extend(head_values)
            activations.extend(activation_values)

    tables = {
        "checkpoint_parameters.parquet": checkpoint_rows,
        "checkpoint_displacements.parquet": displacement,
        "matched_divergence.parquet": divergence,
        "fixed_policy.parquet": fixed_policy,
        "attention.parquet": attention,
        "probe_parameters.parquet": probe_parameters,
        "probe_subsystems.parquet": probe_subsystems,
        "probe_heads.parquet": probe_heads,
        "activations.parquet": activations,
        "wandb_eval.parquet": eval_rows,
        "online_updates.parquet": online_rows,
    }
    for name, rows in tables.items():
        _write_table(rows, output / name)
    _write_plots(output, eval_rows, checkpoint_rows)
    _write_report(output, args, eval_rows, checkpoint_rows, probe_parameters, probe_heads, selected)
    provenance = {
        "schema_version": _SCHEMA_VERSION,
        "args": asdict(args),
        "artifact_prefix": args.artifact_prefix,
        "git_sha": _git_sha(),
        "experiment_path": _EXPERIMENT_PATH,
        "experiment_sha256": _EXPERIMENT_SHA256,
        "checkpoints": [asdict(ref) | {"path": str(ref.path)} for ref in refs],
        "fork_validation": fork_validation,
        "selected_probe_updates": selected,
        "sentinel_parameters": probe_sentinels,
        "held_out_batch_stream_sha256": batch_sha256,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "started_unix": started,
        "finished_unix": time.time(),
        "wandb_access": "read_only_api_no_init",
        "optimizer_steps_taken": 0,
        "training_reservoir_constructed": False,
    }
    _write_json(output / "provenance.json", provenance)
    manifest = _manifest(output)
    _write_json(output / "manifest.json", manifest)
    completion = {
        "schema_version": _SCHEMA_VERSION,
        "status": "complete",
        "git_sha": provenance["git_sha"],
        "through_update": args.through_update,
        "manifest_sha256": _sha256(output / "manifest.json"),
        "finished_unix": provenance["finished_unix"],
    }
    _write_json(output / "complete.json", completion)
    if args.upload:
        _upload(output, args, completion)
        print(f"[complete] r2://{r2.bucket()}/{args.artifact_prefix}/complete.json", flush=True)
    return output


# %%
if __name__ == "__main__":
    run(tyro.cli(Args))
