"""Compute-optimal capacity brackets for the complete O43 treatment.

The five launchable endpoints, together with the completed O43 reference, form
two three-model iso-FLOP slices. They keep the historical controller codec,
sparse ten-offset decoder, next-frame-weighted objective, data, and optimizer
recipe fixed. Only model geometry and the schedule length change.

Describe every endpoint without training:
    uv run experiments/047_legacy_capacity_scaling.py --describe

Train one endpoint:
    uv run experiments/047_legacy_capacity_scaling.py --endpoint c5e16-xs
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import torch
import tyro

import wandb
from hal.training.checkpoints import load_for_resume
from hal.training.ego_stats import load_consolidated_stats


def _load_o43() -> ModuleType:
    path = Path(__file__).with_name("043_legacy_codec.py")
    name = "_hal_experiment_043_for_047"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_o43 = _load_o43()

LOW_COMPUTE_FLOPS = 50_000_000_000_000_000
REFERENCE_COMPUTE_FLOPS = 138_254_443_207_458_816
REFERENCE_RUN_ID = "1imfy8v3"
REFERENCE_RUN_NAME = (
    "260828-203353_043_legacy_codec_mtp043-legacy-v2-d384-L8-h6-Lc128-t128x2-"
    "o1-2-3-4-5-6-9-12-16-20-s1-o1w50-base_ranked-anon-1_"
    "forensic-ranked-legacy-codec-h1-next50"
)
CORPUS_LOSS_POSITIONS = 2_409_583_026
REFERENCE_STEPS = 16_384
REFERENCE_WARMUP_STEPS = 500
WANDB_GROUP = "047-legacy-capacity-scaling"


@dataclass(frozen=True, slots=True)
class Endpoint:
    """One fixed-compute model geometry."""

    name: str
    compute_budget_flops: int
    d_model: int
    n_layers: int
    temporal_d_model: int
    temporal_layers: int
    max_steps: int
    reference_run_id: str | None = None

    @property
    def n_heads(self) -> int:
        return self.d_model // 64

    @property
    def temporal_heads(self) -> int:
        return self.temporal_d_model // 32

    @property
    def warmup_steps(self) -> int:
        return round(REFERENCE_WARMUP_STEPS * self.max_steps / REFERENCE_STEPS)


POINTS: dict[str, Endpoint] = {
    "c5e16-xs": Endpoint("c5e16-xs", LOW_COMPUTE_FLOPS, 192, 4, 128, 2, 15_740),
    "c5e16-s": Endpoint("c5e16-s", LOW_COMPUTE_FLOPS, 256, 5, 128, 2, 12_027),
    "c5e16-m": Endpoint("c5e16-m", LOW_COMPUTE_FLOPS, 320, 7, 128, 2, 8_165),
    "cref-s": Endpoint("cref-s", REFERENCE_COMPUTE_FLOPS, 256, 5, 128, 2, 33_255),
    "cref-m": Endpoint("cref-m", REFERENCE_COMPUTE_FLOPS, 320, 7, 128, 2, 22_576),
    "cref-reference": Endpoint(
        "cref-reference",
        REFERENCE_COMPUTE_FLOPS,
        384,
        8,
        128,
        2,
        REFERENCE_STEPS,
        reference_run_id=REFERENCE_RUN_ID,
    ),
}
TRAIN_ENDPOINTS = tuple(name for name, endpoint in POINTS.items() if endpoint.reference_run_id is None)


def endpoint_config(endpoint: Endpoint) -> _o43.TrainConfig:
    """Build the frozen O43 configuration for one scaling endpoint."""
    return replace(
        _o43.TrainConfig(),
        d_model=endpoint.d_model,
        n_layers=endpoint.n_layers,
        n_heads=endpoint.n_heads,
        temporal_d_model=endpoint.temporal_d_model,
        temporal_layers=endpoint.temporal_layers,
        temporal_heads=endpoint.temporal_heads,
        temporal_ff_dim=2 * endpoint.temporal_d_model,
        group_head_dim=2 * endpoint.temporal_d_model,
        max_steps=endpoint.max_steps,
        warmup_steps=endpoint.warmup_steps,
        eval_every=0,
        final_eval_n_matchups=96,
        final_diag_n_matchups=0,
    )


def endpoint_for_config(cfg: _o43.TrainConfig) -> Endpoint:
    """Resolve a serialized configuration to exactly one declared endpoint."""
    identity = (
        cfg.d_model,
        cfg.n_layers,
        cfg.n_heads,
        cfg.temporal_d_model,
        cfg.temporal_layers,
        cfg.temporal_heads,
        cfg.temporal_ff_dim,
        cfg.group_head_dim,
        cfg.max_steps,
        cfg.warmup_steps,
    )
    matches = [endpoint for endpoint in POINTS.values() if identity == _endpoint_identity(endpoint)]
    if len(matches) != 1:
        raise ValueError(f"configuration does not identify one O47 endpoint: {identity}")
    return matches[0]


def _endpoint_identity(endpoint: Endpoint) -> tuple[int, ...]:
    cfg = endpoint_config(endpoint)
    return (
        cfg.d_model,
        cfg.n_layers,
        cfg.n_heads,
        cfg.temporal_d_model,
        cfg.temporal_layers,
        cfg.temporal_heads,
        cfg.temporal_ff_dim,
        cfg.group_head_dim,
        cfg.max_steps,
        cfg.warmup_steps,
    )


def endpoint_report(endpoint: Endpoint) -> dict[str, int | str | None]:
    """Return the exact model, data, and compute quantities for one point."""
    cfg = endpoint_config(endpoint)
    model = _o43.GPT(cfg)
    counts = _o43.subsystem_parameter_counts(model)
    flops_per_update = _o43.approximate_training_flops_per_update(cfg, counts)
    processed_positions = cfg.max_steps * cfg.batch_size * cfg.L_ctx
    actual_flops = cfg.max_steps * flops_per_update
    if abs(actual_flops - endpoint.compute_budget_flops) > flops_per_update / 2:
        raise RuntimeError(
            f"{endpoint.name} misses its compute budget: {actual_flops} != {endpoint.compute_budget_flops}"
        )
    if processed_positions >= CORPUS_LOSS_POSITIONS:
        raise RuntimeError(f"{endpoint.name} repeats the corpus: D={processed_positions} >= U={CORPUS_LOSS_POSITIONS}")
    effective_parameters = flops_per_update // (6 * cfg.batch_size * cfg.L_ctx)
    return {
        "endpoint": endpoint.name,
        "reference_run_id": endpoint.reference_run_id,
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "temporal_d_model": cfg.temporal_d_model,
        "temporal_layers": cfg.temporal_layers,
        "total_parameters": counts["total"],
        "effective_parameters": effective_parameters,
        "max_steps": cfg.max_steps,
        "warmup_steps": cfg.warmup_steps,
        "processed_positions": processed_positions,
        "corpus_loss_positions": CORPUS_LOSS_POSITIONS,
        "compute_budget_flops": endpoint.compute_budget_flops,
        "actual_flops": actual_flops,
    }


def model_tag(cfg: _o43.TrainConfig) -> str:
    endpoint = endpoint_for_config(cfg)
    return (
        f"legacy-cap-{endpoint.name}-d{cfg.d_model}-L{cfg.n_layers}-h{cfg.n_heads}-"
        f"t{cfg.temporal_d_model}x{cfg.temporal_layers}"
    )


def _init_wandb(cfg: _o43.TrainConfig, run_name: str, resume_state: dict | None) -> None:
    endpoint = endpoint_for_config(cfg)
    report = endpoint_report(endpoint)
    wandb.init(
        project="hal",
        group=WANDB_GROUP,
        name=run_name,
        id=None if resume_state is None else resume_state.get("wandb_id"),
        resume="allow" if resume_state is not None else None,
        tags=["047", "legacy-codec", "capacity-scaling", endpoint.name],
        config={**asdict(cfg), **{f"scaling_{key}": value for key, value in report.items()}},
        settings=wandb.Settings(x_stats_sampling_interval=5.0, x_stats_track_process_tree=True),
    )
    if wandb.run is None:
        return
    wandb.define_metric("global_step")
    wandb.define_metric("*", step_metric="global_step")
    wandb.run.summary["scaling/endpoint"] = endpoint.name
    wandb.run.summary["scaling/compute_budget_flops"] = endpoint.compute_budget_flops
    wandb.run.summary["scaling/processed_positions"] = report["processed_positions"]
    wandb.run.summary["scaling/effective_parameters"] = report["effective_parameters"]
    wandb.run.summary["scaling/corpus_loss_positions"] = CORPUS_LOSS_POSITIONS
    wandb.run.summary["scaling/reference_run_id"] = REFERENCE_RUN_ID
    wandb.run.summary["scaling/reference_run_name"] = REFERENCE_RUN_NAME
    wandb.run.summary["evaluation/suites"] = "terminal char_matchup and fox only"
    wandb.run.summary["training/updates"] = cfg.max_steps
    wandb.run.summary["data/nominal_samples"] = cfg.max_steps * cfg.batch_size
    wandb.run.summary["data/max_context_prefixes"] = report["processed_positions"]
    if cfg.wandb_log_code:
        _o43.log_wandb_code(wandb.run)


# Reuse O43's training implementation under a private module. These replacements
# change only experiment identity and W&B metadata; model and objective code stay
# byte-for-byte O43.
_o43.__file__ = __file__
_o43.model_tag = model_tag
_o43._init_wandb = _init_wandb


@dataclass
class Args:
    endpoint: str | None = None
    """One of c5e16-xs, c5e16-s, c5e16-m, cref-s, or cref-m."""

    comment: str = ""
    resume: str | None = None
    describe: bool = False


def _selected_endpoint(name: str | None) -> Endpoint:
    if name not in TRAIN_ENDPOINTS:
        choices = ", ".join(TRAIN_ENDPOINTS)
        raise SystemExit(f"--endpoint must be one of: {choices}")
    return POINTS[name]


def main(args: Args) -> None:
    if args.describe:
        print(json.dumps([endpoint_report(endpoint) for endpoint in POINTS.values()], indent=2))
        return
    if args.resume is not None:
        resume_state = load_for_resume(args.resume, Path("runs") / args.resume, device=_o43.DEVICE)
        if resume_state is None:
            raise SystemExit(f"no latest.pt for run {args.resume!r}")
        cfg = _o43.config_from_state(resume_state["cfg"])
        endpoint = endpoint_for_config(cfg)
        if args.endpoint is not None and args.endpoint != endpoint.name:
            raise SystemExit(f"resume endpoint is {endpoint.name!r}, not {args.endpoint!r}")
        run_name = args.resume
    else:
        endpoint = _selected_endpoint(args.endpoint)
        if endpoint.reference_run_id is not None:
            raise SystemExit(f"{endpoint.name} is the existing reference and cannot be launched")
        cfg = endpoint_config(endpoint)
        resume_state = None
        run_name = None

    _o43.validate_config(cfg)
    endpoint_report(endpoint)
    stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
    _o43.train(cfg, stats, comment=args.comment, resume_run=run_name, resume_state=resume_state)


if __name__ == "__main__":
    main(tyro.cli(Args))
