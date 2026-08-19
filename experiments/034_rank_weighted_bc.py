"""Rank-weighted behavioral cloning on the exact experiment-026 policy.

The source replay stream, sampled windows, model, optimizer, schedule, and
deployment protocol stay fixed. A replay-ID sidecar attaches the sampled ego
player's ladder tier after window selection, and the full optimizer-step loss is
self-normalized with Platinum/Diamond/Master weights ``(1, 2, 4)``.

Run:
    uv run experiments/034_rank_weighted_bc.py
    uv run experiments/034_rank_weighted_bc.py --eval runs/<run>/final.pt
"""

import functools
import gzip
import hashlib
import importlib.util
import io
import json
import math
import os
import sys
import time
from collections import Counter
from collections.abc import Callable
from collections.abc import Iterator
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from dataclasses import fields
from dataclasses import replace
from pathlib import Path

import fsspec
import numpy as np
import torch
import tyro
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR

from hal.data.feature_stats import FeatureStats
from hal.data.policy_schema import policy_replay_identity
from hal.data.schema import Rank
from hal.data.schema import rank_from_player_name
from hal.training.features import Context
from hal.training.features import TrainBatch
from hal.training.rank_metadata import ReplayRankLookup

_BASE_PATH = Path(__file__).with_name("026_temporal_mtp.py")
_SPEC = importlib.util.spec_from_file_location("hal_exp026_for_034", _BASE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
base = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = base
_SPEC.loader.exec_module(base)

_validate_026_config = base.validate_config
_model_tag_026 = base.model_tag
_TrainConfig026 = base.TrainConfig

for _name in dir(base):
    if not _name.startswith("__"):
        globals()[_name] = getattr(base, _name)


_POLICY_DATA_ROOT = "data/processed/ranked-anonymized-1/mds-policy-v7"
_RANK_SIDECAR_LOCAL = f"{_POLICY_DATA_ROOT}/ranks-v1.jsonl.gz"
_RANK_SIDECAR_SHA256 = "9670b77ef378e762a2d157a3b37cbce35bd05ea54b84a9218a9f0d1269173e70"
_RANK_SIDECAR_ROWS = 114_768
_RANK_SIDECAR_SCHEMA_VERSION = 1
_RANK_MANIFEST_LOCAL = "data/processed/ranked-anonymized-1/mds-v7/manifest.jsonl"
_RANK_MANIFEST_REMOTE = "s3://hal/processed/ranked-anonymized-1/mds-v7/manifest.jsonl"
_RANK_MANIFEST_SHA256 = "a563c62603b8cfdef219cd133324cb090f7e6488fcfc4b6941d03e863a255d16"
_RANK_PLAYER_COUNTS = (98_248, 59_218, 72_070)
_EXPERIMENT_ID = "034_rank_weighted_bc_v1"
_RANK_TIERS = (Rank.PLATINUM, Rank.DIAMOND, Rank.MASTER)
_RANK_NAMES = {
    Rank.PLATINUM: "platinum",
    Rank.DIAMOND: "diamond",
    Rank.MASTER: "master",
}


@dataclass
class TrainConfig(base.TrainConfig):
    # Exact eligible 026 production overrides.
    batch_size: int = 512
    cache_limit_gb: int = 160
    eval_max_parallel: int | None = 32

    # The only training treatment.
    experiment_id: str = _EXPERIMENT_ID
    rank_weights: tuple[float, float, float] = (1.0, 2.0, 4.0)
    minimum_rank_ess: float = 0.60
    rank_sidecar_local: str = _RANK_SIDECAR_LOCAL
    rank_sidecar_sha256: str = _RANK_SIDECAR_SHA256
    rank_sidecar_rows: int = _RANK_SIDECAR_ROWS
    rank_manifest_local: str = _RANK_MANIFEST_LOCAL
    rank_manifest_remote: str = _RANK_MANIFEST_REMOTE
    rank_manifest_sha256: str = _RANK_MANIFEST_SHA256
    rank_player_counts: tuple[int, int, int] = _RANK_PLAYER_COUNTS


_SMOKE_OVERRIDE_FIELDS = frozenset(
    {
        "cache_limit_gb",
        "ckpt_every",
        "compile_temporal",
        "compile_trunk",
        "compiled_inference_bucket",
        "eval_every",
        "eval_max_frames",
        "eval_max_parallel",
        "eval_n_matchups",
        "final_diag_n_matchups",
        "final_eval_n_matchups",
        "grad_accum_steps",
        "inference_mode",
        "max_steps",
        "num_workers",
        "predownload",
        "prefetch_batches",
        "prefetch_factor",
        "push_to_r2",
        "val_batch_size",
        "val_every",
        "val_n_samples",
        "wandb_grad_every",
        "wandb_log_code",
    }
)
_EVALUATION_OVERRIDE_FIELDS = frozenset({"eval_max_parallel", "inference_mode"})


def _config_changes(cfg: TrainConfig, reference: TrainConfig) -> dict[str, tuple[object, object]]:
    return {
        item.name: (getattr(cfg, item.name), getattr(reference, item.name))
        for item in fields(TrainConfig)
        if getattr(cfg, item.name) != getattr(reference, item.name)
    }


def _validate_rank_contract(cfg: TrainConfig) -> None:
    """Validate rank metadata independently of training/evaluation mode."""
    if cfg.experiment_id != _EXPERIMENT_ID:
        raise ValueError(f"experiment_id must be {_EXPERIMENT_ID!r}, got {cfg.experiment_id!r}")
    if cfg.data_root != _POLICY_DATA_ROOT:
        raise ValueError(f"034 must read the exact 026 policy stream at {_POLICY_DATA_ROOT!r}")
    if not cfg.compact_data:
        raise ValueError("034 requires the compact policy replay reservoir")
    if len(cfg.rank_weights) != len(_RANK_TIERS) or any(
        not math.isfinite(weight) or weight <= 0 for weight in cfg.rank_weights
    ):
        raise ValueError(f"rank_weights must contain three finite positive values, got {cfg.rank_weights}")
    if not math.isfinite(cfg.minimum_rank_ess) or not 0 < cfg.minimum_rank_ess <= 1:
        raise ValueError(f"minimum_rank_ess must be in (0, 1], got {cfg.minimum_rank_ess}")
    if len(cfg.rank_sidecar_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in cfg.rank_sidecar_sha256
    ):
        raise ValueError("rank_sidecar_sha256 must be a lowercase SHA-256 digest")
    if len(cfg.rank_manifest_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in cfg.rank_manifest_sha256
    ):
        raise ValueError("rank_manifest_sha256 must be a lowercase SHA-256 digest")
    if cfg.rank_sidecar_rows <= 0:
        raise ValueError("rank_sidecar_rows must be positive")
    if (
        len(cfg.rank_player_counts) != len(_RANK_TIERS)
        or any(not isinstance(count, int) or isinstance(count, bool) or count <= 0 for count in cfg.rank_player_counts)
        or sum(cfg.rank_player_counts) != 2 * cfg.rank_sidecar_rows
    ):
        raise ValueError(
            "rank_player_counts must contain three positive integer counts totaling two players per replay"
        )


def validate_config(cfg: TrainConfig) -> None:
    """Validate 026 invariants plus the rank-treatment contract."""
    _validate_026_config(cfg)
    _validate_rank_contract(cfg)

    reference = TrainConfig()
    if cfg.max_steps > reference.max_steps:
        raise ValueError(f"034 cannot exceed the frozen {reference.max_steps} optimizer steps")
    allowed = _SMOKE_OVERRIDE_FIELDS if cfg.max_steps < reference.max_steps else frozenset()
    changed = _config_changes(cfg, reference)
    forbidden = {name: value for name, value in changed.items() if name not in allowed}
    if forbidden:
        mode = "smoke" if cfg.max_steps < reference.max_steps else "production"
        raise ValueError(f"{mode} 034 config changed frozen scientific fields: {forbidden}")


def model_tag(cfg: TrainConfig) -> str:
    weights = "-".join(f"{weight:g}" for weight in cfg.rank_weights)
    return f"{_model_tag_026(cfg)}-rank{weights}"


def _read_uri_bytes(local: str, remote: str) -> bytes:
    local_path = Path(local)
    if local_path.is_file():
        return local_path.read_bytes()
    endpoint = os.environ.get("S3_ENDPOINT_URL") or os.environ.get("AWS_ENDPOINT_URL")
    options = {"client_kwargs": {"endpoint_url": endpoint}} if endpoint else {}
    with fsspec.open(remote, "rb", **options) as handle:
        return handle.read()


RankLookup = ReplayRankLookup


def _load_rank_sidecar(payload: bytes, cfg: TrainConfig) -> RankLookup:
    digest = hashlib.sha256(payload).hexdigest()
    if digest != cfg.rank_sidecar_sha256:
        raise ValueError(f"rank sidecar SHA-256 {digest} != expected {cfg.rank_sidecar_sha256}")
    try:
        lines = gzip.decompress(payload).splitlines()
    except (EOFError, OSError) as error:
        raise ValueError("rank sidecar is not valid gzip data") from error
    if not lines:
        raise ValueError("rank sidecar is empty")
    header = json.loads(lines[0])
    if header.get("rank_sidecar_schema_version") != _RANK_SIDECAR_SCHEMA_VERSION:
        raise ValueError(f"rank sidecar has unsupported header {header}")
    if header.get("source_manifest_sha256") != cfg.rank_manifest_sha256:
        raise ValueError(
            "rank sidecar source manifest SHA-256 "
            f"{header.get('source_manifest_sha256')} != expected {cfg.rank_manifest_sha256}"
        )
    if header.get("rows") != cfg.rank_sidecar_rows or len(lines) - 1 != cfg.rank_sidecar_rows:
        raise ValueError(
            f"rank sidecar row count is header={header.get('rows')}, payload={len(lines) - 1}, "
            f"expected={cfg.rank_sidecar_rows}"
        )
    expected_counts = {str(int(rank)): count for rank, count in zip(_RANK_TIERS, cfg.rank_player_counts, strict=True)}
    if header.get("player_rank_counts") != expected_counts:
        raise ValueError(
            f"rank sidecar player counts {header.get('player_rank_counts')} != expected {expected_counts}"
        )

    allowed = {int(rank) for rank in _RANK_TIERS}
    by_replay: dict[str, tuple[int, int]] = {}
    observed_counts: Counter[int] = Counter()
    for line_number, line in enumerate(lines[1:], start=2):
        row = json.loads(line)
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError(f"rank sidecar line {line_number} must be [replay_id, p1_rank, p2_rank]")
        replay_id, p1_rank, p2_rank = row
        if not isinstance(replay_id, str) or len(replay_id) != 32:
            raise ValueError(f"rank sidecar line {line_number} has invalid replay ID {replay_id!r}")
        ranks = (int(p1_rank), int(p2_rank))
        if set(ranks) - allowed:
            raise ValueError(f"rank sidecar line {line_number} has unsupported ranks {ranks}")
        if replay_id in by_replay:
            raise ValueError(f"rank sidecar repeats replay ID {replay_id}")
        by_replay[replay_id] = ranks
        observed_counts.update(ranks)
    if {str(rank): observed_counts[rank] for rank in sorted(allowed)} != expected_counts:
        raise ValueError("rank sidecar payload player counts do not match its frozen header")
    return RankLookup(by_replay)


def _load_rank_manifest(payload: bytes, cfg: TrainConfig) -> RankLookup:
    digest = hashlib.sha256(payload).hexdigest()
    if digest != cfg.rank_manifest_sha256:
        raise ValueError(f"rank manifest SHA-256 {digest} != expected {cfg.rank_manifest_sha256}")
    allowed = {int(rank) for rank in _RANK_TIERS}
    by_replay: dict[str, tuple[int, int]] = {}
    observed_counts: Counter[int] = Counter()
    for line_number, line in enumerate(io.BytesIO(payload), start=1):
        row = json.loads(line)
        if row.get("annotation") is None:
            continue
        replay_id = policy_replay_identity(row["path"])
        by_port = {int(player["port"]): int(rank_from_player_name(player.get("name"))) for player in row["players"]}
        if set(by_port) != {1, 2} or set(by_port.values()) - allowed:
            raise ValueError(f"rank manifest line {line_number} has unsupported players {row['players']}")
        if replay_id in by_replay:
            raise ValueError(f"rank manifest repeats replay ID {replay_id}")
        ranks = (by_port[1], by_port[2])
        by_replay[replay_id] = ranks
        observed_counts.update(ranks)
    if len(by_replay) != cfg.rank_sidecar_rows:
        raise ValueError(f"rank manifest produced {len(by_replay)} rows, expected {cfg.rank_sidecar_rows}")
    expected_counts = {int(rank): count for rank, count in zip(_RANK_TIERS, cfg.rank_player_counts, strict=True)}
    if {rank: observed_counts[rank] for rank in sorted(allowed)} != expected_counts:
        raise ValueError(f"rank manifest player counts {dict(observed_counts)} != expected {expected_counts}")
    return RankLookup(by_replay)


def load_rank_lookup(cfg: TrainConfig) -> RankLookup:
    """Load the local sidecar or derive it read-only from the frozen manifest."""
    sidecar = Path(cfg.rank_sidecar_local)
    if sidecar.is_file():
        return _load_rank_sidecar(sidecar.read_bytes(), cfg)
    manifest = _read_uri_bytes(cfg.rank_manifest_local, cfg.rank_manifest_remote)
    return _load_rank_manifest(manifest, cfg)


@dataclass(frozen=True, slots=True)
class RankBatch:
    """An unchanged 026 batch plus replay-level target weights."""

    batch: TrainBatch
    rank: Tensor
    rank_weight: Tensor

    @property
    def context(self) -> Context:
        return self.batch.context

    @property
    def target(self) -> Tensor:
        return self.batch.target

    @property
    def replay_ids(self) -> tuple[str, ...] | None:
        return self.batch.replay_ids

    def to(self, device: str | torch.device) -> RankBatch:
        return RankBatch(
            batch=self.batch.to(device),
            rank=self.rank.to(device, non_blocking=True),
            rank_weight=self.rank_weight.to(device, non_blocking=True),
        )

    def pin_memory(self) -> RankBatch:
        return RankBatch(
            batch=self.batch.pin_memory(),
            rank=self.rank.pin_memory(),
            rank_weight=self.rank_weight.pin_memory(),
        )

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        tensors = [*self.context.features.values(), self.context.ctx_pad, self.target, self.rank, self.rank_weight]
        if self.context.slot_ids is not None:
            tensors.append(self.context.slot_ids)
        if self.context.reset is not None:
            tensors.append(self.context.reset)
        for tensor in tensors:
            tensor.record_stream(stream)

    def valid_rank_weights(self, valid: Tensor) -> Tensor:
        context_length = next(iter(self.context.features.values())).shape[1]
        if valid.shape != (self.rank_weight.shape[0], context_length):
            raise ValueError("valid mask and rank weights have incompatible shapes")
        return self.rank_weight[:, None].expand_as(valid).reshape(-1)[valid.reshape(-1)]


def collate_rank_batch(
    windows: list[dict[str, np.ndarray]],
    batch: TrainBatch,
    *,
    rank_weights: tuple[float, float, float],
) -> RankBatch:
    """Attach one validated ego tier and multiplier to each sampled window."""
    ranks = np.asarray([np.asarray(window["ego_rank"]).item() for window in windows], dtype=np.uint8)
    allowed = np.asarray([int(rank) for rank in _RANK_TIERS], dtype=np.uint8)
    invalid = ~np.isin(ranks, allowed)
    if invalid.any():
        row = int(np.flatnonzero(invalid)[0])
        raise ValueError(f"window {row} has unsupported ego rank {int(ranks[row])}")
    table = np.asarray((0.0, *rank_weights), dtype=np.float32)
    return RankBatch(
        batch=batch,
        rank=torch.from_numpy(ranks),
        rank_weight=torch.from_numpy(table[ranks]),
    )


@dataclass(frozen=True, slots=True)
class RankStep:
    valid_prefixes: int
    weight_sum: float
    metrics: dict[str, float]


def summarize_rank_step(cpu_batches: list[RankBatch], cfg: TrainConfig) -> RankStep:
    """Compute one optimizer step's FP32 normalizer and rank diagnostics."""
    if not cpu_batches:
        raise ValueError("rank step contains no batches")
    ranks = torch.cat([batch.rank for batch in cpu_batches]).to(torch.int64)
    weights = torch.cat([batch.rank_weight for batch in cpu_batches]).to(torch.float32)
    valid_counts = torch.cat([cfg.L_ctx - batch.context.ctx_pad for batch in cpu_batches]).to(torch.float32)
    if ranks.numel() != cfg.batch_size:
        raise ValueError(f"effective rank batch has {ranks.numel()} examples, expected {cfg.batch_size}")
    if not ((valid_counts > 0) & (valid_counts <= cfg.L_ctx)).all():
        raise ValueError("rank step has an invalid context-padding count")
    if not torch.isfinite(weights).all() or not (weights > 0).all():
        raise FloatingPointError("rank step contains a non-finite or non-positive weight")
    valid_prefixes = int(valid_counts.sum().item())
    weight_sum_tensor = (weights * valid_counts).sum(dtype=torch.float32)
    weight_square_sum = (weights.square() * valid_counts).sum(dtype=torch.float32)
    if valid_prefixes <= 0 or not torch.isfinite(weight_sum_tensor) or weight_sum_tensor <= 0:
        raise FloatingPointError("rank step has no finite positive objective mass")
    ess = float(weight_sum_tensor.square() / (valid_prefixes * weight_square_sum))
    if ess < cfg.minimum_rank_ess:
        raise RuntimeError(f"rank effective-sample-size fraction {ess:.4f} < gate {cfg.minimum_rank_ess:.4f}")

    weight_sum = float(weight_sum_tensor)
    metrics = {
        "rank/ess_fraction": ess,
        "rank/raw_weight_min": float(weights.min()),
        "rank/raw_weight_mean": weight_sum / valid_prefixes,
        "rank/raw_weight_max": float(weights.max()),
        "rank/valid_prefixes": float(valid_prefixes),
        "rank/weight_sum": weight_sum,
    }
    for tier in _RANK_TIERS:
        selected = ranks == int(tier)
        tier_prefixes = float(valid_counts[selected].sum())
        tier_mass = float((valid_counts[selected] * weights[selected]).sum(dtype=torch.float32))
        name = _RANK_NAMES[tier]
        metrics[f"rank/example_fraction_{name}"] = float(selected.float().mean())
        metrics[f"rank/prefix_fraction_{name}"] = tier_prefixes / valid_prefixes
        metrics[f"rank/gradient_mass_fraction_{name}"] = tier_mass / weight_sum
    return RankStep(valid_prefixes=valid_prefixes, weight_sum=weight_sum, metrics=metrics)


def rank_weighted_objective(
    nll: Tensor,
    prefix_weight: Tensor,
    *,
    weight_sum: float,
    aux_loss_weight: float,
) -> Tensor:
    """Return this microbatch's share of the full-step weighted objective."""
    if nll.ndim != 3 or nll.shape[1:] != (10, N_GROUPS):
        raise ValueError(f"rank-weighted NLL has unexpected shape {tuple(nll.shape)}")
    if prefix_weight.shape != (nll.shape[0],):
        raise ValueError("one rank weight is required per valid prefix")
    if not math.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError(f"weight_sum must be finite and positive, got {weight_sum}")
    weights = prefix_weight.detach().to(torch.float32)
    if not torch.isfinite(weights).all() or not (weights > 0).all():
        raise FloatingPointError("objective received a non-finite or non-positive rank weight")
    joint_nll = nll.to(torch.float32).sum(dim=-1)
    weighted = joint_nll * weights[:, None]
    primary = weighted[:, :4].sum() / (weight_sum * 4)
    auxiliary = weighted[:, 4:].sum() / (weight_sum * (nll.shape[1] - 4))
    return primary + aux_loss_weight * auxiliary


def device_rank_batches(
    cpu_batches: list[RankBatch],
    device: str | torch.device,
    copy_stream: torch.cuda.Stream | None,
) -> Iterator[RankBatch]:
    """Move one rank batch ahead while preserving CUDA allocator ownership."""
    if not cpu_batches:
        return
    target = torch.device(device)
    if target.type != "cuda" or copy_stream is None:
        for batch in cpu_batches:
            yield batch.to(target)
        return
    compute_stream = torch.cuda.current_stream(target)
    with torch.cuda.stream(copy_stream):
        staged = cpu_batches[0].to(target)
    for index in range(len(cpu_batches)):
        compute_stream.wait_stream(copy_stream)
        ready = staged
        ready.record_stream(compute_stream)
        if index + 1 < len(cpu_batches):
            with torch.cuda.stream(copy_stream):
                staged = cpu_batches[index + 1].to(target)
        yield ready


def _make_loaders(cfg: TrainConfig, stats: dict[str, FeatureStats]):
    kwargs = loader_kwargs(cfg, stats)
    lookup = load_rank_lookup(cfg)
    transform = functools.partial(collate_rank_batch, rank_weights=cfg.rank_weights)
    train_loader = make_reservoir_loader(
        split="train",
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor,
        predownload=cfg.predownload,
        windows_per_replay=cfg.windows_per_replay,
        reservoir_capacity=cfg.reservoir_capacity,
        prefetch_batches=cfg.prefetch_batches,
        replay_format="policy",
        window_transform=lookup,
        batch_transform=transform,
        **kwargs,
    )
    val_kwargs = {**kwargs, "batch_size": cfg.val_batch_size}
    val_loader = make_loader(split=cfg.val_split, num_workers=0, compact=True, **val_kwargs)
    return train_loader, cache_validation(val_loader, cfg.val_n_samples)


def train(
    cfg: TrainConfig,
    stats: dict[str, FeatureStats],
    *,
    comment: str = "",
    resume_run: str | None = None,
    resume_state: dict | None = None,
) -> None:
    """Train 026 with whole-step self-normalized replay-rank weights."""
    validate_config(cfg)
    run_name = resume_run or make_run_name(Path(__file__).stem, model_tag(cfg), cfg.data_root, comment)
    uploader = BackgroundUploader(run_name) if cfg.push_to_r2 else None
    wandb.init(
        project="hal",
        name=run_name,
        id=None if resume_state is None else resume_state.get("wandb_id"),
        resume="allow" if resume_state is not None else None,
        tags=["gpt", "temporal-mtp", "rank-weighted-bc", "034"],
        config=asdict(cfg),
    )
    if wandb.run is not None:
        wandb.define_metric("eval/net_stock_lcb", step_metric="global_step")
        wandb.define_metric("eval/net_dmg_lcb", step_metric="global_step")
        wandb.run.summary["rank/sidecar_sha256"] = cfg.rank_sidecar_sha256
        wandb.run.summary["nll_semantics"] = (
            "rank weights affect backprop only; train/loss and every validation metric remain unweighted"
        )
        if cfg.wandb_log_code:
            log_wandb_code(wandb.run)
    run_dir, replay_dir = setup_run_dir(run_name)
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high" if cfg.allow_tf32 else "highest")
    model = GPT(cfg).to(DEVICE)
    counts = subsystem_parameter_counts(model)
    if wandb.run is not None:
        for name, value in counts.items():
            wandb.run.summary[f"parameters/{name}"] = value
        wandb.run.summary["parameters/total"] = sum(parameter.numel() for parameter in model.parameters())
    optimizer = make_optimizer(model, cfg)
    scheduler = LambdaLR(optimizer, lr_schedule(cfg))
    start_step = 0
    if resume_state is not None:
        model.load_state_dict(resume_state["model"])
        optimizer.load_state_dict(resume_state["opt"])
        scheduler.load_state_dict(resume_state["sched"])
        start_step = int(resume_state["step"]) + 1

    def trunk_fn(features, pad, actions):
        return model(features, pad, actions)

    temporal_fn: Callable = model.temporal.teacher_forced_nll
    if DEVICE == "cuda" and cfg.compile_trunk:
        trunk_fn = torch.compile(trunk_fn, dynamic=False)
    if DEVICE == "cuda" and cfg.compile_temporal:
        temporal_fn = torch.compile(temporal_fn, dynamic=False)

    train_loader, val_cache = _make_loaders(cfg, stats)
    iterator = iter(train_loader)
    copy_stream = torch.cuda.Stream() if DEVICE == "cuda" else None
    run_started = time.monotonic()
    eval_inference: BF16Inference | None = None
    model.train()
    try:
        for step in range(start_step, cfg.max_steps):
            if DEVICE == "cuda":
                torch.cuda.reset_peak_memory_stats()
            loader_started = time.monotonic()
            cpu_batches: list[RankBatch] = []
            for _ in range(cfg.grad_accum_steps):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(train_loader)
                    batch = next(iterator)
                if not isinstance(batch, RankBatch):
                    raise TypeError(f"rank loader yielded {type(batch).__name__}, expected RankBatch")
                validate_batch_geometry(batch, cfg, micro_batch_size(cfg))
                cpu_batches.append(batch)
            loader_wait = time.monotonic() - loader_started
            rank_step = summarize_rank_step(cpu_batches, cfg)
            optimizer.zero_grad()
            nll_sum = torch.zeros(len(cfg.head_offsets), N_GROUPS, device=DEVICE)
            n_prefixes = 0
            weighted_objective_nats = torch.zeros((), device=DEVICE)
            with profile("step") as stopwatch:
                for batch in device_rank_batches(cpu_batches, DEVICE, copy_stream):
                    history, targets, valid = prepared_targets(model, batch)
                    with amp_context(cfg, DEVICE):
                        hidden = trunk_fn(batch.context.features, batch.context.ctx_pad, history)
                        dense_nll = temporal_fn(hidden, history, targets)
                        parts = ActionLoss(nll=dense_nll[valid], targets=targets[valid])
                        loss = rank_weighted_objective(
                            parts.nll,
                            batch.valid_rank_weights(valid),
                            weight_sum=rank_step.weight_sum,
                            aux_loss_weight=cfg.aux_loss_weight,
                        )
                    loss.backward()
                    weighted_objective_nats += loss.detach()
                    nll_sum += parts.nll.detach().sum(dim=0)
                    n_prefixes += parts.nll.shape[0]
                if n_prefixes != rank_step.valid_prefixes:
                    raise RuntimeError(
                        f"GPU valid-prefix count {n_prefixes} != CPU normalizer count {rank_step.valid_prefixes}"
                    )
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError(f"step {step}: non-finite gradient norm {gradient_norm}")
                optimizer.step()
                scheduler.step()
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
            metrics = nll_mean_metrics((nll_sum / n_prefixes).cpu(), cfg.head_offsets)
            log = {
                "global_step": step,
                "samples": (step + 1) * cfg.batch_size,
                **{f"train/{name}": value for name, value in metrics.items()},
                **rank_step.metrics,
                "train/rank_weighted_objective": float(weighted_objective_nats / _LN2),
                "train/grad_norm": float(gradient_norm),
                "throughput/step_s": stopwatch.elapsed,
                "throughput/loader_wait_s": loader_wait,
                "throughput/samples_per_s": cfg.batch_size / stopwatch.elapsed,
                "throughput/samples_per_wall_s": cfg.batch_size / (stopwatch.elapsed + loader_wait),
                "throughput/prefixes_per_s": n_prefixes / stopwatch.elapsed,
                "lr/muon": next(group["lr"] for group in optimizer.param_groups if group["use_muon"]),
                "lr/adam": next(group["lr"] for group in optimizer.param_groups if not group["use_muon"]),
            }
            if DEVICE == "cuda":
                log["hardware/peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 2**30
                log["hardware/peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
            wandb.log(log)
            if step < 10 or step % 50 == 0:
                print(
                    f"[t+{time.monotonic() - run_started:.0f}s] step {step}: "
                    f"{metrics['loss']:.3f} bits unweighted, "
                    f"{float(weighted_objective_nats / _LN2):.3f} bits weighted, "
                    f"{cfg.batch_size / stopwatch.elapsed:.0f} samples/s",
                    flush=True,
                )
            val_due = cfg.val_every > 0 and step > 0 and step % cfg.val_every == 0
            eval_due = cfg.eval_every > 0 and step > 0 and step % cfg.eval_every == 0
            ckpt_due = cfg.ckpt_every > 0 and step > 0 and step % cfg.ckpt_every == 0
            checkpoint_path = run_dir / "latest.pt"
            if val_due or eval_due or ckpt_due:
                save_checkpoint(
                    checkpoint_path,
                    step=step,
                    model=model,
                    opt=optimizer,
                    sched=scheduler,
                    cfg=asdict(cfg),
                    wandb_id=None if wandb.run is None else wandb.run.id,
                    uploader=uploader,
                )
            if val_due:
                values = val_metrics(model, val_cache, cfg)
                wandb.log({"global_step": step, **{f"val/{name}": value for name, value in values.items()}})
            if eval_due:
                if eval_inference is None:
                    eval_inference = BF16Inference(model, cfg)
                values = eval_vs_cpu(
                    model,
                    stats,
                    cfg,
                    n_matchups=cfg.eval_n_matchups,
                    replay_dir=replay_dir / f"step_{step:06d}",
                    checkpoint_sha256=_checkpoint_sha256(checkpoint_path),
                    inference=eval_inference,
                )
                wandb.log({"global_step": step, **{f"eval/{name}": value for name, value in values.items()}})

        final_path = run_dir / "final.pt"
        save_checkpoint(
            final_path,
            step=cfg.max_steps,
            model=model,
            opt=optimizer,
            sched=scheduler,
            cfg=asdict(cfg),
            wandb_id=None if wandb.run is None else wandb.run.id,
            uploader=uploader,
        )
        checkpoint_sha = _checkpoint_sha256(final_path)
        final_val = val_metrics(model, val_cache, cfg)
        wandb.log({"global_step": cfg.max_steps, **{f"val/{name}": value for name, value in final_val.items()}})
        if eval_inference is None:
            eval_inference = BF16Inference(model, cfg)
        final_eval = eval_vs_cpu(
            model,
            stats,
            cfg,
            n_matchups=cfg.final_eval_n_matchups,
            replay_dir=replay_dir / "final",
            checkpoint_sha256=checkpoint_sha,
            inference=eval_inference,
        )
        wandb.log({"global_step": cfg.max_steps, **{f"eval/{name}": value for name, value in final_eval.items()}})
        stride6 = eval_vs_cpu(
            model,
            stats,
            cfg,
            n_matchups=cfg.final_diag_n_matchups,
            replay_dir=replay_dir / "final_s6",
            exec_horizon=cfg.final_diag_exec_horizon,
            checkpoint_sha256=checkpoint_sha,
            inference=eval_inference,
        )
        wandb.log({"global_step": cfg.max_steps, **{f"eval_s6/{name}": value for name, value in stride6.items()}})
    finally:
        if uploader is not None:
            uploader.upload_tree(replay_dir, base=run_dir)
            uploader.close()
        wandb.finish()


_RANK_CHECKPOINT_FIELDS = {
    "experiment_id",
    "minimum_rank_ess",
    "rank_manifest_sha256",
    "rank_player_counts",
    "rank_sidecar_rows",
    "rank_sidecar_sha256",
    "rank_weights",
}


def config_from_state(values: dict) -> TrainConfig:
    """Restore only an explicitly identified experiment-034 checkpoint."""
    missing = (_CHECKPOINT_ARCH_FIELDS | _RANK_CHECKPOINT_FIELDS) - values.keys()
    if missing:
        raise ValueError(f"checkpoint is not experiment 034; missing {sorted(missing)}")
    if values["experiment_id"] != _EXPERIMENT_ID:
        raise ValueError(f"checkpoint experiment_id {values['experiment_id']!r} != required {_EXPERIMENT_ID!r}")
    known = {item.name for item in fields(TrainConfig)}
    return TrainConfig(**{name: value for name, value in values.items() if name in known})


def _validate_evaluation_override(cfg: TrainConfig, checkpoint_cfg: TrainConfig) -> None:
    """Allow only non-scientific execution overrides after checkpoint load."""
    _validate_026_config(cfg)
    _validate_rank_contract(cfg)
    changed = _config_changes(cfg, checkpoint_cfg)
    forbidden = {name: value for name, value in changed.items() if name not in _EVALUATION_OVERRIDE_FIELDS}
    if forbidden:
        raise ValueError(f"evaluation changed checkpoint-scientific fields: {forbidden}")


def eval_checkpoint(
    path: str,
    *,
    exec_horizon: int | None = None,
    n_matchups: int | None = None,
    eager: bool = False,
    max_parallel: int | None = None,
    output_name: str | None = None,
) -> dict[str, float]:
    """Evaluate a provenance-checked 034 checkpoint with runtime overrides."""
    model, checkpoint_cfg, stats, state = load_checkpoint(path)
    cfg = replace(
        checkpoint_cfg,
        inference_mode="eager" if eager else checkpoint_cfg.inference_mode,
        eval_max_parallel=checkpoint_cfg.eval_max_parallel if max_parallel is None else max_parallel,
    )
    _validate_evaluation_override(cfg, checkpoint_cfg)
    horizon = cfg.exec_horizon if exec_horizon is None else exec_horizon
    default_name = "eval_replays_s6" if horizon == 6 else "eval_replays"
    if output_name is not None and (Path(output_name).name != output_name or output_name in ("", ".", "..")):
        raise ValueError(f"evaluation output name must be one directory name, got {output_name!r}")
    replay_dir = Path(path).resolve().parent / (default_name if output_name is None else output_name)
    values = eval_vs_cpu(
        model,
        stats,
        cfg,
        n_matchups=cfg.final_eval_n_matchups if n_matchups is None else n_matchups,
        replay_dir=replay_dir,
        exec_horizon=horizon,
        checkpoint_sha256=_checkpoint_sha256(Path(path)),
    )
    print(f"[eval] step={state['step']} horizon={horizon}: {values}", flush=True)
    return values


@dataclass
class Args(base.Args):
    cfg: TrainConfig = dataclass_field(default_factory=TrainConfig)


base.TrainConfig = TrainConfig
base.validate_config = validate_config
base.model_tag = model_tag
base.config_from_state = config_from_state
base.eval_checkpoint = eval_checkpoint
base.train = train
base.__file__ = __file__


if __name__ == "__main__":
    base.main(tyro.cli(Args))
