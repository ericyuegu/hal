"""Paired real-model benchmark for the policy-world-v8 loader rollout.

The control reads the pilot's retained replay set through the existing physical
shard loader and replay buffer. The treatment reads the same replays through
Mosaic and the policy-world-v8 window wrapper. Both conditions train the same
transformer from the same initialization and record every measured wait.
"""

from __future__ import annotations

import contextlib
import copy
import gc
import hashlib
import json
import os
import resource
import subprocess
import threading
import time
from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import asdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Final
from typing import Literal
from typing import cast

import numpy as np
import torch
import torch.nn as nn
import tyro
from torch import Tensor

import wandb
from hal import r2
from hal import streams
from hal.data.feature_stats import FeatureStats
from hal.data.index import read_jsonl
from hal.data.schema import SCHEMA_VERSION
from hal.training.dataloader import ResumableStreamingDataLoader
from hal.training.dataloader import make_loader
from hal.training.dataloader import train_batch_from_columns
from hal.training.ego_stats import load_consolidated_mixture_stats
from hal.training.features import BASE_ITEMS_PROJECTION
from hal.training.features import ITEM_COLUMNS
from hal.training.features import TrainBatch
from hal.training.mfu import bf16_dense_peak_flops
from hal.training.mfu import bf16_peak_source
from hal.training.physical_shard_loader import MDSStorageAdapter
from hal.training.physical_shard_loader import PhysicalShardReplayLoader
from hal.training.physical_shard_loader import PhysicalShardSelection
from hal.training.physical_shard_loader import SourceRowSelection
from hal.training.physical_shard_loader import build_shard_plan
from hal.training.trunk import Trunk
from hal.training.trunk import TrunkConfig

_SOURCE_NAMES: Final[tuple[str, str]] = (
    "professional-aklo-policy-world-v7",
    "ranked-anonymized-1-policy-world-v7",
)
_CONTEXT = 256
_CHUNK = 20
_CACHE_LIMIT = 4 * 2**30
_WINDOWS_PER_GENERATION = 8


def _git_sha() -> str:
    sha = os.environ.get("HAL_GIT_SHA")
    if sha is None:
        sha = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    if len(sha) != 40 or any(character not in "0123456789abcdef" for character in sha):
        raise ValueError("Git SHA must be a 40-character lowercase hexadecimal value")
    return sha


@dataclass(frozen=True, slots=True)
class Args:
    pilot_prefixes: tuple[str, str]
    num_workers: Literal[24, 32] = 24
    batch_size: int = 512
    warmup_updates: int = 100
    measured_updates: int = 1_000
    replay_slots: int = 4_096
    seed: int = 0
    cache_root: Path = Path("/tmp/hal-policy-world-v8-benchmark")
    report: Path = Path("data/builds/policy-world-v8/benchmark.json")
    compile_model: bool = True
    wandb_entity: str | None = None
    wandb_project: str = "policy-world-v8-loader"
    repetition: int = 0

    def __post_init__(self) -> None:
        if self.batch_size != 512 or self.measured_updates < 1_000:
            raise ValueError("the rollout gate requires batch 512 and at least 1,000 measured updates")
        if self.warmup_updates < 1 or self.replay_slots < self.batch_size:
            raise ValueError("warmup_updates must be positive and replay_slots must cover one batch")
        if len(set(self.pilot_prefixes)) != 2:
            raise ValueError("pilot prefixes must be distinct")


@dataclass(frozen=True, slots=True)
class PilotData:
    treatment_sources: tuple[streams.StreamSource, ...]
    control_selection: PhysicalShardSelection
    source_manifest_sha256: dict[str, str]
    stats: dict[str, FeatureStats]
    rows: dict[str, int]


def _rclone_uri(uri: str) -> str:
    if uri.startswith("r2:"):
        return uri.rstrip("/")
    if uri.startswith("s3://hal/"):
        return f"r2:hal/{uri.removeprefix('s3://hal/')}".rstrip("/")
    raise ValueError(f"unsupported R2 URI {uri!r}")


def _s3_uri(uri: str) -> str:
    remote = _rclone_uri(uri)
    return f"s3://hal/{remote.removeprefix('r2:hal/')}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _copy_metadata(prefix: str, name: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    r2.copy_file(f"{_rclone_uri(prefix)}/{name}", destination)
    return destination


def _selected_control_rows(
    source: streams.StreamSource,
    pilot_prefix: str,
    root: Path,
) -> tuple[SourceRowSelection, str, int]:
    source_manifest = _copy_metadata(
        source.remote,
        "manifest.jsonl",
        root / "control" / source.name / "manifest.jsonl",
    )
    pilot_manifest = _copy_metadata(
        pilot_prefix,
        "manifest.jsonl",
        root / "treatment" / source.name / "manifest.jsonl",
    )
    selection = json.loads(
        _copy_metadata(
            pilot_prefix,
            "selection.json",
            root / "treatment" / source.name / "selection.json",
        ).read_text()
    )
    expected_manifest = selection["source_hashes"]["metadata"]["manifest.jsonl"]["sha256"]
    actual_manifest = _sha256(source_manifest)
    if actual_manifest != expected_manifest:
        raise ValueError(f"{source.name}: v7 manifest hash differs from the pilot selection")

    selected_paths = {
        entry.path
        for entry in read_jsonl(pilot_manifest)
        if entry.annotation is not None and entry.annotation.split == "train"
    }
    source_rows = [
        entry
        for entry in read_jsonl(source_manifest)
        if entry.annotation is not None and entry.annotation.split == "train"
    ]
    by_path = {entry.path: int(entry.annotation.mds_row_idx) for entry in source_rows if entry.annotation is not None}
    missing = sorted(selected_paths - set(by_path))
    if missing:
        raise ValueError(f"{source.name}: {len(missing)} pilot paths are absent from the v7 manifest")
    selected_rows = sorted(by_path[path] for path in selected_paths)
    if not selected_rows:
        raise ValueError(f"{source.name}: pilot has no retained training rows")
    stop = selected_rows[-1] + 1
    selected_set = set(selected_rows)
    excluded = tuple(row for row in range(stop) if row not in selected_set)
    view = SourceRowSelection(source.name, stop, excluded)
    if view.row_count != len(selected_paths):
        raise AssertionError("control selection changed the eligible replay count")
    return view, actual_manifest, view.row_count


def prepare_pilot_data(args: Args) -> PilotData:
    args.cache_root.mkdir(parents=True, exist_ok=True)
    registered = {source.name: source for source in streams.POLICY_WORLD_V7_SOURCES}
    views: list[SourceRowSelection] = []
    manifest_hashes: dict[str, str] = {}
    row_counts: dict[str, int] = {}
    treatment_sources: list[streams.StreamSource] = []
    stats_paths: list[Path] = []
    for source_name, prefix in zip(_SOURCE_NAMES, args.pilot_prefixes, strict=True):
        source = registered[source_name]
        view, manifest_hash, rows = _selected_control_rows(source, prefix, args.cache_root)
        views.append(view)
        manifest_hashes[source_name] = manifest_hash
        row_counts[source_name] = rows
        local = args.cache_root / "mds" / source_name
        treatment_sources.append(streams.StreamSource(f"{source_name}-pilot-v8", _s3_uri(prefix), local))
        stats_paths.append(_copy_metadata(prefix, "stats.json", local / "stats.json"))

    selection_payload = {
        "sources": [asdict(view) for view in views],
        "pilot_prefixes": args.pilot_prefixes,
    }
    selection_hash = hashlib.sha256(json.dumps(selection_payload, sort_keys=True).encode()).hexdigest()
    selection = PhysicalShardSelection(tuple(views), selection_hash)
    if selection.row_count != sum(row_counts.values()):
        raise AssertionError("paired selection row accounting differs")
    stats = load_consolidated_mixture_stats(
        stats_paths,
        [float(row_counts[name]) for name in _SOURCE_NAMES],
        expected_mds_schema_version=SCHEMA_VERSION,
    )
    return PilotData(tuple(treatment_sources), selection, manifest_hashes, stats, row_counts)


def _no_labels(_compact: Mapping[str, object]) -> dict[str, np.ndarray]:
    return {}


def _control_batch(
    replay_ids: tuple[str, ...],
    columns: Mapping[str, np.ndarray],
    *,
    stats: dict[str, FeatureStats],
) -> TrainBatch:
    batch = train_batch_from_columns(
        columns,
        stats=stats,
        L_ctx=_CONTEXT,
        extra=ITEM_COLUMNS,
        projection=BASE_ITEMS_PROJECTION,
    )
    return TrainBatch(batch.context, batch.target, replay_ids)


def make_control_loader(args: Args, data: PilotData) -> PhysicalShardReplayLoader[TrainBatch]:
    adapter = MDSStorageAdapter(data.control_selection, download_retry=8)
    tasks = build_shard_plan(data.control_selection, adapter.manifests)
    return PhysicalShardReplayLoader(
        selection=data.control_selection,
        adapter=adapter,
        tasks=tasks,
        data_protocol="policy-world-v8-paired-pilot-control-v1",
        source_manifest_sha256=data.source_manifest_sha256,
        labels=_no_labels,
        projection=BASE_ITEMS_PROJECTION,
        batch_transform=partial(_control_batch, stats=data.stats),
        batch_size=args.batch_size,
        replay_slots=args.replay_slots,
        seed=args.seed,
        num_workers=args.num_workers,
        context_length=_CONTEXT,
        chunk_length=_CHUNK,
        windows_per_generation=_WINDOWS_PER_GENERATION,
        schema_version=SCHEMA_VERSION,
        reserved_disk_bytes=16 * 2**30,
        pin_memory=True,
    )


def make_treatment_loader(
    args: Args,
    data: PilotData,
    *,
    num_workers: int | None = None,
) -> ResumableStreamingDataLoader:
    loader = make_loader(
        None,
        "train",
        stats=data.stats,
        L_ctx=_CONTEXT,
        L_chunk=_CHUNK,
        batch_size=args.batch_size,
        seed=args.seed,
        sources=data.treatment_sources,
        cache_limit=_CACHE_LIMIT,
        shuffle_block_size=8_192,
        shuffle=True,
        shuffle_seed=args.seed,
        num_workers=args.num_workers if num_workers is None else num_workers,
        prefetch_factor=2,
        predownload=8 * args.batch_size,
        drop_last=True,
        resumable=True,
        in_order=True,
        pin_memory=True,
        schema_version=SCHEMA_VERSION,
        extra=ITEM_COLUMNS,
        projection=BASE_ITEMS_PROJECTION,
        replay_format="policy-world-v8",
        require_full_context=True,
    )
    if not isinstance(loader, ResumableStreamingDataLoader):
        raise TypeError("v8 treatment did not create the resumable Mosaic loader")
    return loader


class BenchmarkPolicy(nn.Module):
    """Production-width causal transformer with a framewise controller loss."""

    def __init__(self) -> None:
        super().__init__()
        self.columns = tuple(sorted(BASE_ITEMS_PROJECTION.columns))
        self.input = nn.Linear(len(self.columns), 512)
        self.trunk = Trunk(
            TrunkConfig(
                d_model=512,
                n_layers=16,
                n_heads=8,
                L_ctx=_CONTEXT,
                attention_backend="varlen_flash",
            )
        )
        self.output = nn.Linear(512, 14)

    def forward(self, batch: TrainBatch) -> Tensor:
        missing = sorted(set(self.columns) - set(batch.context.features))
        if missing:
            raise KeyError(f"benchmark context is missing projected columns: {missing}")
        values = torch.stack([batch.context.features[name].float() for name in self.columns], dim=-1)
        values = values / (1.0 + values.abs())
        hidden = self.trunk.forward_unpadded(self.input(values))
        return self.output(hidden[:, -_CHUNK:])


def _model_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _batch_hash(batch: TrainBatch) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(batch.context.features.items()):
        digest.update(name.encode())
        digest.update(value.numpy().tobytes())
    digest.update(batch.context.ctx_pad.numpy().tobytes())
    digest.update(batch.target.numpy().tobytes())
    return digest.hexdigest()


def _cgroup_memory() -> int:
    try:
        return int(Path("/sys/fs/cgroup/memory.current").read_text())
    except FileNotFoundError, ValueError:
        return 0


def _network_bytes() -> int:
    total = 0
    for line in Path("/proc/net/dev").read_text().splitlines()[2:]:
        fields = line.replace(":", " ").split()
        total += int(fields[1]) + int(fields[9])
    return total


def _tree_bytes(root: Path) -> int:
    total = 0
    for directory, _subdirectories, files in os.walk(root):
        for name in files:
            with contextlib.suppress(FileNotFoundError):
                total += (Path(directory) / name).stat().st_size
    return total


class ResourceSampler:
    def __init__(self, cache_root: Path) -> None:
        self.cache_root = cache_root
        self.stop = threading.Event()
        self.memory_start = _cgroup_memory()
        self.network_start = _network_bytes()
        self.memory_peak = self.memory_start
        self.cache_peak = _tree_bytes(cache_root)
        self.gpu_utilization: list[float] = []
        self.thread = threading.Thread(target=self._sample, name="v8-resource-sampler", daemon=True)

    def __enter__(self) -> ResourceSampler:
        self.thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop.set()
        self.thread.join()

    def _sample(self) -> None:
        tick = 0
        while not self.stop.wait(1.0):
            self.memory_peak = max(self.memory_peak, _cgroup_memory())
            if tick % 10 == 0:
                self.cache_peak = max(self.cache_peak, _tree_bytes(self.cache_root))
            tick += 1
            result = subprocess.run(
                ("nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"),
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                self.gpu_utilization.extend(float(value) for value in result.stdout.split() if value)

    def mark_steady_state(self) -> None:
        """Keep startup peaks but reset rate and utilization measurements."""
        self.network_start = _network_bytes()
        self.gpu_utilization.clear()

    def metrics(self) -> dict[str, float | int]:
        self.memory_peak = max(self.memory_peak, _cgroup_memory())
        self.cache_peak = max(self.cache_peak, _tree_bytes(self.cache_root))
        gpu = np.asarray(self.gpu_utilization, dtype=np.float64)
        return {
            "host_memory_start_bytes": self.memory_start,
            "host_memory_peak_bytes": self.memory_peak,
            "loader_host_memory_growth_bytes": max(0, self.memory_peak - self.memory_start),
            "cache_peak_bytes": self.cache_peak,
            "network_bytes": max(0, _network_bytes() - self.network_start),
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "gpu_utilization_mean": float(gpu.mean()) if len(gpu) else float("nan"),
            "gpu_idle_fraction": float(np.mean(gpu < 10.0)) if len(gpu) else float("nan"),
        }


def _update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: TrainBatch,
    device: torch.device,
) -> tuple[float, float]:
    device_batch = batch.to(device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        prediction = model(device_batch)
        loss = torch.nn.functional.smooth_l1_loss(prediction.float(), device_batch.target.float())
    loss.backward()
    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach()), float(gradient.detach())


def _percentiles(values: list[float], prefix: str) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_mean": float(array.mean()),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_p95": float(np.percentile(array, 95)),
        f"{prefix}_p99": float(np.percentile(array, 99)),
    }


def benchmark_condition(
    name: str,
    loader: Iterable[TrainBatch],
    args: Args,
    *,
    cache_root: Path,
) -> tuple[dict[str, object], list[dict[str, float]]]:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model: nn.Module = BenchmarkPolicy().cuda().train()
    initial_model_sha256 = _model_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    if args.compile_model:
        model = cast(nn.Module, torch.compile(model, dynamic=False, fullgraph=True, mode="reduce-overhead"))
    timings: list[dict[str, float]] = []
    losses: list[float] = []
    gradients: list[float] = []
    with ResourceSampler(cache_root) as resources:
        iterator = iter(loader)
        started = time.monotonic()
        batch = next(iterator)
        first_batch_s = time.monotonic() - started
        for _ in range(args.warmup_updates):
            loss, gradient = _update(model, optimizer, batch, torch.device("cuda"))
            if not np.isfinite(loss) or not np.isfinite(gradient):
                raise FloatingPointError(f"{name}: warmup produced non-finite loss or gradient")
            batch = next(iterator)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        resources.mark_steady_state()
        for update in range(args.measured_updates):
            update_started = time.monotonic()
            wait_started = update_started
            batch = next(iterator)
            loader_wait_s = time.monotonic() - wait_started
            loss, gradient = _update(model, optimizer, batch, torch.device("cuda"))
            torch.cuda.synchronize()
            full_update_s = time.monotonic() - update_started
            timings.append(
                {
                    "update": float(update + 1),
                    "loader_wait_s": loader_wait_s,
                    "full_update_s": full_update_s,
                }
            )
            losses.append(loss)
            gradients.append(gradient)
        resource_metrics = resources.metrics()

    waits = [row["loader_wait_s"] for row in timings]
    updates = [row["full_update_s"] for row in timings]
    throughputs = [args.batch_size / duration for duration in updates]
    parameter_count = sum(value.numel() for value in model.parameters())
    device_name = torch.cuda.get_device_name()
    peak_flops = bf16_dense_peak_flops(device_name)
    update_median = float(np.median(updates))
    approximate_flops = 6 * parameter_count * args.batch_size * _CONTEXT
    metrics: dict[str, object] = {
        "condition": name,
        "initial_model_sha256": initial_model_sha256,
        "parameter_count": parameter_count,
        "batch_size": args.batch_size,
        "context": _CONTEXT,
        "chunk": _CHUNK,
        "workers": args.num_workers,
        "warmup_updates": args.warmup_updates,
        "measured_updates": args.measured_updates,
        "first_batch_s": first_batch_s,
        "training_samples_per_s_mean": float(np.mean(throughputs)),
        "training_samples_per_s_median": float(np.median(throughputs)),
        "stalls_above_2s": int(np.sum(np.asarray(waits) > 2.0)),
        "loss_mean": float(np.mean(losses)),
        "loss_min": float(np.min(losses)),
        "loss_max": float(np.max(losses)),
        "gradient_norm_mean": float(np.mean(gradients)),
        "gradient_norm_max": float(np.max(gradients)),
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "approximate_mfu": None if peak_flops is None else approximate_flops / update_median / peak_flops,
        "peak_flops_source": bf16_peak_source(device_name),
        **_percentiles(waits, "loader_wait_s"),
        **_percentiles(updates, "full_update_s"),
        **resource_metrics,
    }
    if isinstance(loader, PhysicalShardReplayLoader):
        metrics.update(
            {
                "loader_raw_bytes_read": loader.raw_bytes_read,
                "loader_buffer_bytes": loader.buffer_bytes,
                "loader_generations_read": loader.generations_read,
            }
        )
    close = getattr(loader, "close", None)
    if callable(close):
        close()
    return metrics, timings


def _metric(metrics: Mapping[str, object], name: str) -> float:
    value = metrics[name]
    if not isinstance(value, int | float):
        raise TypeError(f"benchmark metric {name!r} is not numeric")
    return float(value)


def _gate(control: Mapping[str, object], treatment: Mapping[str, object], resume_exact: bool) -> dict[str, object]:
    throughput_ratio = _metric(treatment, "training_samples_per_s_median") / _metric(
        control, "training_samples_per_s_median"
    )
    wait_ratio = _metric(treatment, "loader_wait_s_p99") / max(_metric(control, "loader_wait_s_p99"), 1e-12)
    checks = {
        "throughput_at_least_95_percent": throughput_ratio >= 0.95,
        "p99_wait_not_over_110_percent": wait_ratio <= 1.10,
        "no_repeated_stalls_above_2s": _metric(treatment, "stalls_above_2s") < 2,
        "first_batch_below_2_minutes": _metric(treatment, "first_batch_s") < 120.0,
        "loader_host_memory_below_16_gib": _metric(treatment, "loader_host_memory_growth_bytes") < 16 * 2**30,
        "cache_below_4_gib": _metric(treatment, "cache_peak_bytes") <= _CACHE_LIMIT,
        "resume_exact": resume_exact,
        "finite_loss_and_gradients": bool(
            np.isfinite(_metric(treatment, "loss_mean")) and np.isfinite(_metric(treatment, "gradient_norm_mean"))
        ),
    }
    return {
        "green": all(checks.values()),
        "checks": checks,
        "median_throughput_ratio": throughput_ratio,
        "p99_loader_wait_ratio": wait_ratio,
    }


def verify_treatment_resume(args: Args, data: PilotData) -> bool:
    """Reproduce the next batch and one optimizer update from a Mosaic cursor."""
    loader = make_treatment_loader(args, data, num_workers=0)
    iterator = iter(loader)
    next(iterator)
    state = loader.state_dict()
    expected = next(iterator)
    expected_batch_hash = _batch_hash(expected)
    del iterator, loader
    gc.collect()

    restored = make_treatment_loader(args, data, num_workers=0)
    restored.load_state_dict(state)
    actual = next(iter(restored))
    if _batch_hash(actual) != expected_batch_hash:
        return False

    torch.manual_seed(args.seed)
    model = BenchmarkPolicy().cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    model_state = copy.deepcopy(model.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    _update(model, optimizer, expected, torch.device("cuda"))
    torch.cuda.synchronize()
    expected_update = _model_hash(model)
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
    _update(model, optimizer, actual, torch.device("cuda"))
    torch.cuda.synchronize()
    return _model_hash(model) == expected_update


def run(args: Args) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("the paired real-model benchmark requires CUDA")
    data = prepare_pilot_data(args)
    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=f"policy-world-v8-w{args.num_workers}-r{args.repetition}",
        config=asdict(args),
    )
    resume_exact = verify_treatment_resume(args, data)
    conditions: dict[str, dict[str, object]] = {}
    raw_timings: dict[str, list[dict[str, float]]] = {}
    for name, loader, cache in (
        ("control", make_control_loader(args, data), Path("data/processed")),
        ("treatment", make_treatment_loader(args, data), args.cache_root / "mds"),
    ):
        metrics, timings = benchmark_condition(name, loader, args, cache_root=cache)
        conditions[name] = metrics
        raw_timings[name] = timings
        run.log({f"{name}/{key}": value for key, value in metrics.items() if isinstance(value, int | float)})
    if conditions["control"]["initial_model_sha256"] != conditions["treatment"]["initial_model_sha256"]:
        raise RuntimeError("paired conditions did not start from identical model weights")
    gate = _gate(conditions["control"], conditions["treatment"], resume_exact)
    report: dict[str, object] = {
        "schema_version": 1,
        "git_sha": _git_sha(),
        "wandb": {"entity": run.entity, "project": run.project, "run_id": run.id, "url": run.url},
        "configuration": asdict(args),
        "invariants": {
            "eligible_rows": data.rows,
            "selection_sha256": data.control_selection.sha256,
            "seed": args.seed,
            "projection": sorted(BASE_ITEMS_PROJECTION.columns),
            "optimizer": "AdamW(lr=3e-4, weight_decay=1e-3)",
        },
        "conditions": conditions,
        "gate": gate,
        "timings": raw_timings,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    artifact = wandb.Artifact(f"policy-world-v8-benchmark-{run.id}", type="benchmark")
    artifact.add_file(str(args.report))
    run.log_artifact(artifact)
    run.summary.update({"gate_green": gate["green"], "resume_exact": resume_exact})
    run.finish()
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


if __name__ == "__main__":
    run(tyro.cli(Args))
