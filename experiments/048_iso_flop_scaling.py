"""Iso-FLOP scaling with nested, content-deduplicated replay corpora.

Status: canceled before launch; never run.

This experiment preserves O43's legacy controller codec and loss. It scales one
architecture dial, trains without learning-rate decay, and exposes only as much
unique replay data as the endpoint processes.

Inspect the fixed grid without R2 access:

    uv run experiments/048_iso_flop_scaling.py --describe

Build the compact corpus index once from the immutable R2 manifests:

    uv run experiments/048_iso_flop_scaling.py --build-corpus-index

Train one endpoint:

    uv run experiments/048_iso_flop_scaling.py --endpoint c1e16-d256
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import json
import math
import sys
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import replace
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Final

import tyro

import wandb
from hal import r2
from hal import streams
from hal.data.policy_schema import policy_replay_identity
from hal.streams import StreamSource
from hal.training.checkpoints import load_for_resume
from hal.training.ego_stats import load_consolidated_mixture_stats


def _load_o43() -> ModuleType:
    path = Path(__file__).with_name("043_legacy_codec.py")
    name = "_hal_experiment_043_for_048"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_o43 = _load_o43()

_PROTOCOL: Final[str] = "048-sha1-dedup-shard-order-v1"
_CORPUS_SCHEMA: Final[int] = 1
_CORPUS_INDEX_PATH: Final[Path] = Path("data/processed/048_policy_world_unique_v1.tsv.gz")
_CORPUS_INDEX_KEY: Final[str] = "processed/048-policy-world-unique-v1.tsv.gz"
_CORPUS_BUILD_ROOT: Final[Path] = Path("data/processed/048_corpus_build")
_TIER_CACHE_ROOT: Final[Path] = Path("data/processed/048_iso_flop_tiers")
_WANDB_GROUP: Final[str] = "048-iso-flop-scaling"

_BATCH_SIZE: Final[int] = 512
_BATCH_POSITIONS: Final[int] = _BATCH_SIZE * _o43.TrainConfig().L_ctx
_WARMUP_FRACTION: Final[float] = 0.03
_POWERLINES_EXPONENT: Final[float] = 0.52
_REFERENCE_CFG = _o43.TrainConfig()
_REFERENCE_POSITIONS: Final[int] = _REFERENCE_CFG.max_steps * _REFERENCE_CFG.batch_size * _REFERENCE_CFG.L_ctx

# Exact half-decades. The short names retain the familiar rounded "3eN" form.
_C1E16: Final[int] = 10_000_000_000_000_000
_C3E16: Final[int] = 31_622_776_601_683_793
_C1E17: Final[int] = 100_000_000_000_000_000
_C3E17: Final[int] = 316_227_766_016_837_933
_C1E18: Final[int] = 1_000_000_000_000_000_000
_C3E18: Final[int] = 3_162_277_660_168_379_332

_LAYERS: Final[dict[int, int]] = {
    128: 3,
    192: 4,
    256: 5,
    320: 7,
    384: 8,
    448: 9,
    512: 11,
}
_TEMPORAL_WIDTHS: Final[dict[int, int]] = {
    128: 64,
    192: 64,
    256: 96,
    320: 96,
    384: 128,
    448: 160,
    512: 160,
}


@dataclass(frozen=True, slots=True)
class Endpoint:
    """One model on one fixed-compute line."""

    name: str
    compute_budget_flops: int
    d_model: int
    yolo: bool = False

    @property
    def n_layers(self) -> int:
        return _LAYERS[self.d_model]

    @property
    def n_heads(self) -> int:
        return self.d_model // 64

    @property
    def temporal_d_model(self) -> int:
        return _TEMPORAL_WIDTHS[self.d_model]

    @property
    def temporal_heads(self) -> int:
        return self.temporal_d_model // 32


def _endpoint_rows() -> Iterator[Endpoint]:
    lines = (
        ("c1e16", _C1E16, (128, 192, 256, 320, 384)),
        ("c3e16", _C3E16, (128, 192, 256, 320, 384)),
        ("c1e17", _C1E17, (192, 256, 320, 384, 448)),
        ("c3e17", _C3E17, (192, 256, 320, 384, 448)),
        ("c1e18", _C1E18, (256, 320, 384, 448, 512)),
    )
    for line, compute, widths in lines:
        for width in widths:
            yield Endpoint(f"{line}-d{width}", compute, width)
    yield Endpoint("c3e18-d448-yolo", _C3E18, 448, yolo=True)


POINTS: Final[dict[str, Endpoint]] = {endpoint.name: endpoint for endpoint in _endpoint_rows()}
TRAIN_ENDPOINTS: Final[tuple[str, ...]] = tuple(POINTS)


@dataclass(frozen=True, slots=True)
class ModelQuantities:
    """Parameter and compute counts for one architecture width."""

    total_parameters: int
    effective_parameters: int
    flops_per_update: int


def _geometry_config(endpoint: Endpoint):
    temporal_width = endpoint.temporal_d_model
    return replace(
        _o43.TrainConfig(),
        d_model=endpoint.d_model,
        n_layers=endpoint.n_layers,
        n_heads=endpoint.n_heads,
        temporal_d_model=temporal_width,
        temporal_layers=2,
        temporal_heads=endpoint.temporal_heads,
        temporal_ff_dim=2 * temporal_width,
        group_head_dim=2 * temporal_width,
        batch_size=_BATCH_SIZE,
    )


@cache
def model_quantities(d_model: int) -> ModelQuantities:
    """Measure literal and FLOP-equivalent parameters for one family member."""
    endpoint = Endpoint("geometry", _C1E16, d_model)
    cfg = _geometry_config(endpoint)
    model = _o43.GPT(cfg)
    counts = _o43.subsystem_parameter_counts(model)
    flops_per_update = _o43.approximate_training_flops_per_update(cfg, counts)
    effective_parameters = flops_per_update // (6 * cfg.batch_size * cfg.L_ctx)
    return ModelQuantities(
        total_parameters=counts["total"],
        effective_parameters=effective_parameters,
        flops_per_update=flops_per_update,
    )


_REFERENCE_PARAMETERS: Final[int] = model_quantities(384).total_parameters
_REFERENCE_TPP: Final[float] = _REFERENCE_POSITIONS / _REFERENCE_PARAMETERS


def _scaled_weight_decay(reference: float, positions: int, parameters: int) -> float:
    """Scale a decoupled decay rate by the anchored AdamW-timescale law."""
    tpp_ratio = (positions / parameters) / _REFERENCE_TPP
    return reference * (_REFERENCE_POSITIONS / positions) * tpp_ratio**_POWERLINES_EXPONENT


def endpoint_config(endpoint: Endpoint):
    """Build the frozen training configuration for an endpoint."""
    quantities = model_quantities(endpoint.d_model)
    max_steps = max(1, round(endpoint.compute_budget_flops / quantities.flops_per_update))
    processed_positions = max_steps * _BATCH_POSITIONS
    warmup_steps = max(1, math.ceil(_WARMUP_FRACTION * max_steps))
    checkpoint_every = max(250, min(4096, max_steps // 4))
    base = _geometry_config(endpoint)
    return replace(
        base,
        muon_weight_decay=_scaled_weight_decay(
            _REFERENCE_CFG.muon_weight_decay,
            processed_positions,
            quantities.total_parameters,
        ),
        adam_weight_decay=_scaled_weight_decay(
            _REFERENCE_CFG.adam_weight_decay,
            processed_positions,
            quantities.total_parameters,
        ),
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        val_every=0,
        eval_every=0,
        ckpt_every=checkpoint_every,
        final_eval_n_matchups=96,
        final_diag_n_matchups=0,
        data_root=str(streams.POLICY_WORLD_V7_SOURCES[0].local_root),
        replay_format="policy-world",
        val_data_root=str(streams.POLICY_WORLD_V7_SOURCES[0].local_root),
        val_replay_format="policy-world",
        train_replay_paths=str(_CORPUS_INDEX_PATH),
        num_workers=0,
        prefetch_batches=0,
        cache_limit_gb=512,
        reservoir_capacity=4096,
    )


def endpoint_report(endpoint: Endpoint) -> dict[str, int | float | str | bool]:
    """Return the exact model, data, compute, and optimizer quantities."""
    cfg = endpoint_config(endpoint)
    quantities = model_quantities(endpoint.d_model)
    processed_positions = cfg.max_steps * cfg.batch_size * cfg.L_ctx
    actual_flops = cfg.max_steps * quantities.flops_per_update
    if abs(actual_flops - endpoint.compute_budget_flops) > quantities.flops_per_update / 2:
        raise RuntimeError(f"{endpoint.name} misses its nearest-update compute budget")
    return {
        "endpoint": endpoint.name,
        "compute_budget_flops": endpoint.compute_budget_flops,
        "actual_flops": actual_flops,
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "temporal_d_model": cfg.temporal_d_model,
        "temporal_layers": cfg.temporal_layers,
        "total_parameters": quantities.total_parameters,
        "effective_parameters": quantities.effective_parameters,
        "max_steps": cfg.max_steps,
        "warmup_steps": cfg.warmup_steps,
        "batch_size": cfg.batch_size,
        "batch_positions": cfg.batch_size * cfg.L_ctx,
        "processed_positions": processed_positions,
        "raw_corpus_loss_positions": 2
        * (sum(streams.POLICY_WORLD_V7_TRAIN_FRAMES.values()) - sum(streams.POLICY_WORLD_V7_TRAIN_REPLAYS.values())),
        "muon_lr": cfg.muon_lr,
        "adam_lr": cfg.adam_lr,
        "muon_weight_decay": cfg.muon_weight_decay,
        "adam_weight_decay": cfg.adam_weight_decay,
        "yolo": endpoint.yolo,
    }


def _config_identity(cfg) -> tuple[int, ...]:
    return (
        cfg.d_model,
        cfg.n_layers,
        cfg.n_heads,
        cfg.temporal_d_model,
        cfg.temporal_layers,
        cfg.temporal_heads,
        cfg.temporal_ff_dim,
        cfg.group_head_dim,
        cfg.batch_size,
        cfg.max_steps,
        cfg.warmup_steps,
    )


def endpoint_for_config(cfg) -> Endpoint:
    """Resolve a checkpoint configuration to one declared endpoint."""
    matches = [
        endpoint
        for endpoint in POINTS.values()
        if _config_identity(endpoint_config(endpoint)) == _config_identity(cfg)
    ]
    if len(matches) != 1:
        raise ValueError(f"configuration does not identify one O48 endpoint: {_config_identity(cfg)}")
    return matches[0]


@dataclass(frozen=True, slots=True)
class CorpusEntry:
    """One canonical training replay and its physical MDS shard."""

    source: str
    row: int
    shard: str
    replay_id: str
    frames: int
    sha1: str

    @property
    def loss_positions(self) -> int:
        return 2 * max(0, self.frames - 1)


@dataclass(frozen=True, slots=True)
class DatasetAudit:
    """Exact unique-data tier selected for one endpoint."""

    replay_ids: frozenset[str]
    selected_shards: dict[str, frozenset[str]]
    unique_replays: int
    unique_loss_positions: int
    target_loss_positions: int
    episode_hash: str
    corpus_hash: str
    corpus_unique_replays: int
    corpus_unique_loss_positions: int
    source_replays: dict[str, int]


def _split_remote(source: StreamSource, suffix: str) -> tuple[str, str]:
    if not source.remote.startswith("s3://"):
        raise ValueError(f"expected S3 source, got {source.remote!r}")
    bucket, _, key = source.remote.removeprefix("s3://").partition("/")
    return bucket, f"{key}/{suffix}"


def _download_source_file(source: StreamSource, suffix: str, destination: Path) -> Path:
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    bucket, key = _split_remote(source, suffix)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    r2.client().download_file(bucket, key, str(temporary))
    temporary.replace(destination)
    return destination


def _source_build_file(source: StreamSource, name: str) -> Path:
    return _CORPUS_BUILD_ROOT / source.name / name


def _shard_by_row(index: dict) -> tuple[list[tuple[int, int, str]], int]:
    spans: list[tuple[int, int, str]] = []
    start = 0
    for shard in index["shards"]:
        stop = start + int(shard["samples"])
        spans.append((start, stop, str(shard["raw_data"]["basename"])))
        start = stop
    return spans, start


def _manifest_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_source_entries(source: StreamSource) -> tuple[list[CorpusEntry], str]:
    manifest = _download_source_file(source, "manifest.jsonl", _source_build_file(source, "manifest.jsonl"))
    index_path = _download_source_file(source, "train/index.json", _source_build_file(source, "train-index.json"))
    spans, indexed_rows = _shard_by_row(json.loads(index_path.read_text()))
    entries: list[CorpusEntry] = []
    with manifest.open() as handle:
        for line in handle:
            record = json.loads(line)
            annotation = record.get("annotation")
            if annotation is None or annotation["split"] != "train":
                continue
            sha1 = record.get("sha1")
            if not sha1:
                raise ValueError(f"{source.name}: train manifest row has no content SHA-1")
            entries.append(
                CorpusEntry(
                    source=source.name,
                    row=int(annotation["mds_row_idx"]),
                    shard="",
                    replay_id=policy_replay_identity(record["path"]),
                    frames=int(annotation["frame_count_actual"]),
                    sha1=str(sha1),
                )
            )
    entries.sort(key=lambda entry: entry.row)
    if [entry.row for entry in entries] != list(range(len(entries))):
        raise ValueError(f"{source.name}: train manifest rows are not contiguous")
    if len(entries) != indexed_rows:
        raise ValueError(f"{source.name}: manifest has {len(entries)} train rows but index has {indexed_rows}")

    with_shards: list[CorpusEntry] = []
    shard_index = 0
    for entry in entries:
        while entry.row >= spans[shard_index][1]:
            shard_index += 1
        with_shards.append(replace(entry, shard=spans[shard_index][2]))
    return with_shards, _manifest_sha256(manifest)


def _write_corpus_index(path: Path, header: dict[str, object], entries: list[CorpusEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as handle:
                handle.write(f"#{json.dumps(header, sort_keys=True)}\n")
                for entry in entries:
                    handle.write(
                        f"{entry.source}\t{entry.row}\t{entry.shard}\t{entry.replay_id}\t"
                        f"{entry.frames}\t{entry.sha1}\n"
                    )
    temporary.replace(path)


def build_corpus_index(path: Path = _CORPUS_INDEX_PATH, *, upload: bool = True) -> dict[str, object]:
    """Build and optionally upload the immutable canonical replay index."""
    if path.exists():
        raise FileExistsError(f"corpus index already exists: {path}")
    source_entries: list[CorpusEntry] = []
    manifest_hashes: dict[str, str] = {}
    for source in streams.POLICY_WORLD_V7_SOURCES:
        entries, manifest_hash = _read_source_entries(source)
        expected_replays = streams.POLICY_WORLD_V7_TRAIN_REPLAYS[source.name]
        expected_frames = streams.POLICY_WORLD_V7_TRAIN_FRAMES[source.name]
        if len(entries) != expected_replays or sum(entry.frames for entry in entries) != expected_frames:
            raise ValueError(f"{source.name}: immutable manifest totals changed")
        source_entries.extend(entries)
        manifest_hashes[source.name] = manifest_hash

    replay_id_counts = Counter(entry.replay_id for entry in source_entries)
    seen_content: set[str] = set()
    canonical: list[CorpusEntry] = []
    duplicate_content = 0
    ambiguous_ids = 0
    for entry in source_entries:
        if entry.sha1 in seen_content:
            duplicate_content += 1
            continue
        seen_content.add(entry.sha1)
        if replay_id_counts[entry.replay_id] != 1:
            ambiguous_ids += 1
            continue
        canonical.append(entry)

    digest = hashlib.sha256()
    source_positions: Counter[str] = Counter()
    source_replays: Counter[str] = Counter()
    for entry in canonical:
        digest.update(
            f"{entry.source}\t{entry.row}\t{entry.shard}\t{entry.replay_id}\t{entry.frames}\t{entry.sha1}\n".encode()
        )
        source_positions[entry.source] += entry.loss_positions
        source_replays[entry.source] += 1
    header: dict[str, object] = {
        "schema_version": _CORPUS_SCHEMA,
        "protocol": _PROTOCOL,
        "source_names": [source.name for source in streams.POLICY_WORLD_V7_SOURCES],
        "raw_train_replays": len(source_entries),
        "raw_train_frames": sum(entry.frames for entry in source_entries),
        "canonical_replays": len(canonical),
        "canonical_loss_positions": sum(entry.loss_positions for entry in canonical),
        "duplicate_content_occurrences_removed": duplicate_content,
        "ambiguous_replay_ids_removed": ambiguous_ids,
        "corpus_hash": digest.hexdigest(),
        "manifest_sha256": manifest_hashes,
        "source_canonical_replays": dict(source_replays),
        "source_canonical_loss_positions": dict(source_positions),
    }
    _write_corpus_index(path, header, canonical)
    if upload:
        r2.client().upload_file(str(path), r2.bucket(), _CORPUS_INDEX_KEY)
    return header


def _ensure_corpus_index(path: Path = _CORPUS_INDEX_PATH) -> Path:
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    r2.client().download_file(r2.bucket(), _CORPUS_INDEX_KEY, str(temporary))
    temporary.replace(path)
    return path


def _corpus_header(path: Path) -> dict[str, object]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        line = handle.readline()
    if not line.startswith("#"):
        raise ValueError(f"{path}: corpus index has no metadata header")
    header = json.loads(line[1:])
    expected_sources = [source.name for source in streams.POLICY_WORLD_V7_SOURCES]
    if (
        header.get("schema_version") != _CORPUS_SCHEMA
        or header.get("protocol") != _PROTOCOL
        or header.get("source_names") != expected_sources
    ):
        raise ValueError(f"{path}: corpus index protocol does not match O48")
    return header


def _corpus_entries(path: Path) -> Iterator[CorpusEntry]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            source, row, shard, replay_id, frames, sha1 = line.rstrip("\n").split("\t")
            yield CorpusEntry(source, int(row), shard, replay_id, int(frames), sha1)


def _shard_priority(source: str, shard: str) -> bytes:
    return hashlib.blake2b(
        f"{_PROTOCOL}\t{source}\t{shard}".encode(),
        digest_size=16,
        person=b"hal-o48-shards",
    ).digest()


def dataset_audit(path: Path, target_loss_positions: int) -> DatasetAudit:
    """Select the smallest nested set of hashed shards covering a data target."""
    header = _corpus_header(path)
    corpus_positions = int(header["canonical_loss_positions"])
    if target_loss_positions > corpus_positions:
        raise ValueError(f"data target {target_loss_positions:,} exceeds the deduplicated corpus {corpus_positions:,}")

    positions_by_shard: Counter[tuple[str, str]] = Counter()
    for entry in _corpus_entries(path):
        positions_by_shard[(entry.source, entry.shard)] += entry.loss_positions
    ordered_shards = sorted(
        positions_by_shard,
        key=lambda key: (_shard_priority(*key), key),
    )
    selected: set[tuple[str, str]] = set()
    selected_positions = 0
    for key in ordered_shards:
        selected.add(key)
        selected_positions += positions_by_shard[key]
        if selected_positions >= target_loss_positions:
            break

    replay_ids: set[str] = set()
    source_replays: Counter[str] = Counter()
    digest = hashlib.sha256()
    for entry in _corpus_entries(path):
        if (entry.source, entry.shard) not in selected:
            continue
        if entry.replay_id in replay_ids:
            raise ValueError(f"canonical corpus repeats replay ID {entry.replay_id}")
        replay_ids.add(entry.replay_id)
        source_replays[entry.source] += 1
        digest.update(f"{entry.source}\t{entry.row}\t{entry.replay_id}\t{entry.frames}\t{entry.sha1}\n".encode())
    selected_shards: dict[str, frozenset[str]] = {}
    for source in streams.POLICY_WORLD_V7_SOURCES:
        names = frozenset(shard for source_name, shard in selected if source_name == source.name)
        if names:
            selected_shards[source.name] = names
    return DatasetAudit(
        replay_ids=frozenset(replay_ids),
        selected_shards=selected_shards,
        unique_replays=len(replay_ids),
        unique_loss_positions=selected_positions,
        target_loss_positions=target_loss_positions,
        episode_hash=digest.hexdigest(),
        corpus_hash=str(header["corpus_hash"]),
        corpus_unique_replays=int(header["canonical_replays"]),
        corpus_unique_loss_positions=corpus_positions,
        source_replays=dict(source_replays),
    )


def _source_train_index(source: StreamSource) -> Path:
    return _download_source_file(source, "train/index.json", source.local_root / "train" / "index.json")


def _prepare_tier_sources(audit: DatasetAudit) -> tuple[StreamSource, ...]:
    tier_root = _TIER_CACHE_ROOT / audit.episode_hash[:16]
    selected_sources: list[StreamSource] = []
    for source in streams.POLICY_WORLD_V7_SOURCES:
        selected_names = audit.selected_shards.get(source.name)
        if not selected_names:
            continue
        source_index = json.loads(_source_train_index(source).read_text())
        selected_shard_rows = [
            shard for shard in source_index["shards"] if shard["raw_data"]["basename"] in selected_names
        ]
        found = {shard["raw_data"]["basename"] for shard in selected_shard_rows}
        if found != set(selected_names):
            raise ValueError(f"{source.name}: selected shards are missing from the immutable train index")
        local = tier_root / source.name
        destination = local / "train" / "index.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({**source_index, "shards": selected_shard_rows}, separators=(",", ":"))
        if destination.is_file() and destination.read_text() != payload:
            raise ValueError(f"tier cache index changed at {destination}")
        if not destination.is_file():
            temporary = destination.with_suffix(".json.tmp")
            temporary.write_text(payload)
            temporary.replace(destination)
        selected_sources.append(StreamSource(name=source.name, remote=source.remote, local=local))
    if not selected_sources:
        raise ValueError("data tier selected no MDS sources")
    return tuple(selected_sources)


_ACTIVE_AUDIT: DatasetAudit | None = None


def _make_loaders(cfg, stats):
    if _ACTIVE_AUDIT is None:
        raise RuntimeError("O48 dataset audit was not initialized")
    selected_sources = _prepare_tier_sources(_ACTIVE_AUDIT)
    train_loader = _o43.make_reservoir_loader(
        data_root=None,
        sources=selected_sources,
        split="train",
        cache_limit=f"{cfg.cache_limit_gb}gb",
        shuffle_block_size=cfg.shuffle_block_size,
        stats=stats,
        L_ctx=cfg.L_ctx,
        L_chunk=cfg.sample_chunk_length,
        batch_size=_o43.micro_batch_size(cfg),
        seed=cfg.seed,
        schema_version=cfg.mds_schema_version,
        projection=_o43.BASE_ACTION_PROJECTION,
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor,
        predownload=cfg.predownload,
        windows_per_replay=cfg.windows_per_replay,
        reservoir_capacity=cfg.reservoir_capacity,
        prefetch_batches=cfg.prefetch_batches,
        replay_format="policy-world",
        replay_filter=_ACTIVE_AUDIT.replay_ids.__contains__,
    )
    validation_kwargs = _o43.loader_kwargs(cfg, stats)
    validation_kwargs.update(
        data_root=cfg.val_data_root,
        remote=streams.remote_for_local(cfg.val_data_root),
        batch_size=cfg.val_batch_size,
    )
    val_loader = _o43.make_loader(
        split=cfg.val_split,
        num_workers=0,
        replay_format=cfg.val_replay_format,
        **validation_kwargs,
    )
    return train_loader, _o43.cache_validation(val_loader, cfg.val_n_samples)


def lr_schedule(cfg):
    """Warm up, then hold both optimizer learning rates constant."""

    def schedule(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / max(cfg.warmup_steps, 1)
        return 1.0

    return schedule


def model_tag(cfg) -> str:
    endpoint = endpoint_for_config(cfg)
    return (
        f"iso048-{endpoint.name}-legacy-v{cfg.codec_version}-d{cfg.d_model}-L{cfg.n_layers}-"
        f"t{cfg.temporal_d_model}x{cfg.temporal_layers}-H1-all44-unique"
    )


def _init_wandb(cfg, run_name: str, resume_state: dict | None) -> None:
    if _ACTIVE_AUDIT is None:
        raise RuntimeError("O48 dataset audit was not initialized")
    endpoint = endpoint_for_config(cfg)
    report = endpoint_report(endpoint)
    audit = _ACTIVE_AUDIT
    wandb.init(
        project="hal",
        group=_WANDB_GROUP,
        name=run_name,
        id=None if resume_state is None else resume_state.get("wandb_id"),
        resume="allow" if resume_state is not None else None,
        tags=["048", "iso-flop", "legacy-codec", "unique-data", endpoint.name],
        config={
            **asdict(cfg),
            **{f"scaling_{key}": value for key, value in report.items()},
            "data_protocol": _PROTOCOL,
            "data_episode_hash": audit.episode_hash,
            "data_corpus_hash": audit.corpus_hash,
            "data_unique_replays": audit.unique_replays,
            "data_unique_loss_positions": audit.unique_loss_positions,
            "data_target_loss_positions": audit.target_loss_positions,
            "data_selected_shards": sum(map(len, audit.selected_shards.values())),
            "data_source_replays": audit.source_replays,
        },
        settings=wandb.Settings(x_stats_sampling_interval=5.0, x_stats_track_process_tree=True),
    )
    if wandb.run is None:
        return
    wandb.run.summary["scaling/processed_positions"] = report["processed_positions"]
    wandb.run.summary["scaling/effective_parameters"] = report["effective_parameters"]
    wandb.run.summary["data/unique_replays"] = audit.unique_replays
    wandb.run.summary["data/unique_loss_positions"] = audit.unique_loss_positions
    wandb.run.summary["data/target_loss_positions"] = audit.target_loss_positions
    wandb.run.summary["data/episode_hash"] = audit.episode_hash
    wandb.run.summary["evaluation/protocol"] = "terminal validation and H=1 closed loop only"
    if cfg.wandb_log_code:
        _o43.log_wandb_code(wandb.run)


def _mixture_stats(header: dict[str, object]):
    weights = header.get("source_canonical_loss_positions")
    if not isinstance(weights, dict):
        raise TypeError("corpus index lacks source_canonical_loss_positions")
    sources = streams.POLICY_WORLD_V7_SOURCES
    return load_consolidated_mixture_stats(
        [source.local_root / "stats.json" for source in sources],
        [float(weights[source.name]) for source in sources],
        expected_mds_schema_version=7,
    )


# O43 remains the implementation of the model, loss, checkpoint, and evaluator.
# These replacements change only the experiment protocol named above.
_o43.__file__ = __file__
_o43._LIVE_HORIZONS = (1, 2, 4, 6)
_o43._make_loaders = _make_loaders
_o43.lr_schedule = lr_schedule
_o43.model_tag = model_tag
_o43._init_wandb = _init_wandb


@dataclass
class Args:
    endpoint: str | None = None
    """One endpoint printed by --describe."""

    comment: str = ""
    resume: str | None = None
    eval: str | None = None
    """Local checkpoint name or path for a later latency evaluation."""

    eval_run: str | None = None
    eval_exec_horizon: int | None = None
    eval_n_matchups: int | None = None
    eval_eager: bool = False
    eval_max_parallel: int | None = None
    describe: bool = False
    build_corpus_index: bool = False
    upload_corpus_index: bool = True


def _selected_endpoint(name: str | None) -> Endpoint:
    if name not in POINTS:
        choices = ", ".join(POINTS)
        raise SystemExit(f"--endpoint must be one of: {choices}")
    return POINTS[name]


def main(args: Args) -> None:
    global _ACTIVE_AUDIT
    if args.describe:
        print(json.dumps([endpoint_report(endpoint) for endpoint in POINTS.values()], indent=2))
        return
    if args.build_corpus_index:
        print(json.dumps(build_corpus_index(upload=args.upload_corpus_index), indent=2, sort_keys=True))
        return
    if args.eval is not None:
        if args.resume is not None or args.endpoint is not None:
            raise SystemExit("--eval cannot be combined with --resume or --endpoint")
        checkpoint = _o43._resolve_eval_checkpoint(args.eval, args.eval_run)
        _o43.eval_checkpoint(
            str(checkpoint),
            exec_horizon=args.eval_exec_horizon,
            n_matchups=args.eval_n_matchups,
            eager=args.eval_eager,
            max_parallel=args.eval_max_parallel,
            upload_run=args.eval_run,
        )
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
        cfg = endpoint_config(endpoint)
        resume_state = None
        run_name = None

    _o43.validate_config(cfg)
    report = endpoint_report(endpoint)
    corpus_index = _ensure_corpus_index(Path(cfg.train_replay_paths))
    _ACTIVE_AUDIT = dataset_audit(corpus_index, int(report["processed_positions"]))
    header = _corpus_header(corpus_index)
    stats = _mixture_stats(header)
    _o43.train(cfg, stats, comment=args.comment, resume_run=run_name, resume_state=resume_state)


if __name__ == "__main__":
    main(tyro.cli(Args))
