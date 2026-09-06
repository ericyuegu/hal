"""Build one audited policy-world-v8 artifact from policy-world-v7 MDS."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import subprocess
import tempfile
from collections import Counter
from collections.abc import Iterator
from collections.abc import Mapping
from contextlib import ExitStack
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tyro
from loguru import logger
from tqdm import tqdm

from hal import r2
from hal.data.bounded_writer import BoundedMDSWriter
from hal.data.bounded_writer import RcloneMDSWriter
from hal.data.feature_stats import StatsAccumulator
from hal.data.feature_stats import dump_sufficient_stats
from hal.data.feature_stats import load_sufficient_stats
from hal.data.index import ReplayIndexEntry
from hal.data.index import read_jsonl
from hal.data.mds import open_shard
from hal.data.mds import read_shard_index
from hal.data.policy_schema import POLICY_SCHEMA_VERSION
from hal.data.policy_schema import policy_replay_identity
from hal.data.policy_world_schema import POLICY_WORLD_FLOAT_COLUMNS
from hal.data.policy_world_schema import POLICY_WORLD_MDS_COLUMNS
from hal.data.policy_world_schema import POLICY_WORLD_SCHEMA_VERSION
from hal.data.policy_world_v8 import POLICY_WORLD_V8_SCHEMA_VERSION
from hal.data.policy_world_v8 import decode_policy_world_v8_replay
from hal.data.policy_world_v8 import decode_policy_world_v8_reward_events
from hal.data.policy_world_v8 import encode_policy_world_v8_replay
from hal.data.policy_world_v8_selection import V8_QUALITY_POLICY
from hal.data.policy_world_v8_selection import assign_v8_ranks
from hal.data.policy_world_v8_selection import failed_quality_rules
from hal.data.policy_world_v8_selection import known_duplicate_rejection
from hal.data.policy_world_v8_selection import rank_imputation_rule
from hal.data.schema import POLICY_WORLD_V8_MDS_COLUMNS
from hal.data.schema import SCHEMA_VERSION
from hal.data.schema import Rank
from hal.scripts.publish_mds import audit

SPLITS = ("train", "val", "test")
DEFAULT_SCRATCH = Path("/dev/shm/hal_policy_world_v8")
DEFAULT_SHARD_SIZE = 16 * 2**20
SELECTION_SCHEMA_VERSION = 2
RANK_REPORT_SCHEMA_VERSION = 1


def _is_rclone(path: str) -> bool:
    return path.startswith("r2:")


def _join(root: str, relative: str) -> str:
    return f"{root.rstrip('/')}/{relative}"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _hash_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _verify_file(path: Path, info: dict[str, Any]) -> None:
    expected_bytes = int(info["bytes"])
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"{path}: size {path.stat().st_size} != expected {expected_bytes}")
    for algorithm, expected in info.get("hashes", {}).items():
        actual = _hash_file(path, algorithm)
        if actual.lower() != str(expected).lower():
            raise ValueError(f"{path}: {algorithm} {actual} != expected {expected}")


def _copy_remote_file(source: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    r2.copy_file(source, destination)


def _download_source_metadata(source: str, root: Path) -> None:
    names = (
        "_SUCCESS",
        "failures.materialize.jsonl",
        "manifest.jsonl",
        "projection.json",
        "stats.json",
        *(f"{split}/index.json" for split in SPLITS),
    )
    for name in names:
        _copy_remote_file(_join(source, name), root / name)


def _columns(info: dict[str, Any]) -> dict[str, str]:
    names = info.get("column_names", [])
    encodings = info.get("column_encodings", [])
    columns = dict(zip(names, encodings, strict=True))
    if len(columns) != len(names):
        raise ValueError("MDS shard contains duplicate column names")
    return columns


def _source_indexes(root: Path) -> dict[str, list[dict[str, Any]]]:
    indexes = {split: read_shard_index(root, split) for split in SPLITS}
    for split, shards in indexes.items():
        for shard in shards:
            if _columns(shard) != POLICY_WORLD_MDS_COLUMNS:
                raise ValueError(f"{split}: source columns differ from policy-world-v7")
            if shard.get("zip_data") is None or shard.get("compression") != "zstd":
                raise ValueError(f"{split}: policy-world-v7 source shard must be zstd-compressed")
    return indexes


def _output_indexes(root: Path) -> dict[str, list[dict[str, Any]]]:
    indexes = {split: read_shard_index(root, split) for split in SPLITS}
    for split, shards in indexes.items():
        for shard in shards:
            if _columns(shard) != POLICY_WORLD_V8_MDS_COLUMNS:
                raise ValueError(f"{split}: output columns differ from policy-world-v8")
            if shard.get("zip_data") is not None or shard.get("compression") is not None:
                raise ValueError(f"{split}: policy-world-v8 outer shard must be raw")
            if int(shard["raw_data"]["bytes"]) > DEFAULT_SHARD_SIZE:
                raise ValueError(f"{split}: output shard exceeds the 16 MiB limit")
    return indexes


def _rows(indexes: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    return {split: sum(int(shard["samples"]) for shard in indexes[split]) for split in SPLITS}


def _line_count(path: Path) -> int:
    with path.open() as handle:
        return sum(bool(line.strip()) for line in handle)


def _validate_source_metadata(
    root: Path,
    indexes: dict[str, list[dict[str, Any]]],
    source_rows: dict[str, int],
) -> int:
    required = ("_SUCCESS", "failures.materialize.jsonl", "manifest.jsonl", "projection.json", "stats.json")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"source is missing required files {missing}")
    projection = _load_json(root / "projection.json")
    if projection.get("columns") != POLICY_WORLD_MDS_COLUMNS:
        raise ValueError("source projection columns do not match policy-world-v7")
    if int(projection.get("policy_world_schema_version", -1)) != POLICY_WORLD_SCHEMA_VERSION:
        raise ValueError("source policy-world schema version differs")
    if int(projection.get("source_schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("source canonical schema version differs")
    projected_rows = {split: int(value) for split, value in projection.get("rows", {}).items()}
    if projected_rows != source_rows:
        raise ValueError(f"source projection rows {projected_rows} != indexes {source_rows}")
    success = _load_json(root / "_SUCCESS")
    published_rows = {split: int(value) for split, value in success.get("rows", {}).items()}
    if published_rows != source_rows:
        raise ValueError(f"source publication rows {published_rows} != indexes {source_rows}")
    failure_count = _line_count(root / "failures.materialize.jsonl")
    if int(projection.get("failures", -1)) != failure_count:
        raise ValueError("source failure ledger count differs from projection.json")
    return failure_count


def _materialize_metadata_index(
    metadata_index: str | Path | None,
    root: Path,
) -> tuple[Path | None, dict[str, object] | None]:
    if metadata_index is None:
        return None, None
    local = root / "supplemental-index.jsonl"
    if isinstance(metadata_index, str) and _is_rclone(metadata_index):
        _copy_remote_file(metadata_index, local)
        source = metadata_index
        name = Path(metadata_index).name
        if name.startswith("ranked-1-index.") and name.endswith(".jsonl"):
            expected = name.removeprefix("ranked-1-index.").removesuffix(".jsonl")
            if _hash_file(local) != expected:
                raise ValueError(f"supplemental metadata SHA-256 differs from its immutable path: {source}")
    else:
        path = Path(metadata_index)
        if not path.is_file():
            raise FileNotFoundError(f"supplemental metadata index not found: {path}")
        shutil.copy2(path, local)
        source = str(path.resolve())
    return local, {"source": source, "sha256": _hash_file(local), "bytes": local.stat().st_size}


def _metadata_by_path(path: Path | None) -> dict[str, ReplayIndexEntry]:
    if path is None:
        return {}
    rows: dict[str, ReplayIndexEntry] = {}
    for entry in read_jsonl(path, verify_schema_version=False):
        if entry.path in rows:
            raise ValueError(f"supplemental metadata contains duplicate path {entry.path!r}")
        rows[entry.path] = entry
    return rows


def _manifest_by_split(
    root: Path,
    source_rows: dict[str, int],
    metadata_index: Path | None,
) -> dict[str, list[ReplayIndexEntry]]:
    indexed: dict[str, dict[int, ReplayIndexEntry]] = {split: {} for split in SPLITS}
    identities: set[str] = set()
    metadata = _metadata_by_path(metadata_index)
    for entry in read_jsonl(root / "manifest.jsonl"):
        annotation = entry.annotation
        if annotation is None or annotation.schema_version != SCHEMA_VERSION:
            raise ValueError(f"{entry.path}: source manifest annotation is not canonical schema v7")
        if entry.stats is None:
            supplemental = metadata.get(entry.path)
            if supplemental is None or supplemental.stats is None:
                raise ValueError(f"{entry.path}: source manifest has no replay statistics")
            source_players = [(player.port, player.player_type) for player in entry.players]
            extra_players = [(player.port, player.player_type) for player in supplemental.players]
            if supplemental.stage != entry.stage or supplemental.frame_count != entry.frame_count:
                raise ValueError(f"{entry.path}: supplemental metadata differs from the source manifest")
            if extra_players != source_players:
                raise ValueError(f"{entry.path}: supplemental player ports or types differ")
            if [player.port for player in supplemental.stats.players] != [player.port for player in entry.players]:
                raise ValueError(f"{entry.path}: supplemental statistics use different player ports")
            entry = dataclasses.replace(entry, stats=supplemental.stats, sha1=entry.sha1 or supplemental.sha1)
        rows = indexed.get(annotation.split)
        if rows is None or annotation.mds_row_idx in rows:
            raise ValueError(f"{entry.path}: invalid or duplicate source MDS position")
        identity = policy_replay_identity(entry.path)
        if identity in identities:
            raise ValueError(f"duplicate replay identity {identity} in source manifest")
        identities.add(identity)
        rows[annotation.mds_row_idx] = entry
    ordered: dict[str, list[ReplayIndexEntry]] = {}
    for split, count in source_rows.items():
        if sorted(indexed[split]) != list(range(count)):
            raise ValueError(f"{split}: source manifest row indexes are not contiguous over {count} rows")
        ordered[split] = [indexed[split][row] for row in range(count)]
    return ordered


def _scalar_int(sample: dict[str, Any], name: str) -> int:
    value = np.asarray(sample[name])
    if value.shape:
        raise ValueError(f"{name} must be scalar, got {value.shape}")
    return int(value.item())


def _validate_source_sample(sample: dict[str, Any], entry: ReplayIndexEntry, split: str, row: int) -> int:
    if set(sample) != set(POLICY_WORLD_MDS_COLUMNS):
        raise ValueError(f"{split} row {row}: sample columns differ from policy-world-v7")
    versions = {
        "source_schema_version": SCHEMA_VERSION,
        "policy_schema_version": POLICY_SCHEMA_VERSION,
        "policy_world_schema_version": POLICY_WORLD_SCHEMA_VERSION,
    }
    for name, expected in versions.items():
        if _scalar_int(sample, name) != expected:
            raise ValueError(f"{split} row {row}: {name} differs")
    expected_id = policy_replay_identity(entry.path)
    if sample["replay_id"] != expected_id:
        raise ValueError(f"{split} row {row}: replay ID differs from the manifest")
    frames = _scalar_int(sample, "num_frames")
    if entry.annotation is None or frames != entry.annotation.frame_count_actual:
        raise ValueError(f"{split} row {row}: frame count differs from the manifest")
    return frames


def _source_hashes(root: Path, indexes: dict[str, list[dict[str, Any]]]) -> dict[str, object]:
    metadata = {
        name: {"sha256": _hash_file(root / name), "bytes": (root / name).stat().st_size}
        for name in ("_SUCCESS", "failures.materialize.jsonl", "manifest.jsonl", "projection.json", "stats.json")
    }
    shards = {
        split: [
            {
                "basename": shard["zip_data"]["basename"],
                "bytes": int(shard["zip_data"]["bytes"]),
                "hashes": shard["zip_data"].get("hashes", {}),
            }
            for shard in indexes[split]
        ]
        for split in SPLITS
    }
    return {"metadata": metadata, "shards": shards}


def _selection_plan(
    source_name: str,
    manifests: dict[str, list[ReplayIndexEntry]],
    retained_limits: dict[str, int] | None,
) -> tuple[dict[str, list[tuple[str, ...]]], dict[str, list[bool]], dict[str, int]]:
    failures: dict[str, list[tuple[str, ...]]] = {}
    selected: dict[str, list[bool]] = {}
    omitted = dict.fromkeys(SPLITS, 0)
    for split in SPLITS:
        limit = None if retained_limits is None else retained_limits.get(split)
        if limit is not None and limit < 1:
            raise ValueError(f"retained limit for {split} must be positive")
        retained = 0
        split_failures: list[tuple[str, ...]] = []
        split_selected: list[bool] = []
        for row, entry in enumerate(manifests[split]):
            reasons = list(failed_quality_rules(entry))
            duplicate = known_duplicate_rejection(source_name, split, row, entry)
            if duplicate is not None:
                reasons.append(duplicate)
            keep = not reasons and (limit is None or retained < limit)
            if keep:
                retained += 1
            elif not reasons:
                omitted[split] += 1
            split_failures.append(tuple(reasons))
            split_selected.append(keep)
        failures[split] = split_failures
        selected[split] = split_selected
    return failures, selected, omitted


def _stored_info(info: dict[str, Any]) -> dict[str, Any]:
    return info["zip_data"] if info.get("zip_data") is not None else info["raw_data"]


def _download_shard(source: str, source_root: Path, split: str, info: dict[str, Any]) -> Path:
    stored = _stored_info(info)
    name = str(stored["basename"])
    if Path(name).name != name:
        raise ValueError(f"unsafe shard basename {name!r}")
    destination = source_root / split / name
    _copy_remote_file(_join(source, f"{split}/{name}"), destination)
    _verify_file(destination, stored)
    return destination


@contextmanager
def _source_shard(
    source: str,
    source_root: Path,
    split: str,
    info: dict[str, Any],
    decode_scratch: Path,
) -> Iterator[Any]:
    downloaded: Path | None = None
    if _is_rclone(source):
        downloaded = _download_shard(source, source_root, split, info)
    try:
        with open_shard(source_root, split, info, decode_scratch) as reader:
            yield reader
    finally:
        if downloaded is not None:
            downloaded.unlink(missing_ok=True)


def _writer(
    output: str, split: str, local_output: Path | None, upload_root: Path, shard_size: int
) -> BoundedMDSWriter:
    common = {
        "columns": POLICY_WORLD_V8_MDS_COLUMNS,
        "compression": None,
        "hashes": ["md5", "sha256"],
        "size_limit": shard_size,
        "max_workers": 2,
        "max_pending_uploads": 2,
        "exist_ok": False,
    }
    if _is_rclone(output):
        return RcloneMDSWriter(local=upload_root / split, remote=_join(output, split), **common)
    if local_output is None:
        raise AssertionError("local output root is required")
    return BoundedMDSWriter(out=str(local_output / split), **common)


def _write_sidecar(source: Path, output: str, relative: str, local_output: Path | None) -> None:
    if _is_rclone(output):
        r2.copy_file(source, _join(output, relative))
        return
    if local_output is None:
        raise AssertionError("local output root is required")
    destination = local_output / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _implementation_identity() -> dict[str, object]:
    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run(("git", "rev-parse", "HEAD"), cwd=repo, capture_output=True, text=True, check=True)
    dirty = subprocess.run(("git", "diff", "--quiet"), cwd=repo, capture_output=True, check=False).returncode != 0
    files = (
        Path(__file__),
        repo / "hal/data/policy_world_v8.py",
        repo / "hal/data/policy_world_v8_selection.py",
        repo / "hal/data/schema.py",
        repo / "hal/data/bounded_writer.py",
    )
    return {
        "git_sha": result.stdout.strip(),
        "git_dirty": dirty,
        "file_sha256": {str(path.relative_to(repo)): _hash_file(path) for path in files},
    }


def _rank_report_template(source_name: str) -> dict[str, Any]:
    return {
        "schema_version": RANK_REPORT_SCHEMA_VERSION,
        "source_name": source_name,
        "rule": rank_imputation_rule(),
        "splits": {split: {"observed": Counter(), "imputed": Counter(), "sides": 0} for split in SPLITS},
    }


def _record_ranks(report: dict[str, Any], split: str, sample: Mapping[str, object]) -> None:
    block = report["splits"][split]
    mask = int(np.asarray(sample["rank_imputed_mask"]).item())
    for side in (1, 2):
        rank = Rank(int(np.asarray(sample[f"p{side}_rank"]).item())).name
        kind = "imputed" if mask & (1 << (side - 1)) else "observed"
        block[kind][rank] += 1
        block["sides"] += 1


def _finish_rank_report(report: dict[str, Any]) -> dict[str, Any]:
    observed: Counter[str] = Counter()
    imputed: Counter[str] = Counter()
    sides = 0
    for block in report["splits"].values():
        observed.update(block["observed"])
        imputed.update(block["imputed"])
        block["observed"] = dict(sorted(block["observed"].items()))
        block["imputed"] = dict(sorted(block["imputed"].items()))
        sides += block["sides"]
    report["aggregate"] = {
        "observed": dict(sorted(observed.items())),
        "imputed": dict(sorted(imputed.items())),
        "sides": sides,
    }
    return report


def filter_policy_world_mds(
    source: str,
    output: str,
    *,
    source_name: str,
    scratch: Path = DEFAULT_SCRATCH,
    shard_size: int = DEFAULT_SHARD_SIZE,
    stats_batch_rows: int = 64,
    metadata_index: str | Path | None = None,
    retained_limits: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Filter and encode one audited policy-world-v7 source into fresh v8 MDS."""
    if not source_name.endswith("-policy-world-v7"):
        raise ValueError(f"unexpected source name {source_name!r}")
    if not 1 <= shard_size <= DEFAULT_SHARD_SIZE:
        raise ValueError(f"shard_size must be in [1, {DEFAULT_SHARD_SIZE}], got {shard_size}")
    if stats_batch_rows < 1:
        raise ValueError("stats_batch_rows must be positive")
    if _is_rclone(source) != _is_rclone(output):
        raise ValueError("source and output must both be local paths or both be r2: prefixes")
    output_path = None if _is_rclone(output) else Path(output)
    if output_path is not None and output_path.exists():
        raise FileExistsError(f"output already exists: {output}")
    if _is_rclone(output) and r2.list_files(output):
        raise FileExistsError(f"remote output prefix is not empty: {output}")

    scratch.mkdir(parents=True, exist_ok=True)
    source_audit = audit(source) if _is_rclone(source) else None
    with ExitStack() as stack:
        run_root = Path(stack.enter_context(tempfile.TemporaryDirectory(dir=scratch, prefix="rewrite-")))
        source_root = run_root / "source" if _is_rclone(source) else Path(source)
        if _is_rclone(source):
            _download_source_metadata(source, source_root)
        indexes = _source_indexes(source_root)
        source_rows = _rows(indexes)
        inherited_failures = _validate_source_metadata(source_root, indexes, source_rows)
        if source_audit is not None and source_audit["rows"] != source_rows:
            raise ValueError(f"source audit rows {source_audit['rows']} != indexes {source_rows}")
        local_metadata, supplemental = _materialize_metadata_index(metadata_index, run_root)
        manifests = _manifest_by_split(source_root, source_rows, local_metadata)
        failures, selected, omitted_rows = _selection_plan(source_name, manifests, retained_limits)

        if output_path is None:
            local_output = None
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            name = stack.enter_context(
                tempfile.TemporaryDirectory(dir=output_path.parent, prefix=f".{output_path.name}.")
            )
            local_output = Path(name)
        upload_root = run_root / "upload"
        decode_scratch = run_root / "decode"
        decode_scratch.mkdir()
        manifest_path = run_root / "manifest.jsonl"
        rejections_path = run_root / "rejections.v8.jsonl"
        stats_path = run_root / "stats.json"
        projection_path = run_root / "projection.json"
        selection_path = run_root / "selection.json"
        rank_path = run_root / "ranks.json"

        output_rows = dict.fromkeys(SPLITS, 0)
        output_frames = dict.fromkeys(SPLITS, 0)
        rejected_rows = dict.fromkeys(SPLITS, 0)
        rejection_reasons: Counter[str] = Counter()
        stats = StatsAccumulator(POLICY_WORLD_FLOAT_COLUMNS)
        rank_report = _rank_report_template(source_name)
        retained_sha1: dict[str, str] = {}
        max_input_raw_bytes = 0
        max_input_stored_bytes = 0

        with manifest_path.open("w") as manifest_handle, rejections_path.open("w") as rejection_handle:
            for split in SPLITS:
                for row, reasons in enumerate(failures[split]):
                    if not reasons:
                        continue
                    rejected_rows[split] += 1
                    rejection_reasons.update(reasons)
                    rejection_handle.write(
                        json.dumps(
                            {
                                "failed_rules": list(reasons),
                                "path": manifests[split][row].path,
                                "source_mds_row_idx": row,
                                "split": split,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )

                writer = _writer(output, split, local_output, upload_root, shard_size)
                source_row = 0
                try:
                    with tqdm(total=sum(selected[split]), desc=f"policy-world-v8 {split}", unit="replay") as bar:
                        for info in indexes[split]:
                            shard_rows = int(info["samples"])
                            shard_selected = selected[split][source_row : source_row + shard_rows]
                            if not any(shard_selected):
                                source_row += shard_rows
                                continue
                            max_input_raw_bytes = max(max_input_raw_bytes, int(info["raw_data"]["bytes"]))
                            max_input_stored_bytes = max(max_input_stored_bytes, int(_stored_info(info)["bytes"]))
                            stats_values = {name: [] for name in POLICY_WORLD_FLOAT_COLUMNS}
                            with _source_shard(source, source_root, split, info, decode_scratch) as reader:
                                for source_sample in reader:
                                    entry = manifests[split][source_row]
                                    sample = dict(source_sample)
                                    frames = _validate_source_sample(sample, entry, split, source_row)
                                    if selected[split][source_row]:
                                        ports = tuple(sorted(player.port for player in entry.players))
                                        if len(ports) != 2:
                                            raise AssertionError("selected replay passed the two-player policy")
                                        encoded = encode_policy_world_v8_replay(
                                            sample,
                                            source_name=source_name,
                                            p1_port=ports[0],
                                            p2_port=ports[1],
                                        )
                                        writer.write(encoded)
                                        output_row = output_rows[split]
                                        output_rows[split] += 1
                                        output_frames[split] += frames
                                        _record_ranks(rank_report, split, encoded)
                                        if entry.sha1 is None or len(entry.sha1) != 40:
                                            raise ValueError(f"{entry.path}: retained manifest row has no valid SHA-1")
                                        previous = retained_sha1.get(entry.sha1)
                                        if previous is not None:
                                            raise ValueError(
                                                f"unexpected duplicate SHA-1 {entry.sha1}: {previous} and {entry.path}"
                                            )
                                        retained_sha1[entry.sha1] = entry.path
                                        if split == "train":
                                            for stat_name in POLICY_WORLD_FLOAT_COLUMNS:
                                                stats_values[stat_name].append(np.asarray(sample[stat_name]))
                                            if len(stats_values[POLICY_WORLD_FLOAT_COLUMNS[0]]) >= stats_batch_rows:
                                                for stat_name, values in stats_values.items():
                                                    stats.update(stat_name, np.concatenate(values))
                                                    values.clear()
                                        if entry.annotation is None:
                                            raise AssertionError("source manifest was validated")
                                        rewritten = dataclasses.replace(
                                            entry,
                                            annotation=dataclasses.replace(entry.annotation, mds_row_idx=output_row),
                                        )
                                        manifest_handle.write(json.dumps(rewritten.to_dict(), sort_keys=True) + "\n")
                                        bar.update(1)
                                    source_row += 1
                            if split == "train":
                                for stat_name, values in stats_values.items():
                                    if values:
                                        stats.update(stat_name, np.concatenate(values))
                    if source_row != source_rows[split]:
                        raise ValueError(f"{split}: visited {source_row} source rows, expected {source_rows[split]}")
                finally:
                    writer.finish()

        shutil.copy2(source_root / "failures.materialize.jsonl", run_root / "failures.materialize.jsonl")
        dump_sufficient_stats(stats_path, stats.to_sufficient(), split="train", mds_schema_version=SCHEMA_VERSION)
        rank_report = _finish_rank_report(rank_report)
        rank_path.write_text(json.dumps(rank_report, indent=2, sort_keys=True) + "\n")
        projection = {
            "artifact": "mds-policy-world-v8",
            "columns": POLICY_WORLD_V8_MDS_COLUMNS,
            "failures": inherited_failures,
            "policy_schema_version": POLICY_SCHEMA_VERSION,
            "policy_world_schema_version": POLICY_WORLD_V8_SCHEMA_VERSION,
            "rejections": rejected_rows,
            "rows": output_rows,
            "source": source,
            "source_name": source_name,
            "source_rows": source_rows,
            "source_schema_version": SCHEMA_VERSION,
        }
        projection_path.write_text(json.dumps(projection, indent=2, sort_keys=True) + "\n")
        selection: dict[str, Any] = {
            "artifact": "mds-policy-world-v8",
            "schema_version": SELECTION_SCHEMA_VERSION,
            "source": source,
            "source_name": source_name,
            "source_hashes": _source_hashes(source_root, indexes),
            "source_rows": source_rows,
            "rows": output_rows,
            "frames": output_frames,
            "rejections": rejected_rows,
            "omitted_after_pilot_limit": omitted_rows,
            "rejection_reasons": dict(sorted(rejection_reasons.items())),
            "inherited_materialization_failures": inherited_failures,
            "policy": V8_QUALITY_POLICY.to_dict(),
            "rank_imputation": rank_imputation_rule(),
            "schema_identity": {
                "policy_world_schema_version": POLICY_WORLD_V8_SCHEMA_VERSION,
                "policy_schema_version": POLICY_SCHEMA_VERSION,
                "source_schema_version": SCHEMA_VERSION,
                "columns": POLICY_WORLD_V8_MDS_COLUMNS,
            },
            "disk_bounds": {
                "max_input_raw_shard_bytes": max_input_raw_bytes,
                "max_input_stored_shard_bytes": max_input_stored_bytes,
                "output_shard_size_limit": shard_size,
                "max_pending_output_shards": 2,
                "concurrent_input_shards": 1,
            },
            "implementation": _implementation_identity(),
        }
        if supplemental is not None and local_metadata is not None:
            pinned_name = f"metadata/ranked-1-index.{supplemental['sha256']}.jsonl"
            selection["supplemental_metadata"] = {**supplemental, "artifact_path": pinned_name}
            _write_sidecar(local_metadata, output, pinned_name, local_output)
        selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")

        for path in (
            run_root / "failures.materialize.jsonl",
            manifest_path,
            projection_path,
            rank_path,
            rejections_path,
            selection_path,
            stats_path,
        ):
            _write_sidecar(path, output, path.name, local_output)
        if output_path is not None:
            if output_path.exists() or local_output is None:
                raise FileExistsError(f"output appeared during rewrite: {output}")
            local_output.rename(output_path)

    logger.info(f"built policy-world-v8 {source_name}: rows={output_rows}, rejected={rejected_rows}")
    return selection


def _audit_metadata(
    root: Path,
    base_rows: dict[str, int],
) -> tuple[dict[str, Any], dict[str, list[ReplayIndexEntry]]]:
    selection = _load_json(root / "selection.json")
    if (
        selection.get("schema_version") != SELECTION_SCHEMA_VERSION
        or selection.get("artifact") != "mds-policy-world-v8"
    ):
        raise ValueError("selection.json has an unsupported identity")
    if selection.get("policy") != V8_QUALITY_POLICY.to_dict():
        raise ValueError("selection policy differs from the v8 contract")
    if selection.get("rank_imputation") != rank_imputation_rule():
        raise ValueError("selection rank-imputation rule differs from the v8 contract")
    expected_identity = {
        "policy_world_schema_version": POLICY_WORLD_V8_SCHEMA_VERSION,
        "policy_schema_version": POLICY_SCHEMA_VERSION,
        "source_schema_version": SCHEMA_VERSION,
        "columns": POLICY_WORLD_V8_MDS_COLUMNS,
    }
    if selection.get("schema_identity") != expected_identity:
        raise ValueError("selection schema identity differs from the v8 contract")
    rows = {split: int(value) for split, value in selection.get("rows", {}).items()}
    source_rows = {split: int(value) for split, value in selection.get("source_rows", {}).items()}
    rejected = {split: int(value) for split, value in selection.get("rejections", {}).items()}
    omitted = {split: int(value) for split, value in selection.get("omitted_after_pilot_limit", {}).items()}
    frames = {split: int(value) for split, value in selection.get("frames", {}).items()}
    if rows != base_rows:
        raise ValueError(f"selection rows {rows} != MDS indexes {base_rows}")
    for split in SPLITS:
        if source_rows[split] != rows[split] + rejected[split] + omitted[split]:
            raise ValueError(f"{split}: source/kept/rejected/omitted accounting differs")
    projection = _load_json(root / "projection.json")
    if projection.get("columns") != POLICY_WORLD_V8_MDS_COLUMNS:
        raise ValueError("projection columns differ from policy-world-v8")
    for name, expected in (("rows", rows), ("source_rows", source_rows), ("rejections", rejected)):
        actual = {split: int(value) for split, value in projection.get(name, {}).items()}
        if actual != expected:
            raise ValueError(f"projection {name} differs from selection")
    inherited = int(selection.get("inherited_materialization_failures", -1))
    if (
        _line_count(root / "failures.materialize.jsonl") != inherited
        or int(projection.get("failures", -1)) != inherited
    ):
        raise ValueError("inherited materialization failure accounting differs")

    manifests = _manifest_by_split(root, rows, None)
    for entries in manifests.values():
        for entry in entries:
            failures = failed_quality_rules(entry)
            if failures:
                raise ValueError(f"{entry.path}: retained row fails v8 quality rules {failures}")
    actual_frames = {
        split: sum(entry.annotation.frame_count_actual for entry in entries if entry.annotation is not None)
        for split, entries in manifests.items()
    }
    if actual_frames != frames:
        raise ValueError(f"selection frames {frames} != manifest frames {actual_frames}")
    retained_paths = {entry.path for entries in manifests.values() for entry in entries}
    rejection_counts = dict.fromkeys(SPLITS, 0)
    rejection_reasons: Counter[str] = Counter()
    positions: set[tuple[str, int]] = set()
    with (root / "rejections.v8.jsonl").open() as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            split = str(row.get("split"))
            source_row = int(row.get("source_mds_row_idx", -1))
            position = (split, source_row)
            reasons = row.get("failed_rules")
            if split not in rejection_counts or not 0 <= source_row < source_rows[split]:
                raise ValueError(f"rejection line {line_number} has an invalid source position")
            if position in positions or row.get("path") in retained_paths:
                raise ValueError(f"rejection line {line_number} duplicates a position or retained path")
            if not isinstance(reasons, list) or not reasons or not all(isinstance(reason, str) for reason in reasons):
                raise ValueError(f"rejection line {line_number} has invalid failed_rules")
            positions.add(position)
            rejection_counts[split] += 1
            rejection_reasons.update(reasons)
    if rejection_counts != rejected:
        raise ValueError("rejection ledger counts differ from selection")
    expected_reasons = {str(name): int(value) for name, value in selection.get("rejection_reasons", {}).items()}
    if dict(sorted(rejection_reasons.items())) != expected_reasons:
        raise ValueError("rejection reason counts differ from selection")
    stats = load_sufficient_stats(root / "stats.json", expected_mds_schema_version=SCHEMA_VERSION)
    if set(stats) != set(POLICY_WORLD_FLOAT_COLUMNS):
        raise ValueError("stats.json does not cover the policy-world float columns")
    supplemental = selection.get("supplemental_metadata")
    if supplemental is not None:
        if not isinstance(supplemental, dict):
            raise TypeError("supplemental_metadata must be an object")
        relative = str(supplemental.get("artifact_path"))
        path = root / relative
        if path.resolve().parent != (root / "metadata").resolve() or not path.is_file():
            raise ValueError("supplemental metadata artifact path is missing or unsafe")
        if path.stat().st_size != int(supplemental.get("bytes", -1)) or _hash_file(path) != supplemental.get("sha256"):
            raise ValueError("supplemental metadata artifact differs from selection.json")
    return {
        "source_rows": source_rows,
        "rows": rows,
        "frames": frames,
        "rejections": rejected,
        "omitted_after_pilot_limit": omitted,
        "rejection_reasons": expected_reasons,
        "inherited_materialization_failures": inherited,
    }, manifests


def _download_output_metadata(prefix: str, root: Path) -> None:
    names = (
        "failures.materialize.jsonl",
        "manifest.jsonl",
        "projection.json",
        "ranks.json",
        "rejections.v8.jsonl",
        "selection.json",
        "stats.json",
        *(f"{split}/index.json" for split in SPLITS),
    )
    for name in names:
        _copy_remote_file(_join(prefix, name), root / name)
    supplemental = _load_json(root / "selection.json").get("supplemental_metadata")
    if supplemental is not None:
        if not isinstance(supplemental, dict):
            raise TypeError("supplemental_metadata must be an object")
        relative = str(supplemental.get("artifact_path"))
        if Path(relative).parent != Path("metadata"):
            raise ValueError("supplemental metadata artifact path is unsafe")
        _copy_remote_file(_join(prefix, relative), root / relative)


def _audit_rows(
    prefix: str,
    root: Path,
    indexes: dict[str, list[dict[str, Any]]],
    manifests: dict[str, list[ReplayIndexEntry]],
    scratch: Path,
) -> dict[str, Any]:
    source_name = str(_load_json(root / "selection.json")["source_name"])
    report = _rank_report_template(source_name)
    seen_sha1: dict[str, str] = {}
    for split in SPLITS:
        row = 0
        for info in indexes[split]:
            downloaded = _download_shard(prefix, root, split, info) if _is_rclone(prefix) else None
            try:
                with open_shard(root, split, info, scratch) as reader:
                    for raw in reader:
                        sample = dict(raw)
                        entry = manifests[split][row]
                        expected_id = bytes.fromhex(policy_replay_identity(entry.path))
                        if sample.get("replay_id") != expected_id:
                            raise ValueError(f"{split} row {row}: replay ID differs from manifest")
                        if (
                            entry.annotation is None
                            or int(sample["num_frames"]) != entry.annotation.frame_count_actual
                        ):
                            raise ValueError(f"{split} row {row}: frame count differs from manifest")
                        ports = tuple(sorted(player.port for player in entry.players))
                        if (int(sample["p1_port"]), int(sample["p2_port"])) != ports:
                            raise ValueError(f"{split} row {row}: physical ports differ from manifest")
                        assigned = assign_v8_ranks(
                            source_name,
                            expected_id,
                            int(sample["p1_rank"]),
                            int(sample["p2_rank"]),
                        )
                        actual_assignment = (
                            Rank(int(sample["p1_rank"])),
                            Rank(int(sample["p2_rank"])),
                            int(sample["rank_imputed_mask"]),
                        )
                        if assigned != actual_assignment:
                            raise ValueError(f"{split} row {row}: rank assignment differs from policy")
                        decoded = decode_policy_world_v8_replay(sample)
                        decode_policy_world_v8_reward_events(sample)
                        terminated = any(int(np.asarray(decoded[f"p{side}_stock"])[-1]) == 0 for side in (1, 2))
                        if int(sample["mc_terminated"]) != int(terminated):
                            raise ValueError(f"{split} row {row}: mc_terminated differs from decoded stocks")
                        _record_ranks(report, split, sample)
                        if entry.sha1 is None or len(entry.sha1) != 40:
                            raise ValueError(f"{entry.path}: retained row has no valid SHA-1")
                        previous = seen_sha1.get(entry.sha1)
                        if previous is not None:
                            raise ValueError(f"unexpected duplicate SHA-1 {entry.sha1}: {previous} and {entry.path}")
                        seen_sha1[entry.sha1] = entry.path
                        row += 1
            finally:
                if downloaded is not None:
                    downloaded.unlink(missing_ok=True)
        if row != len(manifests[split]):
            raise ValueError(f"{split}: audited {row} rows, expected {len(manifests[split])}")
    actual_report = _finish_rank_report(report)
    if actual_report != _load_json(root / "ranks.json"):
        raise ValueError("rank report differs from decoded output rows")
    return actual_report["aggregate"]


def audit_policy_world_v8(prefix: str, *, scratch: Path = DEFAULT_SCRATCH, deep: bool = True) -> dict[str, Any]:
    """Audit v8 structure, accounting, ranks, hashes, and every internal CRC."""
    scratch.mkdir(parents=True, exist_ok=True)
    base_audit = audit(prefix) if _is_rclone(prefix) else None
    with tempfile.TemporaryDirectory(dir=scratch, prefix="audit-v8-") as name:
        root = Path(name) if _is_rclone(prefix) else Path(prefix)
        if _is_rclone(prefix):
            _download_output_metadata(prefix, root)
        indexes = _output_indexes(root)
        rows = _rows(indexes)
        metadata, manifests = _audit_metadata(root, rows)
        ranks = (
            _audit_rows(prefix, root, indexes, manifests, Path(name))
            if deep
            else _load_json(root / "ranks.json")["aggregate"]
        )
    return {**({} if base_audit is None else base_audit), **metadata, "ranks": ranks}


@dataclass(frozen=True, slots=True)
class FilterPolicyWorldConfig:
    source: str
    output: str
    source_name: str
    scratch: Path = DEFAULT_SCRATCH
    shard_size: int = DEFAULT_SHARD_SIZE
    stats_batch_rows: int = 64
    metadata_index: str | None = None


def run(cfg: FilterPolicyWorldConfig) -> None:
    result = filter_policy_world_mds(
        cfg.source,
        cfg.output,
        source_name=cfg.source_name,
        scratch=cfg.scratch,
        shard_size=cfg.shard_size,
        stats_batch_rows=cfg.stats_batch_rows,
        metadata_index=cfg.metadata_index,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    run(tyro.cli(FilterPolicyWorldConfig))
