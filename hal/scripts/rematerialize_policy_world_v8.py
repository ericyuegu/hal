"""Filter immutable policy-world-v7 MDS corpora into policy-world-v8."""

from __future__ import annotations

import base64
import dataclasses
import datetime as dt
import hashlib
import json
import math
import re
import shutil
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Final
from typing import Protocol
from typing import cast

import numpy as np
from botocore.exceptions import ClientError

from hal.data.bounded_writer import BoundedMDSWriter
from hal.data.feature_stats import STATS_SCHEMA_VERSION
from hal.data.feature_stats import StatsAccumulator
from hal.data.feature_stats import dump_sufficient_stats
from hal.data.index import ReplayIndexEntry
from hal.data.index import Split
from hal.data.mds import open_shard
from hal.data.policy_schema import POLICY_SCHEMA_VERSION
from hal.data.policy_schema import policy_replay_identity
from hal.data.policy_world_schema import POLICY_WORLD_FLOAT_COLUMNS
from hal.data.policy_world_schema import POLICY_WORLD_MDS_COLUMNS
from hal.data.policy_world_schema import POLICY_WORLD_SCHEMA_VERSION
from hal.data.replay_stats import PlayerStatsMins
from hal.data.schema import SCHEMA_VERSION
from hal.policy import INCLUDED_STAGES
from hal.scripts.filter import Predicate
from hal.scripts.filter import build_predicates

SPLITS: Final[tuple[Split, ...]] = ("train", "val", "test")
POLICY_ID: Final[str] = "policy-world-v8"
OUTPUT_SHARD_SIZE: Final[int] = 256 * 2**20
RANKED_ONE_CORPUS: Final[str] = "ranked-anonymized-1-policy-world-v8"
RANKED_ONE_METADATA_SHA256: Final[str] = "22ac48f73a5ba5cd5718701d3de6666b15e75b2d777458da4927561e0b0fa75d"
RANKED_ONE_METADATA_KEY: Final[str] = (
    "processed/_staging/policy-world-v8/626f9dcfeaa9bca67d0776b1bf6fbd16d8b0ef16/_metadata/"
    f"ranked-1-index.{RANKED_ONE_METADATA_SHA256}.jsonl"
)
DRAFT_PUBLICATION_GIT_SHA: Final[str] = "78bf4d7ab561a3edbc886b11514d3283f9b4f460"
DRAFT_PUBLICATION_RUN_ID: Final[str] = "policy-world-v8-pilot-78bf4d7"
DRAFT_PUBLISHED_CORPORA: Final[frozenset[str]] = frozenset(
    {
        "professional-axe-policy-world-v8",
        "professional-billybopeep-policy-world-v8",
        "professional-bobbybigballz-policy-world-v8",
        "professional-rapm-policy-world-v8",
    }
)
_SHA_PATTERN: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{40}")
_REQUIRED_SIDECARS: Final[frozenset[str]] = frozenset(
    {
        "manifest.jsonl",
        "rejections.jsonl",
        "selection.json",
        "stats.json",
        "projection.json",
        "failures.materialize.jsonl",
    }
)


@dataclass(frozen=True, slots=True)
class PolicyWorldV8Policy:
    """The immutable v8 selection contract."""

    policy_id: str = POLICY_ID
    min_frames: int = 1_500
    stages: tuple[str, ...] = tuple(stage.name for stage in INCLUDED_STAGES)
    characters: tuple[str, ...] = ()
    ranks: tuple[str, ...] = ()
    max_frames: int | None = None
    completed_only: bool = False
    stock_zero_only: bool = False
    min_damage_dealt: float = 100.0
    min_damage_taken: float = 100.0
    min_stocks_remaining: int | None = None
    min_inputs: int | None = None
    min_death_count: int = 3
    max_cheap_deaths: int = 2
    cheap_death_pct: float = 10.0
    human_players_only: bool = True
    min_damage_dealt_per_player_exclusive: float = 50.0
    starting_stocks: int = 4


POLICY_WORLD_V8: Final[PolicyWorldV8Policy] = PolicyWorldV8Policy()


@dataclass(frozen=True, slots=True)
class CorpusJob:
    """One retry-safe corpus invocation."""

    name: str
    source_prefix: str
    staging_prefix: str
    final_prefix: str
    run_id: str
    git_sha: str

    def __post_init__(self) -> None:
        for field_name in ("name", "source_prefix", "staging_prefix", "final_prefix", "run_id"):
            value = getattr(self, field_name)
            if not value or value.startswith("/") or value.endswith("/") or ".." in value.split("/"):
                raise ValueError(f"invalid {field_name}: {value!r}")
        if not _SHA_PATTERN.fullmatch(self.git_sha):
            raise ValueError(f"git_sha must be a full lowercase Git SHA, got {self.git_sha!r}")
        if self.source_prefix == self.final_prefix:
            raise ValueError("source and final prefixes must differ")
        staging_parts = self.staging_prefix.split("/")
        if self.run_id not in staging_parts or self.name not in staging_parts:
            raise ValueError("staging_prefix must contain exact run_id and corpus-name path components")


@dataclass(frozen=True, slots=True)
class StoredObject:
    key: str
    size: int
    etag: str
    metadata: tuple[tuple[str, str], ...] = ()

    def metadata_value(self, name: str) -> str | None:
        return dict(self.metadata).get(name.lower())


class ObjectStore(Protocol):
    """The small object-store surface used by one corpus job."""

    def read_bytes(self, key: str) -> bytes: ...

    def download(self, key: str, destination: Path) -> None: ...

    def put_bytes(self, key: str, data: bytes) -> StoredObject: ...

    def upload(self, key: str, source: Path) -> StoredObject: ...

    def head(self, key: str) -> StoredObject | None: ...

    def list(self, prefix: str) -> dict[str, StoredObject]: ...

    def copy(self, source: str, destination: str) -> StoredObject: ...

    def delete_prefix(self, prefix: str) -> None: ...


def _digests(data: bytes) -> tuple[str, str]:
    return hashlib.md5(data).hexdigest(), hashlib.sha256(data).hexdigest()


def _file_digests(path: Path) -> tuple[str, str, int]:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            md5.update(chunk)
            sha256.update(chunk)
            size += len(chunk)
    return md5.hexdigest(), sha256.hexdigest(), size


class Boto3ObjectStore:
    """Single-part S3 operations against one bucket."""

    def __init__(self, client: Any, bucket: str) -> None:
        self._client = client
        self._bucket = bucket

    @staticmethod
    def _stored(key: str, response: Mapping[str, Any]) -> StoredObject:
        return StoredObject(
            key=key,
            size=int(response["ContentLength"]),
            etag=str(response["ETag"]).strip('"').lower(),
            metadata=tuple(sorted((str(k).lower(), str(v)) for k, v in response.get("Metadata", {}).items())),
        )

    def read_bytes(self, key: str) -> bytes:
        response = self._client.get_object(Bucket=self._bucket, Key=key)
        return cast(bytes, response["Body"].read())

    def download(self, key: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._client.download_file(self._bucket, key, str(destination))

    def put_bytes(self, key: str, data: bytes) -> StoredObject:
        md5, sha256 = _digests(data)
        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=data,
            ContentLength=len(data),
            ContentMD5=base64.b64encode(bytes.fromhex(md5)).decode("ascii"),
            Metadata={"md5": md5, "sha256": sha256},
        )
        stored = self.head(key)
        if stored is None:
            raise RuntimeError(f"uploaded object is absent: {key}")
        return stored

    def upload(self, key: str, source: Path) -> StoredObject:
        md5, sha256, size = _file_digests(source)
        with source.open("rb") as body:
            self._client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=body,
                ContentLength=size,
                ContentMD5=base64.b64encode(bytes.fromhex(md5)).decode("ascii"),
                Metadata={"md5": md5, "sha256": sha256},
            )
        stored = self.head(key)
        if stored is None:
            raise RuntimeError(f"uploaded object is absent: {key}")
        return stored

    def head(self, key: str) -> StoredObject | None:
        try:
            response = self._client.head_object(Bucket=self._bucket, Key=key)
        except ClientError as error:
            code = str(error.response.get("Error", {}).get("Code", ""))
            if code in {"404", "NoSuchKey", "NotFound"}:
                return None
            raise
        return self._stored(key, response)

    def list(self, prefix: str) -> dict[str, StoredObject]:
        out: dict[str, StoredObject] = {}
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                key = str(item["Key"])
                out[key] = StoredObject(
                    key=key,
                    size=int(item["Size"]),
                    etag=str(item["ETag"]).strip('"').lower(),
                )
        return out

    def copy(self, source: str, destination: str) -> StoredObject:
        self._client.copy_object(
            Bucket=self._bucket,
            Key=destination,
            CopySource={"Bucket": self._bucket, "Key": source},
            MetadataDirective="COPY",
        )
        stored = self.head(destination)
        if stored is None:
            raise RuntimeError(f"copied object is absent: {destination}")
        return stored

    def delete_prefix(self, prefix: str) -> None:
        keys = list(self.list(prefix))
        for start in range(0, len(keys), 1_000):
            batch = keys[start : start + 1_000]
            self._client.delete_objects(
                Bucket=self._bucket,
                Delete={"Objects": [{"Key": key} for key in batch], "Quiet": True},
            )


class DirectoryObjectStore:
    """Filesystem-backed object store for focused tests and local pilots."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def _path(self, key: str) -> Path:
        path = self._root / key
        if not path.resolve().is_relative_to(self._root.resolve()):
            raise ValueError(f"object key escapes root: {key!r}")
        return path

    def read_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def download(self, key: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._path(key), destination)

    def put_bytes(self, key: str, data: bytes) -> StoredObject:
        destination = self._path(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        return self._describe(key)

    def upload(self, key: str, source: Path) -> StoredObject:
        destination = self._path(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        return self._describe(key)

    def _describe(self, key: str) -> StoredObject:
        path = self._path(key)
        md5, sha256, size = _file_digests(path)
        return StoredObject(key=key, size=size, etag=md5, metadata=(("md5", md5), ("sha256", sha256)))

    def head(self, key: str) -> StoredObject | None:
        return self._describe(key) if self._path(key).is_file() else None

    def list(self, prefix: str) -> dict[str, StoredObject]:
        base = self._path(prefix)
        if base.is_file():
            return {prefix: self._describe(prefix)}
        if not base.exists():
            return {}
        return {
            str(path.relative_to(self._root)): self._describe(str(path.relative_to(self._root)))
            for path in sorted(base.rglob("*"))
            if path.is_file()
        }

    def copy(self, source: str, destination: str) -> StoredObject:
        target = self._path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._path(source), target)
        return self._describe(destination)

    def delete_prefix(self, prefix: str) -> None:
        path = self._path(prefix)
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def _join(prefix: str, relative: str) -> str:
    return f"{prefix}/{relative}"


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _json_object(data: bytes, where: str) -> dict[str, Any]:
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError(f"{where}: expected a JSON object")
    return cast(dict[str, Any], value)


def _jsonl_objects(data: bytes, where: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(data.decode().splitlines(), 1):
        if not raw_line.strip():
            continue
        value = json.loads(raw_line)
        if not isinstance(value, dict):
            raise ValueError(f"{where}:{line_no}: expected a JSON object")
        rows.append(cast(dict[str, Any], value))
    return rows


def _manifest_entries(data: bytes, where: str) -> tuple[ReplayIndexEntry, ...]:
    entries: list[ReplayIndexEntry] = []
    for line_no, row in enumerate(_jsonl_objects(data, where), 1):
        try:
            entries.append(ReplayIndexEntry.from_dict(row))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{where}:{line_no}: invalid manifest row: {error}") from error
    return tuple(entries)


def _policy_dict() -> dict[str, object]:
    payload = dataclasses.asdict(POLICY_WORLD_V8)
    payload.pop("policy_id")
    payload["stages"] = list(POLICY_WORLD_V8.stages)
    payload["characters"] = list(POLICY_WORLD_V8.characters)
    payload["ranks"] = list(POLICY_WORLD_V8.ranks)
    return payload


def _draft_policy_dict() -> dict[str, object]:
    """Return the exact policy metadata used by four interrupted-run publications."""
    return {
        "policy_id": POLICY_ID,
        "min_frames": POLICY_WORLD_V8.min_frames,
        "stages": list(POLICY_WORLD_V8.stages),
        "min_damage_dealt_any": POLICY_WORLD_V8.min_damage_dealt,
        "min_damage_taken_any": POLICY_WORLD_V8.min_damage_taken,
        "min_death_count_any": POLICY_WORLD_V8.min_death_count,
        "max_cheap_deaths_exclusive": POLICY_WORLD_V8.max_cheap_deaths,
        "cheap_death_pct_inclusive": POLICY_WORLD_V8.cheap_death_pct,
        "exactly_two_humans": POLICY_WORLD_V8.human_players_only,
        "min_damage_dealt_each_exclusive": POLICY_WORLD_V8.min_damage_dealt_per_player_exclusive,
        "inferred_starting_stocks_each": POLICY_WORLD_V8.starting_stocks,
    }


def _v7_predicates() -> tuple[tuple[str, Predicate], ...]:
    return tuple(
        build_predicates(
            min_frames=POLICY_WORLD_V8.min_frames,
            stages=set(INCLUDED_STAGES),
            mins=PlayerStatsMins(
                damage_dealt=POLICY_WORLD_V8.min_damage_dealt,
                damage_taken=POLICY_WORLD_V8.min_damage_taken,
            ),
            min_death_count=POLICY_WORLD_V8.min_death_count,
            max_cheap_deaths=POLICY_WORLD_V8.max_cheap_deaths,
            cheap_death_pct=POLICY_WORLD_V8.cheap_death_pct,
        )
    )


def _v8_predicates() -> tuple[tuple[str, Predicate], ...]:
    return tuple(
        build_predicates(
            human_players_only=POLICY_WORLD_V8.human_players_only,
            min_damage_dealt_per_player_exclusive=POLICY_WORLD_V8.min_damage_dealt_per_player_exclusive,
            starting_stocks=POLICY_WORLD_V8.starting_stocks,
        )
    )


@dataclass(frozen=True, slots=True)
class Rejection:
    path: str
    replay_id: str
    split: Split
    source_mds_row_idx: int
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "split": self.split,
            "source_mds_row_idx": self.source_mds_row_idx,
            "failed_predicates": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class Selection:
    entries: tuple[ReplayIndexEntry, ...]
    entries_by_split: dict[Split, tuple[ReplayIndexEntry, ...]]
    kept_rows: dict[Split, tuple[int, ...]]
    rejections: tuple[Rejection, ...]
    input_counts: dict[Split, int]
    retained_counts: dict[Split, int]


def select_manifest(
    entries: tuple[ReplayIndexEntry, ...],
    rows_by_split: Mapping[Split, int],
    *,
    where: str,
) -> Selection:
    """Validate a v7 manifest and evaluate every v8 rejection reason."""
    by_split_row: dict[Split, dict[int, ReplayIndexEntry]] = {split: {} for split in SPLITS}
    paths: set[str] = set()
    replay_ids: set[str] = set()
    v7_failures: list[str] = []
    v7_predicates = _v7_predicates()
    v8_predicates = _v8_predicates()
    rejections: list[Rejection] = []
    kept: dict[Split, list[int]] = {split: [] for split in SPLITS}

    for entry in entries:
        if entry.schema_version != SCHEMA_VERSION:
            raise ValueError(f"{where}: {entry.path}: schema_version={entry.schema_version} != {SCHEMA_VERSION}")
        if entry.path in paths:
            raise ValueError(f"{where}: duplicate manifest path {entry.path!r}")
        paths.add(entry.path)
        replay_id = policy_replay_identity(entry.path)
        if replay_id in replay_ids:
            raise ValueError(f"{where}: duplicate replay identity {replay_id}")
        replay_ids.add(replay_id)
        annotation = entry.annotation
        if annotation is None:
            raise ValueError(f"{where}: {entry.path}: missing MDS annotation")
        if annotation.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"{where}: {entry.path}: annotation schema_version={annotation.schema_version} != {SCHEMA_VERSION}"
            )
        if entry.stats is None:
            raise ValueError(f"{where}: {entry.path}: missing replay statistics")
        for player in entry.stats.players:
            finite_values = (player.damage_dealt, player.damage_taken, *player.death_percents)
            if not all(math.isfinite(value) and value >= 0.0 for value in finite_values):
                raise ValueError(f"{where}: {entry.path}: player {player.port} has invalid statistics")
            if player.stocks_remaining < 0 or player.inputs < 0:
                raise ValueError(f"{where}: {entry.path}: player {player.port} has negative statistics")
        player_ports = sorted(player.port for player in entry.players)
        stat_ports = [player.port for player in entry.stats.players]
        if player_ports != stat_ports:
            raise ValueError(f"{where}: {entry.path}: player ports {player_ports} != statistics ports {stat_ports}")
        claimed = by_split_row[annotation.split]
        if annotation.mds_row_idx in claimed:
            raise ValueError(
                f"{where}: duplicate {annotation.split} mds_row_idx={annotation.mds_row_idx} "
                f"for {entry.path!r} and {claimed[annotation.mds_row_idx].path!r}"
            )
        claimed[annotation.mds_row_idx] = entry

        failed_v7 = tuple(label for label, predicate in v7_predicates if not predicate(entry))
        if failed_v7:
            v7_failures.append(f"{entry.path}: {failed_v7}")
            continue
        reasons = tuple(label for label, predicate in v8_predicates if not predicate(entry))
        if reasons:
            rejections.append(
                Rejection(
                    path=entry.path,
                    replay_id=replay_id,
                    split=annotation.split,
                    source_mds_row_idx=annotation.mds_row_idx,
                    reasons=reasons,
                )
            )
        else:
            kept[annotation.split].append(annotation.mds_row_idx)

    for split in SPLITS:
        expected = rows_by_split[split]
        observed = sorted(by_split_row[split])
        if observed != list(range(expected)):
            raise ValueError(f"{where}: {split} manifest annotations are not contiguous 0..{expected - 1}")
    if v7_failures:
        examples = "; ".join(v7_failures[:10])
        raise ValueError(f"{where}: {len(v7_failures)} source rows fail the immutable v7 policy: {examples}")

    entries_by_split = {
        split: tuple(by_split_row[split][row] for row in range(rows_by_split[split])) for split in SPLITS
    }
    retained_counts = {split: len(kept[split]) for split in SPLITS}
    return Selection(
        entries=entries,
        entries_by_split=entries_by_split,
        kept_rows={split: tuple(kept[split]) for split in SPLITS},
        rejections=tuple(rejections),
        input_counts=dict(rows_by_split),
        retained_counts=retained_counts,
    )


@dataclass(frozen=True, slots=True)
class SourceDataset:
    indexes: dict[Split, dict[str, Any]]
    selection: Selection
    identity: dict[str, object]
    materialization_failures: int
    supplemental_metadata: dict[str, object] | None


def _validate_index(index: dict[str, Any], where: str) -> int:
    if index.get("version") != 2 or not isinstance(index.get("shards"), list):
        raise ValueError(f"{where}: invalid MDS index")
    expected_columns = sorted(POLICY_WORLD_MDS_COLUMNS.items())
    rows = 0
    for shard_id, raw_shard in enumerate(index["shards"]):
        if not isinstance(raw_shard, dict):
            raise ValueError(f"{where}: shard {shard_id} is not an object")
        shard = cast(dict[str, Any], raw_shard)
        columns = list(zip(shard.get("column_names", []), shard.get("column_encodings", []), strict=True))
        if columns != expected_columns:
            raise ValueError(f"{where}: shard {shard_id} column schema differs from policy-world schema 1")
        if shard.get("compression") != "zstd" or shard.get("format") != "mds" or shard.get("version") != 2:
            raise ValueError(f"{where}: shard {shard_id} has unsupported MDS encoding")
        samples = int(shard.get("samples", -1))
        if samples < 1:
            raise ValueError(f"{where}: shard {shard_id} has invalid sample count {samples}")
        for variant in ("raw_data", "zip_data"):
            info = shard.get(variant)
            if not isinstance(info, dict):
                raise ValueError(f"{where}: shard {shard_id} is missing {variant}")
            hashes = info.get("hashes")
            if not isinstance(hashes, dict) or set(hashes) != {"md5", "sha256"}:
                raise ValueError(f"{where}: shard {shard_id} {variant} lacks exact md5 and sha256 hashes")
        rows += samples
    return rows


def _validate_stats_payload(payload: dict[str, Any], where: str) -> None:
    if payload.get("schema_version") != STATS_SCHEMA_VERSION:
        raise ValueError(f"{where}: wrong stats schema version")
    if payload.get("mds_schema_version") != SCHEMA_VERSION or payload.get("split") != "train":
        raise ValueError(f"{where}: statistics do not describe schema-v7 train rows")
    sufficient = payload.get("sufficient")
    if not isinstance(sufficient, dict) or set(sufficient) != set(POLICY_WORLD_FLOAT_COLUMNS):
        raise ValueError(f"{where}: statistics feature set differs from the policy-world schema")
    if payload.get("feature_count") != len(POLICY_WORLD_FLOAT_COLUMNS):
        raise ValueError(f"{where}: incorrect statistics feature_count")
    for name, raw_block in sufficient.items():
        if not isinstance(raw_block, dict) or set(raw_block) != {"count", "mean", "m2", "min", "max"}:
            raise ValueError(f"{where}: invalid sufficient statistics for {name}")
        if int(raw_block["count"]) < 0:
            raise ValueError(f"{where}: negative statistics count for {name}")


def _supplement_manifest_statistics(
    store: ObjectStore,
    job: CorpusJob,
    entries: tuple[ReplayIndexEntry, ...],
) -> tuple[tuple[ReplayIndexEntry, ...], dict[str, object] | None]:
    if job.name != RANKED_ONE_CORPUS:
        return entries, None
    data = store.read_bytes(RANKED_ONE_METADATA_KEY)
    observed_sha256 = hashlib.sha256(data).hexdigest()
    if observed_sha256 != RANKED_ONE_METADATA_SHA256:
        raise ValueError(f"{RANKED_ONE_METADATA_KEY}: sha256 {observed_sha256} != pinned {RANKED_ONE_METADATA_SHA256}")
    supplement: dict[str, ReplayIndexEntry] = {}
    for entry in _manifest_entries(data, RANKED_ONE_METADATA_KEY):
        if entry.path in supplement:
            raise ValueError(f"{RANKED_ONE_METADATA_KEY}: duplicate path {entry.path!r}")
        supplement[entry.path] = entry

    hydrated: list[ReplayIndexEntry] = []
    for entry in entries:
        if entry.stats is not None:
            hydrated.append(entry)
            continue
        extra = supplement.get(entry.path)
        if extra is None or extra.stats is None:
            raise ValueError(f"{entry.path}: source and supplemental manifests have no replay statistics")
        source_players = [(player.port, player.player_type) for player in entry.players]
        extra_players = [(player.port, player.player_type) for player in extra.players]
        if extra.schema_version != SCHEMA_VERSION:
            raise ValueError(f"{entry.path}: supplemental metadata is not canonical schema v7")
        if extra.stage != entry.stage or extra.frame_count != entry.frame_count or extra_players != source_players:
            raise ValueError(f"{entry.path}: supplemental metadata differs from the source manifest")
        if [player.port for player in extra.stats.players] != [player.port for player in entry.players]:
            raise ValueError(f"{entry.path}: supplemental statistics use different player ports")
        hydrated.append(dataclasses.replace(entry, stats=extra.stats, sha1=entry.sha1 or extra.sha1))
    identity: dict[str, object] = {
        "key": RANKED_ONE_METADATA_KEY,
        "sha256": observed_sha256,
        "bytes": len(data),
    }
    return tuple(hydrated), identity


def inspect_source(store: ObjectStore, job: CorpusJob) -> SourceDataset:
    """Validate source sidecars and build the split-row keep mask."""
    indexes: dict[Split, dict[str, Any]] = {}
    index_hashes: dict[str, str] = {}
    rows: dict[Split, int] = {}
    for split in SPLITS:
        key = _join(job.source_prefix, f"{split}/index.json")
        data = store.read_bytes(key)
        indexes[split] = _json_object(data, key)
        rows[split] = _validate_index(indexes[split], key)
        index_hashes[split] = hashlib.sha256(data).hexdigest()

    manifest_key = _join(job.source_prefix, "manifest.jsonl")
    manifest_data = store.read_bytes(manifest_key)
    raw_entries = _manifest_entries(manifest_data, manifest_key)
    entries, supplemental = _supplement_manifest_statistics(store, job, raw_entries)
    selection = select_manifest(entries, rows, where=manifest_key)

    projection_key = _join(job.source_prefix, "projection.json")
    projection = _json_object(store.read_bytes(projection_key), projection_key)
    projected_rows = {split: int(projection.get("rows", {}).get(split, -1)) for split in SPLITS}
    if projected_rows != rows:
        raise ValueError(f"{projection_key}: source accounting differs from MDS indexes")
    if projection.get("policy_world_schema_version") != POLICY_WORLD_SCHEMA_VERSION:
        raise ValueError(f"{projection_key}: wrong policy-world schema version")
    if projection.get("source_schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{projection_key}: wrong source schema version")
    if projection.get("columns") != POLICY_WORLD_MDS_COLUMNS:
        raise ValueError(f"{projection_key}: projected column schema differs")

    failures_key = _join(job.source_prefix, "failures.materialize.jsonl")
    failure_data = store.read_bytes(failures_key)
    failure_rows = _jsonl_objects(failure_data, failures_key)
    failure_paths = [str(row.get("path", "")) for row in failure_rows]
    if any(not path for path in failure_paths) or len(failure_paths) != len(set(failure_paths)):
        raise ValueError(f"{failures_key}: failure ledger has a missing or duplicate path")
    if set(failure_paths) & {entry.path for entry in raw_entries}:
        raise ValueError(f"{failures_key}: failure ledger overlaps the retained manifest")
    failure_count = len(failure_rows)
    if int(projection.get("failures", -1)) != failure_count:
        raise ValueError(f"{projection_key}: failure count differs from the source failure ledger")
    stats_key = _join(job.source_prefix, "stats.json")
    _validate_stats_payload(_json_object(store.read_bytes(stats_key), stats_key), stats_key)
    success_key = _join(job.source_prefix, "_SUCCESS")
    success_data = store.read_bytes(success_key)
    success = _json_object(success_data, success_key)
    if "rows" in success and {split: int(success["rows"].get(split, -1)) for split in SPLITS} != rows:
        raise ValueError(f"{success_key}: source success marker row counts differ")
    if int(success.get("failures", failure_count)) != failure_count:
        raise ValueError(f"{success_key}: source success marker failure count differs")

    identity: dict[str, object] = {
        "prefix": job.source_prefix,
        "manifest_sha256": hashlib.sha256(manifest_data).hexdigest(),
        "index_sha256": index_hashes,
        "success_sha256": hashlib.sha256(success_data).hexdigest(),
        "failures_sha256": hashlib.sha256(failure_data).hexdigest(),
        "rows": rows,
    }
    if supplemental is not None:
        identity["supplemental_metadata"] = supplemental
    return SourceDataset(
        indexes=indexes,
        selection=selection,
        identity=identity,
        materialization_failures=failure_count,
        supplemental_metadata=supplemental,
    )


def _selection_payload(job: CorpusJob, source: SourceDataset) -> dict[str, object]:
    rejected = {
        split: source.selection.input_counts[split] - source.selection.retained_counts[split] for split in SPLITS
    }
    frames = dict.fromkeys(SPLITS, 0)
    for split in SPLITS:
        keep = set(source.selection.kept_rows[split])
        frames[split] = sum(
            entry.annotation.frame_count_actual
            for row, entry in enumerate(source.selection.entries_by_split[split])
            if row in keep and entry.annotation is not None
        )
    rejection_reasons = Counter(reason for rejection in source.selection.rejections for reason in rejection.reasons)
    payload: dict[str, object] = {
        "schema_version": 1,
        "artifact": "mds-policy-world-v8",
        "policy": _policy_dict(),
        "source": f"r2:hal/{job.source_prefix}",
        "source_rows": source.selection.input_counts,
        "rows": source.selection.retained_counts,
        "rejections": rejected,
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "frames": frames,
        "source_materialization_failures": source.materialization_failures,
        "source_identity": source.identity,
        "implementation": {
            "git_sha": job.git_sha,
            "git_dirty": False,
        },
    }
    if source.supplemental_metadata is not None:
        payload["supplemental_metadata"] = source.supplemental_metadata
    return payload


def _draft_source_identity(source: SourceDataset) -> dict[str, object]:
    return {
        name: source.identity[name] for name in ("prefix", "manifest_sha256", "index_sha256", "success_sha256", "rows")
    }


def _validate_draft_selection_payload(
    payload: dict[str, Any],
    job: CorpusJob,
    source: SourceDataset,
) -> None:
    if job.name not in DRAFT_PUBLISHED_CORPORA:
        raise ValueError("draft selection metadata is not allowed for this corpus")
    input_total = sum(source.selection.input_counts.values())
    retained_total = sum(source.selection.retained_counts.values())
    expected_counts = {
        "input": source.selection.input_counts,
        "retained": source.selection.retained_counts,
        "rejected": {
            split: source.selection.input_counts[split] - source.selection.retained_counts[split] for split in SPLITS
        },
        "input_total": input_total,
        "retained_total": retained_total,
        "rejected_total": input_total - retained_total,
    }
    expected = {
        "schema_version": 1,
        "policy": _draft_policy_dict(),
        "corpus": job.name,
        "source": _draft_source_identity(source),
        "counts": expected_counts,
        "run_id": DRAFT_PUBLICATION_RUN_ID,
        "git_sha": DRAFT_PUBLICATION_GIT_SHA,
    }
    if set(payload) != set(expected) or payload != expected:
        raise ValueError("selection differs from the exact interrupted-run v8 contract")


def _validate_selection_payload(
    payload: dict[str, Any],
    job: CorpusJob,
    source: SourceDataset,
    *,
    require_current_run: bool,
) -> None:
    expected = _selection_payload(job, source)
    if require_current_run:
        if payload != expected:
            raise ValueError("staging selection differs from the current job")
        return
    if "artifact" not in payload:
        _validate_draft_selection_payload(payload, job, source)
        return
    stable_fields = (
        "schema_version",
        "artifact",
        "policy",
        "source",
        "source_rows",
        "rows",
        "rejections",
        "rejection_reasons",
        "frames",
        "source_materialization_failures",
    )
    if any(payload.get(name) != expected[name] for name in stable_fields):
        raise ValueError("selection policy, source partition, or counts differ")
    identity = payload.get("source_identity")
    if identity is not None and identity != source.identity:
        raise ValueError("selection source identity differs")
    if source.supplemental_metadata is not None:
        supplemental = payload.get("supplemental_metadata")
        if not isinstance(supplemental, dict):
            raise ValueError("selection omits the required supplemental metadata")
        if supplemental.get("sha256") != source.supplemental_metadata["sha256"]:
            raise ValueError("selection supplemental metadata differs")


_DRAFT_REJECTION_REASONS: Final[dict[str, str]] = {
    "exactly_two_humans": "human_players_only",
    f"damage_dealt_each>{POLICY_WORLD_V8.min_damage_dealt_per_player_exclusive}": (
        f"damage_dealt_per_player>{POLICY_WORLD_V8.min_damage_dealt_per_player_exclusive}"
    ),
    f"inferred_starting_stocks_each={POLICY_WORLD_V8.starting_stocks}": (
        f"starting_stocks={POLICY_WORLD_V8.starting_stocks}"
    ),
}


def _normalize_rejection_row(row: dict[str, Any], where: str) -> dict[str, object]:
    draft = "reasons" in row
    reason_field = "reasons" if draft else "failed_predicates"
    reasons = row.get(reason_field)
    if not isinstance(reasons, list) or not reasons or not all(isinstance(reason, str) for reason in reasons):
        raise ValueError(f"{where}: invalid {reason_field}")
    expected_keys = {"path", "split", "source_mds_row_idx", reason_field}
    if draft:
        expected_keys.add("replay_id")
    if set(row) != expected_keys:
        raise ValueError(f"{where}: invalid rejection fields")
    path = str(row["path"])
    normalized: dict[str, object] = {
        "path": path,
        "split": str(row["split"]),
        "source_mds_row_idx": int(row["source_mds_row_idx"]),
        "failed_predicates": sorted(_DRAFT_REJECTION_REASONS.get(reason, reason) for reason in reasons),
    }
    if draft and row["replay_id"] != policy_replay_identity(path):
        raise ValueError(f"{where}: draft rejection replay identity differs from its path")
    return normalized


class _StoreUploader:
    def __init__(self, store: ObjectStore, local: Path, prefix: str) -> None:
        self._store = store
        self._local = local
        self._prefix = prefix

    def upload_file(self, filename: str) -> None:
        path = self._local / filename
        self._store.upload(_join(self._prefix, filename), path)
        path.unlink()


class _StoreMDSWriter(BoundedMDSWriter):
    def __init__(self, store: ObjectStore, local: Path, prefix: str) -> None:
        super().__init__(
            out=str(local),
            keep_local=False,
            columns=POLICY_WORLD_MDS_COLUMNS,
            compression="zstd",
            hashes=["md5", "sha256"],
            size_limit=OUTPUT_SHARD_SIZE,
            exist_ok=False,
            max_workers=2,
            max_pending_uploads=2,
        )
        self.cloud_writer = _StoreUploader(store, local, prefix)
        self.local = str(local)
        self.remote = prefix


def _scalar_int(sample: Mapping[str, object], name: str) -> int:
    value = np.asarray(sample[name])
    if value.shape:
        raise ValueError(f"{name} must be scalar, got {value.shape}")
    return int(value.item())


def _validate_sample(sample: dict[str, Any], entry: ReplayIndexEntry, where: str) -> None:
    if set(sample) != set(POLICY_WORLD_MDS_COLUMNS):
        raise ValueError(f"{where}: MDS row column schema differs")
    expected_scalars = {
        "policy_schema_version": POLICY_SCHEMA_VERSION,
        "policy_world_schema_version": POLICY_WORLD_SCHEMA_VERSION,
        "source_schema_version": SCHEMA_VERSION,
        "num_frames": entry.annotation.frame_count_actual if entry.annotation is not None else -1,
    }
    for name, expected in expected_scalars.items():
        observed = _scalar_int(sample, name)
        if observed != expected:
            raise ValueError(f"{where}: {name}={observed} != {expected}")
    replay_id = str(sample["replay_id"])
    expected_id = policy_replay_identity(entry.path)
    if replay_id != expected_id:
        raise ValueError(f"{where}: replay_id={replay_id!r} != manifest identity {expected_id!r}")


def _verify_download(path: Path, info: Mapping[str, Any], where: str) -> None:
    md5, sha256, size = _file_digests(path)
    if size != int(info["bytes"]):
        raise ValueError(f"{where}: downloaded size {size} != index size {info['bytes']}")
    hashes = info["hashes"]
    if md5 != hashes["md5"] or sha256 != hashes["sha256"]:
        raise ValueError(f"{where}: downloaded shard hash differs from index")


def _download_source_shard(
    store: ObjectStore,
    job: CorpusJob,
    split: Split,
    shard: Mapping[str, Any],
    source_root: Path,
) -> Path:
    zip_info = cast(dict[str, Any], shard["zip_data"])
    basename = str(zip_info["basename"])
    key = _join(job.source_prefix, f"{split}/{basename}")
    remote = store.head(key)
    if remote is None:
        raise FileNotFoundError(f"source index references missing object {key}")
    if remote.size != int(zip_info["bytes"]):
        raise ValueError(f"{key}: object size differs from source index")
    destination = source_root / split / basename
    store.download(key, destination)
    _verify_download(destination, zip_info, key)
    return destination


def _finish_writers(writers: Mapping[Split, _StoreMDSWriter], active_error: BaseException | None) -> None:
    finish_error: BaseException | None = None
    for writer in writers.values():
        try:
            writer.finish()
        except BaseException as error:
            if finish_error is None:
                finish_error = error
    if active_error is None and finish_error is not None:
        raise finish_error


def _write_sidecars(
    store: ObjectStore,
    job: CorpusJob,
    source: SourceDataset,
    rewritten: Mapping[tuple[Split, int], ReplayIndexEntry],
    stats: StatsAccumulator,
    scratch: Path,
) -> None:
    manifest_rows: list[str] = []
    for entry in source.selection.entries:
        annotation = entry.annotation
        assert annotation is not None
        retained = rewritten.get((annotation.split, annotation.mds_row_idx))
        if retained is not None:
            manifest_rows.append(json.dumps(retained.to_dict(), sort_keys=True))
    manifest = ("\n".join(manifest_rows) + ("\n" if manifest_rows else "")).encode()
    store.put_bytes(_join(job.staging_prefix, "manifest.jsonl"), manifest)

    rejection_rows = [json.dumps(row.to_dict(), sort_keys=True) for row in source.selection.rejections]
    rejections = ("\n".join(rejection_rows) + ("\n" if rejection_rows else "")).encode()
    store.put_bytes(_join(job.staging_prefix, "rejections.jsonl"), rejections)

    selection = _selection_payload(job, source)
    store.put_bytes(_join(job.staging_prefix, "selection.json"), _json_bytes(selection))
    rejected = {
        split: source.selection.input_counts[split] - source.selection.retained_counts[split] for split in SPLITS
    }
    projection = {
        "policy_world_schema_version": POLICY_WORLD_SCHEMA_VERSION,
        "source_schema_version": SCHEMA_VERSION,
        "source": f"r2:hal/{job.source_prefix}",
        "source_rows": source.selection.input_counts,
        "rows": source.selection.retained_counts,
        "rejections": rejected,
        "failures": 0,
        "columns": POLICY_WORLD_MDS_COLUMNS,
    }
    store.put_bytes(_join(job.staging_prefix, "projection.json"), _json_bytes(projection))
    store.put_bytes(_join(job.staging_prefix, "failures.materialize.jsonl"), b"")

    stats_path = scratch / "stats.json"
    dump_sufficient_stats(
        stats_path,
        stats.to_sufficient(),
        split="train",
        mds_schema_version=SCHEMA_VERSION,
    )
    store.upload(_join(job.staging_prefix, "stats.json"), stats_path)


def _build_staging(store: ObjectStore, job: CorpusJob, source: SourceDataset, scratch: Path) -> None:
    source_root = scratch / "source"
    output_root = scratch / "output"
    decompress_root = scratch / "decompress"
    for root in (source_root, output_root, decompress_root):
        root.mkdir(parents=True, exist_ok=True)
    writers = {
        split: _StoreMDSWriter(store, output_root / split, _join(job.staging_prefix, split)) for split in SPLITS
    }
    stats = StatsAccumulator(POLICY_WORLD_FLOAT_COLUMNS)
    rewritten: dict[tuple[Split, int], ReplayIndexEntry] = {}
    active_error: BaseException | None = None
    try:
        for split in SPLITS:
            keep = set(source.selection.kept_rows[split])
            source_row = 0
            output_row = 0
            for shard_id, shard in enumerate(source.indexes[split]["shards"]):
                compressed = _download_source_shard(store, job, split, shard, source_root)
                shard_rows = 0
                try:
                    with open_shard(source_root, split, shard, decompress_root) as reader:
                        for shard_row, sample in enumerate(reader):
                            entry = source.selection.entries_by_split[split][source_row]
                            where = f"{job.name}: {split} source row {source_row}, shard {shard_id} row {shard_row}"
                            _validate_sample(sample, entry, where)
                            if source_row in keep:
                                writers[split].write(sample)
                                if split == "train":
                                    for name in POLICY_WORLD_FLOAT_COLUMNS:
                                        stats.update(name, np.asarray(sample[name]))
                                annotation = entry.annotation
                                assert annotation is not None
                                rewritten[(split, source_row)] = dataclasses.replace(
                                    entry,
                                    annotation=dataclasses.replace(annotation, mds_row_idx=output_row),
                                )
                                output_row += 1
                            source_row += 1
                            shard_rows += 1
                finally:
                    compressed.unlink(missing_ok=True)
                if shard_rows != int(shard["samples"]):
                    raise ValueError(
                        f"{job.name}: {split} source shard {shard_id} yielded {shard_rows} rows, "
                        f"expected {shard['samples']}"
                    )
            if source_row != source.selection.input_counts[split]:
                raise ValueError(f"{job.name}: {split} source MDS row accounting changed while reading")
            if output_row != source.selection.retained_counts[split]:
                raise ValueError(f"{job.name}: {split} retained row accounting changed while writing")
    except BaseException as error:
        active_error = error
        raise
    finally:
        _finish_writers(writers, active_error)
    _write_sidecars(store, job, source, rewritten, stats, scratch)


def _relative_objects(store: ObjectStore, prefix: str) -> dict[str, StoredObject]:
    stem = prefix + "/"
    return {key.removeprefix(stem): value for key, value in store.list(stem).items()}


def _check_object_hash(store: ObjectStore, key: str, expected: Mapping[str, Any]) -> None:
    obj = store.head(key)
    if obj is None:
        raise FileNotFoundError(f"missing indexed object {key}")
    if obj.size != int(expected["bytes"]):
        raise ValueError(f"{key}: stored size differs from index")
    expected_md5 = str(expected["hashes"]["md5"])
    expected_sha256 = str(expected["hashes"]["sha256"])
    if obj.etag != expected_md5:
        raise ValueError(f"{key}: ETag is not the indexed MD5; upload was not single-part or data changed")
    metadata_sha256 = obj.metadata_value("sha256")
    if metadata_sha256 is not None and metadata_sha256 != expected_sha256:
        raise ValueError(f"{key}: sha256 object metadata differs from index")


def _canonical_sufficient(stats: StatsAccumulator) -> dict[str, dict[str, float | int]]:
    return {
        name: {
            "count": block.count,
            "mean": block.mean,
            "m2": block.m2,
            "min": block.min,
            "max": block.max,
        }
        for name, block in stats.to_sufficient().items()
    }


def audit_dataset(
    store: ObjectStore,
    prefix: str,
    *,
    job: CorpusJob,
    source: SourceDataset,
    scratch: Path,
) -> dict[str, object]:
    """Read and validate every retained row and all publication metadata."""
    (scratch / "audit-decompress").mkdir(parents=True, exist_ok=True)
    objects = _relative_objects(store, prefix)
    missing = sorted(_REQUIRED_SIDECARS - objects.keys())
    if missing:
        raise ValueError(f"{prefix}: missing required objects {missing}")
    if "_STAGED" in objects:
        marker_key = _join(prefix, "_STAGED")
        marker = _json_object(store.read_bytes(marker_key), marker_key)
        if marker.get("run_id") != job.run_id or marker.get("git_sha") != job.git_sha:
            raise ValueError(f"{marker_key}: marker provenance differs from the current job")
    if "_SUCCESS" in objects:
        marker_key = _join(prefix, "_SUCCESS")
        marker = _json_object(store.read_bytes(marker_key), marker_key)
        marker_rows = {
            split: int(marker.get("rows", marker.get("audit", {}).get("rows", {})).get(split, -1)) for split in SPLITS
        }
        if marker_rows != source.selection.retained_counts:
            raise ValueError(f"{marker_key}: publication row counts differ from the requested corpus")
        marker_final = str(marker.get("final", marker.get("final_prefix", ""))).removeprefix("r2:hal/")
        if marker_final != job.final_prefix:
            raise ValueError(f"{marker_key}: final prefix differs from the requested corpus")
        allowed_policies = (_policy_dict(),)
        allowed_sources = (source.identity,)
        if job.name in DRAFT_PUBLISHED_CORPORA:
            allowed_policies += (_draft_policy_dict(),)
            allowed_sources += (_draft_source_identity(source),)
        if "policy" in marker and marker.get("policy") not in allowed_policies:
            raise ValueError(f"{marker_key}: policy differs from the v8 contract")
        if marker.get("policy") == _draft_policy_dict() and (
            marker.get("run_id") != DRAFT_PUBLICATION_RUN_ID or marker.get("git_sha") != DRAFT_PUBLICATION_GIT_SHA
        ):
            raise ValueError(f"{marker_key}: draft publication provenance differs")
        marker_source = marker.get("source")
        if isinstance(marker_source, dict) and marker_source not in allowed_sources:
            raise ValueError(f"{marker_key}: source identity differs from the requested corpus")

    selection_key = _join(prefix, "selection.json")
    selection_payload = _json_object(store.read_bytes(selection_key), selection_key)
    try:
        _validate_selection_payload(
            selection_payload,
            job,
            source,
            require_current_run=prefix == job.staging_prefix,
        )
    except ValueError as error:
        raise ValueError(f"{selection_key}: {error}") from error
    projection_key = _join(prefix, "projection.json")
    projection = _json_object(store.read_bytes(projection_key), projection_key)
    if projection.get("policy_world_schema_version") != POLICY_WORLD_SCHEMA_VERSION:
        raise ValueError(f"{projection_key}: wrong policy-world schema version")
    if projection.get("source_schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{projection_key}: wrong source identity or schema")
    expected_source = f"r2:hal/{job.source_prefix}"
    projection_source = projection.get("source")
    allowed_projection_sources = (expected_source,)
    if job.name in DRAFT_PUBLISHED_CORPORA:
        allowed_projection_sources += (_draft_source_identity(source),)
    if projection_source not in allowed_projection_sources:
        raise ValueError(f"{projection_key}: wrong source partition")
    if projection.get("columns") != POLICY_WORLD_MDS_COLUMNS or projection.get("failures") != 0:
        raise ValueError(f"{projection_key}: invalid projection contract")
    if isinstance(projection_source, str):
        source_rows = {split: int(projection.get("source_rows", {}).get(split, -1)) for split in SPLITS}
        if source_rows != source.selection.input_counts:
            raise ValueError(f"{projection_key}: source row counts differ from selection")
    if {split: int(projection.get("rows", {}).get(split, -1)) for split in SPLITS} != source.selection.retained_counts:
        raise ValueError(f"{projection_key}: row counts differ from selection")
    rejected = {
        split: source.selection.input_counts[split] - source.selection.retained_counts[split] for split in SPLITS
    }
    if (
        isinstance(projection_source, str)
        and {split: int(projection.get("rejections", {}).get(split, -1)) for split in SPLITS} != rejected
    ):
        raise ValueError(f"{projection_key}: rejection counts differ from selection")
    failures_key = _join(prefix, "failures.materialize.jsonl")
    if store.read_bytes(failures_key).strip():
        raise ValueError(f"{failures_key}: rematerialization failure ledger is not empty")

    rejection_key = _join(prefix, "rejections.jsonl")
    rejection_rows = _jsonl_objects(store.read_bytes(rejection_key), rejection_key)
    if any("reasons" in row for row in rejection_rows) and job.name not in DRAFT_PUBLISHED_CORPORA:
        raise ValueError(f"{rejection_key}: draft rejection metadata is not allowed for this corpus")
    expected_rejections = [row.to_dict() for row in source.selection.rejections]
    actual_by_position: dict[tuple[str, int], dict[str, object]] = {}
    for row in rejection_rows:
        normalized = _normalize_rejection_row(row, rejection_key)
        position = (str(normalized["split"]), int(cast(int, normalized["source_mds_row_idx"])))
        if position in actual_by_position:
            raise ValueError(f"{rejection_key}: duplicate source position {position}")
        actual_by_position[position] = normalized
    expected_by_position = {
        (str(row["split"]), int(cast(int, row["source_mds_row_idx"]))): _normalize_rejection_row(
            cast(dict[str, Any], row), rejection_key
        )
        for row in expected_rejections
    }
    if actual_by_position != expected_by_position:
        raise ValueError(f"{rejection_key}: rejection ledger differs from complete policy evaluation")

    manifest_key = _join(prefix, "manifest.jsonl")
    retained_manifest = _manifest_entries(store.read_bytes(manifest_key), manifest_key)
    selected_paths = {
        entry.path
        for split in SPLITS
        for row, entry in enumerate(source.selection.entries_by_split[split])
        if row in set(source.selection.kept_rows[split])
    }
    if {entry.path for entry in retained_manifest} != selected_paths or len(retained_manifest) != len(selected_paths):
        raise ValueError(f"{manifest_key}: retained row identities differ from selection")
    retained_by_split: dict[Split, dict[int, ReplayIndexEntry]] = {split: {} for split in SPLITS}
    for entry in retained_manifest:
        annotation = entry.annotation
        if annotation is None:
            raise ValueError(f"{manifest_key}: retained row {entry.path!r} is unannotated")
        rows = retained_by_split[annotation.split]
        if annotation.mds_row_idx in rows:
            raise ValueError(f"{manifest_key}: duplicate retained annotation")
        rows[annotation.mds_row_idx] = entry

    stats = StatsAccumulator(POLICY_WORLD_FLOAT_COLUMNS)
    train_frames = 0
    index_rows: dict[Split, int] = {}
    expected_objects = set(_REQUIRED_SIDECARS)
    for split in SPLITS:
        index_name = f"{split}/index.json"
        expected_objects.add(index_name)
        index_key = _join(prefix, index_name)
        index_data = store.read_bytes(index_key)
        index = _json_object(index_data, index_key)
        index_rows[split] = _validate_index(index, index_key)
        if index_rows[split] != source.selection.retained_counts[split]:
            raise ValueError(f"{index_key}: sample count differs from selection")
        expected_entries = retained_by_split[split]
        if sorted(expected_entries) != list(range(index_rows[split])):
            raise ValueError(f"{manifest_key}: {split} retained annotations are not contiguous")

        output_root = scratch / "audit-output"
        output_row = 0
        for shard_id, shard in enumerate(index["shards"]):
            zip_info = cast(dict[str, Any], shard["zip_data"])
            basename = str(zip_info["basename"])
            relative = f"{split}/{basename}"
            expected_objects.add(relative)
            key = _join(prefix, relative)
            _check_object_hash(store, key, zip_info)
            local = output_root / relative
            store.download(key, local)
            _verify_download(local, zip_info, key)
            shard_rows = 0
            try:
                with open_shard(output_root, split, shard, scratch / "audit-decompress") as reader:
                    for shard_row, sample in enumerate(reader):
                        entry = expected_entries[output_row]
                        _validate_sample(
                            sample,
                            entry,
                            f"{job.name}: audit {split} row {output_row}, shard {shard_id} row {shard_row}",
                        )
                        if split == "train":
                            train_frames += _scalar_int(sample, "num_frames")
                            for name in POLICY_WORLD_FLOAT_COLUMNS:
                                stats.update(name, np.asarray(sample[name]))
                        output_row += 1
                        shard_rows += 1
            finally:
                local.unlink(missing_ok=True)
            if shard_rows != int(shard["samples"]):
                raise ValueError(f"{key}: reader yielded the wrong number of rows")
        if output_row != index_rows[split]:
            raise ValueError(f"{index_key}: reader row count differs from index")

    allowed_markers = {"_STAGED", "_SUCCESS"}
    unexpected = sorted(set(objects) - expected_objects - allowed_markers)
    missing_objects = sorted(expected_objects - set(objects))
    if unexpected or missing_objects:
        raise ValueError(f"{prefix}: unexpected objects={unexpected}, missing objects={missing_objects}")
    stats_key = _join(prefix, "stats.json")
    stats_payload = _json_object(store.read_bytes(stats_key), stats_key)
    _validate_stats_payload(stats_payload, stats_key)
    if stats_payload["sufficient"] != _canonical_sufficient(stats):
        raise ValueError(f"{stats_key}: sufficient statistics differ from retained train rows")

    return {
        "rows": index_rows,
        "rejections": len(source.selection.rejections),
        "retained": sum(index_rows.values()),
        "train_frames": train_frames,
        "objects": len(expected_objects),
        "bytes": sum(objects[name].size for name in expected_objects),
    }


def _objects_match(store: ObjectStore, first: str, second: str) -> bool:
    a = store.head(first)
    b = store.head(second)
    if a is None or b is None:
        return False
    return (
        a.size == b.size
        and a.etag == b.etag
        and (
            a.metadata_value("sha256") is None
            or b.metadata_value("sha256") is None
            or a.metadata_value("sha256") == b.metadata_value("sha256")
        )
    )


def publish_dataset(
    store: ObjectStore,
    job: CorpusJob,
    source: SourceDataset,
    scratch: Path,
) -> dict[str, object]:
    """Resume matching server-side copies and write the success marker last."""
    staged = _relative_objects(store, job.staging_prefix)
    if "_STAGED" not in staged:
        raise ValueError(f"{job.staging_prefix}: staging audit marker is absent")
    audit_dataset(store, job.staging_prefix, job=job, source=source, scratch=scratch)
    copy_names = sorted(name for name in staged if name not in {"_STAGED", "_SUCCESS"})
    final = _relative_objects(store, job.final_prefix)
    if "_SUCCESS" in final:
        return audit_dataset(store, job.final_prefix, job=job, source=source, scratch=scratch)
    unexpected = sorted(set(final) - set(copy_names))
    if unexpected:
        raise FileExistsError(f"{job.final_prefix}: partial final prefix has unexpected objects {unexpected}")
    for name in sorted(final):
        if not _objects_match(store, _join(job.staging_prefix, name), _join(job.final_prefix, name)):
            raise FileExistsError(f"{job.final_prefix}: existing object differs from staging: {name}")
    for name in copy_names:
        destination = _join(job.final_prefix, name)
        if name not in final:
            store.copy(_join(job.staging_prefix, name), destination)
        if not _objects_match(store, _join(job.staging_prefix, name), destination):
            raise RuntimeError(f"{destination}: server-side copy differs from staging")
    audit = audit_dataset(store, job.final_prefix, job=job, source=source, scratch=scratch)
    success = {
        "schema_version": 1,
        "policy": _policy_dict(),
        "corpus": job.name,
        "source": source.identity,
        "staging_prefix": job.staging_prefix,
        "final_prefix": job.final_prefix,
        "run_id": job.run_id,
        "git_sha": job.git_sha,
        "published_at": dt.datetime.now(dt.UTC).isoformat(),
        "audit": audit,
    }
    store.put_bytes(_join(job.final_prefix, "_SUCCESS"), _json_bytes(success))
    return audit_dataset(store, job.final_prefix, job=job, source=source, scratch=scratch)


def _run_record_key(job: CorpusJob) -> str:
    return f"processed/_runs/{POLICY_ID}/{job.run_id}/{job.name}.json"


def _record(store: ObjectStore, job: CorpusJob, state: str, **values: object) -> None:
    payload = {
        "schema_version": 1,
        "state": state,
        "corpus": job.name,
        "run_id": job.run_id,
        "git_sha": job.git_sha,
        "source_prefix": job.source_prefix,
        "staging_prefix": job.staging_prefix,
        "final_prefix": job.final_prefix,
        "updated_at": dt.datetime.now(dt.UTC).isoformat(),
        **values,
    }
    store.put_bytes(_run_record_key(job), _json_bytes(payload))


def rematerialize_corpus(
    store: ObjectStore,
    job: CorpusJob,
    scratch: Path,
    *,
    staging_only: bool = False,
) -> dict[str, object]:
    """Build, audit, and optionally publish one corpus."""
    scratch.mkdir(parents=True, exist_ok=True)
    _record(store, job, "inspecting")
    try:
        source = inspect_source(store, job)
        if store.head(_join(job.final_prefix, "_SUCCESS")) is not None:
            audit = audit_dataset(store, job.final_prefix, job=job, source=source, scratch=scratch)
            _record(store, job, "validated-existing", audit=audit)
            return {"state": "validated-existing", "corpus": job.name, **audit}

        staged = _relative_objects(store, job.staging_prefix)
        if staged and "_STAGED" not in staged:
            store.delete_prefix(job.staging_prefix + "/")
            staged = {}
        if not staged:
            _record(store, job, "building")
            _build_staging(store, job, source, scratch)
            audit = audit_dataset(store, job.staging_prefix, job=job, source=source, scratch=scratch)
            store.put_bytes(
                _join(job.staging_prefix, "_STAGED"),
                _json_bytes({"schema_version": 1, "run_id": job.run_id, "git_sha": job.git_sha, "audit": audit}),
            )
        else:
            audit = audit_dataset(store, job.staging_prefix, job=job, source=source, scratch=scratch)
        if staging_only:
            _record(store, job, "staged", audit=audit)
            return {"state": "staged", "corpus": job.name, **audit}
        published = publish_dataset(store, job, source, scratch)
        _record(store, job, "published", audit=published)
        return {"state": "published", "corpus": job.name, **published}
    except BaseException as error:
        _record(store, job, "failed", error=repr(error))
        raise


def independent_manifest_counts(store: ObjectStore, job: CorpusJob) -> dict[str, object]:
    """Count v8 rows with a second, direct implementation for the pilot."""
    source = inspect_source(store, job)
    retained = dict.fromkeys(SPLITS, 0)
    rejected = dict.fromkeys(SPLITS, 0)
    for entry in source.selection.entries:
        stats = entry.stats
        annotation = entry.annotation
        assert stats is not None and annotation is not None
        humans = len(entry.players) == 2 and all(player.player_type == "HUMAN" for player in entry.players)
        damage = all(
            player.damage_dealt > POLICY_WORLD_V8.min_damage_dealt_per_player_exclusive for player in stats.players
        )
        stocks = all(
            player.stocks_remaining + len(player.death_percents) == POLICY_WORLD_V8.starting_stocks
            for player in stats.players
        )
        target = retained if humans and damage and stocks else rejected
        target[annotation.split] += 1
    return {
        "retained": retained,
        "rejected": rejected,
        "retained_total": sum(retained.values()),
        "rejected_total": sum(rejected.values()),
    }
