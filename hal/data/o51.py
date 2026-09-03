"""O51 nested, content-unique replay bands.

The O48 gzip index is the immutable inventory.  This module only assigns its
already-deduplicated rows to source-stratified O51 bands; it never revisits the
raw manifests or treats older schema projections as new data.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections import Counter
from collections import defaultdict
from collections.abc import Iterable
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from hal import streams
from hal.streams import StreamSource

NESTED_PROTOCOL: Final[str] = "o51-nested-v1"
O48_PROTOCOL: Final[str] = "048-sha1-dedup-shard-order-v1"
O48_SCHEMA_VERSION: Final[int] = 1
NESTED_SCHEMA_VERSION: Final[int] = 1
NESTING_DOMAIN: Final[bytes] = b"o51-nested-v1\0"

D0: Final[int] = 2**30
TIER_SCALES: Final[tuple[int, ...]] = (1, 2, 4, 8)
OFFICIAL_RAW_REPLAYS: Final[int] = 1_300_640
OFFICIAL_UNIQUE_REPLAYS: Final[int] = 1_300_638
OFFICIAL_CORPUS_TARGETS: Final[int] = 26_582_742_076
OFFICIAL_TIER_REPLAYS: Final[dict[int, int]] = {
    1: 162_598,
    2: 325_176,
    4: 650_331,
    8: 1_300_638,
}
OFFICIAL_TIER_TARGETS: Final[dict[int, int]] = {
    1: 3_321_597_594,
    2: 6_647_731_852,
    4: 13_297_093_392,
    8: 26_582_742_076,
}

DEFAULT_O48_INDEX: Final[Path] = Path("data/processed/048_policy_world_unique_v1.tsv.gz")
DEFAULT_O48_REMOTE: Final[str] = "s3://hal/processed/048-policy-world-unique-v1.tsv.gz"
DEFAULT_BAND_ROOT: Final[Path] = Path("data/processed/o51-nested-v1")
DEFAULT_BAND_REMOTE: Final[str] = "s3://hal/processed/o51-nested-v1"


@dataclass(frozen=True, slots=True)
class InventoryEntry:
    """One content-unique O48 training replay."""

    source: str
    row: int
    shard: str
    replay_id: str
    frames: int
    content_sha1: str

    def __post_init__(self) -> None:
        if self.frames < 1:
            raise ValueError(f"replay {self.replay_id!r} has invalid frame count {self.frames}")
        if len(self.content_sha1) != 40 or any(character not in "0123456789abcdef" for character in self.content_sha1):
            raise ValueError(f"replay {self.replay_id!r} has invalid content SHA-1 {self.content_sha1!r}")

    @property
    def potential_targets(self) -> int:
        """Both ports at every position that has a subsequent frame."""
        return 2 * (self.frames - 1)

    @property
    def nesting_key(self) -> tuple[bytes, str, str]:
        digest = hashlib.blake2b(NESTING_DOMAIN + self.content_sha1.encode("ascii"), digest_size=16).digest()
        return digest, self.content_sha1, self.replay_id


@dataclass(frozen=True, slots=True)
class Inventory:
    header: dict[str, object]
    entries: tuple[InventoryEntry, ...]

    @property
    def corpus_hash(self) -> str:
        return str(self.header["corpus_hash"])


@dataclass(frozen=True, slots=True)
class NestedBand:
    """One disjoint physical band added at a scale boundary."""

    scale: int
    entries: tuple[InventoryEntry, ...]
    sha256: str
    source_replays: dict[str, int]
    source_frames: dict[str, int]

    @property
    def unique_replays(self) -> int:
        return len(self.entries)

    @property
    def potential_targets(self) -> int:
        return sum(entry.potential_targets for entry in self.entries)


@dataclass(frozen=True, slots=True)
class NestedTier:
    """The union of one band and every preceding band."""

    scale: int
    entries: tuple[InventoryEntry, ...]
    sha256: str
    source_replays: dict[str, int]
    source_frames: dict[str, int]

    @property
    def unique_replays(self) -> int:
        return len(self.entries)

    @property
    def potential_targets(self) -> int:
        return sum(entry.potential_targets for entry in self.entries)

    @property
    def target_positions(self) -> int:
        return self.scale * D0

    def updates(self, batch_size: int, *, supervised_positions_per_window: int = 128) -> int:
        positions_per_update = batch_size * supervised_positions_per_window
        quotient, remainder = divmod(self.target_positions, positions_per_update)
        if remainder:
            raise ValueError(f"D={self.target_positions} is not divisible by batch geometry {positions_per_update}")
        return quotient

    @property
    def windows_per_replay(self) -> float:
        return self.target_positions / (128 * self.unique_replays)


@dataclass(frozen=True, slots=True)
class NestedCorpus:
    corpus_hash: str
    bands: dict[int, NestedBand]
    tiers: dict[int, NestedTier]


def _source_names() -> tuple[str, ...]:
    return tuple(source.name for source in streams.POLICY_WORLD_V7_SOURCES)


def read_o48_inventory(path: Path = DEFAULT_O48_INDEX) -> Inventory:
    """Read and authenticate the frozen O48 content index."""
    entries: list[InventoryEntry] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        first = handle.readline()
        if not first.startswith("#"):
            raise ValueError(f"{path}: O48 inventory has no metadata header")
        header = json.loads(first[1:])
        for line_number, line in enumerate(handle, start=2):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 6:
                raise ValueError(f"{path}:{line_number}: expected six tab-separated fields")
            source, row, shard, replay_id, frames, content_sha1 = fields
            entries.append(InventoryEntry(source, int(row), shard, replay_id, int(frames), content_sha1))

    expected_sources = list(_source_names())
    if (
        header.get("schema_version") != O48_SCHEMA_VERSION
        or header.get("protocol") != O48_PROTOCOL
        or header.get("source_names") != expected_sources
    ):
        raise ValueError(f"{path}: inventory is not the frozen all-source O48 corpus")
    if int(header.get("canonical_replays", -1)) != len(entries):
        raise ValueError(f"{path}: header and body replay counts differ")
    if int(header.get("canonical_loss_positions", -1)) != sum(entry.potential_targets for entry in entries):
        raise ValueError(f"{path}: header and body target counts differ")

    digest = hashlib.sha256()
    for entry in entries:
        digest.update(_o48_line(entry))
    if digest.hexdigest() != header.get("corpus_hash"):
        raise ValueError(f"{path}: O48 corpus hash does not match its rows")
    if len({entry.content_sha1 for entry in entries}) != len(entries):
        raise ValueError(f"{path}: O48 inventory is not content-unique")
    if len({entry.replay_id for entry in entries}) != len(entries):
        raise ValueError(f"{path}: O48 inventory repeats a replay ID")
    return Inventory(header=header, entries=tuple(entries))


def _o48_line(entry: InventoryEntry) -> bytes:
    return (
        f"{entry.source}\t{entry.row}\t{entry.shard}\t{entry.replay_id}\t{entry.frames}\t{entry.content_sha1}\n"
    ).encode()


def _nested_line(entry: InventoryEntry) -> bytes:
    return (f"{entry.source}\t{entry.row}\t{entry.replay_id}\t{entry.frames}\t{entry.content_sha1}\n").encode()


def _summary(entries: Iterable[InventoryEntry]) -> tuple[str, dict[str, int], dict[str, int]]:
    digest = hashlib.sha256()
    replay_counts: Counter[str] = Counter()
    frame_counts: Counter[str] = Counter()
    for entry in entries:
        digest.update(_nested_line(entry))
        replay_counts[entry.source] += 1
        frame_counts[entry.source] += entry.frames
    return digest.hexdigest(), dict(replay_counts), dict(frame_counts)


def build_nested_corpus(inventory: Inventory, *, strict_official: bool = True) -> NestedCorpus:
    """Assign each inventory row to one deterministic disjoint band."""
    if len({entry.content_sha1 for entry in inventory.entries}) != len(inventory.entries):
        raise ValueError("O51 inventory is not content-unique")
    if len({entry.replay_id for entry in inventory.entries}) != len(inventory.entries):
        raise ValueError("O51 inventory repeats a replay ID")
    by_source: dict[str, list[InventoryEntry]] = defaultdict(list)
    for entry in inventory.entries:
        by_source[entry.source].append(entry)
    expected_sources = _source_names()
    if set(by_source) != set(expected_sources):
        missing = set(expected_sources) - by_source.keys()
        extra = by_source.keys() - set(expected_sources)
        raise ValueError(f"inventory source coverage differs: missing={sorted(missing)}, extra={sorted(extra)}")
    for entries in by_source.values():
        entries.sort(key=lambda entry: entry.nesting_key)

    band_entries: dict[int, list[InventoryEntry]] = {scale: [] for scale in TIER_SCALES}
    previous: dict[str, int] = {source: 0 for source in expected_sources}
    for scale in TIER_SCALES:
        for source in expected_sources:
            entries = by_source[source]
            stop = math.ceil(scale * len(entries) / 8)
            band_entries[scale].extend(entries[previous[source] : stop])
            previous[source] = stop

    bands: dict[int, NestedBand] = {}
    tiers: dict[int, NestedTier] = {}
    cumulative: list[InventoryEntry] = []
    seen_content: set[str] = set()
    seen_replays: set[str] = set()
    for scale in TIER_SCALES:
        # The key is random-like and stable, so writing in this order also
        # pre-shuffles rows before Mosaic sees them.
        entries = tuple(sorted(band_entries[scale], key=lambda entry: (entry.nesting_key, entry.source)))
        band_hash, source_replays, source_frames = _summary(entries)
        bands[scale] = NestedBand(scale, entries, band_hash, source_replays, source_frames)
        cumulative.extend(entries)
        tier_entries = tuple(cumulative)
        tier_hash, tier_source_replays, tier_source_frames = _summary(tier_entries)
        tiers[scale] = NestedTier(scale, tier_entries, tier_hash, tier_source_replays, tier_source_frames)

        content = {entry.content_sha1 for entry in tier_entries}
        replay_ids = {entry.replay_id for entry in tier_entries}
        if not seen_content < content and scale != TIER_SCALES[0]:
            raise ValueError(f"tier U{scale} is not a strict content superset")
        if not seen_replays < replay_ids and scale != TIER_SCALES[0]:
            raise ValueError(f"tier U{scale} is not a strict replay superset")
        seen_content, seen_replays = content, replay_ids
        if set(tier_source_replays) != set(expected_sources):
            raise ValueError(f"tier U{scale} does not cover all 44 sources")

    full = tiers[8]
    if {entry.content_sha1 for entry in full.entries} != {entry.content_sha1 for entry in inventory.entries}:
        raise ValueError("the full O51 tier does not contain the complete O48 inventory")
    if sum(band.unique_replays for band in bands.values()) != len(inventory.entries):
        raise ValueError("O51 bands are not a disjoint partition")

    corpus = NestedCorpus(inventory.corpus_hash, bands, tiers)
    if strict_official:
        validate_official_corpus(inventory, corpus)
    return corpus


def validate_official_corpus(inventory: Inventory, corpus: NestedCorpus) -> None:
    """Pin the published O51 counts while deriving hashes from the inventory."""
    raw_rows = int(inventory.header.get("raw_train_replays", -1))
    duplicates = int(inventory.header.get("duplicate_content_occurrences_removed", -1))
    ambiguous = int(inventory.header.get("ambiguous_replay_ids_removed", -1))
    if raw_rows != OFFICIAL_RAW_REPLAYS or duplicates != 2 or ambiguous != 0:
        raise ValueError(
            f"O48 inventory drift: rows={raw_rows}, duplicate occurrences={duplicates}, ambiguous IDs={ambiguous}"
        )
    if len(inventory.entries) != OFFICIAL_UNIQUE_REPLAYS:
        raise ValueError(f"O48 unique replay count drift: {len(inventory.entries)}")
    if sum(entry.potential_targets for entry in inventory.entries) != OFFICIAL_CORPUS_TARGETS:
        raise ValueError("O48 potential target count drift")
    for scale in TIER_SCALES:
        tier = corpus.tiers[scale]
        expected = (OFFICIAL_TIER_REPLAYS[scale], OFFICIAL_TIER_TARGETS[scale])
        actual = (tier.unique_replays, tier.potential_targets)
        if actual != expected:
            raise ValueError(f"U{scale} inventory drift: {actual} != {expected}")


def band_manifest_path(root: Path, scale: int) -> Path:
    if scale not in TIER_SCALES:
        raise ValueError(f"band scale must be one of {TIER_SCALES}, got {scale}")
    return root / f"band-{scale}" / "manifest.o51.jsonl.gz"


def write_band_manifests(corpus: NestedCorpus, root: Path = DEFAULT_BAND_ROOT) -> tuple[Path, ...]:
    """Write deterministic selection manifests; existing unequal files fail."""
    paths: list[Path] = []
    for scale in TIER_SCALES:
        band = corpus.bands[scale]
        path = band_manifest_path(root, scale)
        header = {
            "schema_version": NESTED_SCHEMA_VERSION,
            "protocol": NESTED_PROTOCOL,
            "band_scale": scale,
            "corpus_hash": corpus.corpus_hash,
            "band_sha256": band.sha256,
            "unique_replays": band.unique_replays,
            "potential_targets": band.potential_targets,
            "source_replays": band.source_replays,
            "source_frames": band.source_frames,
        }
        lines = [json.dumps(header, sort_keys=True, separators=(",", ":"))]
        lines.extend(
            json.dumps(
                {
                    "source": entry.source,
                    "row": entry.row,
                    "shard": entry.shard,
                    "replay_id": entry.replay_id,
                    "frames": entry.frames,
                    "content_sha1": entry.content_sha1,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            for entry in band.entries
        )
        payload = ("\n".join(lines) + "\n").encode()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if gzip.decompress(path.read_bytes()) != payload:
                raise ValueError(f"existing O51 band manifest differs: {path}")
        else:
            temporary = path.with_suffix(path.suffix + ".tmp")
            with (
                temporary.open("wb") as raw,
                gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
            ):
                compressed.write(payload)
            temporary.replace(path)
        paths.append(path)
    return tuple(paths)


def band_sources(
    tier_scale: int,
    *,
    local_root: Path = DEFAULT_BAND_ROOT,
    remote_root: str = DEFAULT_BAND_REMOTE,
) -> tuple[StreamSource, ...]:
    """Return the physical band streams whose union is one nested tier."""
    if tier_scale not in TIER_SCALES:
        raise ValueError(f"tier scale must be one of {TIER_SCALES}, got {tier_scale}")
    return tuple(
        StreamSource(
            name=f"o51-band-{scale}",
            remote=f"{remote_root.rstrip('/')}/band-{scale}",
            local=local_root / f"band-{scale}",
        )
        for scale in TIER_SCALES
        if scale <= tier_scale
    )


def iter_band_manifest(path: Path) -> Iterator[InventoryEntry]:
    """Read one O51 band manifest and verify its row hash and counts."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        header = json.loads(handle.readline())
        entries = []
        for line in handle:
            row = json.loads(line)
            entries.append(
                InventoryEntry(
                    source=row["source"],
                    row=int(row["row"]),
                    shard=row["shard"],
                    replay_id=row["replay_id"],
                    frames=int(row["frames"]),
                    content_sha1=row["content_sha1"],
                )
            )
    digest, source_replays, source_frames = _summary(entries)
    if (
        header.get("schema_version") != NESTED_SCHEMA_VERSION
        or header.get("protocol") != NESTED_PROTOCOL
        or int(header.get("band_scale", -1)) not in TIER_SCALES
        or int(header.get("unique_replays", -1)) != len(entries)
        or int(header.get("potential_targets", -1)) != sum(entry.potential_targets for entry in entries)
        or header.get("source_replays") != source_replays
        or header.get("source_frames") != source_frames
        or header.get("band_sha256") != digest
    ):
        raise ValueError(f"invalid O51 band manifest {path}")
    if len({entry.content_sha1 for entry in entries}) != len(entries):
        raise ValueError(f"O51 band manifest is not content-unique: {path}")
    if len({entry.replay_id for entry in entries}) != len(entries):
        raise ValueError(f"O51 band manifest repeats a replay ID: {path}")
    yield from entries
