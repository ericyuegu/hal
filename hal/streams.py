"""Cloud-streamed training datasets.

Where `hal/fixtures.py` mirrors small dev artifacts to disk and verifies
sha256, `streams.py` names training-scale MDS datasets that are too big
to fully materialize. The MosaicML `streaming` library handles
download-on-demand: shards are pulled into `local` as the dataloader
reads them, and the cache can be evicted under pressure.

Usage:

    from streaming import StreamingDataset
    from hal.streams import RANKED_ANONYMIZED_1

    remote, local = RANKED_ANONYMIZED_1.for_split("train")
    ds = StreamingDataset(remote=remote, local=str(local), batch_size=...)

Credentials come from the same env vars as `hal/fixtures.py`:
`AWS_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`. boto3 —
and therefore streaming — pick them up automatically; `s3://hal/...` URIs
resolve against R2's endpoint with no further configuration.

Cache layout mirrors the R2 prefix: `<repo>/data/<remote-key-path>/<split>/`,
already gitignored via `/data/`. Treat the cache as streaming-managed.
To pre-warm before going offline, iterate the dataset once end-to-end.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from loguru import logger

from hal import r2
from hal.paths import REPO_DIR


@dataclass(frozen=True, slots=True)
class StreamSource:
    """One MDS dataset with `{train, val, test}/` splits served from R2.

    `remote` is the s3:// URI of the MDS root; `local` is its cache mirror
    relative to repo root. `for_split(name)` returns the (remote, local)
    pair ready to drop into `StreamingDataset`.
    """

    name: str
    remote: str
    local: Path

    def for_split(self, split: str) -> tuple[str, Path]:
        return f"{self.remote}/{split}", Path(REPO_DIR) / self.local / split

    @property
    def local_root(self) -> Path:
        return Path(REPO_DIR) / self.local


RANKED_ANONYMIZED_1: Final[StreamSource] = StreamSource(
    name="ranked-anonymized-1",
    remote="s3://hal/processed/ranked-anonymized-1/mds",
    local=Path("data/processed/ranked-anonymized-1/mds"),
)

RANKED_ANONYMIZED_1_V6: Final[StreamSource] = StreamSource(
    name="ranked-anonymized-1-v6",
    remote="s3://hal/processed/ranked-anonymized-1/mds-v6",
    local=Path("data/processed/ranked-anonymized-1/mds-v6"),
)

RANKED_ANONYMIZED_1_V7: Final[StreamSource] = StreamSource(
    name="ranked-anonymized-1-v7",
    remote="s3://hal/processed/ranked-anonymized-1/mds-v7",
    local=Path("data/processed/ranked-anonymized-1/mds-v7"),
)

RANKED_ANONYMIZED_1_POLICY_V7: Final[StreamSource] = StreamSource(
    name="ranked-anonymized-1-policy-v7",
    remote="s3://hal/processed/ranked-anonymized-1/mds-policy-v7",
    local=Path("data/processed/ranked-anonymized-1/mds-policy-v7"),
)


def _ranked_policy_world_source(rank: int) -> StreamSource:
    name = f"ranked-anonymized-{rank}-policy-world-v7"
    root = f"processed/ranked-anonymized-{rank}/mds-policy-world-v7"
    return StreamSource(name=name, remote=f"s3://hal/{root}", local=Path("data") / root)


RANKED_ANONYMIZED_POLICY_WORLD_V7: Final[tuple[StreamSource, ...]] = tuple(
    _ranked_policy_world_source(rank) for rank in range(1, 7)
)

PROFESSIONAL_PLAYER_SLUGS: Final[tuple[str, ...]] = (
    "aklo",
    "amsa",
    "axe",
    "billybopeep",
    "bobbybigballz",
    "cody",
    "cookbook",
    "daniel",
    "desertsnoopy",
    "druggedfox",
    "fknsilver",
    "franz",
    "frenzy",
    "friend",
    "ginger",
    "gosu",
    "grab2win",
    "iliketurtles",
    "isdsar",
    "jchu",
    "jahridin",
    "kjh",
    "kodorin",
    "krudo",
    "m2k",
    "mang0",
    "mof",
    "monotheon",
    "nicki",
)


def _professional_policy_world_source(slug: str) -> StreamSource:
    name = f"professional-{slug}-policy-world-v7"
    root = f"processed/professional/{slug}/mds-policy-world-v7"
    return StreamSource(name=name, remote=f"s3://hal/{root}", local=Path("data") / root)


PROFESSIONAL_POLICY_WORLD_V7: Final[dict[str, StreamSource]] = {
    slug: _professional_policy_world_source(slug) for slug in PROFESSIONAL_PLAYER_SLUGS
}

# v5 and v6 stay registered: the frozen experiments still read them.
ALL: Final[tuple[StreamSource, ...]] = (
    RANKED_ANONYMIZED_1,
    RANKED_ANONYMIZED_1_V6,
    RANKED_ANONYMIZED_1_V7,
    RANKED_ANONYMIZED_1_POLICY_V7,
    *RANKED_ANONYMIZED_POLICY_WORLD_V7,
    *PROFESSIONAL_POLICY_WORLD_V7.values(),
)
BY_NAME: Final[dict[str, StreamSource]] = {s.name: s for s in ALL}
# Reverse map from a cache root to its registered source. This lets training
# resolve a plain data_root while leaving local dev and scratch paths alone.
_SOURCE_BY_LOCAL: Final[dict[Path, StreamSource]] = {s.local: s for s in ALL}


def _source_for_local(local: str | Path) -> StreamSource | None:
    """Registered stream whose cache root is ``local``, if any."""
    path = Path(local)
    if path.is_absolute():
        try:
            path = path.relative_to(REPO_DIR)
        except ValueError:
            return None
    return _SOURCE_BY_LOCAL.get(path)


def remote_for_local(local: str | Path) -> str | None:
    """R2 remote URI backing a local cache root, or None if it's local-only."""
    src = _source_for_local(local)
    return src.remote if src is not None else None


def _split_uri(remote: str) -> tuple[str, str]:
    """`s3://bucket/key/path` -> ('bucket', 'key/path')."""
    if not remote.startswith("s3://"):
        raise ValueError(f"expected an s3:// URI, got {remote!r}")
    bucket, _, key = remote[len("s3://") :].partition("/")
    return bucket, key


def pull_stats(src: StreamSource) -> Path:
    """Download the dataset's root ``stats.json`` into the local cache.

    StreamingDataset pulls per-split shards on demand, but ``stats.json`` sits at
    the MDS *root* (outside any split), so the streaming layer never fetches it.
    Training needs it before the first batch. ``ensure_stats`` performs this
    pull lazily for the selected data root. Shards still stream lazily.
    """
    bucket, key = _split_uri(src.remote)
    dest = src.local_root / "stats.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    r2.client().download_file(bucket, f"{key}/stats.json", str(dest))
    logger.info(f"[streams] {src.name}: stats.json -> {dest}")
    return dest


def ensure_stats(path: str | Path) -> Path:
    """Fetch a missing stats file when its parent is a registered stream root."""
    stats_path = Path(path)
    if stats_path.is_file() or stats_path.name != "stats.json":
        return stats_path
    src = _source_for_local(stats_path.parent)
    return pull_stats(src) if src is not None else stats_path
