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

from botocore.exceptions import ClientError
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
# Reverse map: local cache root (string) -> remote URI. Lets the dataloader turn
# a plain `data_root` into its R2 origin, while purely-local paths (dev MDS,
# overfit scratch) that aren't registered here resolve to None and stay local.
_REMOTE_BY_LOCAL: Final[dict[str, str]] = {str(s.local_root): s.remote for s in ALL}
_NOT_FOUND_CODES: Final[frozenset[str]] = frozenset({"404", "NoSuchKey", "NotFound"})


def remote_for_local(local: str | Path) -> str | None:
    """R2 remote URI backing a local cache root, or None if it's local-only."""
    return _REMOTE_BY_LOCAL.get(str(Path(local) if Path(local).is_absolute() else Path(REPO_DIR) / local))


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
    Training needs it before the first batch — hence this explicit, idempotent
    pull, called from the cloud setup script. Shards still stream lazily.
    """
    bucket, key = _split_uri(src.remote)
    dest = src.local_root / "stats.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    r2.client().download_file(bucket, f"{key}/stats.json", str(dest))
    logger.info(f"[streams] {src.name}: stats.json -> {dest}")
    return dest


def pull_stats_if_available(src: StreamSource) -> Path | None:
    """Download stats when the registered stream exists in remote storage.

    Some registered sources are intentionally retained for frozen experiments
    after their remote artifact has been removed. Missing optional sources must
    not prevent unrelated cloud experiments from starting. Authentication and
    transport failures still propagate.
    """
    try:
        return pull_stats(src)
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code not in _NOT_FOUND_CODES:
            raise
        logger.warning(f"[streams] {src.name}: remote stats.json is unavailable; skipping")
        return None
