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
    "rapm",
    "redx",
    "siddward",
    "solobattle",
    "technospider",
    "trif",
    "uhhei",
    "ycz",
    "zain",
)


def _professional_policy_world_source(slug: str) -> StreamSource:
    name = f"professional-{slug}-policy-world-v7"
    root = f"processed/professional/{slug}/mds-policy-world-v7"
    return StreamSource(name=name, remote=f"s3://hal/{root}", local=Path("data") / root)


PROFESSIONAL_POLICY_WORLD_V7: Final[dict[str, StreamSource]] = {
    slug: _professional_policy_world_source(slug) for slug in PROFESSIONAL_PLAYER_SLUGS
}

POLICY_WORLD_V7_SOURCES: Final[tuple[StreamSource, ...]] = (
    *RANKED_ANONYMIZED_POLICY_WORLD_V7,
    *PROFESSIONAL_POLICY_WORLD_V7.values(),
)

# Verified against the immutable train splits on R2 on 2026-08-23. The replay
# counts also define the natural multi-stream sampling and normalization mix.
POLICY_WORLD_V7_TRAIN_REPLAYS: Final[dict[str, int]] = {
    **{
        f"ranked-anonymized-{rank}-policy-world-v7": count
        for rank, count in enumerate((112_409, 146_756, 124_689, 143_750, 129_131, 166_559), start=1)
    },
    **{
        f"professional-{slug}-policy-world-v7": count
        for slug, count in {
            "aklo": 18_903,
            "amsa": 23_749,
            "axe": 1_610,
            "billybopeep": 750,
            "bobbybigballz": 2_532,
            "cody": 62_723,
            "cookbook": 20_476,
            "daniel": 8_002,
            "desertsnoopy": 26_946,
            "druggedfox": 439,
            "fknsilver": 5_510,
            "franz": 15_222,
            "frenzy": 19_736,
            "friend": 8_496,
            "ginger": 20_272,
            "gosu": 20_465,
            "grab2win": 5_640,
            "iliketurtles": 14_813,
            "isdsar": 5_570,
            "jchu": 3_237,
            "jahridin": 26_189,
            "kjh": 2_176,
            "kodorin": 8_768,
            "krudo": 9_588,
            "m2k": 8_291,
            "mang0": 30_219,
            "mof": 1_241,
            "monotheon": 16_333,
            "nicki": 2_517,
            "rapm": 605,
            "redx": 1_466,
            "siddward": 16_243,
            "solobattle": 27_063,
            "technospider": 4_447,
            "trif": 14_059,
            "uhhei": 6_893,
            "ycz": 7_390,
            "zain": 8_767,
        }.items()
    },
}

POLICY_WORLD_V7_TRAIN_FRAMES: Final[dict[str, int]] = {
    **{
        f"ranked-anonymized-{rank}-policy-world-v7": count
        for rank, count in enumerate(
            (1_204_903_922, 1_576_992_919, 1_303_498_202, 1_520_552_392, 1_368_355_306, 1_772_910_144),
            start=1,
        )
    },
    **{
        f"professional-{slug}-policy-world-v7": count
        for slug, count in {
            "aklo": 170_323_702,
            "amsa": 199_696_752,
            "axe": 15_860_033,
            "billybopeep": 6_777_005,
            "bobbybigballz": 23_308_107,
            "cody": 546_728_054,
            "cookbook": 203_876_868,
            "daniel": 87_270_667,
            "desertsnoopy": 263_988_278,
            "druggedfox": 3_801_870,
            "fknsilver": 65_322_125,
            "franz": 146_365_722,
            "frenzy": 186_564_727,
            "friend": 95_999_430,
            "ginger": 172_168_951,
            "gosu": 171_964_884,
            "grab2win": 60_801_489,
            "iliketurtles": 136_955_947,
            "isdsar": 48_171_832,
            "jchu": 36_154_253,
            "jahridin": 253_415_539,
            "kjh": 20_929_570,
            "kodorin": 83_103_402,
            "krudo": 92_537_936,
            "m2k": 76_475_554,
            "mang0": 273_932_573,
            "mof": 12_195_613,
            "monotheon": 164_934_541,
            "nicki": 24_198_496,
            "rapm": 6_171_906,
            "redx": 18_634_454,
            "siddward": 157_490_540,
            "solobattle": 254_347_491,
            "technospider": 52_310_073,
            "trif": 147_228_758,
            "uhhei": 83_763_834,
            "ycz": 80_852_018,
            "zain": 100_857_261,
        }.items()
    },
}

_policy_world_names = {source.name for source in POLICY_WORLD_V7_SOURCES}
if set(POLICY_WORLD_V7_TRAIN_REPLAYS) != _policy_world_names:
    raise RuntimeError("policy-world replay counts do not cover the registered source set")
if set(POLICY_WORLD_V7_TRAIN_FRAMES) != _policy_world_names:
    raise RuntimeError("policy-world frame counts do not cover the registered source set")

# v5 and v6 stay registered: the frozen experiments still read them.
ALL: Final[tuple[StreamSource, ...]] = (
    RANKED_ANONYMIZED_1,
    RANKED_ANONYMIZED_1_V6,
    RANKED_ANONYMIZED_1_V7,
    RANKED_ANONYMIZED_1_POLICY_V7,
    *POLICY_WORLD_V7_SOURCES,
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
