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


def _ranked_policy_world_v8_source(rank: int) -> StreamSource:
    name = f"ranked-anonymized-{rank}-policy-world-v8"
    root = f"processed/ranked-anonymized-{rank}/mds-policy-world-v8"
    return StreamSource(name=name, remote=f"s3://hal/{root}", local=Path("data") / root)


RANKED_ANONYMIZED_POLICY_WORLD_V8: Final[tuple[StreamSource, ...]] = tuple(
    _ranked_policy_world_v8_source(rank) for rank in range(1, 7)
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


def _professional_policy_world_v8_source(slug: str) -> StreamSource:
    name = f"professional-{slug}-policy-world-v8"
    root = f"processed/professional/{slug}/mds-policy-world-v8"
    return StreamSource(name=name, remote=f"s3://hal/{root}", local=Path("data") / root)


PROFESSIONAL_POLICY_WORLD_V8: Final[dict[str, StreamSource]] = {
    slug: _professional_policy_world_v8_source(slug) for slug in PROFESSIONAL_PLAYER_SLUGS
}

POLICY_WORLD_V7_SOURCES: Final[tuple[StreamSource, ...]] = (
    *RANKED_ANONYMIZED_POLICY_WORLD_V7,
    *PROFESSIONAL_POLICY_WORLD_V7.values(),
)

POLICY_WORLD_V8_SOURCES: Final[tuple[StreamSource, ...]] = (
    *RANKED_ANONYMIZED_POLICY_WORLD_V8,
    *PROFESSIONAL_POLICY_WORLD_V8.values(),
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

# Verified against the immutable train splits on R2 on 2026-09-06.
POLICY_WORLD_V8_TRAIN_REPLAYS: Final[dict[str, int]] = {
    **{
        f"ranked-anonymized-{rank}-policy-world-v8": count
        for rank, count in enumerate((112_188, 146_455, 124_398, 143_465, 128_830, 166_189), start=1)
    },
    **{
        f"professional-{slug}-policy-world-v8": count
        for slug, count in {
            "aklo": 18_820,
            "amsa": 23_641,
            "axe": 1_604,
            "billybopeep": 749,
            "bobbybigballz": 2_521,
            "cody": 62_189,
            "cookbook": 20_103,
            "daniel": 7_984,
            "desertsnoopy": 26_728,
            "druggedfox": 436,
            "fknsilver": 5_503,
            "franz": 15_159,
            "frenzy": 19_640,
            "friend": 8_433,
            "ginger": 20_062,
            "gosu": 20_314,
            "grab2win": 5_624,
            "iliketurtles": 14_667,
            "isdsar": 5_528,
            "jchu": 3_235,
            "jahridin": 25_987,
            "kjh": 2_172,
            "kodorin": 8_677,
            "krudo": 9_577,
            "m2k": 7_892,
            "mang0": 30_157,
            "mof": 1_225,
            "monotheon": 16_301,
            "nicki": 2_482,
            "rapm": 604,
            "redx": 1_424,
            "siddward": 16_208,
            "solobattle": 26_915,
            "technospider": 4_442,
            "trif": 13_872,
            "uhhei": 6_888,
            "ycz": 7_358,
            "zain": 8_724,
        }.items()
    },
}

POLICY_WORLD_V8_TRAIN_FRAMES: Final[dict[str, int]] = {
    **{
        f"ranked-anonymized-{rank}-policy-world-v8": count
        for rank, count in enumerate(
            (1_203_888_017, 1_575_627_575, 1_302_230_197, 1_519_295_587, 1_367_018_722, 1_771_263_420),
            start=1,
        )
    },
    **{
        f"professional-{slug}-policy-world-v8": count
        for slug, count in {
            "aklo": 169_867_381,
            "amsa": 199_244_133,
            "axe": 15_834_068,
            "billybopeep": 6_773_698,
            "bobbybigballz": 23_245_358,
            "cody": 543_859_859,
            "cookbook": 201_932_010,
            "daniel": 87_158_051,
            "desertsnoopy": 262_588_671,
            "druggedfox": 3_786_376,
            "fknsilver": 65_291_666,
            "franz": 145_977_828,
            "frenzy": 186_046_660,
            "friend": 95_700_935,
            "ginger": 171_176_830,
            "gosu": 171_283_798,
            "grab2win": 60_667_361,
            "iliketurtles": 136_304_585,
            "isdsar": 47_972_508,
            "jchu": 36_145_258,
            "jahridin": 252_442_071,
            "kjh": 20_910_975,
            "kodorin": 82_658_337,
            "krudo": 92_484_831,
            "m2k": 74_365_699,
            "mang0": 273_478_132,
            "mof": 12_084_681,
            "monotheon": 164_767_418,
            "nicki": 24_006_093,
            "rapm": 6_168_038,
            "redx": 18_306_310,
            "siddward": 157_351_234,
            "solobattle": 253_644_935,
            "technospider": 52_286_514,
            "trif": 146_230_291,
            "uhhei": 83_719_769,
            "ycz": 80_696_353,
            "zain": 100_581_942,
        }.items()
    },
}

# SHA-256 of each immutable policy-world-v8 train/index.json, verified on R2
# on 2026-09-06 with the row and frame counts above.
POLICY_WORLD_V8_TRAIN_MANIFEST_SHA256: Final[dict[str, str]] = {
    "ranked-anonymized-1-policy-world-v8": "b97eab90e761bcf2bf03b48981f0ab6acc1ac3057157c58ae0c5a72c76c43bd8",
    "ranked-anonymized-2-policy-world-v8": "f629a8a01eada6904ad16be2ddebe3037c9564e0af812794b75c2a826a8a204c",
    "ranked-anonymized-3-policy-world-v8": "8acb765a597f965bc59bf6c982a2a784f051ac2f5366906f180a04b909943c57",
    "ranked-anonymized-4-policy-world-v8": "b4424eebc9724e4fb94e3b16f21d2f357cd178212915604ec675f782daea17b5",
    "ranked-anonymized-5-policy-world-v8": "94b56ea16d6549564b39342a8c882f214d8b3d747b99a28e291d12bfa4a5c7cc",
    "ranked-anonymized-6-policy-world-v8": "ee58ce5241a510c609e15e9283903163ff941187ccf9a4bff0716b45e157a083",
    "professional-aklo-policy-world-v8": "1ae04b2ffd57fe0bb1bac86933f61b4fbad151bfe7956f607f6ff19521895f64",
    "professional-amsa-policy-world-v8": "11945989cd7a99fb38a0e52fc6306fa2b9e72eb6c7094d656f069b02878ac17f",
    "professional-axe-policy-world-v8": "263f972bb6e629c7e87106d1e2f57d6477ab32d291515dc7fe38f6f316e6e270",
    "professional-billybopeep-policy-world-v8": "7da1a1c4157937ff28c28af4da8ba1103ee81fedb93e4fe820789ca29e2709c7",
    "professional-bobbybigballz-policy-world-v8": "53cae8c1df6a2c13e0e3ab32bf8595be4b9c6438ee54ac0e54bbaa5669f0dcbe",
    "professional-cody-policy-world-v8": "abb6d3e1f790302096270edb6a82c69a646dbd99b544e3ed35507bef293dfb8b",
    "professional-cookbook-policy-world-v8": "e855d5c4e871259092bd2f7772a5f80f1ea7c732d223eeb729e1760dadd1c585",
    "professional-daniel-policy-world-v8": "f2683e42e5a3e2dd2525c5643af7ed9b4536b398aa295a0ffec56b6493144d42",
    "professional-desertsnoopy-policy-world-v8": "23b0af01ab6102edc6be234bfa6055f0c22e7591d5965763524a3ecd4f9b9099",
    "professional-druggedfox-policy-world-v8": "8e4c425e9a62b317df3c11d85d8d172f2be16d2270a8f6279180f2bb79a48687",
    "professional-fknsilver-policy-world-v8": "5ff82418f1a3be4d8c9ea0822b204f88cbea0165c93cb5ba5f19fbadb2692966",
    "professional-franz-policy-world-v8": "5e5720294243254007952fb8f2289b81485fc98ac1df3db304c286286ef80cd7",
    "professional-frenzy-policy-world-v8": "46a9cce7517bc03fde2b3cf70efedade5092bc3841c23711bafc1fac30da18e5",
    "professional-friend-policy-world-v8": "05c6821f857d21c20ae1582cdd7362ad230f8f72c933e44aecf9f0cb351bcd67",
    "professional-ginger-policy-world-v8": "b494bee53cb3c48b2ac673d1aae0ee82926e6cdf8f940d35a20ef6cbfaab1203",
    "professional-gosu-policy-world-v8": "08e743b829bcb7c46abdae6d31386f2eb384f90998ded32967b4287a9bb3043d",
    "professional-grab2win-policy-world-v8": "acafaf1e3406166a35f42d5fafb269009218ec6f43237834338236aed875d1e6",
    "professional-iliketurtles-policy-world-v8": "7690dc9b46de6b2ff7a8f0e99f3f01a21f91388c8a0ab2d4f593a080bf6ae70e",
    "professional-isdsar-policy-world-v8": "6ce04e9378f414f6cfc5dc27bd2ff3e5a58db798aa38abc37a659408f0da4ecf",
    "professional-jchu-policy-world-v8": "403f4f38ab482976c61d76748ff0400f1d502e51ab022347ace87feb9fa9fc35",
    "professional-jahridin-policy-world-v8": "899064901131096198ed9855bd606ff887a20fdb996bfeffacbb72b4a5802a38",
    "professional-kjh-policy-world-v8": "32deb8672642e653f89de565e58c3737abf4b7d03110286cb05d8d355ddb2b4b",
    "professional-kodorin-policy-world-v8": "24d36b73c2f5d06b4fffc1aa014f369cf93c99cdc0097ff1f98c87649b18a41b",
    "professional-krudo-policy-world-v8": "a595bec356530f73a898d6fd0c7fa98eab3efecc05072f0536c87ecc29265f2c",
    "professional-m2k-policy-world-v8": "da529b454a8a84860c328572f9c2a31694e5983ed46c5ab6579c9ac2c674fbca",
    "professional-mang0-policy-world-v8": "cb1a23eebefc78e98dcf36f3bf6d32e66e1f1c4b00ee908e78e5988536445340",
    "professional-mof-policy-world-v8": "eeebeb685c3be5e77cedb8446ea8cc358f5d16286c8d7fdce469b2a9b51bc174",
    "professional-monotheon-policy-world-v8": "8d1e2f996b6aae3fd17f971f8310b1b4beb5b2d97d7faed402dab42714da4492",
    "professional-nicki-policy-world-v8": "ac07fe70c0e0a2ea31a3f3a764b53d1216248d3689d5b4558781bf843b5c0e6c",
    "professional-rapm-policy-world-v8": "58bf73d0868f5e51ef06c1f9f0eb64c5f8153cc77f4053d60c7568d5f20852d1",
    "professional-redx-policy-world-v8": "d54d66e26e97327452da4762eeddf8e2700ec9ef88cff9960a039eda97e766f5",
    "professional-siddward-policy-world-v8": "69b182eda72df5700505cf52c9f7302fcf9acb4771f42bf3f4fb6d2a0538b895",
    "professional-solobattle-policy-world-v8": "eb5084e6340490b9f312f9b0a8ca194d5755a8e83c63043d65668fc482c94bb7",
    "professional-technospider-policy-world-v8": "1558eadd94b43150e6ab0bd2edb15c036a2136b5aa2766a675df83ee8474cb52",
    "professional-trif-policy-world-v8": "16f2905d2894807f2f950063bb8387ce69b5a918523f42556b997385f3c5f462",
    "professional-uhhei-policy-world-v8": "f5414ac89182f6a2af547149cac3d52ce38cfad5549db45e6e1e9979fa6b6101",
    "professional-ycz-policy-world-v8": "597c813dd86e341848d0436d3dac80316fc76ee9d29683a699c114df832c5897",
    "professional-zain-policy-world-v8": "6f35dfbb1f5353138b73866a99e3599034f93d859f70cfb421adc484cedf6549",
}

_policy_world_v7_names = {source.name for source in POLICY_WORLD_V7_SOURCES}
if set(POLICY_WORLD_V7_TRAIN_REPLAYS) != _policy_world_v7_names:
    raise RuntimeError("policy-world v7 replay counts do not cover the registered source set")
if set(POLICY_WORLD_V7_TRAIN_FRAMES) != _policy_world_v7_names:
    raise RuntimeError("policy-world v7 frame counts do not cover the registered source set")

_policy_world_v8_names = {source.name for source in POLICY_WORLD_V8_SOURCES}
if set(POLICY_WORLD_V8_TRAIN_REPLAYS) != _policy_world_v8_names:
    raise RuntimeError("policy-world v8 replay counts do not cover the registered source set")
if set(POLICY_WORLD_V8_TRAIN_FRAMES) != _policy_world_v8_names:
    raise RuntimeError("policy-world v8 frame counts do not cover the registered source set")
if set(POLICY_WORLD_V8_TRAIN_MANIFEST_SHA256) != _policy_world_v8_names:
    raise RuntimeError("policy-world v8 manifest hashes do not cover the registered source set")
if any(
    len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest)
    for digest in POLICY_WORLD_V8_TRAIN_MANIFEST_SHA256.values()
):
    raise RuntimeError("policy-world v8 manifest hashes contain an invalid SHA-256 digest")

# v5 and v6 stay registered: the frozen experiments still read them.
ALL: Final[tuple[StreamSource, ...]] = (
    RANKED_ANONYMIZED_1,
    RANKED_ANONYMIZED_1_V6,
    RANKED_ANONYMIZED_1_V7,
    RANKED_ANONYMIZED_1_POLICY_V7,
    *POLICY_WORLD_V7_SOURCES,
    *POLICY_WORLD_V8_SOURCES,
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
