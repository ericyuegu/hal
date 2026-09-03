"""Direct O51 source selection and runtime replay labels."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np

from hal import streams
from hal.training import returns as returns_lib
from hal.training.player_identity import ReplayPlayerLookup

DATA_PROTOCOL: Final[str] = "o51-direct-source-prefix-v1"
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
    1: 3_353_805_100,
    2: 6_686_081_812,
    4: 13_291_247_716,
    8: 26_582_742_076,
}

CORPUS_SHA256: Final[str] = "e6a83dfb8c98d2cf1ddfacce41c1c2a3e3a8db0a50496f6ecce1c686f5a6ef95"
TIER_SHA256: Final[dict[int, str]] = {
    1: "0df4c7f80d7bcc4ae135a8ec0151a4e50e8cae0a76107dcab4637a9a79b801ce",
    2: "3d57c4836ecd50c9dbaea87434527baef847870ea63286600aa5816770614bb6",
    4: "ce671cc83ca4fe7fd251bf27993c91451ff5812162123528482dbe206629f9d7",
    8: "b8ead0b579b725fa67d6b0a5176251c0057b947886350b61d591cf7b82a0a15c",
}
EXCLUDED_SOURCE_ROWS: Final[dict[str, tuple[int, ...]]] = {
    "professional-monotheon-policy-world-v7": (14_160, 14_163),
}
SOURCE_MANIFEST_SHA256: Final[dict[str, str]] = {
    "professional-aklo-policy-world-v7": "c8635058e791874e76371e3b473fcd0552e6bf69a9f3d30fa35f7967de5a3b79",
    "professional-amsa-policy-world-v7": "c7b213e6ada0bbaec4899b142bfc6deb52716162f07ef3e4dd92fd07165839db",
    "professional-axe-policy-world-v7": "b9568dbb6c4bf931e7c7786ba49bf56614b07c7cb9530857529c0b8fb78df212",
    "professional-billybopeep-policy-world-v7": ("d07fe88191500a77c70e540163a3d645cdd71331a0cbfd4d4b5f9660f4681b52"),
    "professional-bobbybigballz-policy-world-v7": ("670a8920ea533f83d3e8e3bae9fe6a935161657570a598d41752e79f1b802e4c"),
    "professional-cody-policy-world-v7": "d931e14d275451f61a4bcf11af30fb435ff7f4a7b02e8da32778cfc6ab972311",
    "professional-cookbook-policy-world-v7": "6380ebefced8a37154f85780a0eb4ec975c1bd32f70d3bfacb0759037981b1fc",
    "professional-daniel-policy-world-v7": "08496d5eaa8819065303ee9e12e7e29f5192a0c81115727db94e8e9f3c4138f0",
    "professional-desertsnoopy-policy-world-v7": ("0694ee88c6d63993be361442159e9b4a814551bbc1339bfaa85fe15a2d5ca0ef"),
    "professional-druggedfox-policy-world-v7": ("5f382cb09714eb3cb369bc02991ac36db3665efa79483dd92c8c031c8ac8346b"),
    "professional-fknsilver-policy-world-v7": ("92af994560b698ae4c3e5a7dcccdbe6e0897a946f0bb6b21893542875573449b"),
    "professional-franz-policy-world-v7": "0e2972e21fb0dd536022ddd07aaf57a2cd5c8ece412c000532fac606ade02705",
    "professional-frenzy-policy-world-v7": "a78ce331906741bee9841bea20821e1da49dc557ffe4545dcfe67450c8ef435a",
    "professional-friend-policy-world-v7": "012e9f2878f50813984b43dd612cd4fe7a9d8a7e1ad3f9df4ecb0ac457ac6443",
    "professional-ginger-policy-world-v7": "5fce487fb31b7c954f396d7b0bc47f2254d464ada683a92ad7abc88b07d41c29",
    "professional-gosu-policy-world-v7": "410f7456656a72158be5326d247e69a731a6e98a85819ca97a794bc1f9711e4d",
    "professional-grab2win-policy-world-v7": ("3a386a12110b407ec0158a62c941c5a10bfb92a7f0d5b67cc5e95cc9f25f07d3"),
    "professional-iliketurtles-policy-world-v7": ("ba241befd7cafc216d60cdf4b9c357bf5931c51b4ea17213927961cd4f12b3f9"),
    "professional-isdsar-policy-world-v7": "9f4b6f543e9a0a3bcdab5d4304d8bbfc6b31992fc20f2fd6f9478c01920352ca",
    "professional-jahridin-policy-world-v7": "428fa4e7051925fecf8fb0545f5cbb84f10524f965fed92403f1c19ae1c170e9",
    "professional-jchu-policy-world-v7": "9096fcd2424135280c9f0163ae9c04c2fdffa4dbc71f4ca9fdb9472f65a1f4ab",
    "professional-kjh-policy-world-v7": "09d78720633ca3809ccbeb92be2a5f377cc173584435cc7e054dfc5c4b1d6d03",
    "professional-kodorin-policy-world-v7": "8b0d9809da17d1be49f4c8eb5e0a82061b992f84dfbf33636bbeebe94118dae0",
    "professional-krudo-policy-world-v7": "e3b572be20e6c6ee6d249c669a2559b69d22ce24fae653001deb04cc16fc7f54",
    "professional-m2k-policy-world-v7": "b372dd7ba6a46ee01b977f9cf992d18a08e9213354796b45a1449b7f1ff2a5aa",
    "professional-mang0-policy-world-v7": "f78a84bdf70885fb691a202933b936b1921bbf74b8daaea92cffb4d21250ac21",
    "professional-mof-policy-world-v7": "1ee81b47cc8dddf1f7ebae5f9139f00a406188bc7b7c75638a9fcd1e9476e030",
    "professional-monotheon-policy-world-v7": ("9ff5057f8826ef78e1d31f3b687cd6cadc7d66d1656ba2cd98c8e396dd68c38a"),
    "professional-nicki-policy-world-v7": "738157bed96c6843b5b9825b4abdb3000a4faa82c210ead17121b61120250034",
    "professional-rapm-policy-world-v7": "31e0ee64a551c5dffb23ad8dca7a18ff713fe011fb7f55d5b8b6e347479048cc",
    "professional-redx-policy-world-v7": "4bf776186d9470b0280b951ff9a7eab804a4354c277ef855e16518ff3aba0cc2",
    "professional-siddward-policy-world-v7": ("c4c483c0165b379089ee1f20084a2c46107d9faf0a5e7c58c50f53614c94bf8b"),
    "professional-solobattle-policy-world-v7": ("4c29eab3e77c7fdd56f8a816733af2f837c82b560f8f77d2291ae50e20bd993e"),
    "professional-technospider-policy-world-v7": ("624f3e27df17c043b7368ca56a758742dbf827f4777244c6f6c7c972812222e2"),
    "professional-trif-policy-world-v7": "55c0a5154159ebded5eaece1efd3a9b01f1d8cf63c68b008c22f8d84ac58b37c",
    "professional-uhhei-policy-world-v7": "2fcbf81b2afc3bc7f9e21a22fa8ec93f809903fb594b9ddbc8666fcc4be65e64",
    "professional-ycz-policy-world-v7": "7e97cb1c4dd6b75afd0d65c69f0dfb05b582e5349e391b9ff53f75c48700554a",
    "professional-zain-policy-world-v7": "0a3e39757fd68d4aabd37831f63f5b8e0e9aa9dafa580fa7ee5494b445c9ee12",
    "ranked-anonymized-1-policy-world-v7": "a563c62603b8cfdef219cd133324cb090f7e6488fcfc4b6941d03e863a255d16",
    "ranked-anonymized-2-policy-world-v7": "f2bb6f33d53bdcffb0038dca83e926c8102d9f6b82cf0f3a346064e93ec73d29",
    "ranked-anonymized-3-policy-world-v7": "8e68c15dfbe898e44c7104c355ae86cbac61ba3bbf28c711266405fd5a633694",
    "ranked-anonymized-4-policy-world-v7": "946d76f03282e5b23ef4cee4552c6dabd2668423408ae1f7510ee9a0671e31f8",
    "ranked-anonymized-5-policy-world-v7": "ebbb72a7a4e26221ee20a187b0ac60684657b70c7188cab26c81ec4be29ed8e3",
    "ranked-anonymized-6-policy-world-v7": "471a945aab225483735db9dbef05cfd26e215178630115e4753a3312116bb019",
}
O51_RETURN_SUFFIX: Final[str] = "awr_return"


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


@dataclass(frozen=True, slots=True)
class SourceSlice:
    """The selected rows from one existing source MDS."""

    source: str
    stop: int
    excluded_rows: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("source name must not be empty")
        if self.stop < 1:
            raise ValueError(f"source slice stop must be positive for {self.source}")
        if self.excluded_rows != tuple(sorted(set(self.excluded_rows))):
            raise ValueError(f"excluded rows are not sorted and unique for {self.source}")
        if any(not 0 <= row < self.stop for row in self.excluded_rows):
            raise ValueError(f"excluded row is outside {self.source}[0:{self.stop}]")

    @property
    def unique_replays(self) -> int:
        return self.stop - len(self.excluded_rows)


@dataclass(frozen=True, slots=True)
class TierSelection:
    """The direct source slices for one O51 data tier."""

    scale: int
    sources: tuple[SourceSlice, ...]
    potential_targets: int
    sha256: str

    def __post_init__(self) -> None:
        if self.scale not in TIER_SCALES:
            raise ValueError(f"invalid tier scale {self.scale}")
        if not self.sources or len({source.source for source in self.sources}) != len(self.sources):
            raise ValueError(f"U{self.scale} source slices are empty or repeated")
        if self.potential_targets < 1 or self.potential_targets % 2:
            raise ValueError(f"U{self.scale} potential-target count is invalid")
        if not _is_sha256(self.sha256):
            raise ValueError(f"U{self.scale} selection hash is invalid")

    @property
    def unique_replays(self) -> int:
        return sum(source.unique_replays for source in self.sources)

    @property
    def frames(self) -> int:
        return self.potential_targets // 2 + self.unique_replays

    def source_replay_counts(self) -> dict[str, int]:
        return {source.source: source.unique_replays for source in self.sources}


@dataclass(frozen=True, slots=True)
class CorpusSelection:
    """All pinned O51 views of the existing policy-world-v7 sources."""

    corpus_hash: str
    source_manifest_sha256: Mapping[str, str]
    tiers: Mapping[int, TierSelection]

    def tier(self, scale: int) -> TierSelection:
        try:
            return self.tiers[scale]
        except KeyError as error:
            raise ValueError(f"tier scale must be one of {TIER_SCALES}, got {scale}") from error


def _prefix_stop(unique_replays: int, excluded_rows: tuple[int, ...]) -> int:
    """Convert a unique-row count to an exclusive raw-row cutoff."""
    stop = unique_replays
    for row in excluded_rows:
        if row < stop:
            stop += 1
    return stop


def corpus_selection() -> CorpusSelection:
    """Return the four pinned tiers without reading or copying replay data."""
    source_names = tuple(source.name for source in streams.POLICY_WORLD_V7_SOURCES)
    raw_counts = streams.POLICY_WORLD_V7_TRAIN_REPLAYS
    if set(raw_counts) != set(source_names):
        raise RuntimeError("policy-world-v7 replay counts do not cover all 44 sources")
    if sum(raw_counts.values()) != OFFICIAL_RAW_REPLAYS:
        raise RuntimeError("policy-world-v7 replay count changed")
    if set(SOURCE_MANIFEST_SHA256) != set(source_names):
        raise RuntimeError("pinned source manifests do not cover all 44 sources")
    if any(not _is_sha256(digest) for digest in SOURCE_MANIFEST_SHA256.values()):
        raise RuntimeError("a pinned source manifest hash is invalid")
    if sum(map(len, EXCLUDED_SOURCE_ROWS.values())) != OFFICIAL_RAW_REPLAYS - OFFICIAL_UNIQUE_REPLAYS:
        raise RuntimeError("the duplicate-row sidecar must exclude exactly two rows")

    tiers: dict[int, TierSelection] = {}
    for scale in TIER_SCALES:
        slices: list[SourceSlice] = []
        for source in source_names:
            excluded = EXCLUDED_SOURCE_ROWS.get(source, ())
            unique_source_replays = raw_counts[source] - len(excluded)
            selected_replays = math.ceil(scale * unique_source_replays / 8)
            stop = _prefix_stop(selected_replays, excluded)
            selected_exclusions = tuple(row for row in excluded if row < stop)
            source_slice = SourceSlice(source, stop, selected_exclusions)
            if source_slice.unique_replays != selected_replays:
                raise RuntimeError(f"U{scale} slice accounting failed for {source}")
            slices.append(source_slice)
        tier = TierSelection(
            scale=scale,
            sources=tuple(slices),
            potential_targets=OFFICIAL_TIER_TARGETS[scale],
            sha256=TIER_SHA256[scale],
        )
        if tier.unique_replays != OFFICIAL_TIER_REPLAYS[scale]:
            raise RuntimeError(f"U{scale} replay count changed")
        tiers[scale] = tier

    full_exclusions = {source.source: source.excluded_rows for source in tiers[8].sources if source.excluded_rows}
    if full_exclusions != EXCLUDED_SOURCE_ROWS:
        raise RuntimeError("the full direct view does not contain the duplicate-row sidecar")
    return CorpusSelection(
        corpus_hash=CORPUS_SHA256,
        source_manifest_sha256={name: SOURCE_MANIFEST_SHA256[name] for name in source_names},
        tiers=tiers,
    )


@dataclass(frozen=True, slots=True)
class DirectO51ReplayLabels:
    """Compute O51 returns and player IDs after reading an existing MDS row."""

    player_lookup: ReplayPlayerLookup
    gamma: float
    damage_shaping: float
    win_reward: float
    stock_value: float

    def __call__(self, compact: Mapping[str, object]) -> dict[str, np.ndarray]:
        labels = returns_lib.compact_policy_returns(
            compact,
            gamma=self.gamma,
            damage_shaping=self.damage_shaping,
            win_reward=self.win_reward,
            stock_value=self.stock_value,
            suffix=O51_RETURN_SUFFIX,
        )
        labels.update(self.player_lookup(compact))
        return labels
