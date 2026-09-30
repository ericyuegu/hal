"""Generate the netplay imitation roster from the professional identity sidecar and its manifests."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import tyro

from hal import streams
from hal.data.player_identity import load_player_identity_sidecar

# A code is a professional's alternate account when it covers at least this share of their train replays,
# never shares a replay with their main code, and no single opponent fills half of its replays.
ALT_MIN_REPLAY_SHARE: Final[float] = 0.03
ALT_MAX_PARTNER_SHARE: Final[float] = 0.5

# Display tags for the professionals whose replays the sidecar was built from.
PROFESSIONAL_TAGS: Final[dict[str, str]] = {
    "aklo": "Aklo",
    "amsa": "aMSa",
    "axe": "Axe",
    "billybopeep": "BillyBoPeep",
    "bobbybigballz": "BobbyBigBallz",
    "cody": "iBDW",
    "cookbook": "CookBook",
    "daniel": "Daniel",
    "desertsnoopy": "DesertSnoopy",
    "druggedfox": "DruggedFox",
    "fknsilver": "FknSilver",
    "franz": "Franz",
    "frenzy": "Frenzy",
    "friend": "Friend",
    "ginger": "Ginger",
    "gosu": "Gosu",
    "grab2win": "Grab2Win",
    "iliketurtles": "ILikeTurtles",
    "isdsar": "Isdsar",
    "jchu": "JChu",
    "jahridin": "JahRidin",
    "kjh": "KJH",
    "kodorin": "KoDoRiN",
    "krudo": "Krudo",
    "m2k": "M2K",
    "mang0": "Mang0",
    "mof": "MOF",
    "monotheon": "Monotheon",
    "nicki": "Nicki",
    "rapm": "RapM",
    "redx": "RedX",
    "siddward": "Siddward",
    "solobattle": "Solobattle",
    "technospider": "TechnoSpider",
    "trif": "Trif",
    "uhhei": "Uhhei",
    "ycz": "YCZ",
    "zain": "Zain",
}


@dataclass(frozen=True, slots=True)
class Replay:
    """One train replay's occupied connect codes, with each code's display name (empty when absent)."""

    frames: int
    codes: tuple[str, ...]
    names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RosterEntry:
    code: str
    label: str
    train_frames: int


def read_train_replays(path: Path) -> tuple[tuple[Replay, ...], str]:
    """Return the train replays and the manifest SHA-256 that the sidecar header records."""
    replays: list[Replay] = []
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            if not line.strip():
                continue
            raw = json.loads(line)
            annotation = raw.get("annotation")
            if annotation is None or annotation["split"] != "train":
                continue
            players = [
                ((player.get("code") or "").strip(), (player.get("name") or "").strip())
                for player in raw["players"]
                if (player.get("code") or "").strip()
            ]
            replays.append(
                Replay(
                    frames=int(annotation["frame_count_actual"]),
                    codes=tuple(code for code, _ in players),
                    names=tuple(name for _, name in players),
                )
            )
    return tuple(replays), digest.hexdigest()


def main_and_alternate_codes(replays: tuple[Replay, ...]) -> tuple[str, frozenset[str]]:
    """Find a professional's main code (present in the most replays) and their alternate accounts."""
    appearances = Counter(code for replay in replays for code in set(replay.codes))
    main = appearances.most_common(1)[0][0]
    alternates: set[str] = set()
    for code, count in appearances.items():
        if code == main or count < ALT_MIN_REPLAY_SHARE * len(replays):
            continue
        together = [replay for replay in replays if code in replay.codes]
        if any(main in replay.codes for replay in together):
            continue
        partners = Counter(other for replay in together for other in replay.codes if other != code)
        if partners and partners.most_common(1)[0][1] >= ALT_MAX_PARTNER_SHARE * count:
            continue
        alternates.add(code)
    return main, frozenset(alternates)


def _label(code: str, tags: Mapping[str, str], names: Mapping[str, Counter[str]]) -> str:
    """A professional's tag, else the code's most common display name, else the code itself."""
    if code in tags:
        return tags[code]
    if code in names:
        return names[code].most_common(1)[0][0]
    return code


def build_roster(
    replays_by_slug: Mapping[str, tuple[Replay, ...]], vocabulary_codes: frozenset[str], size: int
) -> tuple[RosterEntry, ...]:
    frames: Counter[str] = Counter()
    names: dict[str, Counter[str]] = {}
    tags: dict[str, str] = {}
    alternates: set[str] = set()
    for slug, replays in replays_by_slug.items():
        main, slug_alternates = main_and_alternate_codes(replays)
        tags[main] = PROFESSIONAL_TAGS[slug]
        alternates |= slug_alternates
        for replay in replays:
            for code in replay.codes:
                frames[code] += replay.frames
            for code, name in zip(replay.codes, replay.names, strict=True):
                if name:
                    names.setdefault(code, Counter())[name] += 1
    # A professional's alternate account can be another professional's opponent; their main code wins.
    alternates -= tags.keys()
    ranked = [code for code, _ in frames.most_common() if code not in alternates][:size]
    missing = sorted(code for code in ranked if code not in vocabulary_codes)
    if missing:
        raise ValueError(f"roster codes are absent from the sidecar vocabulary: {missing}")
    entries = tuple(RosterEntry(code, _label(code, tags, names), frames[code]) for code in ranked)
    labels = Counter(entry.label.casefold() for entry in entries)
    duplicated = sorted(label for label, count in labels.items() if count > 1)
    if duplicated:
        raise ValueError(f"roster labels repeat: {duplicated}")
    return entries


def render_module(entries: tuple[RosterEntry, ...], sidecar_sha256: str, vocabulary_sha256: str) -> str:
    rows = "\n".join(
        f"    ({json.dumps(entry.code)}, {json.dumps(entry.label, ensure_ascii=False)}, {entry.train_frames}),"
        for entry in entries
    )
    return f'''"""Netplay imitation roster. Generated by hal/scripts/build_netplay_roster.py; do not edit by hand.

The {len(entries)} connect codes with the most train frames in the professional manifests of the player
identity sidecar, excluding the professionals' alternate accounts. Each row is (code, label, train frames).
"""

from typing import Final

PLAYER_IDENTITY_SIDECAR_SHA256: Final[str] = "{sidecar_sha256}"
PLAYER_VOCABULARY_SHA256: Final[str] = "{vocabulary_sha256}"

PROFESSIONAL_ROSTER: Final[tuple[tuple[str, str, int], ...]] = (
{rows}
)
'''


def build(
    sidecar: Path = Path("data/processed/player-identity-v1/professional-code-v1.jsonl.gz"),
    sidecar_sha256: str = "54ccf8a2497fe240313117297ca2ea31158e08db2cc53c67e7aa46853a8dac1c",
    professional_root: Path = Path("data/processed/professional"),
    size: int = 50,
    output: Path = Path("hal/netplay_service/roster.py"),
) -> None:
    """Read ``<root>/<slug>/mds-policy-world-v7/manifest.jsonl``, the sidecar's own sources."""
    identity = load_player_identity_sidecar(sidecar, expected_sha256=sidecar_sha256)
    recorded: Mapping[str, str] = identity.header["manifest_sha256"]
    replays_by_slug: dict[str, tuple[Replay, ...]] = {}
    for slug in streams.PROFESSIONAL_PLAYER_SLUGS:
        name = streams.PROFESSIONAL_POLICY_WORLD_V7[slug].name
        replays, digest = read_train_replays(professional_root / slug / "mds-policy-world-v7" / "manifest.jsonl")
        if recorded.get(name) != digest:
            raise ValueError(f"manifest {name} differs from the one the sidecar was built from")
        replays_by_slug[slug] = replays
    entries = build_roster(replays_by_slug, frozenset(identity.vocabulary.codes), size)
    output.write_text(render_module(entries, identity.sha256, identity.vocabulary.sha256))
    for entry in entries:
        print(f"{entry.code:12} {entry.train_frames / 216_000:7.1f} h  {entry.label}")


if __name__ == "__main__":
    tyro.cli(build)
