"""Prepare deduplicated, filtered professional-player corpora for Stage 3.

The Dropbox manifest names one directory per professional player.  Each
archive is indexed independently into a shared per-player index and receives
an atomic completion marker.  Successful replays are therefore resumable,
while a small number of hard parser failures do not cause every future run to
retry an otherwise completed archive.

This command does not materialize or upload MDS.  It emits, per player:

* ``index.jsonl``: all successfully parsed replays;
* ``index.failures.jsonl``: parser failures from all archives;
* ``deduped-index.jsonl``: deterministic SHA-1 deduplication;
* ``paths.txt``: the standard quality-filtered replay paths;
* ``rank-overrides.jsonl``: owner-only ``PRO`` labels when identity is known;
* ``professional-report.json``: source, dedupe, filter, and label coverage.

The rank override is conservative.  It labels the archive owner only when a
name/connect-code identity can be linked across the corpus.  An opponent is
never promoted merely because they appear in a professional player's archive,
and replays with no usable identity metadata remain ``UNKNOWN``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Literal

import tyro
from loguru import logger

from hal.data.archive import list_archive_slps
from hal.data.index import PlayerEntry
from hal.data.index import ReplayIndexEntry
from hal.data.index import read_jsonl
from hal.data.index import write_jsonl
from hal.data.schema import Rank
from hal.scripts.build_index import build_index
from hal.scripts.filter import FilterConfig
from hal.scripts.filter import run as filter_replays


def player_slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")
    if not slug:
        raise ValueError(f"cannot make a player slug from {name!r}")
    return "mang0" if slug in {"mango", "mang0"} else slug


def _source_id(path: Path) -> str:
    digest = hashlib.sha256(os.fsencode(path.resolve())).hexdigest()[:16]
    return f"{path.name}.{digest}"


def load_sources(raw_root: Path, manifest: Path) -> dict[str, list[Path]]:
    """Resolve and validate every archive declared by the download manifest."""
    grouped: dict[str, list[Path]] = defaultdict(list)
    with manifest.open() as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            declared = str(row["path"])
            relative = Path(declared.lstrip("/"))
            if len(relative.parts) < 2:
                raise ValueError(f"{manifest}:{line_no}: expected /player/archive, got {declared!r}")
            source = raw_root / relative
            if not source.is_file():
                raise FileNotFoundError(f"manifest source is missing: {source}")
            expected_bytes = int(row["bytes"])
            actual_bytes = source.stat().st_size
            if actual_bytes != expected_bytes:
                raise ValueError(f"manifest size mismatch for {source}: expected {expected_bytes}, got {actual_bytes}")
            grouped[player_slug(relative.parts[0])].append(source)

    # The separately supplied mang0 corpus belongs to the same player stream.
    standalone_mang0 = raw_root / "mang0.7z"
    if standalone_mang0.is_file():
        grouped["mang0"].append(standalone_mang0)

    return {slug: sorted(set(paths)) for slug, paths in sorted(grouped.items())}


def dedupe_index(source: Path, destination: Path) -> tuple[int, int]:
    """Write one deterministic entry per content SHA-1.

    Entries without a SHA-1 are retained by path instead of being collapsed
    together.  Production indexes compute SHA-1, but this makes failure loud
    and harmless if a metadata-only index is supplied accidentally.
    """
    entries = sorted(read_jsonl(source), key=lambda entry: entry.path)
    selected: list[ReplayIndexEntry] = []
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        identity = ("sha1", entry.sha1) if entry.sha1 is not None else ("path", entry.path)
        if identity in seen:
            continue
        seen.add(identity)
        selected.append(entry)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(destination, selected)
    return len(entries), len(selected)


def _normal(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").casefold())


_EXTRA_ALIASES: dict[str, set[str]] = {
    "m2k": {"m2k", "mew2king"},
    "mang0": {"mango", "mang0"},
    "bobbybigballz": {"bobbybigballz", "bbb"},
    "iliketurtles": {"iliketurtles", "turtles"},
}


class _Components:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _tokens(player: PlayerEntry) -> set[str]:
    tokens: set[str] = set()
    name = _normal(player.name)
    code = _normal(player.code)
    if name:
        tokens.add(f"name:{name}")
    if code:
        tokens.add(f"code:{code}")
    return tokens


def _owner_component(slug: str, entries: list[ReplayIndexEntry]) -> tuple[set[str], dict[str, Any]]:
    components = _Components()
    token_counts: Counter[str] = Counter()
    for entry in entries:
        for player in entry.players:
            tokens = sorted(_tokens(player))
            for token in tokens:
                token_counts[token] += 1
            for token in tokens[1:]:
                components.union(tokens[0], token)

    component_tokens: dict[str, set[str]] = defaultdict(set)
    component_counts: Counter[str] = Counter()
    for token, count in token_counts.items():
        root = components.find(token)
        component_tokens[root].add(token)
        component_counts[root] += count

    aliases = {_normal(slug.replace("-", ""))}
    aliases.update(_EXTRA_ALIASES.get(slug.replace("-", ""), set()))
    aliases = {_normal(alias) for alias in aliases}
    matches = [
        root
        for root, tokens in component_tokens.items()
        if any(token.removeprefix("name:") in aliases for token in tokens if token.startswith("name:"))
    ]
    method = "alias"
    owner: str | None = None
    if len(matches) == 1:
        owner = matches[0]
    elif not matches and component_counts:
        # Some folders use a legal name while the replay metadata uses a tag.
        # Accept a dominant identity only when it appears in at least half the
        # corpus and has a clear lead over the runner-up.
        ranked = component_counts.most_common(2)
        best, best_count = ranked[0]
        second_count = ranked[1][1] if len(ranked) > 1 else 0
        if best_count >= max(3, len(entries) // 2) and best_count >= 2 * max(1, second_count):
            owner = best
            method = "dominant"
    elif len(matches) > 1:
        method = "ambiguous-alias"
    else:
        method = "unresolved"

    owner_tokens = component_tokens.get(owner, set()) if owner is not None else set()
    report = {
        "method": method,
        "aliases": sorted(aliases),
        "owner_tokens": sorted(owner_tokens),
        "top_components": [
            {"count": count, "tokens": sorted(component_tokens[root])}
            for root, count in component_counts.most_common(5)
        ],
    }
    return owner_tokens, report


def write_owner_rank_overrides(
    slug: str,
    index: Path,
    selected_paths: Path,
    destination: Path,
) -> dict[str, Any]:
    entries = {entry.path: entry for entry in read_jsonl(index)}
    paths = [line.strip() for line in selected_paths.read_text().splitlines() if line.strip()]
    selected = [entries[path] for path in paths]
    owner_tokens, identity_report = _owner_component(slug, selected)

    rows: list[dict[str, object]] = []
    labeled_replays = 0
    ambiguous_replays = 0
    for entry in selected:
        owner_ports = [player.port for player in entry.players if _tokens(player) & owner_tokens]
        if len(owner_ports) == 1:
            # p1/p2 are the two occupied physical ports in ascending order,
            # not literal controller ports 1 and 2 (see extract_replay).
            occupied = sorted(player.port for player in entry.players)
            ranks = [Rank.UNKNOWN, Rank.UNKNOWN]
            try:
                logical_port = occupied.index(owner_ports[0])
            except ValueError:
                logical_port = -1
            if logical_port in (0, 1):
                ranks[logical_port] = Rank.PRO
                labeled_replays += 1
            rows.append(
                {
                    "path": entry.path,
                    "p1_rank": int(ranks[0]),
                    "p2_rank": int(ranks[1]),
                }
            )
        else:
            ambiguous_replays += len(owner_ports) > 1
            rows.append(
                {
                    "path": entry.path,
                    "p1_rank": int(Rank.UNKNOWN),
                    "p2_rank": int(Rank.UNKNOWN),
                }
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return {
        **identity_report,
        "selected_replays": len(selected),
        "owner_labeled_replays": labeled_replays,
        "unlabeled_replays": len(selected) - labeled_replays,
        "ambiguous_replays": ambiguous_replays,
    }


def write_corpus_rank_overrides(selected_paths: Path, destination: Path) -> dict[str, Any]:
    """Label both sides PRO when corpus-level provenance is explicitly chosen."""
    paths = [line.strip() for line in selected_paths.read_text().splitlines() if line.strip()]
    with destination.open("w") as handle:
        for path in paths:
            handle.write(
                json.dumps(
                    {"path": path, "p1_rank": int(Rank.PRO), "p2_rank": int(Rank.PRO)},
                    sort_keys=True,
                )
                + "\n"
            )
    return {
        "method": "corpus",
        "aliases": [],
        "owner_tokens": ["corpus-provenance"],
        "top_components": [],
        "selected_replays": len(paths),
        "owner_labeled_replays": len(paths),
        "unlabeled_replays": 0,
        "ambiguous_replays": 0,
    }


@dataclass
class PrepareProfessionalConfig:
    raw_root: Path = Path("/home/ericgu/data/raw")
    manifest: Path = Path("/home/ericgu/data/raw/dropbox-pro-replays-manifest.jsonl")
    build_root: Path = Path("data/builds/policy-world-20260816/professional")
    player: str | None = None
    workers: int = 10
    queue_size: int = 8
    tmpfs_root: Path = Path("/dev/shm/hal_professional_index")
    rank_mode: Literal["owner", "corpus"] = "owner"


def prepare_professional(cfg: PrepareProfessionalConfig) -> None:
    grouped = load_sources(cfg.raw_root, cfg.manifest)
    requested = player_slug(cfg.player) if cfg.player is not None else None
    if requested is not None:
        if requested not in grouped:
            raise ValueError(f"unknown player {cfg.player!r}; choices: {sorted(grouped)}")
        grouped = {requested: grouped[requested]}

    for slug, sources in grouped.items():
        player_root = cfg.build_root / slug
        markers = player_root / "archives-complete"
        index = player_root / "index.jsonl"
        failures = player_root / "index.failures.jsonl"
        markers.mkdir(parents=True, exist_ok=True)
        source_report: list[dict[str, object]] = []
        for source in sources:
            members = list_archive_slps(source)
            marker = markers / f"{_source_id(source)}.json"
            if marker.is_file():
                logger.info(f"{slug}: already complete: {source.name}")
            else:
                build_index(
                    output=index,
                    archive=source,
                    incremental=index.exists(),
                    compute_sha1=True,
                    with_stats=True,
                    workers=cfg.workers,
                    tmpfs_root=cfg.tmpfs_root / slug,
                    queue_size=cfg.queue_size,
                    failure_log=failures,
                )
                marker.write_text(
                    json.dumps(
                        {
                            "source": str(source.resolve()),
                            "bytes": source.stat().st_size,
                            "members": len(members),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                )
            source_report.append({"path": str(source), "members": len(members), "complete": marker.is_file()})

        deduped = player_root / "deduped-index.jsonl"
        before, after = dedupe_index(index, deduped)
        paths = player_root / "paths.txt"
        kept = filter_replays(FilterConfig(index=deduped, output=paths))
        rank_path = player_root / "rank-overrides.jsonl"
        rank_report = (
            write_owner_rank_overrides(slug, deduped, paths, rank_path)
            if cfg.rank_mode == "owner"
            else write_corpus_rank_overrides(paths, rank_path)
        )
        report = {
            "player": slug,
            "sources": source_report,
            "indexed": before,
            "deduplicated": after,
            "duplicates_removed": before - after,
            "quality_filtered": kept,
            "rank": rank_report,
        }
        (player_root / "professional-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        logger.info(f"{slug}: prepared {kept} filtered replays; rank coverage={rank_report}")


if __name__ == "__main__":
    prepare_professional(tyro.cli(PrepareProfessionalConfig))
