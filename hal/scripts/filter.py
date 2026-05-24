"""Stage 2: query `index.jsonl` and emit a `paths.txt`.

Pure function on the index — no slp opens. All predicates run in-memory and
compose with AND. Output is a deterministically-sorted newline-delimited list
of absolute slp paths, one per line, ready to feed into `process_replays.py`.

CLI defaults bake in the "sensible" filter for tournament-style training:
  - min 1500 frames (~25 sec, drops insta-quits and CSS-only replays)
  - tournament-legal six stages
  - min 100% damage dealt and taken by some player
  - some player loses >= 3 stocks
  - no player loses >= 2 stocks at <= 10% (cheap-death sniff for AFK / griefing)

Override or disable any of these via flags. Pass `--stages` an empty list
(or a different list) to drop the stage filter; `--no-completed-only` to
include unfinished games; `--min-frames 0` to keep everything; set the stats
knobs to None / 0 to disable individually.

Stages and characters accept names (case-insensitive) from the tables below,
OR slp-native integer ids (e.g. `--stages 31 32` or `--stages BATTLEFIELD
FINAL_DESTINATION`).

Stats predicates (damage / stocks / inputs / death counts / cheap deaths)
require an index built with `python -m hal.scripts.build_index --with-stats`. If
the index has no stats, `filter_index` raises rather than silently producing
empty output.
"""

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from pathlib import Path

import melee
import tyro
from loguru import logger

from hal.data.index import ReplayIndexEntry
from hal.data.index import read_jsonl
from hal.data.replay_stats import PlayerStatsMins
from hal.policy import INCLUDED_STAGES
from hal.wire import CHARACTERS_BY_NAME
from hal.wire import slp_stage_to_libmelee

Predicate = Callable[[ReplayIndexEntry], bool]

# Project policy: tournament-legal stages, keyed by libmelee enum name.
INCLUDED_STAGES_BY_NAME: dict[str, melee.Stage] = {stage.name: stage for stage in INCLUDED_STAGES}


def _resolve_ids(values: list[str], table: dict[str, int], kind: str) -> set[int]:
    out: set[int] = set()
    for v in values:
        v = v.strip()
        if not v:
            continue
        if v.isdigit():
            out.add(int(v))
            continue
        key = v.upper()
        if key not in table:
            raise ValueError(f"unknown {kind} {v!r}; known names: {sorted(table)}")
        out.add(table[key])
    return out


def _resolve_stages(values: list[str]) -> set[melee.Stage]:
    """Resolve stage names or slp-native ints to libmelee ``Stage`` enums."""
    out: set[melee.Stage] = set()
    for v in values:
        v = v.strip()
        if not v:
            continue
        if v.isdigit():
            out.add(slp_stage_to_libmelee(int(v)))
            continue
        key = v.upper()
        if key not in INCLUDED_STAGES_BY_NAME:
            raise ValueError(f"unknown stage {v!r}; known names: {sorted(INCLUDED_STAGES_BY_NAME)}")
        out.add(INCLUDED_STAGES_BY_NAME[key])
    return out


def build_predicates(
    *,
    min_frames: int | None = None,
    max_frames: int | None = None,
    completed_only: bool = False,
    stages: set[melee.Stage] | None = None,
    characters: set[int] | None = None,
    ranks: set[str] | None = None,
    mins: PlayerStatsMins | None = None,
    min_death_count: int | None = None,
    max_cheap_deaths: int | None = None,
    cheap_death_pct: float = 10.0,
) -> list[tuple[str, Predicate]]:
    """Return (label, predicate) pairs. The label is used for diagnostics.

    Per-player floors (`characters`, `mins`, `min_death_count`) are satisfied
    if ANY player matches. Per-player ceilings (`max_cheap_deaths`) require
    EVERY player to stay under. Stats predicates require entries with `stats`
    populated — `filter_index` raises on `entry.stats is None` before any
    stats predicate is evaluated, so predicate bodies here assume
    `e.stats is not None`.
    """
    preds: list[tuple[str, Predicate]] = []

    if min_frames is not None:
        preds.append((f"min_frames={min_frames}", lambda e: e.frame_count >= min_frames))
    if max_frames is not None:
        preds.append((f"max_frames={max_frames}", lambda e: e.frame_count <= max_frames))
    if completed_only:
        preds.append(("completed_only", lambda e: e.outcome is not None and e.outcome.completed))
    if stages:

        def _stage_in_set(e: ReplayIndexEntry, s: set[melee.Stage] = stages) -> bool:
            try:
                return slp_stage_to_libmelee(e.stage) in s
            except ValueError:
                return False

        preds.append((f"stages={sorted(s.name for s in stages)}", _stage_in_set))
    if characters:
        preds.append(
            (f"characters={sorted(characters)}", lambda e, c=characters: any(p.character in c for p in e.players))
        )
    if ranks:
        preds.append((f"ranks={sorted(ranks)}", lambda e: e.rank_filename in ranks))

    if mins is not None:
        for f in fields(mins):
            t = getattr(mins, f.name)
            if t is None:
                continue
            preds.append(
                (
                    f"min_{f.name}={t}",
                    lambda e, t=t, n=f.name: any(getattr(p, n) >= t for p in e.stats.players),
                )
            )

    if min_death_count is not None:
        preds.append(
            (
                f"min_death_count={min_death_count}",
                lambda e, t=min_death_count: any(len(p.death_percents) >= t for p in e.stats.players),
            )
        )

    if max_cheap_deaths is not None:
        preds.append(
            (
                f"max_cheap_deaths<{max_cheap_deaths}@{cheap_death_pct}%",
                lambda e, m=max_cheap_deaths, c=cheap_death_pct: all(
                    sum(1 for dp in p.death_percents if dp <= c) < m for p in e.stats.players
                ),
            )
        )

    return preds


def filter_index(
    index: Path,
    output: Path,
    *,
    min_frames: int | None = None,
    max_frames: int | None = None,
    completed_only: bool = False,
    stages: set[melee.Stage] | None = None,
    characters: set[int] | None = None,
    ranks: set[str] | None = None,
    mins: PlayerStatsMins | None = None,
    min_death_count: int | None = None,
    max_cheap_deaths: int | None = None,
    cheap_death_pct: float = 10.0,
    log_per_filter: bool = True,
) -> int:
    if not index.exists():
        raise FileNotFoundError(f"--index {index} not found")

    preds = build_predicates(
        min_frames=min_frames,
        max_frames=max_frames,
        completed_only=completed_only,
        stages=stages,
        characters=characters,
        ranks=ranks,
        mins=mins,
        min_death_count=min_death_count,
        max_cheap_deaths=max_cheap_deaths,
        cheap_death_pct=cheap_death_pct,
    )

    needs_stats = (mins is not None and mins.any_set()) or min_death_count is not None or max_cheap_deaths is not None

    paths: list[str] = []
    total = 0
    entries_failing_by_label: dict[str, int] = {label: 0 for label, _ in preds}

    for entry in read_jsonl(index):
        total += 1
        if needs_stats and entry.stats is None:
            raise ValueError(
                f"entry {entry.path} has stats=None but stats predicates were requested. "
                "Rebuild the index with: python -m hal.scripts.build_index --with-stats ..."
            )
        kept = True
        for label, pred in preds:
            if not pred(entry):
                entries_failing_by_label[label] += 1
                kept = False
                if not log_per_filter:
                    break
        if kept:
            paths.append(entry.path)

    paths.sort()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(paths) + ("\n" if paths else ""))

    logger.info(f"index: {total}  kept: {len(paths)}  dropped: {total - len(paths)}")
    sum_caveat = " (sum > dropped when an entry fails multiple predicates)" if log_per_filter else ""
    logger.info(f"entries failing each predicate{sum_caveat}:")
    for label, n in entries_failing_by_label.items():
        logger.info(f"  fail[{label}]: {n}")
    logger.info(f"wrote {len(paths)} paths -> {output}")
    return len(paths)


@dataclass
class FilterConfig:
    """Filter `index.jsonl` to a `paths.txt` for Stage 3.

    Defaults bake in a 1500-frame minimum, the six tournament-legal stages,
    a 100% damage floor for both sides, a 3-stock-loss floor, and a
    cheap-death sniff (no player loses >= 2 stocks at <= 10%). Override or
    disable any of these via flags.
    """

    index: Path
    """Path to index.jsonl from build_index."""

    output: Path
    """Destination paths.txt."""

    min_frames: int = 1500
    """Drop replays shorter than this. Set to 0 to disable."""

    max_frames: int | None = None
    """Drop replays longer than this. None = unbounded."""

    completed_only: bool = False
    """Keep only replays that ended via stocks / time / sudden-death.
    Pass --no-completed-only to include NO_CONTEST and unresolved games."""

    stages: list[str] = field(
        default_factory=lambda: [
            "BATTLEFIELD",
            "FINAL_DESTINATION",
            "FOUNTAIN_OF_DREAMS",
            "POKEMON_STADIUM",
            "DREAMLAND",
            "YOSHIS_STORY",
        ]
    )
    """Stage names (or slp-native ints). Pass --stages with no values to
    disable, e.g. via `--stages` (no items) — keeps every stage."""

    characters: list[str] = field(default_factory=list)
    """Character names (or slp-native ints). Empty = no character filter."""

    ranks: list[str] = field(default_factory=list)
    """Rank substrings to keep, e.g. master,diamond,platinum. Empty = no
    rank filter."""

    min_damage_dealt: float | None = 100.0
    """Keep if any player dealt >= this damage. Requires --with-stats index."""

    min_damage_taken: float | None = 100.0
    """Keep if any player took >= this damage. Requires --with-stats index."""

    min_stocks_remaining: int | None = None
    """Keep if any player ended with >= this many stocks. Requires --with-stats."""

    min_inputs: int | None = None
    """Keep if any player had >= this many button presses. Requires --with-stats."""

    min_death_count: int | None = 3
    """Keep if any player lost >= this many stocks. Requires --with-stats."""

    max_cheap_deaths: int | None = 2
    """Reject if any player lost >= this many stocks at <= `cheap_death_pct`.
    Set to None to disable the cheap-death sniff entirely."""

    cheap_death_pct: float = 10.0
    """Percent threshold below which a stock loss counts as "cheap" for the
    `max_cheap_deaths` predicate. Ignored if `max_cheap_deaths` is None."""


def run(cfg: FilterConfig) -> int:
    stages = _resolve_stages(cfg.stages) if cfg.stages else None
    chars = _resolve_ids(cfg.characters, CHARACTERS_BY_NAME, "character") if cfg.characters else None
    ranks = {r.strip().lower() for r in cfg.ranks} if cfg.ranks else None
    mins = PlayerStatsMins(
        damage_dealt=cfg.min_damage_dealt,
        damage_taken=cfg.min_damage_taken,
        stocks_remaining=cfg.min_stocks_remaining,
        inputs=cfg.min_inputs,
    )

    return filter_index(
        index=cfg.index,
        output=cfg.output,
        min_frames=cfg.min_frames if cfg.min_frames > 0 else None,
        max_frames=cfg.max_frames,
        completed_only=cfg.completed_only,
        stages=stages,
        characters=chars,
        ranks=ranks,
        mins=mins if mins.any_set() else None,
        min_death_count=cfg.min_death_count,
        max_cheap_deaths=cfg.max_cheap_deaths,
        cheap_death_pct=cfg.cheap_death_pct,
    )


if __name__ == "__main__":
    run(tyro.cli(FilterConfig))
