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

`--stock-zero-only` restores the historical HAL completion rule: at least one
player must have zero stocks in the final parsed frame. This is stricter than
`--completed-only`, which also accepts games completed by time or other parsed
outcomes.

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
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import fields
from functools import partial
from pathlib import Path

import melee
from loguru import logger

from hal.data.index import ReplayIndexEntry
from hal.data.index import read_jsonl
from hal.data.replay_stats import PlayerStats
from hal.data.replay_stats import PlayerStatsMins
from hal.data.slippi import CHARACTERS_BY_NAME
from hal.data.slippi import slp_stage_to_libmelee
from hal.policy import INCLUDED_STAGES

Predicate = Callable[[ReplayIndexEntry], bool]

# Project policy: tournament-legal stages, keyed by libmelee enum name.
INCLUDED_STAGES_BY_NAME: dict[str, melee.Stage] = {stage.name: stage for stage in INCLUDED_STAGES}


def _stats_players(entry: ReplayIndexEntry) -> tuple[PlayerStats, ...]:
    if entry.stats is None:
        raise ValueError("a statistics predicate requires replay statistics")
    return entry.stats.players


def _resolve_ids(values: Sequence[str], table: dict[str, int], kind: str) -> set[int]:
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


def _resolve_stages(values: Sequence[str]) -> set[melee.Stage]:
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


def _human_players_only(entry: ReplayIndexEntry) -> bool:
    return len(entry.players) == 2 and all(player.player_type == "HUMAN" for player in entry.players)


def _min_frames(entry: ReplayIndexEntry, *, minimum: int) -> bool:
    return entry.frame_count >= minimum


def _max_frames(entry: ReplayIndexEntry, *, maximum: int) -> bool:
    return entry.frame_count <= maximum


def _completed_only(entry: ReplayIndexEntry) -> bool:
    return entry.outcome is not None and entry.outcome.completed


def _stock_zero_only(entry: ReplayIndexEntry) -> bool:
    return any(player.stocks_remaining == 0 for player in _stats_players(entry))


def _stage_in_set(entry: ReplayIndexEntry, *, stages: set[melee.Stage]) -> bool:
    try:
        return slp_stage_to_libmelee(entry.stage) in stages
    except ValueError:
        return False


def _character_in_set(entry: ReplayIndexEntry, *, characters: set[int]) -> bool:
    return any(player.character in characters for player in entry.players)


def _rank_in_set(entry: ReplayIndexEntry, *, ranks: set[str]) -> bool:
    return entry.rank_filename in ranks


def _min_player_stat(entry: ReplayIndexEntry, *, name: str, threshold: float) -> bool:
    return any(getattr(player, name) >= threshold for player in _stats_players(entry))


def _min_death_count(entry: ReplayIndexEntry, *, minimum: int) -> bool:
    return any(len(player.death_percents) >= minimum for player in _stats_players(entry))


def _max_cheap_deaths(entry: ReplayIndexEntry, *, maximum: int, percentage: float) -> bool:
    return all(
        sum(1 for death_percentage in player.death_percents if death_percentage <= percentage) < maximum
        for player in _stats_players(entry)
    )


def _min_damage_dealt_per_player_exclusive(entry: ReplayIndexEntry, *, minimum: float) -> bool:
    return all(player.damage_dealt > minimum for player in _stats_players(entry))


def _starting_stocks(entry: ReplayIndexEntry, *, expected: int) -> bool:
    return all(player.stocks_remaining + len(player.death_percents) == expected for player in _stats_players(entry))


def build_predicates(
    *,
    min_frames: int | None = None,
    max_frames: int | None = None,
    completed_only: bool = False,
    stock_zero_only: bool = False,
    stages: set[melee.Stage] | None = None,
    characters: set[int] | None = None,
    ranks: set[str] | None = None,
    mins: PlayerStatsMins | None = None,
    min_death_count: int | None = None,
    max_cheap_deaths: int | None = None,
    cheap_death_pct: float = 10.0,
    human_players_only: bool = False,
    min_damage_dealt_per_player_exclusive: float | None = None,
    starting_stocks: int | None = None,
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

    if human_players_only:
        preds.append(("human_players_only", _human_players_only))
    if min_frames is not None:
        preds.append((f"min_frames={min_frames}", partial(_min_frames, minimum=min_frames)))
    if max_frames is not None:
        preds.append((f"max_frames={max_frames}", partial(_max_frames, maximum=max_frames)))
    if completed_only:
        preds.append(("completed_only", _completed_only))
    if stock_zero_only:
        preds.append(("stock_zero_only", _stock_zero_only))
    if stages:
        preds.append((f"stages={sorted(stage.name for stage in stages)}", partial(_stage_in_set, stages=stages)))
    if characters:
        preds.append((f"characters={sorted(characters)}", partial(_character_in_set, characters=characters)))
    if ranks:
        preds.append((f"ranks={sorted(ranks)}", partial(_rank_in_set, ranks=ranks)))

    if mins is not None:
        for f in fields(mins):
            t = getattr(mins, f.name)
            if t is None:
                continue
            preds.append(
                (
                    f"min_{f.name}={t}",
                    partial(_min_player_stat, name=f.name, threshold=t),
                )
            )

    if min_death_count is not None:
        preds.append(
            (
                f"min_death_count={min_death_count}",
                partial(_min_death_count, minimum=min_death_count),
            )
        )

    if max_cheap_deaths is not None:
        preds.append(
            (
                f"max_cheap_deaths<{max_cheap_deaths}@{cheap_death_pct}%",
                partial(_max_cheap_deaths, maximum=max_cheap_deaths, percentage=cheap_death_pct),
            )
        )

    if min_damage_dealt_per_player_exclusive is not None:
        preds.append(
            (
                f"damage_dealt_per_player>{min_damage_dealt_per_player_exclusive}",
                partial(_min_damage_dealt_per_player_exclusive, minimum=min_damage_dealt_per_player_exclusive),
            )
        )

    if starting_stocks is not None:
        preds.append(
            (
                f"starting_stocks={starting_stocks}",
                partial(_starting_stocks, expected=starting_stocks),
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
    stock_zero_only: bool = False,
    stages: set[melee.Stage] | None = None,
    characters: set[int] | None = None,
    ranks: set[str] | None = None,
    mins: PlayerStatsMins | None = None,
    min_death_count: int | None = None,
    max_cheap_deaths: int | None = None,
    cheap_death_pct: float = 10.0,
    human_players_only: bool = False,
    min_damage_dealt_per_player_exclusive: float | None = None,
    starting_stocks: int | None = None,
    log_per_filter: bool = True,
) -> int:
    if not index.exists():
        raise FileNotFoundError(f"--index {index} not found")

    preds = build_predicates(
        min_frames=min_frames,
        max_frames=max_frames,
        completed_only=completed_only,
        stock_zero_only=stock_zero_only,
        stages=stages,
        characters=characters,
        ranks=ranks,
        mins=mins,
        min_death_count=min_death_count,
        max_cheap_deaths=max_cheap_deaths,
        cheap_death_pct=cheap_death_pct,
        human_players_only=human_players_only,
        min_damage_dealt_per_player_exclusive=min_damage_dealt_per_player_exclusive,
        starting_stocks=starting_stocks,
    )

    needs_stats = (
        stock_zero_only
        or (mins is not None and mins.any_set())
        or min_death_count is not None
        or max_cheap_deaths is not None
        or min_damage_dealt_per_player_exclusive is not None
        or starting_stocks is not None
    )

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


@dataclass(frozen=True, slots=True)
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

    stock_zero_only: bool = False
    """Keep only replays where a player ends with zero stocks.

    This is the completion predicate used by the historical Cody build.
    """

    stages: tuple[str, ...] = (
        "BATTLEFIELD",
        "FINAL_DESTINATION",
        "FOUNTAIN_OF_DREAMS",
        "POKEMON_STADIUM",
        "DREAMLAND",
        "YOSHIS_STORY",
    )
    """Stage names (or slp-native ints). Pass --stages with no values to
    disable, e.g. via `--stages` (no items) — keeps every stage."""

    characters: tuple[str, ...] = ()
    """Character names (or libmelee Character ints, e.g. FOX==1). Empty = no
    character filter. Matches the index's normalized internal ids."""

    ranks: tuple[str, ...] = ()
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

    human_players_only: bool = False
    """Keep only replays with exactly two players, both human."""

    min_damage_dealt_per_player_exclusive: float | None = None
    """Keep only if every player dealt strictly more than this value."""

    starting_stocks: int | None = None
    """Required `stocks_remaining + death_count` for every player."""


def select_replay_paths(cfg: FilterConfig) -> int:
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
        stock_zero_only=cfg.stock_zero_only,
        stages=stages,
        characters=chars,
        ranks=ranks,
        mins=mins if mins.any_set() else None,
        min_death_count=cfg.min_death_count,
        max_cheap_deaths=cfg.max_cheap_deaths,
        cheap_death_pct=cfg.cheap_death_pct,
        human_players_only=cfg.human_players_only,
        min_damage_dealt_per_player_exclusive=cfg.min_damage_dealt_per_player_exclusive,
        starting_stocks=cfg.starting_stocks,
    )
