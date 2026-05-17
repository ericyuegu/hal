"""Stage 2: query `index.jsonl` and emit a `paths.txt`.

Pure function on the index — no slp opens. All predicates run in-memory and
compose with AND. Output is a deterministically-sorted newline-delimited list
of absolute slp paths, one per line, ready to feed into `process_replays.py`.

CLI defaults bake in the "sensible" filter for tournament-style training:
  - completed games only (no NO_CONTEST / unresolved)
  - min 1500 frames (~25 sec, drops insta-quits and CSS-only replays)
  - tournament-legal six stages

Override or disable any of these via flags. Pass `--stages` an empty list
(or a different list) to drop the stage filter; `--no-completed-only` to
include unfinished games; `--min-frames 0` to keep everything.

Stages and characters accept names (case-insensitive) from the tables below,
OR slp-native integer ids (e.g. `--stages 31 32` or `--stages BATTLEFIELD
FINAL_DESTINATION`). Player-code filters accept inline names or
`@path/to/file.txt` for one-per-line lists.

Damage / stocks / inputs predicates require an index built with
`python -m hal.scripts.index --with-stats`. If the index has no stats,
`filter_index` raises rather than silently producing empty output.
"""

import operator
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
from hal.policy import INCLUDED_STAGES
from hal.wire import CHARACTERS_BY_NAME
from hal.wire import slp_stage_to_libmelee

Predicate = Callable[[ReplayIndexEntry], bool]

# Project policy: tournament-legal stages, keyed by libmelee enum name.
INCLUDED_STAGES_BY_NAME: dict[str, melee.Stage] = {stage.name: stage for stage in INCLUDED_STAGES}


@dataclass(frozen=True, slots=True)
class PlayerStatsThresholds:
    """Minimums for per-player stats. None = no filter on that field.

    Field names mirror `PlayerStats` (minus `port`). Predicates use
    "any player matches" semantics, consistent with `characters` /
    `codes_include`. Requires an index built with `index --with-stats`.
    """

    damage_dealt: float | None = None
    damage_taken: float | None = None
    stocks_remaining: int | None = None
    inputs: int | None = None
    sds: int | None = None
    early_sds: int | None = None

    def any_set(self) -> bool:
        return any(getattr(self, f.name) is not None for f in fields(self))


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


def _parse_codes(arg: str) -> set[str]:
    if arg.startswith("@"):
        path = Path(arg[1:])
        return {line.strip() for line in path.read_text().splitlines() if line.strip()}
    return {c.strip() for c in arg.split(",") if c.strip()}


def _parse_version(s: str) -> tuple[int, int, int]:
    parts = s.split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise ValueError(f"slp version must be MAJOR.MINOR.PATCH; got {s!r}")
    a, b, c = (int(p) for p in parts)
    return (a, b, c)


def build_predicates(
    *,
    min_frames: int | None = None,
    max_frames: int | None = None,
    completed_only: bool = False,
    stages: set[melee.Stage] | None = None,
    characters: set[int] | None = None,
    ranks: set[str] | None = None,
    codes_include: set[str] | None = None,
    codes_exclude: set[str] | None = None,
    slp_version_min: tuple[int, int, int] | None = None,
    stats_thresholds: PlayerStatsThresholds | None = None,
) -> list[tuple[str, Predicate]]:
    """Return (label, predicate) pairs. The label is used for diagnostics.

    Per-player conditions (`characters`, `codes_include`, stats) are
    satisfied if ANY player matches. Stats predicates require entries with
    `stats` populated — `filter_index` raises on `entry.stats is None`
    before any stats predicate is evaluated, so predicate bodies here
    assume `e.stats is not None`.
    """
    preds: list[tuple[str, Predicate]] = []

    if min_frames is not None:
        preds.append((f"min_frames={min_frames}", lambda e: e.frame_count >= min_frames))
    if max_frames is not None:
        preds.append((f"max_frames={max_frames}", lambda e: e.frame_count <= max_frames))
    if completed_only:
        preds.append(("completed_only", lambda e: e.outcome is not None and e.outcome.completed))
    if stages:
        stages_set = stages
        preds.append(
            (
                f"stages={sorted(s.name for s in stages_set)}",
                lambda e, s=stages_set: slp_stage_to_libmelee(e.stage) in s,
            )
        )
    if characters:
        chars = characters
        preds.append((f"characters={sorted(chars)}", lambda e, c=chars: any(p.character in c for p in e.players)))
    if ranks:
        preds.append((f"ranks={sorted(ranks)}", lambda e: e.rank_filename in ranks))
    if codes_include:
        inc = codes_include
        preds.append(
            (
                f"codes_include={sorted(inc)}",
                lambda e, c=inc: any(p.code in c for p in e.players if p.code),
            )
        )
    if codes_exclude:
        exc = codes_exclude
        preds.append(
            (
                f"codes_exclude={sorted(exc)}",
                lambda e, c=exc: not any(p.code in c for p in e.players if p.code),
            )
        )
    if slp_version_min is not None:
        preds.append((f"slp_version>={slp_version_min}", lambda e: e.slp_version >= slp_version_min))

    if stats_thresholds is not None:
        for f in fields(stats_thresholds):
            t = getattr(stats_thresholds, f.name)
            if t is None:
                continue
            get = operator.attrgetter(f.name)
            preds.append(
                (
                    f"min_{f.name}={t}",
                    lambda e, t=t, g=get: any(g(p) >= t for p in e.stats.players),
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
    codes_include: set[str] | None = None,
    codes_exclude: set[str] | None = None,
    slp_version_min: tuple[int, int, int] | None = None,
    stats_thresholds: PlayerStatsThresholds | None = None,
    log_per_filter: bool = True,
) -> int:
    if not index.exists():
        raise FileNotFoundError(f"--index {index} not found")
    if codes_include and codes_exclude:
        raise ValueError("--player-codes-include and --player-codes-exclude are mutually exclusive")

    preds = build_predicates(
        min_frames=min_frames,
        max_frames=max_frames,
        completed_only=completed_only,
        stages=stages,
        characters=characters,
        ranks=ranks,
        codes_include=codes_include,
        codes_exclude=codes_exclude,
        slp_version_min=slp_version_min,
        stats_thresholds=stats_thresholds,
    )

    needs_stats = stats_thresholds is not None and stats_thresholds.any_set()

    paths: list[str] = []
    total = 0
    entries_failing_by_label: dict[str, int] = {label: 0 for label, _ in preds}

    for entry in read_jsonl(index):
        total += 1
        if needs_stats and entry.stats is None:
            raise ValueError(
                f"entry {entry.path} has stats=None but stats predicates were requested. "
                "Rebuild the index with: python -m hal.scripts.index --with-stats ..."
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

    Defaults bake in completed-only, 1500-frame minimum, and the six
    tournament-legal stages. Override or disable any of these via flags.
    """

    index: Path
    """Path to index.jsonl from build_index."""

    output: Path
    """Destination paths.txt."""

    min_frames: int = 1500
    """Drop replays shorter than this. Set to 0 to disable."""

    max_frames: int | None = None
    """Drop replays longer than this. None = unbounded."""

    completed_only: bool = True
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

    player_codes_include: str | None = None
    """Inline codes (ZAIN#0,IBDW#1) or `@path/to/file.txt`."""

    player_codes_exclude: str | None = None
    """Same syntax as --player-codes-include."""

    slp_version_min: str | None = None
    """Minimum slp version, e.g. 3.7.0. None = no version filter."""

    min_damage_dealt: float | None = None
    """Keep if any player dealt >= this damage. Requires --with-stats index."""

    min_damage_taken: float | None = None
    """Keep if any player took >= this damage. Requires --with-stats index."""

    min_stocks_remaining: int | None = None
    """Keep if any player ended with >= this many stocks. Requires --with-stats."""

    min_inputs: int | None = None
    """Keep if any player had >= this many button presses. Requires --with-stats."""

    min_sds: int | None = None
    """Keep if any player had >= this many self-destructs. Requires --with-stats."""

    min_early_sds: int | None = None
    """Keep if any player had >= this many SDs within ~8s of spawning. Requires --with-stats."""


def run(cfg: FilterConfig) -> int:
    stages = _resolve_stages(cfg.stages) if cfg.stages else None
    chars = _resolve_ids(cfg.characters, CHARACTERS_BY_NAME, "character") if cfg.characters else None
    ranks = {r.strip().lower() for r in cfg.ranks} if cfg.ranks else None
    codes_in = _parse_codes(cfg.player_codes_include) if cfg.player_codes_include else None
    codes_ex = _parse_codes(cfg.player_codes_exclude) if cfg.player_codes_exclude else None
    version_min = _parse_version(cfg.slp_version_min) if cfg.slp_version_min else None
    thresholds = PlayerStatsThresholds(
        damage_dealt=cfg.min_damage_dealt,
        damage_taken=cfg.min_damage_taken,
        stocks_remaining=cfg.min_stocks_remaining,
        inputs=cfg.min_inputs,
        sds=cfg.min_sds,
        early_sds=cfg.min_early_sds,
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
        codes_include=codes_in,
        codes_exclude=codes_ex,
        slp_version_min=version_min,
        stats_thresholds=thresholds if thresholds.any_set() else None,
    )


if __name__ == "__main__":
    run(tyro.cli(FilterConfig))
