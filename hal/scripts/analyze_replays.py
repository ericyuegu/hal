"""Sweep ``.slp`` replays into one behavior CSV, one row per player per match.

Every metric comes from ``hal.eval.behavior``; this module only walks the
directories, spreads the parsing over processes, writes the CSV and prints the
per-model means. Keep it that way — a metric defined here would be invisible to
the tests that pin the libraries.

Port labels come from the replays themselves (the head-to-head runner stamps the
policy name into the game-start display names). Pass ``--records`` with one or
more head-to-head ``matches.jsonl`` files to name the ports from the run instead,
which also covers replays recorded before stamping.

Run:
    python -m hal.scripts.analyze_replays \\
        --inputs 019-factored=<dir> 016-base=<dir> \\
        --records <h2h out_dir>/matches.jsonl \\
        --out behavior_rows.csv
"""

import csv
import math
from collections.abc import Mapping
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from typing import Literal

import tyro
from loguru import logger

from hal.eval.behavior import analyze_replay
from hal.eval.h2h import load_records

Row = dict[str, int | float | str]
# (label, replay path, port -> model, minimum active frames): one parse job.
Job = tuple[str, str, dict[int, str] | None, int]

# What the grouped summary prints. Everything else stays in the CSV.
SUMMARY_METRICS: Final[tuple[str, ...]] = (
    "won",
    "stocks_lost",
    "stocks_taken",
    "damage_dealt_per_min",
    "damage_taken_per_min",
    "death_percent_mean",
    "deaths_bottom",
    "deaths_side",
    "deaths_top",
    "deaths_sd_like",
    "sd_count",
    "deaths_helpless",
    "deaths_edgeguarded",
    "center_control_frac",
    "mean_center_dist",
    "frac_offstage",
    "frac_near_ledge_onstage",
    "openings_per_min",
    "opening_damage_mean",
    "damage_per_opening",
    "openings_per_kill",
    "neutral_win_ratio",
    "counter_hit_ratio",
    "successful_conversion_ratio",
    "wavedashes_per_min",
    "dash_dances_per_min",
    "ledge_grabs_per_min",
    "idle_frac",
    "shield_frac",
)


@dataclass(frozen=True)
class Args:
    """Behavioral comparison of Melee policies from their .slp replays."""

    inputs: tuple[str, ...]
    """One or more ``LABEL=DIR`` pairs; DIR is searched recursively for .slp."""
    out: Path = Path("behavior_rows.csv")
    """Destination CSV (one row per player per match)."""
    records: tuple[Path, ...] = ()
    """Head-to-head ``matches.jsonl`` files that name each replay's ports."""
    workers: int = 8
    """Parser processes; 1 parses in this process."""
    group_by: Literal["model", "label"] = "model"
    """Summary grouping: the resolved policy name, or the input label."""
    models_only: bool = True
    """Leave CPU players out of the summary (they stay in the CSV)."""
    min_active_frames: int = 300
    """Drop matches with fewer live frames (aborted boots, instant restarts)."""


def port_map(paths: Sequence[Path]) -> dict[str, dict[int, str]]:
    """Replay basename -> port -> model, read from head-to-head match records.

    The basename is the key because a record's ``replay_path`` is absolute on the
    machine that ran the sweep, while the match id (which IS the basename) is
    globally unique. Two records that name the same file differently are a
    contradiction and raise.
    """
    out: dict[str, dict[int, str]] = {}
    for path in paths:
        for record in load_records(path):
            names = {1: record.model_port_1, 2: record.model_port_2}
            keys = {f"{record.match_id}.slp"}
            if record.replay_path is not None:
                keys.add(Path(record.replay_path).name)
            for key in keys:
                if out.get(key, names) != names:
                    raise ValueError(f"{key} is mapped to both {out[key]} and {names}")
                out[key] = names
    return out


def replay_paths(root: Path) -> list[Path]:
    return sorted(root.rglob("*.slp")) if root.is_dir() else [root]


def analyze_one(job: Job) -> list[Row]:
    """One replay -> its flat rows, tagged with the input label. Empty if the
    replay is unusable; the sweep reports the shortfall rather than failing."""
    label, path, names, min_active_frames = job
    rows = analyze_replay(path, names=names, min_active_frames=min_active_frames)
    if rows is None:
        return []
    return [{"label": label, **row.as_flat_dict()} for row in rows]


def _mean(rows: Sequence[Mapping[str, int | float | str]], metric: str) -> float:
    """Mean over the rows that carry a finite number for ``metric``."""
    values = [
        float(r[metric]) for r in rows if isinstance(r[metric], (int, float)) and math.isfinite(float(r[metric]))
    ]
    return sum(values) / len(values) if values else math.nan


def summary_table(rows: Sequence[Row], *, group_by: str) -> str:
    """Mean of every summary metric per group, one column per group."""
    groups: dict[str, list[Row]] = {}
    for row in rows:
        groups.setdefault(str(row[group_by]), []).append(row)
    if not groups:
        return "no rows to summarize"
    columns = sorted(groups)
    width = max(14, *(len(c) + 2 for c in columns))
    lines = [
        f"=== behavior means by {group_by} ===",
        "metric".ljust(30) + "".join(c.rjust(width) for c in columns),
        "player-matches".ljust(30) + "".join(str(len(groups[c])).rjust(width) for c in columns),
    ]
    for metric in SUMMARY_METRICS:
        cells = "".join(f"{_mean(groups[c], metric):.3f}".rjust(width) for c in columns)
        lines.append(metric.ljust(30) + cells)
    return "\n".join(lines)


def main(args: Args) -> None:
    names_by_replay = port_map(args.records)
    jobs: list[Job] = []
    for spec in args.inputs:
        if "=" not in spec:
            raise ValueError(f"--inputs entries must be LABEL=DIR, got {spec!r}")
        label, _, directory = spec.partition("=")
        paths = replay_paths(Path(directory).expanduser())
        if not paths:
            raise ValueError(f"no .slp under {directory}")
        logger.info(f"{label}: {len(paths)} replays under {directory}")
        jobs.extend((label, str(p), names_by_replay.get(p.name), args.min_active_frames) for p in paths)

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(analyze_one, jobs, chunksize=4))
    else:
        results = [analyze_one(job) for job in jobs]
    rows = [row for result in results for row in result]
    if not rows:
        raise RuntimeError(f"no usable match in {len(jobs)} replays")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"{len(rows)} player-match rows from {len(rows) // 2}/{len(jobs)} replays -> {args.out}")

    summarized = [r for r in rows if not r["is_cpu"]] if args.models_only else list(rows)
    print(summary_table(summarized, group_by=args.group_by))


if __name__ == "__main__":
    main(tyro.cli(Args))
