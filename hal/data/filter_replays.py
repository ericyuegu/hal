"""Stage 2: query `index.jsonl` and emit a `paths.txt`.

Pure function on the index — no slp opens. All predicates run in-memory and
compose with AND. Output is a deterministically-sorted newline-delimited list
of absolute slp paths, one per line, ready to feed into `process_replays.py`.

Usage:
    python -m hal.data.filter_replays \\
        --index /path/to/index.jsonl \\
        --output /path/to/paths.txt \\
        [--min-frames 1500] [--max-frames N] \\
        [--completed-only] \\
        [--stages BATTLEFIELD,FINAL_DESTINATION,...] \\
        [--characters FOX,MARTH] \\
        [--rank master,diamond] \\
        [--player-codes-include CODES] [--player-codes-exclude CODES] \\
        [--slp-version-min 3.7.0]

`--stages` and `--characters` accept names (case-insensitive) from the tables
below, OR slp-native integer ids. `--player-codes-*` accept either a
comma-separated list inline (`ZAIN#0,IBDW#1`) or `@path/to/codes.txt`
(one per line).

`--require-damage` is intentionally absent: the index doesn't store damage
because Stage 1 reads start/end blocks only (peppi `skip_frames=True`). If
a damage filter becomes necessary, add a peek-the-last-frame pass to
`build_index.py`, write `had_damage` to the manifest, and re-expose it
here.
"""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from loguru import logger

from hal.data.manifest import ReplayIndexEntry
from hal.data.manifest import read_jsonl

# Tournament-legal stages, slp-native ids. Names are libmelee-style for ergonomics.
LEGAL_STAGES_BY_NAME: dict[str, int] = {
    "FOUNTAIN_OF_DREAMS": 2,
    "POKEMON_STADIUM": 3,
    "YOSHIS_STORY": 8,
    "DREAMLAND": 28,
    "BATTLEFIELD": 31,
    "FINAL_DESTINATION": 32,
}

# Standard cast, slp-native ids. Values coincide with libmelee.Character enum
# (verified in notebooks/peppi_vs_libmelee.py).
CHARACTERS_BY_NAME: dict[str, int] = {
    "MARIO": 0,
    "FOX": 1,
    "CPTFALCON": 2,
    "DK": 3,
    "KIRBY": 4,
    "BOWSER": 5,
    "LINK": 6,
    "SHEIK": 7,
    "NESS": 8,
    "PEACH": 9,
    "POPO": 10,
    "NANA": 11,
    "PIKACHU": 12,
    "SAMUS": 13,
    "YOSHI": 14,
    "JIGGLYPUFF": 15,
    "MEWTWO": 16,
    "LUIGI": 17,
    "MARTH": 18,
    "ZELDA": 19,
    "YLINK": 20,
    "DOC": 21,
    "FALCO": 22,
    "PICHU": 23,
    "GAMEANDWATCH": 24,
    "GANONDORF": 25,
    "ROY": 26,
}


Predicate = Callable[[ReplayIndexEntry], bool]


def _resolve_ids(values: list[str], table: dict[str, int], kind: str) -> set[int]:
    out: set[int] = set()
    for v in values:
        v = v.strip()
        if not v:
            continue
        if v.isdigit() or (v.startswith("-") and v[1:].isdigit()):
            out.add(int(v))
            continue
        key = v.upper()
        if key not in table:
            raise ValueError(f"unknown {kind} {v!r}; known names: {sorted(table)}")
        out.add(table[key])
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
    stages: set[int] | None = None,
    characters: set[int] | None = None,
    ranks: set[str] | None = None,
    codes_include: set[str] | None = None,
    codes_exclude: set[str] | None = None,
    slp_version_min: tuple[int, int, int] | None = None,
) -> list[tuple[str, Predicate]]:
    """Return (label, predicate) pairs. The label is used for diagnostics."""
    preds: list[tuple[str, Predicate]] = []

    if min_frames is not None:
        preds.append((f"min_frames={min_frames}", lambda e: e.frame_count >= min_frames))
    if max_frames is not None:
        preds.append((f"max_frames={max_frames}", lambda e: e.frame_count <= max_frames))
    if completed_only:
        preds.append(("completed_only", lambda e: e.outcome is not None and e.outcome.completed))
    if stages:
        preds.append((f"stages={sorted(stages)}", lambda e: e.stage in stages))
    if characters:
        preds.append((f"characters={sorted(characters)}", lambda e: any(p.character in characters for p in e.players)))
    if ranks:
        preds.append((f"ranks={sorted(ranks)}", lambda e: e.rank_filename in ranks))
    if codes_include:
        preds.append(
            (
                f"codes_include={sorted(codes_include)}",
                lambda e: any(p.code in codes_include for p in e.players if p.code),
            )
        )
    if codes_exclude:
        preds.append(
            (
                f"codes_exclude={sorted(codes_exclude)}",
                lambda e: not any(p.code in codes_exclude for p in e.players if p.code),
            )
        )
    if slp_version_min is not None:
        preds.append((f"slp_version>={slp_version_min}", lambda e: e.slp_version >= slp_version_min))

    return preds


def filter_index(
    index: Path,
    output: Path,
    *,
    min_frames: int | None = None,
    max_frames: int | None = None,
    completed_only: bool = False,
    stages: set[int] | None = None,
    characters: set[int] | None = None,
    ranks: set[str] | None = None,
    codes_include: set[str] | None = None,
    codes_exclude: set[str] | None = None,
    slp_version_min: tuple[int, int, int] | None = None,
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
    )

    paths: list[str] = []
    total = 0
    drops_by_label: dict[str, int] = {label: 0 for label, _ in preds}

    for entry in read_jsonl(index):
        total += 1
        kept = True
        for label, pred in preds:
            if not pred(entry):
                drops_by_label[label] += 1
                kept = False
                if not log_per_filter:
                    break
        if kept:
            paths.append(entry.path)

    paths.sort()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(paths) + ("\n" if paths else ""))

    logger.info(f"index: {total}  kept: {len(paths)}  dropped: {total - len(paths)}")
    for label, n in drops_by_label.items():
        logger.info(f"  drop[{label}]: {n}")
    logger.info(f"wrote {len(paths)} paths -> {output}")
    return len(paths)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-frames", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--completed-only", action="store_true")
    parser.add_argument(
        "--stages",
        type=str,
        default=None,
        help=f"comma list of stage names or slp-native ints; names: {sorted(LEGAL_STAGES_BY_NAME)}",
    )
    parser.add_argument(
        "--characters",
        type=str,
        default=None,
        help=f"comma list of character names or slp-native ints; names: {sorted(CHARACTERS_BY_NAME)}",
    )
    parser.add_argument("--rank", type=str, default=None, help="comma list, e.g. master,diamond,platinum")
    parser.add_argument(
        "--player-codes-include",
        type=str,
        default=None,
        help="comma list (ZAIN#0,IBDW#1) or @path/to/file.txt",
    )
    parser.add_argument("--player-codes-exclude", type=str, default=None, help="same syntax as --player-codes-include")
    parser.add_argument("--slp-version-min", type=str, default=None, help="MAJOR.MINOR.PATCH (e.g. 3.7.0)")
    args = parser.parse_args()

    stages = _resolve_ids(args.stages.split(","), LEGAL_STAGES_BY_NAME, "stage") if args.stages else None
    chars = _resolve_ids(args.characters.split(","), CHARACTERS_BY_NAME, "character") if args.characters else None
    ranks = {r.strip().lower() for r in args.rank.split(",")} if args.rank else None
    codes_in = _parse_codes(args.player_codes_include) if args.player_codes_include else None
    codes_ex = _parse_codes(args.player_codes_exclude) if args.player_codes_exclude else None
    version_min = _parse_version(args.slp_version_min) if args.slp_version_min else None

    filter_index(
        index=args.index,
        output=args.output,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        completed_only=args.completed_only,
        stages=stages,
        characters=chars,
        ranks=ranks,
        codes_include=codes_in,
        codes_exclude=codes_ex,
        slp_version_min=version_min,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
