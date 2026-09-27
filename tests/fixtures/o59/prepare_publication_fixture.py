"""Select a small, deterministic full-MDS publication fixture from dev.7z."""

import argparse
import json
from pathlib import Path

from hal.data.index import ReplayIndexEntry
from hal.data.index import read_jsonl
from hal.data.index import replay_uuid_from_path
from hal.data.mds_materialization import bucket_fraction


def _sort_key(entry: ReplayIndexEntry) -> tuple[int, str]:
    return entry.frame_count, entry.path


def main(index: Path, eligible_paths: Path, output: Path) -> None:
    eligible = set(eligible_paths.read_text().splitlines())
    by_split: dict[str, list[ReplayIndexEntry]] = {"train": [], "val": [], "test": []}
    for entry in read_jsonl(index):
        if entry.path not in eligible:
            continue
        fraction = bucket_fraction(replay_uuid_from_path(entry.path))
        split = "train" if fraction < 0.34 else "val" if fraction < 0.67 else "test"
        by_split[split].append(entry)

    selected: list[tuple[str, ReplayIndexEntry]] = []
    for split, entries in by_split.items():
        if len(entries) < 2:
            raise ValueError(f"{split}: expected at least two eligible replays")
        selected.extend((split, entry) for entry in sorted(entries, key=_sort_key)[:2])

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write("".join(f"{entry.path}\n" for _, entry in selected))
    provenance = output.with_suffix(".jsonl")
    with provenance.open("x") as handle:
        for split, entry in selected:
            handle.write(
                json.dumps(
                    {
                        "path": entry.path,
                        "sha1": entry.sha1,
                        "frame_count": entry.frame_count,
                        "split": split,
                    },
                    sort_keys=True,
                )
                + "\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--eligible-paths", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.index, args.eligible_paths, args.output)
