"""Attribute GPU activity to each complete two-buffer rollout tick."""

import bisect
import dataclasses
import hashlib
import json
import sqlite3
from collections import defaultdict
from contextlib import closing
from itertools import pairwise
from pathlib import Path

import tyro


@dataclasses.dataclass(frozen=True, slots=True)
class Tick:
    start: int
    end: int
    boundary: bool


def busy_seconds(intervals: list[tuple[int, int]]) -> float:
    total = previous = 0
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError("negative device interval")
        total += max(0, end - max(start, previous))
        previous = max(previous, end)
    return total / 1e9


def analyze(database: Path) -> dict[str, object]:
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        strings = dict(db.execute("SELECT id,value FROM StringIds"))
        ticks = sorted(
            (
                Tick(start, end, (text or strings[text_id]).endswith("/boundary_tick"))
                for start, end, text, text_id in db.execute(
                    "SELECT start,end,text,textId FROM NVTX_EVENTS WHERE end IS NOT NULL"
                )
                if (text or strings.get(text_id)) in ("o65/rollout/tick", "o65/rollout/boundary_tick")
            ),
            key=lambda tick: tick.start,
        )
        if not ticks or any(left.end > right.start for left, right in pairwise(ticks)):
            raise ValueError("trace needs non-overlapping complete rollout tick ranges")
        starts = [tick.start for tick in ticks]
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        apis: dict[tuple[int, int], tuple[int, int, str]] = {}
        for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
            if table not in tables:
                continue
            for start, tid, correlation, name in db.execute(
                f"SELECT start,globalTid,correlationId,nameId FROM {table}"
            ):
                index = bisect.bisect_right(starts, start) - 1
                if index >= 0 and start >= ticks[index].end:
                    index = -1
                key = tid & 0xFFFFFF000000, correlation
                if key not in apis or apis[key][0] < start:
                    apis[key] = start, index, strings[name]
        intervals: dict[int, list[tuple[int, int]]] = defaultdict(list)
        kernel_counts: dict[int, int] = defaultdict(int)
        eager: dict[int, int] = defaultdict(int)
        graphs: dict[int, int] = defaultdict(int)
        for _, index, name in apis.values():
            if index >= 0:
                graphs[index] += int("GraphLaunch" in name)
                eager[index] += int("LaunchKernel" in name)
        for table in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
            if table not in tables:
                continue
            for start, end, pid, correlation in db.execute(f"SELECT start,end,globalPid,correlationId FROM {table}"):
                api = apis.get((pid & 0xFFFFFF000000, correlation))
                if api is None:
                    raise ValueError("GPU activity has no launching API")
                index = api[1]
                if index >= 0:
                    intervals[index].append((start, end))
                    kernel_counts[index] += int(table == "CUPTI_ACTIVITY_KIND_KERNEL")
        if not sum(kernel_counts.values()):
            raise ValueError("rollout ticks contain no GPU kernels")
        rows = [
            {
                "tick": index,
                "boundary": tick.boundary,
                "host_seconds": (tick.end - tick.start) / 1e9,
                "gpu_busy_seconds": busy_seconds(intervals[index]),
                "kernels": kernel_counts[index],
                "eager_launches": eager[index],
                "graph_launches": graphs[index],
            }
            for index, tick in enumerate(ticks)
        ]
    with database.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    return {
        "format": "hal.o65.async-ticks.v1",
        "database_sha256": digest,
        "ticks": rows,
        "attribution": "GPU interval union by the issuing API's logical tick; includes memcpy and memset",
    }


@dataclasses.dataclass(frozen=True, slots=True)
class Args:
    database: Path
    output: Path


def main() -> None:
    args = tyro.cli(Args)
    args.output.write_text(json.dumps(analyze(args.database), indent=2) + "\n")


if __name__ == "__main__":
    main()
