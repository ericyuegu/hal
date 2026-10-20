"""Attribute a complete Nsight O65 trace and export one sampled tick graph."""

import csv
import dataclasses
import json
import re
import sqlite3
from collections import Counter
from collections import defaultdict
from contextlib import closing
from pathlib import Path

import tyro

from hal.eval.self_play_checkpoint import sha256

PHASES = {"rollout", "reference", "preparation", "critic", "actor", "muon"}


@dataclasses.dataclass(frozen=True, slots=True)
class Range:
    start: int
    end: int
    tid: int
    name: str


@dataclasses.dataclass(frozen=True, slots=True)
class Api:
    start: int
    end: int
    tid: int
    name: str
    phase: str


def union_ns(intervals: list[tuple[int, int]]) -> int:
    total = end = 0
    for start, stop in sorted(intervals):
        if stop < start:
            raise ValueError("negative trace interval")
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def phase_at(timestamp: int, ranges: list[Range]) -> str:
    active = [span for span in ranges if span.start <= timestamp < span.end]
    if not active:
        return "outside"
    return min(active, key=lambda span: span.end - span.start).name.rsplit("/", 1)[1]


def tick_range(ranges: list[Range]) -> Range:
    choices = [(span, re.search(r"/B([0-9]+)/append2/R4$", span.name)) for span in ranges]
    batches = [(span, int(match.group(1))) for span, match in choices if match is not None]
    if not batches:
        raise ValueError("Nsight trace has no attributed ordinary R=4 tick")
    largest = max(batch for _, batch in batches)
    selected = sorted((span for span, batch in batches if batch == largest), key=lambda span: span.start)
    return selected[min(8, len(selected) - 1)]


def analyze(database: Path, output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=False)
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        strings = dict(db.execute("SELECT id, value FROM StringIds"))
        ranges = [
            Range(start, end, tid, text if text is not None else strings[text_id])
            for start, end, tid, text, text_id in db.execute(
                "SELECT start, end, globalTid, text, textId FROM NVTX_EVENTS WHERE end IS NOT NULL"
            )
        ]
        phases = [
            span for span in ranges if span.name in {f"o65/{name}" for name in PHASES} or span.name == "o65/actor/muon"
        ]
        names = {span.name.rsplit("/", 1)[1] for span in phases}
        if names != PHASES:
            raise ValueError(f"incomplete phase trace: {names}")
        tick = tick_range(ranges)
        apis: dict[tuple[int, int], Api] = {}
        tick_launches: set[tuple[int, int]] = set()
        api_times: dict[str, Counter[str]] = defaultdict(Counter)
        api_counts: dict[str, Counter[str]] = defaultdict(Counter)
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
            if table not in tables:
                continue
            for start, end, tid, name_id, correlation in db.execute(
                f"SELECT start, end, globalTid, nameId, correlationId FROM {table}"
            ):
                name = strings[name_id]
                phase = phase_at(start, phases)
                api = Api(start, end, tid, name, phase)
                key = (tid & 0xFFFFFF000000, correlation)
                # Runtime and driver records can describe the same call chain.
                if key not in apis or apis[key].start < start:
                    apis[key] = api
                api_times[phase][name] += end - start
                api_counts[phase][name] += 1
                if tid == tick.tid and tick.start <= start < tick.end and "GraphLaunch" in name:
                    tick_launches.add(key)
        if not tick_launches:
            raise ValueError("ordinary tick has no correlated CUDA graph launch")
        intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
        kernel_times: dict[str, Counter[str]] = defaultdict(Counter)
        counts: dict[str, Counter[str]] = defaultdict(Counter)
        copies: dict[str, int] = defaultdict(int)
        missing = Counter()
        tick_rows = []
        query = """SELECT start, end, globalPid, correlationId, demangledName,
                          gridX, gridY, gridZ, blockX, blockY, blockZ, graphId, graphNodeId
                   FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"""
        for start, end, pid, correlation, name_id, gx, gy, gz, bx, by, bz, graph, node in db.execute(query):
            key = (pid & 0xFFFFFF000000, correlation)
            api = apis.get(key)
            if api is None:
                missing["kernels"] += 1
                continue
            phase = api.phase
            name = strings[name_id]
            intervals[phase].append((start, end))
            kernel_times[phase][name] += end - start
            counts[phase]["graph_kernels" if graph else "eager_kernels"] += 1
            if key in tick_launches:
                tick_rows.append(
                    {
                        "name": name,
                        "duration_ns": end - start,
                        "start_ns": start,
                        "grid_x": gx,
                        "grid_y": gy,
                        "grid_z": gz,
                        "block_x": bx,
                        "block_y": by,
                        "block_z": bz,
                        "graph_id": graph,
                        "graph_node_id": node,
                        "module": "experiments/065_selfplay_ppo.py:TransformerPolicy.sample_tick.tick",
                    }
                )
        for table in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
            if table not in tables:
                continue
            for start, end, pid, correlation, size in db.execute(
                f"SELECT start, end, globalPid, correlationId, bytes FROM {table}"
            ):
                api = apis.get((pid & 0xFFFFFF000000, correlation))
                if api is None:
                    missing[table] += 1
                    continue
                intervals[api.phase].append((start, end))
                copies[api.phase] += size
        if missing:
            raise ValueError(f"GPU work has no launching API; cannot attribute the complete trace: {missing}")
        if not tick_rows:
            raise ValueError("ordinary tick graph has no kernel nodes")
        with (output / "tick-kernels.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(tick_rows[0]))
            writer.writeheader()
            writer.writerows(tick_rows)
        result: dict[str, object] = {
            "format": "hal.o65.floor-trace.v1",
            "database_sha256": sha256(database),
            "tick_range": dataclasses.asdict(tick),
            "tick_kernel_count": len(tick_rows),
        }
        summary = {}
        for phase in sorted(PHASES | {"outside"}):
            wall = sum(span.end - span.start for span in phases if span.name.rsplit("/", 1)[1] == phase)
            if phase == "actor":
                wall -= sum(span.end - span.start for span in phases if span.name.endswith("/muon"))
            if phase == "preparation":
                wall -= sum(span.end - span.start for span in phases if span.name.endswith("/reference"))
            summary[phase] = {
                "nvtx_host_seconds": wall / 1e9,
                "gpu_busy_seconds": union_ns(intervals[phase]) / 1e9,
                "kernel_seconds": sum(kernel_times[phase].values()) / 1e9,
                "counts": dict(counts[phase]),
                "copy_and_memset_bytes": copies[phase],
                "api_seconds": {name: value / 1e9 for name, value in api_times[phase].most_common()},
                "api_counts": dict(api_counts[phase]),
                "kernel_seconds_by_name": {name: value / 1e9 for name, value in kernel_times[phase].most_common()},
            }
        result["phases"] = summary
        result["gpu_busy_seconds"] = union_ns([span for spans in intervals.values() for span in spans]) / 1e9
        (output / "trace.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        return result


@dataclasses.dataclass(frozen=True, slots=True)
class Args:
    database: Path
    output: Path


if __name__ == "__main__":
    args = tyro.cli(Args)
    analyze(args.database, args.output)
