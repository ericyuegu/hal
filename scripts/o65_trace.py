"""Summarize one O65 profiler trace: a tick timeline and a kernel breakdown per graph.

The rollout profile annotates each tick graph launch as ``o65_graph/...``. This script maps every
GPU kernel to the annotation around its launch, prints when each graph ran and how long the GPU
sat idle, then groups each graph's kernels by type and lists the most expensive ones.
"""

import collections
import dataclasses
import json
import re
from pathlib import Path

import tyro

KERNEL_TYPES = (
    ("gemm", "gemm|cutlass|nvjet|xmma|cublas"),
    ("attention", "flash|flex|fmha|attention"),
    ("triton reduction", "red_fused|per_fused"),
    ("triton pointwise", "poi_fused"),
    ("index or scatter", "index|scatter|gather"),
)


@dataclasses.dataclass(frozen=True, slots=True)
class Args:
    trace: tyro.conf.Positional[Path]
    # Print the kernel breakdown only for annotations containing this text.
    graph: str = ""
    top: int = 20


def kernel_type(name: str) -> str:
    lowered = name.lower()
    for label, pattern in KERNEL_TYPES:
        if re.search(pattern, lowered):
            return label
    return "other"


def main(args: Args) -> None:
    events = json.loads(args.trace.read_text())["traceEvents"]
    gpu = [event for event in events if event.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    annotations = sorted(
        (event for event in events if event.get("cat") == "user_annotation"), key=lambda event: event["ts"]
    )
    origin = min(event["ts"] for event in gpu)

    def enclosing(ts: float) -> str:
        best = None
        for annotation in annotations:
            inside = annotation["ts"] <= ts <= annotation["ts"] + annotation["dur"]
            if inside and (best is None or annotation["dur"] < best["dur"]):
                best = annotation
        return "eager" if best is None else best["name"]

    launches = {}
    for event in events:
        if event.get("cat") == "cuda_runtime":
            graph = event["name"] == "cudaGraphLaunch"
            launches[event["args"].get("correlation")] = (enclosing(event["ts"]) if graph else "eager", graph)
    for annotation in annotations:
        print(
            f"host {annotation['name'][:56]:56s} {(annotation['ts'] - origin) / 1e3:8.2f} +{annotation['dur'] / 1e3:6.2f} ms"
        )
    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for event in gpu:
        label, graph = launches.get(event["args"].get("correlation"), ("eager", False))
        groups[label if graph else ("copies" if event["cat"] != "kernel" else "eager kernels")].append(event)
    for label, kernels in sorted(groups.items(), key=lambda item: min(event["ts"] for event in item[1])):
        start = min(event["ts"] for event in kernels) - origin
        end = max(event["ts"] + event["dur"] for event in kernels) - origin
        busy = sum(event["dur"] for event in kernels)
        print(
            f"gpu  {label[:56]:56s} {start / 1e3:8.2f} .. {end / 1e3:8.2f} ms busy {busy / 1e3:6.2f} n={len(kernels)}"
        )
    intervals = sorted((event["ts"], event["ts"] + event["dur"]) for event in gpu)
    idle, cursor, gaps = 0.0, intervals[0][1], []
    for start, end in intervals[1:]:
        if start > cursor:
            idle += start - cursor
            if start - cursor > 500:
                gaps.append(((cursor - origin) / 1e3, (start - cursor) / 1e3))
        cursor = max(cursor, end)
    print(f"span {(cursor - origin) / 1e3:.2f} ms idle {idle / 1e3:.2f} ms; gaps over 0.5 ms (at, length):")
    print("  " + ", ".join(f"{at:.1f}+{length:.2f}" for at, length in gaps))
    for label, kernels in groups.items():
        if args.graph not in label or not label.startswith("o65_graph"):
            continue
        busy = sum(event["dur"] for event in kernels)
        print(f"\n{label}: {len(kernels)} kernels, {busy / 1e3:.2f} ms busy")
        totals: dict[str, list[float]] = collections.defaultdict(lambda: [0, 0.0])
        names: dict[str, list[float]] = collections.defaultdict(lambda: [0, 0.0])
        for event in kernels:
            for table, key in ((totals, kernel_type(event["name"])), (names, event["name"][:100])):
                table[key][0] += 1
                table[key][1] += event["dur"]
        for key, (count, total) in sorted(totals.items(), key=lambda item: -item[1][1]):
            print(
                f"  {total / 1e3:6.2f} ms {100 * total / busy:5.1f}% n={count:4.0f} mean {total / count:6.1f} us {key}"
            )
        for key, (count, total) in sorted(names.items(), key=lambda item: -item[1][1])[: args.top]:
            print(f"  {total / 1e3:6.2f} ms n={count:3.0f} mean {total / count:6.1f} us {key}")


if __name__ == "__main__":
    main(tyro.cli(Args))
