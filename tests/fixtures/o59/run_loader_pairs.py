"""Run three alternating, reduced-geometry 059 loader measurements."""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from dataclasses import asdict
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Final
from typing import cast

CONTROL_SHA: Final[str] = "d7454f9d1a136f7c745d2d4478af22e19cd65faf"
SOURCE: Final[str] = "professional-aklo-policy-world-v8"
SELECTION_SHA: Final[str] = "5b7c0e8e34d0502f3284c57060db9a1dac10f7129344715a7b5acea209ac029e"
WARM_BATCHES: Final[int] = 200
MEASURED_BATCHES: Final[int] = 500
BATCH_SIZE: Final[int] = 64
REPLAY_SLOTS: Final[int] = 1600
NUM_WORKERS: Final[int] = 2
CORPUS_INDEX: Final[Path] = Path("data/processed/professional/aklo/mds-policy-world-v8/train/index.json")
MANIFEST_SHA: Final[str] = "1ae04b2ffd57fe0bb1bac86933f61b4fbad151bfe7956f607f6ff19521895f64"


@dataclass(frozen=True, slots=True)
class Args:
    repo: Path
    control: Path
    output_root: Path
    rclone_config: Path


@dataclass(frozen=True, slots=True)
class PageCachePreparation:
    raw_shard_count: int
    raw_bytes: int
    seconds: float


def _prepare_page_cache(index_path: Path, control_index_path: Path, *, expected_sha256: str) -> PageCachePreparation:
    if not index_path.samefile(control_index_path):
        raise ValueError("control and candidate must use the same local corpus cache")
    payload = index_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("local corpus manifest differs from the pinned comparison")
    manifest = cast(dict[str, Any], json.loads(payload))
    started = time.perf_counter()
    total_bytes = 0
    shards = cast(list[dict[str, Any]], manifest["shards"])
    for shard in shards:
        raw = cast(dict[str, Any], shard["raw_data"])
        name = cast(str, raw["basename"])
        if Path(name).name != name:
            raise ValueError("raw shard name must stay inside the corpus directory")
        with (index_path.parent / name).open("rb") as handle:
            size = os.fstat(handle.fileno()).st_size
            if size != raw["bytes"]:
                raise ValueError(f"incomplete local raw shard: {name}")
            # This only advises eviction of the selected clean files. It does not
            # change their contents or drop unrelated system caches.
            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            total_bytes += size
    return PageCachePreparation(len(shards), total_bytes, time.perf_counter() - started)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _run_trial(
    args: Args,
    *,
    script: Path,
    kind: str,
    number: int,
    python: Path,
    temporary: Path,
) -> tuple[Path, dict[str, object]]:
    label = f"{kind}-{number}"
    output = args.output_root / f"{label}.json"
    log = args.output_root / f"{label}.log"
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "MKL_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "PYTHONPATH": str(args.control if kind == "control" else args.repo),
            "TMPDIR": str(temporary),
        }
    )
    command = [
        str(python),
        str(script),
        "--sources",
        SOURCE,
        "--warm-batches",
        str(WARM_BATCHES),
        "--measured-batches",
        str(MEASURED_BATCHES),
        "--batch-size",
        str(BATCH_SIZE),
        "--replay-slots",
        str(REPLAY_SLOTS),
        "--num-workers",
        str(NUM_WORKERS),
        "--rclone-config",
        str(args.rclone_config),
        "--label",
        label,
        "--output",
        str(output),
    ]
    cache = _prepare_page_cache(
        args.repo / CORPUS_INDEX,
        args.control / CORPUS_INDEX,
        expected_sha256=MANIFEST_SHA,
    )
    cache_path = args.output_root / f"{label}-cache.json"
    with cache_path.open("x") as handle:
        handle.write(json.dumps(asdict(cache), indent=2, sort_keys=True) + "\n")
    with log.open("x") as handle:
        subprocess.run(command, cwd=args.repo, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
    result = cast(dict[str, object], json.loads(output.read_text()))
    if (
        result["selection_sha256"] != SELECTION_SHA
        or result["sources"] != [SOURCE]
        or result["warm_batches"] != WARM_BATCHES
        or result["measured_batches"] != MEASURED_BATCHES
        or result["batch_size"] != BATCH_SIZE
        or result["replay_slots"] != REPLAY_SLOTS
        or result["num_workers"] != NUM_WORKERS
        or not cast(dict[str, object], result["memory_preflight"])["safe_to_start"]
    ):
        raise ValueError(f"{label}: measured geometry or corpus differs from the declared reduced comparison")
    print(f"{label}: {result['samples_per_second']:.1f} samples/s", flush=True)
    result["cache_preparation"] = asdict(cache)
    result["cache_record"] = str(cache_path)
    result["cache_record_sha256"] = _sha256(cache_path)
    return output, result


def main(args: Args) -> None:
    args = Args(
        args.repo.resolve(),
        args.control.resolve(),
        args.output_root.resolve(),
        args.rclone_config.resolve(),
    )
    repo = args.repo
    control = args.control
    if _git_sha(control) != CONTROL_SHA:
        raise ValueError(f"control checkout must be {CONTROL_SHA}")
    if not args.rclone_config.is_file():
        raise FileNotFoundError(args.rclone_config)
    if not (repo / "scripts/benchmark_replay_loader.py").is_file():
        raise FileNotFoundError(repo / "scripts/benchmark_replay_loader.py")
    python = repo / ".venv/bin/python"
    if not python.is_file():
        raise FileNotFoundError(python)
    args.output_root.mkdir(parents=True, exist_ok=False)
    temporary = args.output_root / "tmp"
    temporary.mkdir()
    control_script = args.output_root / "control-benchmark.py"
    subprocess.run(
        [
            str(python),
            str(repo / "tests/fixtures/o59/make_loader_control_benchmark.py"),
            "--source",
            str(repo / "scripts/benchmark_replay_loader.py"),
            "--output",
            str(control_script),
        ],
        cwd=repo,
        check=True,
    )
    candidate_benchmark_sha = _sha256(repo / "scripts/benchmark_replay_loader.py")
    control_benchmark_sha = _sha256(control_script)

    trials: list[dict[str, object]] = []
    paired_ratios: list[float] = []
    peak_memory_ratios: list[float] = []
    identities: dict[str, tuple[object, object]] = {}
    for number in range(1, 4):
        if (
            _sha256(repo / "scripts/benchmark_replay_loader.py") != candidate_benchmark_sha
            or _sha256(control_script) != control_benchmark_sha
        ):
            raise RuntimeError("common loader measurement source changed during the paired trials")
        control_path, control_result = _run_trial(
            args,
            script=control_script,
            kind="control",
            number=number,
            python=python,
            temporary=temporary,
        )
        candidate_path, candidate_result = _run_trial(
            args,
            script=repo / "scripts/benchmark_replay_loader.py",
            kind="candidate",
            number=number,
            python=python,
            temporary=temporary,
        )
        for key in (
            "protocol",
            "selection_sha256",
            "source_manifest_sha256",
            "mosaic_version",
            "torch_version",
            "python_version",
            "cpu_affinity",
            "cpu_model",
            "batch_array_bytes",
        ):
            if control_result[key] != candidate_result[key]:
                raise ValueError(f"pair {number}: {key} differs between control and candidate")
        for kind, path, result in (
            ("control", control_path, control_result),
            ("candidate", candidate_path, candidate_result),
        ):
            identity = result["loader_sha256"], result["source_manifest_sha256"]
            if identities.setdefault(kind, identity) != identity:
                raise RuntimeError(f"{kind} loader code or corpus manifest changed between trials")
            trials.append(
                {
                    "kind": kind,
                    "pair": number,
                    "path": str(path),
                    "sha256": _sha256(path),
                    "samples_per_second": result["samples_per_second"],
                    "startup_seconds_total": result["startup_seconds_total"],
                    "measurement_cpu_seconds": result["measurement_cpu_seconds"],
                    "measurement_disk_read_bytes": result["measurement_disk_read_bytes"],
                    "measurement_disk_write_bytes": result["measurement_disk_write_bytes"],
                    "process_tree_peak_rss_upper_bound_bytes": result["process_tree_peak_rss_upper_bound_bytes"],
                    "cache_preparation": result["cache_preparation"],
                    "cache_record": result["cache_record"],
                    "cache_record_sha256": result["cache_record_sha256"],
                }
            )
        paired_ratios.append(
            cast(float, candidate_result["samples_per_second"]) / cast(float, control_result["samples_per_second"])
        )
        control_peak = control_result["process_tree_peak_rss_upper_bound_bytes"]
        candidate_peak = candidate_result["process_tree_peak_rss_upper_bound_bytes"]
        if (
            not isinstance(control_peak, int)
            or not isinstance(candidate_peak, int)
            or min(control_peak, candidate_peak) <= 0
        ):
            raise ValueError(
                "peak memory is unavailable; the saved throughput measurements do not qualify memory parity"
            )
        peak_memory_ratios.append(candidate_peak / control_peak)

    report = {
        "scope": "matched reduced loader-core comparison; not production 131072-slot or transformed-training parity",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "control_sha": CONTROL_SHA,
        "candidate_head_sha": _git_sha(repo),
        "source": SOURCE,
        "selection_sha256": SELECTION_SHA,
        "source_manifest_sha256": control_result["source_manifest_sha256"],
        "geometry": {
            "warm_batches": WARM_BATCHES,
            "measured_batches": MEASURED_BATCHES,
            "batch_size": BATCH_SIZE,
            "replay_slots": REPLAY_SLOTS,
            "num_workers": NUM_WORKERS,
        },
        "cache_protocol": "complete shared raw cache; POSIX_FADV_DONTNEED on each selected shard before each trial, then 200 warm batches",
        "candidate_benchmark_sha256": candidate_benchmark_sha,
        "control_benchmark_sha256": control_benchmark_sha,
        "trials": trials,
        "paired_throughput_ratios": paired_ratios,
        "median_paired_throughput_ratio": statistics.median(paired_ratios),
        "paired_peak_memory_ratios": peak_memory_ratios,
        "maximum_paired_peak_memory_ratio": max(peak_memory_ratios),
    }
    with (args.output_root / "summary.json").open("x") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--control", type=Path, default=Path("/tmp/hal-059-control-d7454f9"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rclone-config", type=Path, default=Path.home() / ".config/rclone/rclone.conf")
    values = parser.parse_args()
    main(Args(values.repo, values.control, values.output_root, values.rclone_config))
