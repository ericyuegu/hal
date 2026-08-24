"""Low-overhead cgroup, process-tree, and dataset-cache telemetry."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Final

_GIB: Final[int] = 2**30
_KIB: Final[int] = 2**10
_CGROUP_MEMORY_KEYS: Final[tuple[str, ...]] = (
    "anon",
    "file",
    "inactive_file",
    "active_file",
    "file_dirty",
    "shmem",
    "kernel",
    "pagetables",
)
_SMAPS_KEYS: Final[tuple[str, ...]] = ("Rss", "Pss", "Private_Dirty", "Anonymous")


def _read_int(path: Path) -> int | None:
    try:
        value = path.read_text().strip()
    except OSError:
        return None
    if not value or value == "max":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _read_keyed_ints(path: Path) -> dict[str, int]:
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return {}
    values: dict[str, int] = {}
    for line in lines:
        fields = line.split()
        if len(fields) < 2:
            continue
        try:
            values[fields[0].rstrip(":")] = int(fields[1])
        except ValueError:
            continue
    return values


def _resolve_cgroup_root(cgroup_root: Path, cgroup_file: Path) -> Path:
    if (cgroup_root / "memory.current").is_file():
        return cgroup_root
    try:
        lines = cgroup_file.read_text().splitlines()
    except OSError:
        return cgroup_root
    for line in lines:
        hierarchy, controllers, relative = line.split(":", maxsplit=2)
        if hierarchy == "0" and not controllers:
            candidate = cgroup_root / relative.lstrip("/")
            if (candidate / "memory.current").is_file():
                return candidate
    return cgroup_root


def read_cgroup_memory(
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    cgroup_file: Path = Path("/proc/self/cgroup"),
) -> dict[str, float]:
    """Return byte-valued cgroup-v2 memory counters as GiB metrics."""
    cgroup_root = _resolve_cgroup_root(cgroup_root, cgroup_file)
    metrics: dict[str, float] = {}
    current = _read_int(cgroup_root / "memory.current")
    limit = _read_int(cgroup_root / "memory.max")
    if current is not None:
        metrics["system/cgroup/current_gib"] = current / _GIB
    if limit is not None:
        metrics["system/cgroup/limit_gib"] = limit / _GIB
        if current is not None and limit > 0:
            metrics["system/cgroup/usage_fraction"] = current / limit
    memory_stat = _read_keyed_ints(cgroup_root / "memory.stat")
    for name in _CGROUP_MEMORY_KEYS:
        if name in memory_stat:
            metrics[f"system/cgroup/{name}_gib"] = memory_stat[name] / _GIB
    return metrics


def _process_parent(proc_root: Path, pid: int) -> int | None:
    values = _read_keyed_ints(proc_root / str(pid) / "status")
    return values.get("PPid")


def process_tree_pids(root_pid: int, proc_root: Path = Path("/proc")) -> tuple[int, ...]:
    """Return the live descendants of ``root_pid``, including the root."""
    parents: dict[int, int] = {}
    try:
        entries = tuple(proc_root.iterdir())
    except OSError:
        return (root_pid,)
    for entry in entries:
        if not entry.name.isdecimal():
            continue
        pid = int(entry.name)
        parent = _process_parent(proc_root, pid)
        if parent is not None:
            parents[pid] = parent

    selected = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if pid not in selected and parent in selected:
                selected.add(pid)
                changed = True
    return tuple(sorted(selected))


def _read_smaps_rollup(path: Path) -> dict[str, int]:
    values = _read_keyed_ints(path)
    return {name: values[name] * _KIB for name in _SMAPS_KEYS if name in values}


def read_process_tree_memory(
    root_pid: int | None = None,
    proc_root: Path = Path("/proc"),
) -> dict[str, float]:
    """Aggregate PSS and related counters over one process tree."""
    root_pid = os.getpid() if root_pid is None else root_pid
    pids = process_tree_pids(root_pid, proc_root)
    totals = {name: 0 for name in _SMAPS_KEYS}
    observed = 0
    child_pss: list[int] = []
    root_pss = 0
    for pid in pids:
        values = _read_smaps_rollup(proc_root / str(pid) / "smaps_rollup")
        if not values:
            continue
        observed += 1
        for name, value in values.items():
            totals[name] += value
        pss = values.get("Pss", 0)
        if pid == root_pid:
            root_pss = pss
        else:
            child_pss.append(pss)

    metrics = {
        "system/process_tree/process_count": float(observed),
        "system/process_tree/rss_gib": totals["Rss"] / _GIB,
        "system/process_tree/pss_gib": totals["Pss"] / _GIB,
        "system/process_tree/private_dirty_gib": totals["Private_Dirty"] / _GIB,
        "system/process_tree/anonymous_gib": totals["Anonymous"] / _GIB,
        "system/process_tree/main_pss_gib": root_pss / _GIB,
    }
    if child_pss:
        metrics["system/process_tree/max_child_pss_gib"] = max(child_pss) / _GIB
    return metrics


def read_cache_usage(cache_roots: Sequence[Path]) -> dict[str, float]:
    """Measure apparent and allocated bytes below the configured cache roots."""
    apparent_bytes = 0
    allocated_bytes = 0
    file_count = 0
    for root in cache_roots:
        for directory, _, filenames in os.walk(root):
            directory_path = Path(directory)
            for filename in filenames:
                try:
                    stat = (directory_path / filename).stat()
                except OSError:
                    continue
                apparent_bytes += stat.st_size
                allocated_bytes += stat.st_blocks * 512
                file_count += 1
    return {
        "system/cache/apparent_gib": apparent_bytes / _GIB,
        "system/cache/allocated_gib": allocated_bytes / _GIB,
        "system/cache/file_count": float(file_count),
    }


class HostMetricsSampler:
    """Collect host metrics off the training thread and expose the latest sample."""

    def __init__(
        self,
        cache_roots: Sequence[Path],
        *,
        interval_s: float = 5.0,
        process_interval_s: float = 30.0,
        cache_interval_s: float = 30.0,
    ) -> None:
        if interval_s <= 0 or process_interval_s <= 0 or cache_interval_s <= 0:
            raise ValueError("telemetry intervals must be positive")
        self._cache_roots = tuple(cache_roots)
        self._interval_s = interval_s
        self._process_interval_s = process_interval_s
        self._cache_interval_s = cache_interval_s
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: dict[str, float] = {}
        self._sampled_at = 0.0
        self._errors = 0

    def start(self) -> None:
        """Start one daemon sampling thread."""
        if self._thread is not None:
            raise RuntimeError("host metrics sampler is already started")
        self._thread = threading.Thread(target=self._run, name="host-metrics", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        next_process_sample = 0.0
        next_cache_sample = 0.0
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                sample = read_cgroup_memory()
                if started >= next_process_sample:
                    sample.update(read_process_tree_memory())
                    next_process_sample = started + self._process_interval_s
                if started >= next_cache_sample:
                    sample.update(read_cache_usage(self._cache_roots))
                    next_cache_sample = started + self._cache_interval_s
                with self._lock:
                    retained = {
                        name: value
                        for name, value in self._latest.items()
                        if name.startswith(("system/cache/", "system/process_tree/"))
                    }
                    self._latest = {**retained, **sample}
                    self._sampled_at = time.monotonic()
            except Exception:
                # Telemetry must not kill a training run. The error counter makes
                # persistent collection failures visible in W&B.
                with self._lock:
                    self._errors += 1
            elapsed = time.monotonic() - started
            self._stop.wait(max(0.0, self._interval_s - elapsed))

    def snapshot(self) -> dict[str, float]:
        """Return the most recent complete sample without blocking on collection."""
        with self._lock:
            values = dict(self._latest)
            sampled_at = self._sampled_at
            errors = self._errors
        if sampled_at:
            values["system/telemetry_age_s"] = max(0.0, time.monotonic() - sampled_at)
        values["system/telemetry_errors"] = float(errors)
        return values

    def close(self) -> None:
        """Stop the sampler and wait briefly for its thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 2 * self._interval_s))

    def __enter__(self) -> HostMetricsSampler:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
