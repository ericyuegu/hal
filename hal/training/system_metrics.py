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
_CGROUP_V1_KEYS: Final[dict[str, str]] = {
    "rss": "anon",
    "cache": "file",
    "inactive_file": "inactive_file",
    "active_file": "active_file",
    "dirty": "file_dirty",
    "shmem": "shmem",
    "kernel_stack": "kernel",
    "page_tables": "pagetables",
}
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


def _resolve_cgroup_v2_root(cgroup_root: Path, cgroup_file: Path) -> Path | None:
    if (cgroup_root / "memory.current").is_file():
        return cgroup_root
    try:
        lines = cgroup_file.read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        hierarchy, controllers, relative = line.split(":", maxsplit=2)
        if hierarchy == "0" and not controllers:
            candidate = cgroup_root / relative.lstrip("/")
            if (candidate / "memory.current").is_file():
                return candidate
    return None


def _resolve_cgroup_v1_root(cgroup_root: Path, cgroup_file: Path) -> Path | None:
    candidates: list[Path] = []
    try:
        lines = cgroup_file.read_text().splitlines()
    except OSError:
        lines = []
    for line in lines:
        _, controllers, relative = line.split(":", maxsplit=2)
        if "memory" not in controllers.split(","):
            continue
        relative = relative.lstrip("/")
        if relative:
            candidates.extend((cgroup_root / "memory" / relative, cgroup_root / relative))
    # A cgroup namespace can expose the container cgroup directly at the memory
    # controller mount root, even though /proc/self/cgroup retains the host path.
    candidates.extend((cgroup_root / "memory", cgroup_root))
    return next((path for path in candidates if (path / "memory.usage_in_bytes").is_file()), None)


def read_cgroup_memory(
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    cgroup_file: Path = Path("/proc/self/cgroup"),
) -> dict[str, float]:
    """Return byte-valued cgroup-v1 or cgroup-v2 memory counters as GiB metrics."""
    metrics: dict[str, float] = {}
    resolved = _resolve_cgroup_v2_root(cgroup_root, cgroup_file)
    if resolved is not None:
        current = _read_int(resolved / "memory.current")
        peak = _read_int(resolved / "memory.peak")
        limit = _read_int(resolved / "memory.max")
        memory_stat = _read_keyed_ints(resolved / "memory.stat")
        normalized_stat = {name: memory_stat[name] for name in _CGROUP_MEMORY_KEYS if name in memory_stat}
        metrics["system/cgroup/version"] = 2.0
    else:
        resolved = _resolve_cgroup_v1_root(cgroup_root, cgroup_file)
        if resolved is None:
            return metrics
        current = _read_int(resolved / "memory.usage_in_bytes")
        peak = _read_int(resolved / "memory.max_usage_in_bytes")
        limit = _read_int(resolved / "memory.limit_in_bytes")
        memory_stat = _read_keyed_ints(resolved / "memory.stat")
        normalized_stat = {
            target: memory_stat[source] for source, target in _CGROUP_V1_KEYS.items() if source in memory_stat
        }
        metrics["system/cgroup/version"] = 1.0
    if current is not None:
        metrics["system/cgroup/current_gib"] = current / _GIB
    if peak is not None:
        metrics["system/cgroup/peak_gib"] = peak / _GIB
    if limit is not None:
        metrics["system/cgroup/limit_gib"] = limit / _GIB
        if current is not None and limit > 0:
            metrics["system/cgroup/usage_fraction"] = current / limit
        if peak is not None and limit > 0:
            metrics["system/cgroup/peak_usage_fraction"] = peak / limit
    for name, value in normalized_stat.items():
        metrics[f"system/cgroup/{name}_gib"] = value / _GIB
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


def _read_process_smaps(process_root: Path) -> dict[str, int]:
    values = _read_smaps_rollup(process_root / "smaps_rollup")
    if values:
        return values
    try:
        lines = (process_root / "smaps").read_text().splitlines()
    except OSError:
        return {}
    totals = {name: 0 for name in _SMAPS_KEYS}
    found = False
    for line in lines:
        fields = line.split()
        if len(fields) < 2:
            continue
        name = fields[0].rstrip(":")
        if name not in totals:
            continue
        try:
            totals[name] += int(fields[1]) * _KIB
            found = True
        except ValueError:
            continue
    return totals if found else {}


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
        values = _read_process_smaps(proc_root / str(pid))
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


def read_system_counters(
    *,
    proc_stat: Path = Path("/proc/stat"),
    proc_diskstats: Path = Path("/proc/diskstats"),
    proc_net_dev: Path = Path("/proc/net/dev"),
    proc_vmstat: Path = Path("/proc/vmstat"),
    proc_meminfo: Path = Path("/proc/meminfo"),
    sys_block: Path = Path("/sys/block"),
) -> tuple[dict[str, int], dict[str, float]]:
    """Read monotonic host counters plus instantaneous pinned-memory use."""
    counters: dict[str, int] = {}
    gauges: dict[str, float] = {}
    try:
        cpu = proc_stat.read_text().splitlines()[0].split()
        if cpu and cpu[0] == "cpu":
            values = [int(value) for value in cpu[1:]]
            idle = sum(values[index] for index in (3, 4) if index < len(values))
            counters["cpu_total"] = sum(values)
            counters["cpu_busy"] = sum(values) - idle
    except OSError, ValueError, IndexError:
        pass

    try:
        devices = {path.name for path in sys_block.iterdir()}
    except OSError:
        devices = set()
    try:
        disk_values = [0, 0, 0, 0]
        for line in proc_diskstats.read_text().splitlines():
            fields = line.split()
            if len(fields) < 14 or (devices and fields[2] not in devices):
                continue
            disk_values[0] += int(fields[3])
            disk_values[1] += int(fields[5]) * 512
            disk_values[2] += int(fields[7])
            disk_values[3] += int(fields[9]) * 512
        for name, value in zip(
            ("disk_reads", "disk_read_bytes", "disk_writes", "disk_write_bytes"),
            disk_values,
            strict=True,
        ):
            counters[name] = value
    except OSError, ValueError:
        pass

    try:
        received = transmitted = 0
        for line in proc_net_dev.read_text().splitlines()[2:]:
            interface, separator, payload = line.partition(":")
            if not separator or interface.strip() == "lo":
                continue
            values = payload.split()
            received += int(values[0])
            transmitted += int(values[8])
        counters["network_receive_bytes"] = received
        counters["network_transmit_bytes"] = transmitted
    except OSError, ValueError, IndexError:
        pass

    vm = _read_keyed_ints(proc_vmstat)
    if "pgfault" in vm:
        counters["page_faults"] = vm["pgfault"]
    if "pgmajfault" in vm:
        counters["major_page_faults"] = vm["pgmajfault"]
    memory = _read_keyed_ints(proc_meminfo)
    if "Mlocked" in memory:
        gauges["system/pinned_memory_gib"] = memory["Mlocked"] * _KIB / _GIB
    elif "Unevictable" in memory:
        gauges["system/pinned_memory_gib"] = memory["Unevictable"] * _KIB / _GIB
    return counters, gauges


def system_counter_rates(current: dict[str, int], previous: dict[str, int], elapsed_s: float) -> dict[str, float]:
    """Convert two host counter snapshots into O51 resource rates."""
    if elapsed_s <= 0:
        raise ValueError("counter interval must be positive")

    def rate(name: str) -> float | None:
        if name not in current or name not in previous:
            return None
        return max(0, current[name] - previous[name]) / elapsed_s

    metrics: dict[str, float] = {}
    total_delta = rate("cpu_total")
    busy_delta = rate("cpu_busy")
    if total_delta is not None and busy_delta is not None and total_delta > 0:
        metrics["system/cpu/utilization"] = min(busy_delta / total_delta, 1.0)
    mappings = {
        "disk_read_bytes": ("system/disk/read_mib_s", 2**20),
        "disk_write_bytes": ("system/disk/write_mib_s", 2**20),
        "disk_reads": ("system/disk/read_iops", 1),
        "disk_writes": ("system/disk/write_iops", 1),
        "network_receive_bytes": ("system/network/read_mib_s", 2**20),
        "network_transmit_bytes": ("system/network/write_mib_s", 2**20),
        "page_faults": ("system/page_faults/s", 1),
        "major_page_faults": ("system/major_page_faults/s", 1),
    }
    for counter, (metric, divisor) in mappings.items():
        value = rate(counter)
        if value is not None:
            metrics[metric] = value / divisor
    return metrics


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
        self._last_counters: dict[str, int] = {}
        self._last_counter_at = 0.0

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
                counters, gauges = read_system_counters()
                sample.update(gauges)
                if self._last_counter_at:
                    sample.update(system_counter_rates(counters, self._last_counters, started - self._last_counter_at))
                self._last_counters = counters
                self._last_counter_at = started
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
