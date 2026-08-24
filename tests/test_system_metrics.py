"""Tests for host telemetry used by long-running trainers."""

from pathlib import Path

import pytest

from hal.training import system_metrics


def _write_proc_process(root: Path, pid: int, parent: int, *, pss_kib: int) -> None:
    process = root / str(pid)
    process.mkdir()
    (process / "status").write_text(f"Name:\ttest\nPPid:\t{parent}\n")
    (process / "smaps_rollup").write_text(
        f"Rss: {2 * pss_kib} kB\nPss: {pss_kib} kB\nPrivate_Dirty: {pss_kib // 2} kB\nAnonymous: {pss_kib // 4} kB\n"
    )


def test_read_cgroup_memory_reports_absolute_bytes_and_fraction(tmp_path: Path) -> None:
    (tmp_path / "memory.current").write_text(str(3 * 2**30))
    (tmp_path / "memory.max").write_text(str(12 * 2**30))
    (tmp_path / "memory.stat").write_text(f"anon {2**30}\nfile {2 * 2**30}\ninactive_file {2**29}\nfile_dirty 4096\n")

    metrics = system_metrics.read_cgroup_memory(tmp_path)

    assert metrics["system/cgroup/current_gib"] == 3.0
    assert metrics["system/cgroup/limit_gib"] == 12.0
    assert metrics["system/cgroup/usage_fraction"] == 0.25
    assert metrics["system/cgroup/anon_gib"] == 1.0
    assert metrics["system/cgroup/file_gib"] == 2.0


def test_read_cgroup_memory_resolves_nested_v2_path(tmp_path: Path) -> None:
    cgroup_root = tmp_path / "cgroup"
    nested = cgroup_root / "jobs" / "trainer"
    nested.mkdir(parents=True)
    (nested / "memory.current").write_text(str(5 * 2**30))
    (nested / "memory.max").write_text("max")
    (nested / "memory.stat").write_text(f"anon {3 * 2**30}\nfile {2 * 2**30}\n")
    cgroup_file = tmp_path / "self.cgroup"
    cgroup_file.write_text("0::/jobs/trainer\n")

    metrics = system_metrics.read_cgroup_memory(cgroup_root, cgroup_file)

    assert metrics["system/cgroup/current_gib"] == 5.0
    assert "system/cgroup/limit_gib" not in metrics
    assert metrics["system/cgroup/anon_gib"] == 3.0


def test_read_process_tree_memory_excludes_unrelated_processes(tmp_path: Path) -> None:
    _write_proc_process(tmp_path, 10, 1, pss_kib=1024)
    _write_proc_process(tmp_path, 11, 10, pss_kib=2048)
    _write_proc_process(tmp_path, 12, 11, pss_kib=4096)
    _write_proc_process(tmp_path, 20, 1, pss_kib=8192)

    metrics = system_metrics.read_process_tree_memory(10, tmp_path)

    assert system_metrics.process_tree_pids(10, tmp_path) == (10, 11, 12)
    assert metrics["system/process_tree/process_count"] == 3.0
    assert metrics["system/process_tree/pss_gib"] == pytest.approx(7 / 1024)
    assert metrics["system/process_tree/main_pss_gib"] == pytest.approx(1 / 1024)
    assert metrics["system/process_tree/max_child_pss_gib"] == pytest.approx(4 / 1024)


def test_read_cache_usage_sums_multiple_roots(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "a.bin").write_bytes(b"a" * 1024)
    (second / "b.bin").write_bytes(b"b" * 2048)

    metrics = system_metrics.read_cache_usage((first, second))

    assert metrics["system/cache/apparent_gib"] == pytest.approx(3072 / 2**30)
    assert metrics["system/cache/allocated_gib"] >= metrics["system/cache/apparent_gib"]
    assert metrics["system/cache/file_count"] == 2.0
