"""Tests for host telemetry used by long-running trainers."""

from pathlib import Path

import pytest

from hal.training import system_metrics


def _write_proc_process(
    root: Path,
    pid: int,
    parent: int,
    *,
    pss_kib: int,
    rollup: bool = True,
) -> None:
    process = root / str(pid)
    process.mkdir()
    (process / "status").write_text(f"Name:\ttest\nPPid:\t{parent}\n")
    smaps_name = "smaps_rollup" if rollup else "smaps"
    (process / smaps_name).write_text(
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


def test_read_cgroup_memory_supports_namespaced_v1_mount(tmp_path: Path) -> None:
    memory_root = tmp_path / "memory"
    memory_root.mkdir()
    (memory_root / "memory.usage_in_bytes").write_text(str(6 * 2**30))
    (memory_root / "memory.limit_in_bytes").write_text(str(24 * 2**30))
    (memory_root / "memory.stat").write_text(f"cache {4 * 2**30}\nrss {2 * 2**30}\ninactive_file {3 * 2**30}\n")
    cgroup_file = tmp_path / "self.cgroup"
    cgroup_file.write_text("6:memory:/host/container-id\n")

    metrics = system_metrics.read_cgroup_memory(tmp_path, cgroup_file)

    assert metrics["system/cgroup/version"] == 1.0
    assert metrics["system/cgroup/current_gib"] == 6.0
    assert metrics["system/cgroup/limit_gib"] == 24.0
    assert metrics["system/cgroup/usage_fraction"] == 0.25
    assert metrics["system/cgroup/anon_gib"] == 2.0
    assert metrics["system/cgroup/file_gib"] == 4.0


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


def test_read_process_tree_memory_falls_back_to_smaps(tmp_path: Path) -> None:
    _write_proc_process(tmp_path, 10, 1, pss_kib=1024, rollup=False)
    _write_proc_process(tmp_path, 11, 10, pss_kib=2048, rollup=False)

    metrics = system_metrics.read_process_tree_memory(10, tmp_path)

    assert metrics["system/process_tree/process_count"] == 2.0
    assert metrics["system/process_tree/pss_gib"] == pytest.approx(3 / 1024)


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


def test_read_system_counters_covers_cpu_disk_network_faults_and_pinned_memory(tmp_path: Path) -> None:
    proc_stat = tmp_path / "stat"
    proc_stat.write_text("cpu  100 10 20 400 30 5 3 2\n")
    proc_diskstats = tmp_path / "diskstats"
    proc_diskstats.write_text(
        "259 0 nvme0n1 11 0 101 0 13 0 103 0 0 0 0 0\n259 1 nvme0n1p1 7 0 70 0 9 0 90 0 0 0 0 0\n"
    )
    proc_net_dev = tmp_path / "net_dev"
    proc_net_dev.write_text(
        "Inter-| Receive | Transmit\n"
        " face |bytes packets errs drop fifo frame compressed multicast|bytes packets errs drop fifo colls carrier compressed\n"
        " lo: 100 0 0 0 0 0 0 0 200 0 0 0 0 0 0 0\n"
        " eth0: 300 0 0 0 0 0 0 0 500 0 0 0 0 0 0 0\n"
    )
    proc_vmstat = tmp_path / "vmstat"
    proc_vmstat.write_text("pgfault 1200\npgmajfault 12\n")
    proc_meminfo = tmp_path / "meminfo"
    proc_meminfo.write_text("Mlocked: 1048576 kB\nUnevictable: 2097152 kB\n")
    sys_block = tmp_path / "block"
    (sys_block / "nvme0n1").mkdir(parents=True)

    counters, gauges = system_metrics.read_system_counters(
        proc_stat=proc_stat,
        proc_diskstats=proc_diskstats,
        proc_net_dev=proc_net_dev,
        proc_vmstat=proc_vmstat,
        proc_meminfo=proc_meminfo,
        sys_block=sys_block,
    )

    assert counters == {
        "cpu_total": 570,
        "cpu_busy": 140,
        "disk_reads": 11,
        "disk_read_bytes": 101 * 512,
        "disk_writes": 13,
        "disk_write_bytes": 103 * 512,
        "network_receive_bytes": 300,
        "network_transmit_bytes": 500,
        "page_faults": 1200,
        "major_page_faults": 12,
    }
    assert gauges["system/pinned_memory_gib"] == 1.0


def test_system_counter_rates_are_clamped_and_unit_scaled() -> None:
    previous = {
        "cpu_total": 100,
        "cpu_busy": 30,
        "disk_read_bytes": 2**20,
        "disk_write_bytes": 4 * 2**20,
        "disk_reads": 10,
        "disk_writes": 20,
        "network_receive_bytes": 3 * 2**20,
        "network_transmit_bytes": 5 * 2**20,
        "page_faults": 100,
        "major_page_faults": 10,
    }
    current = {
        "cpu_total": 200,
        "cpu_busy": 70,
        "disk_read_bytes": 5 * 2**20,
        "disk_write_bytes": 2 * 2**20,
        "disk_reads": 18,
        "disk_writes": 30,
        "network_receive_bytes": 7 * 2**20,
        "network_transmit_bytes": 11 * 2**20,
        "page_faults": 140,
        "major_page_faults": 14,
    }

    metrics = system_metrics.system_counter_rates(current, previous, 2.0)

    assert metrics["system/cpu/utilization"] == 0.4
    assert metrics["system/disk/read_mib_s"] == 2.0
    assert metrics["system/disk/write_mib_s"] == 0.0
    assert metrics["system/disk/read_iops"] == 4.0
    assert metrics["system/disk/write_iops"] == 5.0
    assert metrics["system/network/read_mib_s"] == 2.0
    assert metrics["system/network/write_mib_s"] == 3.0
    assert metrics["system/page_faults/s"] == 20.0
    assert metrics["system/major_page_faults/s"] == 2.0
    with pytest.raises(ValueError, match="positive"):
        system_metrics.system_counter_rates(current, previous, 0.0)
