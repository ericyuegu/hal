"""Compiler-cache persistence for short O59 benchmark iterations."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts/benchmark_o59_modal.py"
_SPEC = importlib.util.spec_from_file_location("o59_benchmark_launcher_test", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
launcher = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = launcher
_SPEC.loader.exec_module(launcher)


def test_compiler_cache_round_trip_and_corruption_rejection(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "kernel.py").write_text("kernel source\n")
    (cache / "triton").mkdir()
    (cache / "triton" / "kernel.cubin").write_bytes(b"compiled kernel")
    saved = tmp_path / "saved"
    saved.mkdir()
    launcher.save_compiler_cache(cache, saved)
    restored = tmp_path / "restored"
    launcher.restore_compiler_cache(saved, restored)
    assert (restored / "kernel.py").read_text() == "kernel source\n"
    assert (restored / "triton" / "kernel.cubin").read_bytes() == b"compiled kernel"
    with (saved / "compiler-cache.tar.gz").open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        launcher.restore_compiler_cache(saved, tmp_path / "corrupt")
    assert not (tmp_path / "corrupt").exists()


def test_compiler_cache_rejects_unknown_format(tmp_path: Path) -> None:
    (tmp_path / "compiler-cache.json").write_text(json.dumps({"format": "unknown"}))
    with pytest.raises(ValueError, match="unsupported compiler cache"):
        launcher.restore_compiler_cache(tmp_path, tmp_path / "restored")


@pytest.mark.parametrize("seconds", [0, 6901])
def test_invalid_budget_fails_before_launch(seconds: int) -> None:
    with pytest.raises(ValueError, match="budget_seconds"):
        launcher.main(launcher.Args(budget_seconds=seconds))
