import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location("o65_pilot", Path(__file__).parents[1] / "experiments/o65/pilot.py")
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_kill_waits_for_published_boundary(tmp_path: Path) -> None:
    manifest = tmp_path / "checkpoint.json"
    child = tmp_path / "child.py"
    child.write_text(
        "import json,os,signal,sys\n"
        "from pathlib import Path\n"
        "path=Path(sys.argv[1])\n"
        "temp=path.with_suffix('.partial')\n"
        "temp.write_text(json.dumps({'pid':os.getpid()}))\n"
        "temp.replace(path)\n"
        "signal.pause()\n"
    )
    _MODULE.kill_at_checkpoint(
        [sys.executable, str(child), str(manifest)], manifest, tmp_path / "child.log", dict(os.environ)
    )
    with pytest.raises(ProcessLookupError):
        os.kill(json.loads(manifest.read_text())["pid"], 0)


def test_exit_before_checkpoint_fails(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="before the kill boundary"):
        _MODULE.kill_at_checkpoint(
            [sys.executable, "-c", "raise SystemExit(3)"],
            tmp_path / "absent.json",
            tmp_path / "child.log",
            dict(os.environ),
        )


def test_qualification_rejects_bad_scaling_and_memory() -> None:
    def row(rate: float, gib: int = 20) -> list[dict[str, float]]:
        return [{"systems/match_frames_per_second": rate, "systems/peak_reserved_bytes": float(gib * 2**30)}]

    _MODULE.qualify_pool([row(3000), row(5000), row(5100)])
    with pytest.raises(ValueError, match="over 10% slower"):
        _MODULE.qualify_pool([row(3000), row(5000), row(4000)])
    with pytest.raises(ValueError, match="80 GiB"):
        _MODULE.qualify_pool([row(3000), row(5000), row(5100, 80)])
    with pytest.raises(ValueError, match="finite positive"):
        _MODULE.qualify_pool([row(3000), row(5000), row(float("nan"))])
