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


def test_pilot_uses_separate_evaluation_and_deterministic_training(tmp_path: Path) -> None:
    command = _MODULE.training_command(
        tmp_path, tmp_path / "run", (0, 1), matches=1, groups=2, updates=4, checkpoint_every=1, warmup=0, verify=True
    )
    assert command[command.index("--evaluation-every") + 1] == "0"
    assert "--training.deterministic" in command
    assert "--verify-resume" in command
    evaluation = _MODULE.evaluation_command(
        tmp_path, tmp_path / "model", tmp_path / "protocol", tmp_path / "eval", held_out=False, simulators=("msl",)
    )
    assert evaluation[evaluation.index("--simulators") + 1 :] == ["msl"]
    assert evaluation[evaluation.index("--anchors") + 1] == str(tmp_path / "payloads/final.pt")


def test_pilot_logs_existing_interval_names(tmp_path: Path) -> None:
    interval = {"mean": 0.6, "low": 0.5, "high": 0.7}
    (tmp_path / "results.json").write_text(
        json.dumps(
            {
                "format": "hal.o65.evaluation-results.v1",
                "groups": [
                    {
                        "metrics": {
                            "msl": {
                                "net_stocks_per_min": interval,
                                "win_rate": interval,
                                "matches": 4,
                                "completed_pairs": 2,
                                "net_stock_lcb": 0.2,
                            }
                        }
                    }
                ],
            }
        )
    )
    _MODULE.log_evaluation(tmp_path, 32, {}, online=False)
    metrics = json.loads((tmp_path / "wandb-metrics.json").read_text())
    assert metrics["evaluation/msl_cell0_win_rate_low"] == 0.5
    assert metrics["evaluation/msl_cell0_net_stocks_per_min_high"] == 0.7
