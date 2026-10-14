import importlib.util
import sys
from pathlib import Path

import tyro

ROOT = Path(__file__).parents[1]


def _load(name: str, path: Path):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


WATCH = _load("hal_watch_062_evals", ROOT / "scripts" / "watch_062_evals.py")


def test_pending_evaluations_skip_finished_metrics_and_non_milestones() -> None:
    args = WATCH.Args(run="r", delay_frames=(2, 18))
    keys = {
        "latest.pt",
        "final.pt",
        "checkpoints/step-0016384.pt",
        "checkpoints/step-0008192.pt",
        "eval96-step-0008192-d2r2/metrics.json",
        "eval96-step-0016384-d18r2/match_rows.json",
    }
    assert WATCH.pending_evaluations(args, keys) == [(8192, 18), (16384, 2), (16384, 18)]


def test_eval_command_parses_as_the_experiment_eval_variant() -> None:
    experiment = _load("hal_watch_062_experiment", ROOT / WATCH.EXPERIMENT)
    args = WATCH.Args(run="run name")
    command = WATCH.eval_command(args, 8192, 18)
    assert command[:4] == ["uv", "run", WATCH.EXPERIMENT, "eval"]
    parsed = tyro.cli(experiment.Command, args=command[3:])
    assert isinstance(parsed, experiment.EvalArgs)
    assert (parsed.run, parsed.checkpoint) == ("run name", "checkpoints/step-0008192.pt")
    assert (parsed.delay_frames, parsed.replan_interval_frames, parsed.n_matchups) == (18, 2, 96)
    assert parsed.shared_wandb and parsed.wandb_namespace == "eval_d18r2"
    assert parsed.output_name == "eval96-step-0008192-d18r2"
