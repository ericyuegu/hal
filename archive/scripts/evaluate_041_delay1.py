"""Evaluate one experiment-041 checkpoint at one-frame closed-loop delay.

The checkpoint is downloaded from R2 once, then evaluated in self-play and
against the level-9 CPU. Both protocols schedule 96 Dolphin boots in waves of
16 and upload their replay trees and machine-readable evidence to the source
run's R2 prefix.
"""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import sys
import tempfile
import time
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any
from typing import Final

import tyro

from hal.eval.cross_stage import match_rows
from hal.eval.cross_stage import vs_cpu_metrics
from hal.eval.harness import DEFAULT_START_RETRIES
from hal.eval.harness import default_session_cfg
from hal.eval.harness import run_matches_vec
from hal.eval.match_summary import summarize_trajectory
from hal.eval.self_play import DecodeTelemetry
from hal.eval.self_play import self_play_matches
from hal.sim.process_vec import ProcessVecTelemetry
from hal.training.checkpoints import BackgroundUploader
from hal.training.checkpoints import download_latest

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
EXPERIMENT: Final[Path] = ROOT / "experiments" / "041_architectural_stability.py"
CONTROL_DELAY: Final[int] = 1


@dataclass(frozen=True, slots=True)
class Args:
    run_name: str
    """Exact checkpoint run name under the R2 runs/ prefix."""
    boots: int = 96
    """Dolphin boots in each of the two evaluation protocols."""
    max_parallel: int = 16
    """Maximum concurrent Dolphin boots (the wave size)."""
    max_frames: int = 7200
    """Frame budget per boot, including countdown frames."""
    checkpoint_name: str = "latest.pt"
    """Checkpoint object within the run's R2 prefix."""
    artifact_label: str = "delay1_p16_b96"
    """Directory label for the uploaded evaluation artifacts."""


def _load_experiment() -> ModuleType:
    spec = importlib.util.spec_from_file_location("evaluate_exp041_delay1", EXPERIMENT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {EXPERIMENT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _validate_args(args: Args) -> None:
    for name in ("run_name", "checkpoint_name", "artifact_label"):
        value = getattr(args, name)
        if Path(value).name != value or value in ("", ".", ".."):
            raise SystemExit(f"--{name.replace('_', '-')} must be one directory-safe name")
    if not args.checkpoint_name.endswith(".pt"):
        raise SystemExit("--checkpoint-name must name a .pt object")
    if args.boots != 96:
        raise SystemExit("this evaluation protocol requires exactly 96 boots")
    if args.max_parallel != 16:
        raise SystemExit("this evaluation protocol requires waves of exactly 16 boots")
    if args.max_frames < 2:
        raise SystemExit("--max-frames must be at least 2")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


def _self_play_eval(
    exp: ModuleType,
    model: Any,
    stats: Any,
    cfg: Any,
    inference: Any,
    *,
    replay_dir: Path,
    args: Args,
    checkpoint_sha256: str,
) -> dict[str, float]:
    matches = self_play_matches(args.boots)
    telemetry = DecodeTelemetry()
    process_telemetry = ProcessVecTelemetry()
    policy_index = itertools.count()

    def policy_factory() -> Any:
        return exp.make_policy(
            model,
            stats,
            cfg,
            exec_horizon=CONTROL_DELAY,
            decode_seed=cfg.eval_seed + next(policy_index),
            inference=inference,
            telemetry=telemetry,
        )

    compile_seconds = inference.prewarm(2 * args.max_parallel, CONTROL_DELAY)
    started = time.perf_counter()
    boots = run_matches_vec(
        default_session_cfg(replay_dir, instant_match_restart=False),
        matches,
        policy_factory,
        max_frames=args.max_frames,
        max_parallel=args.max_parallel,
        start_retries=DEFAULT_START_RETRIES,
        process_telemetry=process_telemetry,
    )
    elapsed = time.perf_counter() - started
    results = [
        (
            match.matchup.stage,
            boot_index,
            summarize_trajectory(boot[0]) if boot else None,
        )
        for boot_index, (match, boot) in enumerate(zip(matches, boots, strict=True))
    ]
    rows = match_rows(boots, matches, ego_port=1)
    metrics = vs_cpu_metrics(results, seed=cfg.eval_seed)
    metrics.update(
        {
            "control_delay": float(CONTROL_DELAY),
            "replan_interval": float(CONTROL_DELAY),
            "decode_horizon": float(CONTROL_DELAY),
            "eval_wall_seconds": elapsed,
            "inference_compile_seconds": compile_seconds,
            **telemetry.metrics(),
            **process_telemetry.metrics(),
        }
    )
    exp.require_complete_eval(metrics, args.boots)
    protocol = {
        "kind": "self_play",
        "scheduled_boots": args.boots,
        "max_parallel": args.max_parallel,
        "max_frames": args.max_frames,
        "model_ports": [1, 2],
        "control_delay": CONTROL_DELAY,
        "replan_interval": CONTROL_DELAY,
        "decode_offsets": [1],
        "checkpoint_sha256": checkpoint_sha256,
        "seed": cfg.eval_seed,
        "start_retries": DEFAULT_START_RETRIES,
    }
    _write_json(
        replay_dir / "match_rows.json",
        {"schema_version": 1, "protocol": protocol, "rows": [row.as_dict() for row in rows]},
    )
    _write_json(replay_dir / "metrics.json", {"schema_version": 1, "protocol": protocol, "metrics": metrics})
    return metrics


def main(args: Args) -> None:
    _validate_args(args)
    exp = _load_experiment()
    with tempfile.TemporaryDirectory(prefix="eval-041-delay1-") as directory:
        run_dir = Path(directory) / args.run_name
        checkpoint = download_latest(args.run_name, run_dir, name=args.checkpoint_name)
        if checkpoint is None:
            raise FileNotFoundError(f"R2 checkpoint not found: runs/{args.run_name}/{args.checkpoint_name}")
        checkpoint_sha256 = _sha256(checkpoint)
        model, checkpoint_cfg, stats, state = exp.load_checkpoint(str(checkpoint))
        cfg = replace(checkpoint_cfg, eval_max_parallel=args.max_parallel, eval_max_frames=args.max_frames)
        exp.validate_config(cfg)
        inference = exp.BF16Inference(model, cfg, compiled_buckets=(16, 32), compile_mode="reduce-overhead")
        step = int(state["step"])
        artifact_root = run_dir / "replays" / f"{args.artifact_label}_step{step:07d}"
        cpu_dir = artifact_root / "vs_cpu_lvl9"
        self_play_dir = artifact_root / "self_play"
        uploader = BackgroundUploader(args.run_name)
        summary: dict[str, Any] = {
            "schema_version": 1,
            "run_name": args.run_name,
            "source_checkpoint": args.checkpoint_name,
            "checkpoint_step": step,
            "checkpoint_sha256": checkpoint_sha256,
            "control_delay": CONTROL_DELAY,
            "replan_interval": CONTROL_DELAY,
            "decode_offsets": [1],
            "boots_per_protocol": args.boots,
            "max_parallel": args.max_parallel,
            "max_frames": args.max_frames,
        }
        try:
            cpu_metrics = exp.eval_vs_cpu(
                model,
                stats,
                cfg,
                n_matchups=args.boots,
                replay_dir=cpu_dir,
                exec_horizon=CONTROL_DELAY,
                checkpoint_sha256=checkpoint_sha256,
                inference=inference,
            )
            exp.require_complete_eval(cpu_metrics, args.boots)
            summary["vs_cpu_lvl9"] = cpu_metrics
            _write_json(artifact_root / "summary.partial.json", summary)
            uploader.upload_tree(cpu_dir, base=run_dir)

            self_play_metrics = _self_play_eval(
                exp,
                model,
                stats,
                cfg,
                inference,
                replay_dir=self_play_dir,
                args=args,
                checkpoint_sha256=checkpoint_sha256,
            )
            summary["self_play_port1_perspective"] = self_play_metrics
            _write_json(artifact_root / "summary.json", summary)
            print(f"[eval-041-delay1-result] {json.dumps(summary, sort_keys=True)}", flush=True)
        finally:
            if artifact_root.is_dir():
                uploader.upload_tree(artifact_root, base=run_dir)
            uploader.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
