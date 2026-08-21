"""Run reliable final experiment-037 evaluations from an R2 checkpoint.

The production training jobs requested 32 concurrent Dolphin boots on hosts
with 16 physical CPU cores. When those built-in evaluations fail at startup,
this tool preserves their artifacts and evaluates the same final checkpoint at
16-way parallelism. Matchups, seeds, frames, policy sampling, and all checkpoint
configuration remain unchanged.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any
from typing import Final
from typing import Literal

import tyro
from botocore.exceptions import ClientError

import wandb
from hal import r2
from hal.training.checkpoints import BackgroundUploader

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
EXPERIMENT: Final[Path] = ROOT / "experiments" / "037_factorization_matrix.py"
WANDB_PATH: Final[str] = "ericyuegu/hal"
RUNS: Final[dict[str, tuple[str, str]]] = {
    "D0": ("98r9smrj", "037-D0-future-independent-group-independent-bc-seed0"),
    "D1": ("a117chkw", "037-D1-future-independent-group-ar-bc-seed0"),
    "D2": ("50q39o9j", "037-D2-future-ar-group-independent-bc-seed0"),
    "D3": ("5wfk2esf", "037-D3-future-ar-group-ar-bc-seed0"),
}
HORIZONS: Final[tuple[int, ...]] = (1, 2, 4, 6)


@dataclass(frozen=True, slots=True)
class Args:
    cell: Literal["D0", "D1", "D2", "D3"]
    max_parallel: int = 16
    """Concurrent Dolphin boots; 16 matches the paid host's physical CPU count."""
    artifact_suffix: str = "p16_repair"
    """Suffix that keeps the failed built-in artifacts intact."""


def _load_experiment() -> ModuleType:
    spec = importlib.util.spec_from_file_location("evaluate_exp037", EXPERIMENT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {EXPERIMENT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_r2_json(client: Any, bucket: str, key: str) -> dict[str, Any] | None:
    try:
        body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise AssertionError(f"{key} is not a JSON object")
    return payload


def _archive_builtin_summaries(run: wandb.apis.public.Run, payloads: dict[int, dict[str, Any]]) -> None:
    values: dict[str, Any] = {}
    for horizon, payload in payloads.items():
        metrics = payload.get("metrics")
        protocol = payload.get("protocol")
        if not isinstance(metrics, dict) or not isinstance(protocol, dict):
            raise AssertionError(f"built-in H{horizon} metrics artifact has invalid schema")
        values[f"audit/builtin_eval_h{horizon}/r2_metrics_key"] = f"replays/final_h{horizon}/metrics.json"
        values[f"audit/builtin_eval_h{horizon}/protocol_sha256"] = protocol.get("protocol_sha256")
        for name, value in metrics.items():
            values[f"audit/builtin_eval_h{horizon}/{name}"] = value
    for name, value in values.items():
        run.summary[name] = value
    run.summary.update()


def main(args: Args) -> None:
    if args.max_parallel < 1 or args.max_parallel & (args.max_parallel - 1):
        raise SystemExit("--max-parallel must be a positive power of two")
    if not args.artifact_suffix or Path(args.artifact_suffix).name != args.artifact_suffix:
        raise SystemExit("--artifact-suffix must be one directory-safe name")
    exp = _load_experiment()
    run_id, run_name = RUNS[args.cell]
    client = r2.client()
    bucket = r2.bucket()
    api = wandb.Api(timeout=90)
    run = api.run(f"{WANDB_PATH}/{run_id}")
    if run.name != run_name:
        raise AssertionError(f"W&B run name {run.name!r} != {run_name!r}")

    builtins: dict[int, dict[str, Any]] = {}
    for horizon in HORIZONS:
        key = f"runs/{run_name}/replays/final_h{horizon}/metrics.json"
        payload = _read_r2_json(client, bucket, key)
        if payload is not None:
            builtins[horizon] = payload
    _archive_builtin_summaries(run, builtins)

    with tempfile.TemporaryDirectory(prefix=f"eval-037-{args.cell}-") as directory:
        run_dir = Path(directory) / run_name
        run_dir.mkdir(parents=True)
        checkpoint = run_dir / "final.pt"
        checkpoint_key = f"runs/{run_name}/final.pt"
        client.download_file(bucket, checkpoint_key, str(checkpoint))
        checkpoint_sha = _sha256(checkpoint)
        model, checkpoint_cfg, stats, state = exp.load_checkpoint(str(checkpoint))
        exp.validate_production_config(checkpoint_cfg)
        if state.get("step") != 16_384 or exp.cell_for_config(checkpoint_cfg) != args.cell:
            raise AssertionError("final checkpoint step or factorization cell mismatch")
        cfg = replace(checkpoint_cfg, eval_max_parallel=args.max_parallel)
        inference = exp.BF16Inference(model, cfg)
        uploader = BackgroundUploader(run_name)
        try:
            for horizon in HORIZONS:
                output_name = f"final_h{horizon}_{args.artifact_suffix}"
                replay_dir = run_dir / "replays" / output_name
                print(f"[eval-repair] {args.cell} H{horizon} -> {output_name}", flush=True)
                metrics = exp.eval_vs_cpu(
                    model,
                    stats,
                    cfg,
                    n_matchups=cfg.final_eval_n_matchups,
                    replay_dir=replay_dir,
                    exec_horizon=horizon,
                    checkpoint_sha256=checkpoint_sha,
                    inference=inference,
                )
                uploader.upload_tree(replay_dir, base=run_dir)
                values = {f"eval_h{horizon}/{name}": value for name, value in metrics.items()}
                values.update(
                    {
                        f"audit/selected_eval_h{horizon}/r2_metrics_key": (f"replays/{output_name}/metrics.json"),
                        f"audit/selected_eval_h{horizon}/max_parallel": args.max_parallel,
                        f"audit/selected_eval_h{horizon}/artifact_suffix": args.artifact_suffix,
                    }
                )
                if horizon == 4:
                    values.update({f"eval/{name}": value for name, value in metrics.items()})
                for name, value in values.items():
                    run.summary[name] = value
                run.summary.update()
                print(f"[eval-repair] {args.cell} H{horizon}: {metrics}", flush=True)
        finally:
            uploader.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
