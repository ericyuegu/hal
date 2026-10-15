"""Freeze read-only W&B evidence for the earlier small Muon candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
from datetime import UTC
from datetime import datetime
from pathlib import Path

import wandb

RUN_IDS = ("a117chkw", "5wfk2esf", "cqbbbg77", "1imfy8v3")
TRAIN_KEYS = ("global_step", "train/loss", "train/grad_norm", "lr/muon", "lr/adam")


def summarize_history(rows: list[dict[str, object]]) -> dict[str, object]:
    """Keep exact finite-value bounds and late-run medians, not sampled curves."""
    if not rows:
        raise ValueError("The requested training history is empty")
    records: list[tuple[int, float, float]] = []
    for row in rows:
        step = row.get("global_step")
        if isinstance(step, bool) or not isinstance(step, int):
            raise ValueError("A training row has no integer global_step")
        loss = row.get("train/loss")
        gradient = row.get("train/grad_norm")
        if not isinstance(loss, (int, float)) or not isinstance(gradient, (int, float)):
            raise ValueError("A training row is missing loss or gradient norm")
        records.append((step, float(loss), float(gradient)))
    records.sort()
    finite = all(math.isfinite(record[1]) and math.isfinite(record[2]) for record in records)
    if not finite:
        return {"rows": len(records), "all_finite": False}
    metrics: dict[str, object] = {
        "rows": len(records),
        "all_finite": True,
        "first_global_step": records[0][0],
        "last_global_step": records[-1][0],
    }
    for lower, upper in ((512, 4095), (12288, 14335), (14336, 16383)):
        window = [record for record in records if lower <= record[0] <= upper]
        if not window:
            raise ValueError(f"No training rows in declared window {lower}..{upper}")
        values: dict[str, object] = {"rows": len(window)}
        for column, name in ((1, "loss"), (2, "gradient_norm")):
            ordered = sorted(record[column] for record in window)
            middle = len(ordered) // 2
            median = (ordered[middle] + ordered[(len(ordered) - 1) // 2]) / 2
            values[name] = {"min": ordered[0], "median": median, "max": ordered[-1]}
        metrics[f"updates_{lower}_{upper}"] = values
    return metrics


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")


def _json_history(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("x", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def collect_run(api: wandb.Api, run_id: str, output_root: Path) -> dict[str, object]:
    api.flush()
    run = api.run(f"ericyuegu/hal/{run_id}")
    run_root = output_root / run_id
    run_root.mkdir()
    config = dict(run.config)
    summary = dict(run.summary)
    for key, expected in {
        "max_steps": 16384,
        "L_ctx": 128,
        "n_layers": 8,
        "observation_bundle": "base",
        "mds_schema_version": 7,
        "data_root": "data/processed/ranked-anonymized-1/mds-policy-v7",
    }.items():
        if config.get(key) != expected:
            raise ValueError(f"{run_id}: {key}={config.get(key)!r}, expected {expected!r}")
    if config.get("muon_lr") != 0.02 or summary.get("global_step") != 16384:
        raise ValueError(f"{run_id}: Muon learning rate or final global_step mismatch")
    training = [dict(row) for row in run.scan_history(keys=list(TRAIN_KEYS), page_size=2000)]
    _json_history(run_root / "training.jsonl", training)
    validation = [dict(row) for row in run.scan_history(keys=["global_step", "val/loss"], page_size=1000)]
    _json_history(run_root / "validation.jsonl", validation)
    artifacts = [
        {"name": artifact.name, "digest": artifact.digest, "type": artifact.type, "size": artifact.size}
        for artifact in run.logged_artifacts()
    ]
    available_files = {item.name: item for item in run.files()}
    files = []
    for name in ("config.yaml", "wandb-metadata.json", "requirements.txt", "output_training.log"):
        item = available_files.get(name)
        if item is None:
            files.append({"name": name, "available": False})
            continue
        item.download(root=str(run_root), replace=False)
        file_path = run_root / name
        files.append(
            {
                "name": name,
                "available": True,
                "bytes": file_path.stat().st_size,
                "sha256": hashlib.sha256(file_path.read_bytes()).hexdigest(),
            }
        )
    record = {
        "id": run.id,
        "name": run.name,
        "url": run.url,
        "state": run.state,
        "created_at": run.created_at,
        "git_commit": run.commit,
        "config": config,
        "summary": summary,
        "logged_artifacts": artifacts,
        "files": files,
        "history_health": summarize_history(training),
        "validation_rows": len(validation),
        "history_keys_requested": list(TRAIN_KEYS),
        "history_method": "scan_history; global_step, never W&B _step",
        "selection": "Fixed final endpoint; gameplay-tested prior controls, not selected by attention or validation",
    }
    _write_json(run_root / "run.json", record)
    print(json.dumps({"id": run.id, "state": run.state, "health": record["history_health"]}), flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-ids", nargs="+", default=list(RUN_IDS))
    args = parser.parse_args()
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=False)
    records = [collect_run(wandb.Api(timeout=90), run_id, output) for run_id in args.run_ids]
    source = Path(__file__).resolve()
    _write_json(
        output / "inventory.json",
        {
            "schema_version": 1,
            "recorded_at_utc": datetime.now(UTC).isoformat(),
            "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "python": platform.python_version(),
            "wandb": wandb.__version__,
            "run_ids": [record["id"] for record in records],
            "remote_mutations": False,
        },
    )


if __name__ == "__main__":
    main()
