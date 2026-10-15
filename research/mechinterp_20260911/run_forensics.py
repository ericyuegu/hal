"""Download exact W&B evidence for the O50 optimizer comparison.

Run from the repository root with a configured W&B API login:

    uv run research/mechinterp_20260911/run_forensics.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Final

import wandb

SCRIPT_PATH: Final[Path] = Path(__file__).resolve()
ROOT: Final[Path] = SCRIPT_PATH.parents[2] if len(SCRIPT_PATH.parents) > 2 else Path.cwd()
DEFAULT_OUTPUT: Final[Path] = ROOT / "research" / "mechinterp_20260911" / "artifacts" / "forensics"
ENTITY: Final[str] = "ericyuegu"
PROJECT: Final[str] = "hal"

TRAIN_KEYS: Final[tuple[str, ...]] = (
    "global_step",
    "train/loss",
    "train/nll",
    "train/near_loss",
    "train/far_nll",
    "optimizer/grad_norm",
    "optimizer/clip_fraction",
    "stability/action_grad_abs_max",
    "schedule/muon_lr",
    "schedule/adam_lr",
)
VALIDATION_KEYS: Final[tuple[str, ...]] = (
    "global_step",
    "val/nll",
    "val/near_nll",
    "val/far_nll",
    "val/rollout_nll",
    "val/change_f1",
    "val/exact_frame_acc",
    "val/sequence_acc",
)
EVALUATION_KEYS: Final[tuple[str, ...]] = (
    "global_step",
    "eval/checkpoint_step",
    "eval/boots",
    "eval/crashed",
    "eval/net_stock_per_min",
    "eval/net_stock_lcb",
    "eval/net_dmg_per_min",
    "eval/net_dmg_lcb",
    "eval/damage_dealt_per_min",
    "eval/damage_taken_per_min",
)
OPTIMIZATION_HEALTH_KEYS: Final[tuple[str, ...]] = (
    "global_step",
    "value/loss",
    "value/rmse",
    "awr/weight_mean",
    "awr/ess_fraction",
    "awr/cap_fraction",
    "stability/button_input_abs_p999",
    "stability/button_logit_abs_p999",
    "stability/button_pre_norm_rms_min",
    "stability/button_margin_mean",
)
RUN_FILES: Final[tuple[str, ...]] = (
    "config.yaml",
    "output_training.log",
    "requirements.txt",
    "wandb-metadata.json",
    "wandb-summary.json",
)
CHECKPOINT_UPLOAD: Final[re.Pattern[str]] = re.compile(
    r"^(?P<timestamp>\S+ \S+) \| .*\[ckpt\] uploaded "
    r"(?P<relative>checkpoints/step-(?P<step>\d{7})\.pt|final\.pt) -> (?P<uri>r2://\S+)$"
)
SELECTED_MUON_STEPS: Final[dict[int, str]] = {
    24_576: "best early gameplay evaluation; before the sustained regression",
    57_344: "lowest stock-rate evaluation in the sustained regression",
    98_304: "last durable checkpoint before the step-99,875 gradient spike",
    131_072: "recovered endpoint under the annealed learning rate",
}
EXPORT_STEPS: Final[dict[str, tuple[int, ...]]] = {
    "p1fyyp1z": (24_576, 57_344, 98_304, 131_072),
    "tgpuo1be": (57_344, 131_072),
}


@dataclass(frozen=True, slots=True)
class RunSpec:
    run_id: str
    role: str
    source_commit: str
    experiment_id: str


RUNS: Final[tuple[RunSpec, ...]] = (
    RunSpec(
        run_id="p1fyyp1z",
        role="Muon matrices with AdamW remainder",
        source_commit="cdf84f904d322d2a8c342fe3ea75d926e10346ba",
        experiment_id="050_scaled_temporal_awr_v4",
    ),
    RunSpec(
        run_id="tgpuo1be",
        role="all AdamW",
        source_commit="d67c68cb36c830bb4100ba8ef1b4cefc452539de",
        experiment_id="050_scaled_temporal_awr_v6",
    ),
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def _write_json_lines(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    return count


def _history_counts(run: Any) -> dict[str, int]:
    keys = run._attrs.get("historyKeys", {}).get("keys", {})  # pylint: disable=protected-access
    return {
        key: sum(int(item.get("count", 0)) for item in descriptor.get("typeCounts", []))
        for key, descriptor in keys.items()
    }


def _artifact_record(artifact: Any) -> dict[str, object]:
    return {
        "digest": artifact.digest,
        "name": artifact.name,
        "size": artifact.size,
        "state": artifact.state,
        "type": artifact.type,
    }


def _run_record(run: Any, spec: RunSpec) -> dict[str, object]:
    return {
        "config": dict(run.config),
        "created_at": run.created_at,
        "experiment_id_from_source": spec.experiment_id,
        "files": [{"md5": item.md5, "name": item.name, "size": item.size} for item in run.files()],
        "group": run.group,
        "heartbeat_at": run.heartbeat_at,
        "history_counts": _history_counts(run),
        "id": run.id,
        "job_type": run.job_type,
        "logged_artifacts": [_artifact_record(item) for item in run.logged_artifacts()],
        "name": run.name,
        "notes": run.notes,
        "role": spec.role,
        "source_commit_verified_from_logged_files": spec.source_commit,
        "state": run.state,
        "summary": dict(run.summary),
        "tags": run.tags,
        "url": run.url,
        "used_artifacts": [_artifact_record(item) for item in run.used_artifacts()],
        "wandb_commit_field": run.commit,
    }


def _download_run(api: wandb.Api, spec: RunSpec, output_root: Path) -> dict[str, object]:
    api.flush()
    run = api.run(f"{ENTITY}/{PROJECT}/{spec.run_id}")
    run_root = output_root / spec.run_id
    run_root.mkdir(parents=True, exist_ok=True)

    _write_json(run_root / "run.json", _run_record(run, spec))
    family_counts = {
        "evaluation": _write_json_lines(
            run_root / "evaluation.jsonl",
            run.scan_history(keys=list(EVALUATION_KEYS), page_size=100),
        ),
        "optimization_health": _write_json_lines(
            run_root / "optimization_health.jsonl",
            run.scan_history(keys=list(OPTIMIZATION_HEALTH_KEYS), page_size=1000),
        ),
        "training": _write_json_lines(
            run_root / "training.jsonl",
            run.scan_history(keys=list(TRAIN_KEYS), page_size=1000),
        ),
        "validation": _write_json_lines(
            run_root / "validation.jsonl",
            run.scan_history(keys=list(VALIDATION_KEYS), page_size=100),
        ),
    }
    for name in RUN_FILES:
        run.file(name).download(root=run_root, replace=True)
    return {"history_rows": family_counts, "name": run.name, "run_id": spec.run_id}


def _checkpoint_inventory(output_root: Path) -> None:
    exports_path = output_root / "model_exports" / "model_exports.json"
    verified_exports = {}
    if exports_path.exists():
        export_manifest = json.loads(exports_path.read_text())
        verified_exports = {item["source_uri"]: item for item in export_manifest["exports"]}
    runs: dict[str, list[dict[str, object]]] = {}
    for spec in RUNS:
        log_path = output_root / spec.run_id / "output_training.log"
        checkpoints = []
        for line in log_path.read_text().splitlines():
            match = CHECKPOINT_UPLOAD.fullmatch(line)
            if match is None:
                continue
            step_text = match.group("step")
            step = int(step_text) if step_text is not None else 131_072
            uri = match.group("uri")
            verified = verified_exports.get(uri, {})
            checkpoints.append(
                {
                    "bytes": verified.get("source_content_length"),
                    "etag": verified.get("source_r2_etag"),
                    "evidence": "upload confirmation in the immutable W&B output_training.log",
                    "relative_name": match.group("relative"),
                    "selected_reason": SELECTED_MUON_STEPS.get(step),
                    "sha256": verified.get("source_checkpoint_sha256"),
                    "step": step,
                    "uploaded_at": match.group("timestamp"),
                    "uri": uri,
                }
            )
        runs[spec.run_id] = checkpoints
    _write_json(
        output_root / "checkpoint_inventory.json",
        {
            "note": (
                "The six model-export sources have verified R2 size, ETag, and downloaded SHA-256. "
                "Other objects are established by immutable upload logs but were not read from R2."
            ),
            "runs": runs,
            "schema_version": 1,
        },
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _r2_client() -> Any:
    import boto3
    from botocore.config import Config

    required = (
        "AWS_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_BUCKET",
    )
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"missing R2 environment variables: {missing}")
    return boto3.client(
        "s3",
        endpoint_url=os.environ["AWS_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )


def _parse_r2_uri(uri: str) -> tuple[str, str]:
    match = re.fullmatch(r"r2://([^/]+)/(.+)", uri)
    if match is None:
        raise ValueError(f"invalid R2 URI: {uri!r}")
    return match.group(1), match.group(2)


def _export_models(inventory_path: Path, output_root: Path) -> None:
    """Export model tensors and identity fields from trusted HAL checkpoints."""
    import torch

    inventory = json.loads(inventory_path.read_text())
    client = _r2_client()
    output_root.mkdir(parents=True, exist_ok=True)
    specs = {spec.run_id: spec for spec in RUNS}
    records = []
    temporary = Path("/tmp/hal-source-checkpoint.pt")
    for run_id, selected_steps in EXPORT_STEPS.items():
        spec = specs[run_id]
        objects = {
            int(item["step"]): item for item in inventory["runs"][run_id] if item["relative_name"] != "final.pt"
        }
        for boundary_step in selected_steps:
            item = objects[boundary_step]
            uri = str(item["uri"])
            bucket, key = _parse_r2_uri(uri)
            if bucket != os.environ["AWS_BUCKET"]:
                raise ValueError(f"checkpoint bucket {bucket!r} does not match configured bucket")
            print(f"downloading {run_id} boundary step {boundary_step:,}", flush=True)
            source_head = client.head_object(Bucket=bucket, Key=key)
            client.download_file(bucket, key, str(temporary))
            source_sha256 = _sha256(temporary)
            checkpoint = torch.load(temporary, map_location="cpu", weights_only=False)
            checkpoint_step = checkpoint.get("step")
            config = checkpoint.get("cfg")
            loader = checkpoint.get("loader")
            model = checkpoint.get("model")
            if checkpoint_step != boundary_step - 1:
                raise ValueError(f"{run_id} boundary {boundary_step} stores checkpoint step {checkpoint_step!r}")
            if not isinstance(config, dict) or config.get("experiment_id") != spec.experiment_id:
                raise ValueError(f"{run_id} boundary {boundary_step} has an unexpected config identity")
            if not isinstance(loader, dict):
                raise ValueError(f"{run_id} boundary {boundary_step} has no loader identity")
            source_manifests = loader.get("source_manifest_sha256")
            if not isinstance(source_manifests, dict) or not source_manifests:
                raise ValueError(f"{run_id} boundary {boundary_step} has no source-manifest identity")
            if not isinstance(model, dict) or not model:
                raise ValueError(f"{run_id} boundary {boundary_step} has no model state")
            if any(not isinstance(value, torch.Tensor) for value in model.values()):
                raise TypeError(f"{run_id} boundary {boundary_step} model state contains non-tensors")

            output_path = output_root / f"{run_id}-step-{boundary_step:07d}.model.pt"
            torch.save(
                {
                    "cfg": config,
                    "checkpoint_step": checkpoint_step,
                    "data_protocol": loader.get("data_protocol"),
                    "model": model,
                    "source_manifest_sha256": source_manifests,
                    "source_selection_sha256": loader.get("source_selection_sha256"),
                    "source_checkpoint_sha256": source_sha256,
                    "source_r2_etag": source_head["ETag"],
                    "source_uri": uri,
                    "wandb_id": checkpoint.get("wandb_id"),
                },
                output_path,
            )
            record = {
                "boundary_step": boundary_step,
                "bytes": output_path.stat().st_size,
                "checkpoint_step": checkpoint_step,
                "experiment_id": config["experiment_id"],
                "model_bytes": sum(value.numel() * value.element_size() for value in model.values()),
                "model_tensors": len(model),
                "output": output_path.name,
                "output_sha256": _sha256(output_path),
                "raw_cfg": config,
                "run_id": run_id,
                "source_checkpoint_sha256": source_sha256,
                "source_content_length": source_head["ContentLength"],
                "source_last_modified": source_head["LastModified"],
                "source_manifest_sha256": source_manifests,
                "source_r2_etag": source_head["ETag"],
                "source_selection_sha256": loader.get("source_selection_sha256"),
                "source_uri": uri,
            }
            records.append(record)
            _write_json(output_path.with_suffix(".metadata.json"), record)
            print(
                f"exported {output_path.name}: {record['model_tensors']} tensors, "
                f"{record['model_bytes']:,} tensor bytes",
                flush=True,
            )
            del checkpoint, model
            temporary.unlink()
    _write_json(
        output_root / "model_exports.json",
        {"exports": records, "schema_version": 1},
    )


def _write_manifest(output_root: Path, runs: list[dict[str, object]]) -> None:
    files = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append(
                {
                    "bytes": path.stat().st_size,
                    "path": path.relative_to(output_root).as_posix(),
                    "sha256": _sha256(path),
                }
            )
    _write_json(
        output_root / "manifest.json",
        {
            "fetched_at": datetime.now(UTC).isoformat(),
            "history_method": "wandb.Api.run(...).scan_history(keys=...) using global_step",
            "project": f"{ENTITY}/{PROJECT}",
            "runs": runs,
            "schema_version": 1,
            "files": files,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--export-models",
        type=Path,
        metavar="INVENTORY",
        help="export selected R2 checkpoints into model-only files (trusted checkpoints only)",
    )
    args = parser.parse_args()
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.export_models is not None:
        _export_models(args.export_models.resolve(), output_root)
        return

    api = wandb.Api(timeout=120)
    records = [_download_run(api, spec, output_root) for spec in RUNS]
    _checkpoint_inventory(output_root)
    _write_manifest(output_root, records)
    print(f"wrote exact W&B evidence to {output_root}")


if __name__ == "__main__":
    main()
