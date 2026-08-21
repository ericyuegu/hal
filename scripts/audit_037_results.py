"""Audit completed experiment-037 runs on one shared L40S runtime.

This is a post-training tool. It downloads the four final checkpoints and every
uploaded evaluation metric file from R2, checks them against W&B, measures all
four checkpoints on one GPU/runtime, and records a finite learned-logit exposure
diagnostic. It never changes checkpoint configuration or policy weights.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any
from typing import Final

import torch
import torch.nn.functional as F
import tyro

import wandb
from hal import r2

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
EXPECTED_UPDATES: Final[int] = 16_384
EXPECTED_SAMPLES: Final[int] = 8_388_608
EXPECTED_PARAMETERS: Final[dict[str, int]] = {
    "total": 7_147_504,
    "policy": 7_081_711,
    "value": 65_793,
    "trainable": 7_147_504,
    "receiving_grad": 7_147_242,
}


@dataclass(frozen=True, slots=True)
class Args:
    latency_iterations: int = 100
    """Measured replans per cell and horizon after three compile warm-ups."""
    update_wandb: bool = True
    """Write the verified audit and shared-runtime results into each run summary."""
    upload_r2: bool = True
    """Upload the complete audit JSON outside the four production run directories."""


def _load_experiment() -> ModuleType:
    spec = importlib.util.spec_from_file_location("audit_exp037", EXPERIMENT)
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


def _json_sha256(value: dict[str, Any], *, omit: str | None = None) -> str:
    payload = dict(value)
    if omit is not None:
        payload.pop(omit)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _list_objects(client: Any, bucket: str, prefix: str) -> list[dict[str, Any]]:
    paginator = client.get_paginator("list_objects_v2")
    return [item for page in paginator.paginate(Bucket=bucket, Prefix=prefix) for item in page.get("Contents", [])]


def _download_json(client: Any, bucket: str, key: str) -> tuple[dict[str, Any], str, int]:
    body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise AssertionError(f"{key} is not a JSON object")
    return parsed, hashlib.sha256(body).hexdigest(), len(body)


def _assert_close(actual: Any, expected: Any, label: str) -> None:
    if isinstance(expected, (float, int)) and isinstance(actual, (float, int)):
        if not math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-9):
            raise AssertionError(f"{label}: W&B {actual!r} != R2 {expected!r}")
    elif actual != expected:
        raise AssertionError(f"{label}: W&B {actual!r} != R2 {expected!r}")


def _json_value(value: Any) -> Any:
    """Normalize tuples and other JSON-compatible config containers."""
    return json.loads(json.dumps(value))


def _audit_eval(
    payload: dict[str, Any],
    *,
    cell: str,
    horizon: int,
    checkpoint_sha256: str,
    summary: dict[str, Any],
    expected_max_parallel: int,
    compare_wandb: bool,
) -> None:
    protocol = payload.get("protocol")
    metrics = payload.get("metrics")
    if payload.get("schema_version") != 1 or not isinstance(protocol, dict) or not isinstance(metrics, dict):
        raise AssertionError(f"{cell} H{horizon}: invalid metrics.json schema")
    expected_protocol = {
        "n_matchups": 96,
        "max_frames": 7200,
        "seed": 0,
        "cpu_level": 9,
        "ego_port": 1,
        "seed_stage": 24,
        "exec_horizon": horizon,
        "checkpoint_sha256": checkpoint_sha256,
        "actor_weighting": "uniform",
        "sampling_temperature": 1.0,
        "action_hygiene": "structured_codec_trigger_button_mask_v1",
        "instant_legal_stage_restarts": True,
        "bootstrap_resamples": 2000,
        "max_parallel": expected_max_parallel,
    }
    for name, expected in expected_protocol.items():
        if protocol.get(name) != expected:
            raise AssertionError(f"{cell} H{horizon}: protocol {name}={protocol.get(name)!r}, expected {expected!r}")
    if protocol.get("protocol_sha256") != _json_sha256(protocol, omit="protocol_sha256"):
        raise AssertionError(f"{cell} H{horizon}: protocol digest mismatch")
    expected_axes = {
        "D0": ("independent", "independent"),
        "D1": ("independent", "autoregressive"),
        "D2": ("selected_ar", "independent"),
        "D3": ("selected_ar", "autoregressive"),
    }[cell]
    if (protocol.get("future_conditioning"), protocol.get("group_conditioning")) != expected_axes:
        raise AssertionError(f"{cell} H{horizon}: factorization flags do not match the cell")
    if compare_wandb:
        for name, expected in metrics.items():
            _assert_close(summary.get(f"eval_h{horizon}/{name}"), expected, f"{cell} H{horizon} {name}")
            if horizon == 4:
                _assert_close(summary.get(f"eval/{name}"), expected, f"{cell} mirrored H4 {name}")


@torch.no_grad()
def finite_learned_logit_metrics(exp: ModuleType, model: Any, batches: list[Any], cfg: Any) -> dict[str, float]:
    """Measure exposure before the fixed trigger/button support mask.

    Greedy sampled triggers still apply the production legality mask when
    selecting buttons. Only the logits used to score the recorded target omit
    the hard mask, so every reported cross-entropy is finite and separately
    labeled from the exact rollout-conditioned NLL.
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    teacher_sum = torch.zeros(len(model.head_offsets), exp.N_GROUPS, dtype=torch.float64)
    rollout_sum = torch.zeros_like(teacher_sum)
    support_sum = torch.zeros(len(model.head_offsets), dtype=torch.float64)
    count = 0
    try:
        for cpu_batch in batches:
            batch = cpu_batch.to(device)
            history, targets, _ = exp.prepared_targets(model, batch)
            with exp.amp_context(cfg, device):
                hidden = model(batch.context.features, batch.context.ctx_pad, history)
                teacher = model.temporal.teacher_forced_learned_logits_by_group(hidden, history, targets)

            target = targets[:, -1]
            trunk = exp.decoder_rmsnorm(hidden[:, -1])
            trunk_logits = {name: model.temporal.trunk_outputs[name](trunk) for name in exp.GROUP_NAMES}
            state_bias = model.temporal._state_bias(trunk)
            film = model.temporal._film_params(trunk)
            previous = history[:, -1]
            caches: list[Any] = [None] * len(model.temporal.blocks)
            for depth, offset in enumerate(model.head_offsets):
                with exp.amp_context(cfg, device):
                    state, caches = model.temporal._decode_step(previous, offset, state_bias, film, caches)
                    embedded: dict[str, torch.Tensor] = {}
                    picks: dict[str, torch.Tensor] = {}
                    rollout: dict[str, torch.Tensor] = {}
                    for name in exp.GROUP_ORDER:
                        logits = (
                            model.temporal.outputs[name](model.temporal.group_features(state, name, embedded))
                            + trunk_logits[name]
                        )
                        rollout[name] = logits
                        sample_logits = logits
                        if name == "buttons":
                            sample_logits = logits.masked_fill(
                                model.codec.button_mask(picks["triggers"]), float("-inf")
                            )
                        pick = sample_logits.argmax(dim=-1)
                        picks[name] = pick
                        embedded[name] = model.codec.group_embedding(name, pick)
                sampled = torch.stack([picks[name] for name in exp.GROUP_NAMES], dim=-1)
                if cfg.future_conditioning == "selected_ar":
                    previous = sampled
                expected_trigger = picks["triggers"]
                expected_button = target[:, depth, exp.BUTTONS_G]
                support_sum[depth] += (
                    model.codec.button_valid_for_trigger[expected_trigger, expected_button].double().sum().cpu()
                )
                for group, name in enumerate(exp.GROUP_NAMES):
                    expected = target[:, depth, group]
                    teacher_sum[depth, group] += (
                        F.cross_entropy(teacher[name][:, -1, depth].float(), expected, reduction="sum").double().cpu()
                    )
                    rollout_sum[depth, group] += (
                        F.cross_entropy(rollout[name].float(), expected, reduction="sum").double().cpu()
                    )
            count += target.shape[0]
    finally:
        model.train(was_training)
    if count == 0:
        raise AssertionError("validation cache is empty")
    teacher_bits = teacher_sum / count / math.log(2.0)
    rollout_bits = rollout_sum / count / math.log(2.0)
    out: dict[str, float] = {}
    for depth, offset in enumerate(model.head_offsets):
        for group, name in enumerate(exp.GROUP_NAMES):
            teacher_value = float(teacher_bits[depth, group])
            rollout_value = float(rollout_bits[depth, group])
            out[f"teacher_nll_o{offset:02d}_{name}"] = teacher_value
            out[f"rollout_nll_o{offset:02d}_{name}"] = rollout_value
            out[f"gap_o{offset:02d}_{name}"] = rollout_value - teacher_value
        teacher_joint = float(teacher_bits[depth].sum())
        rollout_joint = float(rollout_bits[depth].sum())
        out[f"teacher_nll_o{offset:02d}"] = teacher_joint
        out[f"rollout_nll_o{offset:02d}"] = rollout_joint
        out[f"gap_o{offset:02d}"] = rollout_joint - teacher_joint
        out[f"target_button_support_rate_o{offset:02d}"] = float(support_sum[depth] / count)
    out["teacher_nll"] = float(teacher_bits.sum(dim=1).mean())
    out["rollout_nll"] = float(rollout_bits.sum(dim=1).mean())
    out["gap"] = out["rollout_nll"] - out["teacher_nll"]
    out["target_button_support_rate"] = float(support_sum.mean() / count)
    if not all(math.isfinite(value) for value in out.values()):
        raise AssertionError("finite learned-logit audit produced a non-finite metric")
    return out


def _validation_cache(exp: ModuleType, cfg: Any, stats: dict[str, Any]) -> list[Any]:
    common = exp.loader_kwargs(cfg, stats)
    common["batch_size"] = cfg.val_batch_size
    loader = exp.make_loader(split=cfg.val_split, num_workers=0, compact=True, **common)
    return exp.cache_validation(loader, cfg.val_n_samples)


def _hardware() -> dict[str, Any]:
    smi = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,driver_version",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "nvidia_smi": smi,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "compile_mode": "default",
        "precision": "bfloat16",
        "inference_batch": 1,
        "latency_iterations": None,
    }


def _audit_run(
    exp: ModuleType,
    *,
    cell: str,
    run_id: str,
    run_name: str,
    api: wandb.Api,
    client: Any,
    bucket: str,
    root: Path,
    latency_iterations: int,
    validation: list[Any] | None,
) -> tuple[dict[str, Any], list[Any]]:
    run = api.run(f"{WANDB_PATH}/{run_id}")
    summary = dict(run.summary)
    if run.name != run_name:
        raise AssertionError(f"{cell}: W&B name mismatch: {run.name!r}")
    history_max_step = summary.get("global_step")
    history_max_samples = summary.get("samples")
    if history_max_step != EXPECTED_UPDATES or history_max_samples != EXPECTED_SAMPLES:
        progress = list(run.scan_history(keys=["global_step", "samples"], page_size=10_000))
        history_max_step = max((row.get("global_step", -1) for row in progress), default=-1)
        history_max_samples = max((row.get("samples", -1) for row in progress), default=-1)
        if history_max_step != EXPECTED_UPDATES or history_max_samples != EXPECTED_SAMPLES:
            raise AssertionError(f"{cell}: W&B history ends at step={history_max_step}, samples={history_max_samples}")
    for name, expected in EXPECTED_PARAMETERS.items():
        if summary.get(f"parameters/{name}") != expected:
            raise AssertionError(f"{cell}: parameters/{name} mismatch")

    prefix = f"runs/{run_name}/"
    objects = _list_objects(client, bucket, prefix)
    keys = [str(item["Key"]) for item in objects]
    if not keys:
        raise AssertionError(f"{cell}: missing R2 run directory {prefix}")
    final_key = f"{prefix}final.pt"
    final_items = [item for item in objects if item["Key"] == final_key]
    if len(final_items) != 1:
        raise AssertionError(f"{cell}: expected one {final_key}, got {len(final_items)}")
    checkpoint_path = root / f"{cell}.pt"
    client.download_file(bucket, final_key, str(checkpoint_path))
    checkpoint_sha = _sha256(checkpoint_path)
    checkpoint_bytes = checkpoint_path.stat().st_size
    if checkpoint_bytes != int(final_items[0]["Size"]):
        raise AssertionError(f"{cell}: downloaded final.pt size does not match R2")

    model, cfg, stats, state = exp.load_checkpoint(str(checkpoint_path))
    exp.validate_production_config(cfg)
    if state.get("step") != EXPECTED_UPDATES or cfg.exec_horizon != 4:
        raise AssertionError(f"{cell}: checkpoint step or stored execution horizon mismatch")
    if exp.cell_for_config(cfg) != cell or cfg.actor_weighting != "uniform":
        raise AssertionError(f"{cell}: checkpoint factorization flags mismatch")
    counts = exp.parameter_counts(model)
    for name in ("total", "trainable"):
        if counts[name] != EXPECTED_PARAMETERS[name]:
            raise AssertionError(f"{cell}: checkpoint parameter count mismatch")
    for name, expected in asdict(cfg).items():
        actual = run.config.get(name)
        if _json_value(actual) != _json_value(expected):
            raise AssertionError(f"{cell}: W&B config {name}={actual!r}, checkpoint has {expected!r}")

    metric_keys = sorted(key for key in keys if key.endswith("metrics.json"))
    builtin_metric_keys = [f"{prefix}replays/final_h{horizon}/metrics.json" for horizon in HORIZONS]
    selected_metric_keys = [f"{prefix}replays/final_h{horizon}_p16-repair/metrics.json" for horizon in HORIZONS]
    expected_metric_keys = sorted(builtin_metric_keys + selected_metric_keys)
    if metric_keys != expected_metric_keys:
        raise AssertionError(f"{cell}: R2 metric files {metric_keys!r} != {expected_metric_keys!r}")
    metrics_audit: dict[str, Any] = {}
    for key in metric_keys:
        payload, digest, size = _download_json(client, bucket, key)
        protocol = payload.get("protocol", {})
        horizon = protocol.get("exec_horizon")
        if horizon not in HORIZONS:
            raise AssertionError(f"{cell}: {key} has invalid horizon {horizon!r}")
        selected = key == selected_metric_keys[HORIZONS.index(horizon)]
        builtin = key == builtin_metric_keys[HORIZONS.index(horizon)]
        if selected == builtin:
            raise AssertionError(f"{cell}: cannot classify evaluation artifact {key}")
        _audit_eval(
            payload,
            cell=cell,
            horizon=horizon,
            checkpoint_sha256=checkpoint_sha,
            summary=summary,
            expected_max_parallel=16 if selected else 32,
            compare_wandb=selected,
        )
        label = f"{'selected' if selected else 'builtin'}_h{horizon}"
        metrics_audit[label] = {
            "key": key,
            "bytes": size,
            "sha256": digest,
            "protocol": payload["protocol"],
            "metrics": payload["metrics"],
        }

    if validation is None:
        validation = _validation_cache(exp, cfg, stats)
    finite = finite_learned_logit_metrics(exp, model, validation, cfg)
    latency = exp.benchmark_model(model, cfg, iterations=latency_iterations, rows=1)
    report = {
        "cell": cell,
        "run_id": run_id,
        "run_name": run_name,
        "wandb_url": run.url,
        "wandb_pre_audit": {
            "state": run.state,
            "summary_global_step": summary.get("global_step"),
            "summary_samples": summary.get("samples"),
            "history_max_global_step": history_max_step,
            "history_max_samples": history_max_samples,
        },
        "r2_prefix": prefix,
        "r2_object_count": len(objects),
        "final_pt": {"key": final_key, "bytes": checkpoint_bytes, "sha256": checkpoint_sha},
        "checkpoint_step": state["step"],
        "checkpoint_config": asdict(cfg),
        "metrics_artifacts": metrics_audit,
        "finite_learned_logits": finite,
        "shared_latency": latency,
    }
    del model
    torch.cuda.empty_cache()
    torch.compiler.reset()
    return report, validation


def main(args: Args) -> None:
    if args.latency_iterations < 1:
        raise SystemExit("--latency-iterations must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("the shared latency audit requires a CUDA GPU")
    exp = _load_experiment()
    api = wandb.Api(timeout=90)
    client = r2.client()
    bucket = r2.bucket()
    hardware = _hardware()
    hardware["latency_iterations"] = args.latency_iterations
    report: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "experiment-037 final artifact audit and shared-L40S latency",
        "hardware": hardware,
        "runs": {},
    }
    validation: list[Any] | None = None
    with tempfile.TemporaryDirectory(prefix="audit-037-") as directory:
        root = Path(directory)
        for cell, (run_id, run_name) in RUNS.items():
            print(f"[audit] {cell}: {run_name}", flush=True)
            cell_report, validation = _audit_run(
                exp,
                cell=cell,
                run_id=run_id,
                run_name=run_name,
                api=api,
                client=client,
                bucket=bucket,
                root=root,
                latency_iterations=args.latency_iterations,
                validation=validation,
            )
            report["runs"][cell] = cell_report

    baseline: dict[str, Any] | None = None
    variable_protocol_fields = {
        "exec_horizon",
        "checkpoint_sha256",
        "future_conditioning",
        "group_conditioning",
        "protocol_sha256",
    }
    for cell, cell_report in report["runs"].items():
        for horizon, artifact in cell_report["metrics_artifacts"].items():
            if not horizon.startswith("selected_"):
                continue
            common = {
                name: value for name, value in artifact["protocol"].items() if name not in variable_protocol_fields
            }
            if baseline is None:
                baseline = common
            elif common != baseline:
                raise AssertionError(f"{cell} {horizon}: evaluation protocol differs from the other cells/horizons")

    encoded = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    report_sha = hashlib.sha256(encoded).hexdigest()
    audit_key = "runs/037_factorization_matrix_audit/shared_l40s_audit.json"
    if args.upload_r2:
        client.put_object(Bucket=bucket, Key=audit_key, Body=encoded, ContentType="application/json")
    if args.update_wandb:
        for cell, (run_id, _) in RUNS.items():
            cell_report = report["runs"][cell]
            values: dict[str, Any] = {
                "audit/report_sha256": report_sha,
                "audit/r2_report_key": audit_key,
                "audit/final_pt_sha256": cell_report["final_pt"]["sha256"],
                "audit/final_pt_bytes": cell_report["final_pt"]["bytes"],
                "audit/r2_object_count": cell_report["r2_object_count"],
                "audit/shared_hardware": hardware["nvidia_smi"],
                "audit/shared_torch_version": hardware["torch_version"],
                "audit/shared_cuda_runtime": hardware["cuda_runtime"],
                "audit/shared_compile_mode": hardware["compile_mode"],
                "audit/shared_precision": hardware["precision"],
                "audit/shared_inference_batch": hardware["inference_batch"],
            }
            for horizon in HORIZONS:
                artifact = cell_report["metrics_artifacts"][f"selected_h{horizon}"]
                values.update({f"eval_h{horizon}/{name}": value for name, value in artifact["metrics"].items()})
                values[f"audit/selected_eval_h{horizon}/r2_metrics_key"] = artifact["key"].removeprefix(
                    f"runs/{cell_report['run_name']}/"
                )
                values[f"audit/selected_eval_h{horizon}/max_parallel"] = 16
                if horizon == 4:
                    values.update({f"eval/{name}": value for name, value in artifact["metrics"].items()})
            values.update(
                {
                    f"audit/finite_learned_logits/{name}": value
                    for name, value in cell_report["finite_learned_logits"].items()
                }
            )
            values.update(
                {f"audit/shared_latency/{name}": value for name, value in cell_report["shared_latency"].items()}
            )
            tracking = wandb.init(
                entity="ericyuegu",
                project="hal",
                id=run_id,
                resume="must",
                reinit="finish_previous",
            )
            tracking.log(
                {
                    "global_step": EXPECTED_UPDATES,
                    "samples": EXPECTED_SAMPLES,
                    **values,
                }
            )
            tracking.summary["global_step"] = EXPECTED_UPDATES
            tracking.summary["samples"] = EXPECTED_SAMPLES
            for name, value in values.items():
                tracking.summary[name] = value
            tracking.finish()
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), flush=True)
    print(f"[audit] R2 {audit_key} sha256={report_sha} bytes={len(encoded)}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
