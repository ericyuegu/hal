"""Evaluate retained KV history against the final checkpoint's CPU protocol."""

import contextlib
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import time
from pathlib import Path
from typing import cast

import torch

from hal import r2
from hal.eval.cross_stage import PRIOR_SWEEP_SEED_STAGE
from hal.eval.cross_stage import sweep_vs_cpu_prior_with_rows
from hal.eval.cross_stage import vs_cpu_metrics
from hal.eval.harness import default_session_cfg
from hal.eval.matchups import matchups_for_vs_cpu
from hal.eval.policy import PolicyBatchAdapter
from hal.fixtures import DOLPHIN_EXIAI
from hal.fixtures import ISO
from hal.inference.api import RuntimeConfig
from hal.inference.bundle import extract_policy_bundle
from hal.inference.checkpoints import resolve_checkpoint
from hal.inference.loader import load_policy
from hal.inference.o59 import O59Policy
from hal.inference.o59 import export_o59_policy
from hal.sim.process_vec import ProcessVecTelemetry
from hal.training.checkpoints import BackgroundUploader

CHECKPOINT_URI = "r2://hal/runs/260921-092548_059_muon_history_decoder_o59v5-cooldown32768/checkpoints/step-0131072.pt"
BASELINE_KEY = "runs/260921-092548_059_muon_history_decoder_o59v5-cooldown32768/eval96-step-0131072/metrics.json"
CHECKPOINT_SHA256 = "52b5233ed506f59f514f7e90a6a6111206152db7413f451dc35be5c30d1e671b"
MATCHUPS = 96
MAX_FRAMES = 7200
EVAL_SEED = 0
BASELINE_TOLERANCE_NSM = 0.2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _baseline() -> dict[str, float]:
    with contextlib.closing(r2.client()) as client:
        response = client.get_object(Bucket=r2.bucket(), Key=BASELINE_KEY)
        with contextlib.closing(response["Body"]) as body:
            values = json.loads(body.read())
    if not isinstance(values, dict):
        raise ValueError("baseline metrics are not an object")
    metrics = cast(dict[str, object], values)
    if any(metrics.get(key) != MATCHUPS for key in ("scheduled_boots", "completed_boots", "boots")):
        raise ValueError("baseline is not a complete 96-boot evaluation")
    nsm = metrics.get("net_stock_per_min")
    if not isinstance(nsm, int | float) or not math.isfinite(nsm):
        raise ValueError("baseline NSM is missing")
    return cast(dict[str, float], metrics)


def _save_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, allow_nan=False))
    temporary.replace(path)


def main() -> None:
    git_sha = os.environ["HAL_GIT_SHA"]
    if len(git_sha) != 40 or any(char not in "0123456789abcdef" for char in git_sha):
        raise ValueError("HAL_GIT_SHA must identify the committed source")
    if not torch.cuda.is_available() or "L40S" not in torch.cuda.get_device_name(0):
        raise RuntimeError("the 96-matchup comparison requires a Modal L40S")

    baseline = _baseline()
    print(f"[eval] published baseline NSM={baseline['net_stock_per_min']:.4f}", flush=True)
    checkpoint = resolve_checkpoint(CHECKPOINT_URI)
    if _sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("downloaded checkpoint differs from the qualified policy bundle")
    bundle = Path("/tmp/hal-kv-cache-eval.hal")
    manifest = export_o59_policy(checkpoint, bundle)
    if manifest.source_sha256 != CHECKPOINT_SHA256:
        raise ValueError("exported bundle refers to a different checkpoint")
    with extract_policy_bundle(bundle) as (_, root):
        configuration = json.loads((root / "backend.json").read_text())
    p90 = configuration["return_p90"]
    if not isinstance(p90, float) or not math.isfinite(p90) or not 0.0 <= p90 <= 40.0:
        raise ValueError("checkpoint p90 return target is invalid")

    runtime = RuntimeConfig(max_batch_size=1, transport_delays=(2,), replan_interval_frames=2)
    policy = load_policy(bundle, device="cuda", seed=EVAL_SEED, compiled=True, history_mode="kv_cache")
    if not isinstance(policy, O59Policy):
        raise ValueError("checkpoint bundle did not load the expected history decoder")
    policy.prepare(runtime)
    print(f"[eval] prepared KV cache; p90 return={p90:.6f}", flush=True)

    schedule = [(int(ego.value), int(cpu.value)) for ego, cpu in matchups_for_vs_cpu(MATCHUPS)]
    schedule_hash = hashlib.sha256(json.dumps(schedule, separators=(",", ":")).encode()).hexdigest()
    run_name = f"kv-cache-eval-vywk3cih-{git_sha[:8]}"
    output = Path("runs") / run_name / "eval96-p90"
    output.mkdir(parents=True, exist_ok=False)
    telemetry = ProcessVecTelemetry()
    wave = itertools.count()

    def factory() -> PolicyBatchAdapter:
        seed = EVAL_SEED + next(wave)
        print(f"[eval] starting matchup wave with sampling seed {seed}", flush=True)
        policy.reset_chunks(seed=seed)
        return PolicyBatchAdapter(policy, runtime, desired_return=p90)

    started = time.perf_counter()
    with torch.compiler.set_stance("fail_on_recompile"):
        results, rows = sweep_vs_cpu_prior_with_rows(
            factory,
            session_cfg=default_session_cfg(output, instant_match_restart=True),
            n_matchups=MATCHUPS,
            max_parallel=1,
            cpu_level=9,
            ego_port=1,
            seed_stage=PRIOR_SWEEP_SEED_STAGE,
            max_frames=MAX_FRAMES,
            process_telemetry=telemetry,
        )
    wall_seconds = time.perf_counter() - started
    metrics = vs_cpu_metrics(results, seed=EVAL_SEED)
    metrics.update(telemetry.metrics())
    metrics["eval_wall_seconds"] = wall_seconds
    metrics["captured_emulator_frames"] = float(sum(row.total_frames for row in rows))
    metrics["emulator_fps"] = metrics["captured_emulator_frames"] / wall_seconds
    metrics["baseline_net_stock_per_min"] = baseline["net_stock_per_min"]
    if "net_stock_per_min" in metrics:
        metrics["net_stock_delta"] = metrics["net_stock_per_min"] - baseline["net_stock_per_min"]
        metrics["within_0p2_nsm"] = float(metrics["net_stock_delta"] >= -BASELINE_TOLERANCE_NSM)

    protocol = {
        "schema_version": 1,
        "git_sha": git_sha,
        "checkpoint_uri": CHECKPOINT_URI,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "bundle_sha256": _sha256(bundle),
        "baseline_r2_key": BASELINE_KEY,
        "baseline_net_stock_per_min": baseline["net_stock_per_min"],
        "matchup_schedule_sha256": schedule_hash,
        "n_matchups": MATCHUPS,
        "matchup_unit": "instant-restart boot; each boot can contain multiple games",
        "max_frames_per_boot": MAX_FRAMES,
        "max_parallel": 1,
        "cpu_level": 9,
        "ego_port": 1,
        "seed_stage": int(PRIOR_SWEEP_SEED_STAGE.value),
        "eval_seed": EVAL_SEED,
        "prediction_frames": 4,
        "transport_delay_frames": 2,
        "replan_interval_frames": 2,
        "return_target": "p90",
        "desired_return": p90,
        "player_identity": None,
        "temperature": 1.0,
        "history_mode": "kv_cache",
        "kv_update_frames": 2,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "melee_version": importlib.metadata.version("melee"),
        "iso_fixture_sha256": ISO.sha256,
        "dolphin_fixture_sha256": DOLPHIN_EXIAI.sha256,
    }
    _save_json(output / "protocol.json", protocol)
    _save_json(output / "baseline_metrics.json", baseline)
    _save_json(output / "match_rows.json", {"schema_version": 1, "rows": [row.as_dict() for row in rows]})
    _save_json(output / "metrics.json", metrics)
    artifacts = {str(path.relative_to(output)): _sha256(path) for path in sorted(output.rglob("*")) if path.is_file()}
    _save_json(output / "artifact_sha256.json", {"schema_version": 1, "files": artifacts})
    uploader = BackgroundUploader(run_name)
    uploader.upload_tree(output, base=output.parent)
    uploader.close()
    print(json.dumps({"run_name": run_name, "protocol": protocol, "metrics": metrics}, sort_keys=True), flush=True)
    if any(int(metrics.get(key, 0.0)) != MATCHUPS for key in ("scheduled_boots", "completed_boots", "boots")):
        raise RuntimeError("KV cache evaluation did not complete all 96 matchup boots")


if __name__ == "__main__":
    main()
