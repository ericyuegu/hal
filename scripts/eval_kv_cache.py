"""Run the pinned 059 level-9 CPU protocol under a named inference profile."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from typing import cast

import torch
import tyro
import wandb

from hal.eval.cross_stage import PRIOR_SWEEP_SEED_STAGE
from hal.eval.cross_stage import sweep_vs_cpu_prior_with_rows
from hal.eval.cross_stage import vs_cpu_metrics
from hal.eval.harness import default_session_cfg
from hal.eval.matchups import matchups_for_vs_cpu
from hal.eval.policy import PolicyBatchAdapter
from hal.eval.scheduling import FrameTiming
from hal.fixtures import DOLPHIN_EXIAI
from hal.fixtures import ISO
from hal.inference.action_sequence_artifact import export_action_sequence_policy
from hal.inference.action_sequence_artifact import load_action_sequence_policy
from hal.inference.action_sequence_policy import ActionSequencePolicy
from hal.inference.api import RuntimeConfig
from hal.inference.checkpoints import resolve_checkpoint
from hal.inference.engine import configure_inference_process
from hal.inference.engine import freeze_inference_runtime
from hal.sim.process_vec import ProcessVecTelemetry
from hal.training.checkpoints import BackgroundUploader

CHECKPOINT_URI = "r2://hal/runs/260921-092548_059_muon_history_decoder_o59v5-cooldown32768/checkpoints/step-0131072.pt"
BASELINE_RUN = "ericyuegu/hal/vywk3cih"
BASELINE_HISTORY_STEP = 1322
BASELINE_NSM = 1.204894549414064
CHECKPOINT_SHA256 = "52b5233ed506f59f514f7e90a6a6111206152db7413f451dc35be5c30d1e671b"
MATCHUPS = 96
MAX_FRAMES = 7200
EVAL_SEED = 0
BASELINE_TOLERANCE_NSM = 0.2

type ProfileName = Literal["official-059", "cached-prefix-two", "local"]


@dataclass(frozen=True, slots=True)
class Args:
    profile: ProfileName = "official-059"
    """Named execution and timing protocol; every profile uses the pinned 96 boots."""


@dataclass(frozen=True, slots=True)
class EvalProfile:
    name: ProfileName
    history_mode: Literal["window", "kv_cache"]
    timing: FrameTiming
    observed_actions: bool


def profile_for(name: ProfileName) -> EvalProfile:
    """Keep each new timing treatment distinct from the scientific 059 control."""
    if name == "official-059":
        return EvalProfile(name, "window", FrameTiming(0, 0, 2, 2, 4), observed_actions=False)
    if name == "cached-prefix-two":
        return EvalProfile(name, "kv_cache", FrameTiming(0, 0, 2, 2, 4), observed_actions=True)
    if name == "local":
        return EvalProfile(name, "kv_cache", FrameTiming(0, 0, 0, 2, 4), observed_actions=True)
    raise ValueError(f"unsupported evaluation profile {name!r}")


@dataclass(slots=True)
class EvaluationPolicyFactory:
    """Advance one sampling seed per matchup wave without a captured closure."""

    policy: ActionSequencePolicy
    runtime: RuntimeConfig
    timing: FrameTiming
    desired_return: float
    observed_actions: bool
    next_wave: int = 0

    def __call__(self) -> PolicyBatchAdapter:
        seed = EVAL_SEED + self.next_wave
        self.next_wave += 1
        print(f"[eval] starting matchup wave with sampling seed {seed}", flush=True)
        self.policy.reset_prediction(seed=seed)
        return PolicyBatchAdapter(
            self.policy,
            self.runtime,
            self.timing,
            desired_return=self.desired_return,
            observed_actions=self.observed_actions,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _baseline() -> dict[str, float]:
    """Pin the historical dense 059 W&B row without reinterpreting its delay label."""
    run = wandb.Api(timeout=60).run(BASELINE_RUN)
    for row in run.scan_history(page_size=1500):
        if row.get("_step") != BASELINE_HISTORY_STEP:
            continue
        values = cast(dict[str, object], row)
        expected = {
            "eval/checkpoint_step": 131072,
            "eval/boots": MATCHUPS,
            "eval/crashed": 0,
            "eval/captured_emulator_frames": MATCHUPS * MAX_FRAMES,
            "eval/prediction_frames": 4,
            "eval/delay_frames": 2,
            "eval/replan_interval_frames": 2,
        }
        if any(values.get(key) != value for key, value in expected.items()):
            raise ValueError("W&B baseline has a different evaluation protocol")
        nsm = values.get("eval/net_stock_per_min")
        if not isinstance(nsm, int | float) or not math.isclose(nsm, BASELINE_NSM, abs_tol=1e-12):
            raise ValueError("W&B baseline NSM changed")
        return {
            key.removeprefix("eval/"): float(value)
            for key, value in values.items()
            if key.startswith("eval/") and isinstance(value, int | float)
        }
    raise ValueError("final p90 W&B evaluation row is missing")


def _save_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, allow_nan=False))
    temporary.replace(path)


def _git_sha() -> str:
    git_sha = os.environ["HAL_GIT_SHA"]
    if len(git_sha) != 40 or any(char not in "0123456789abcdef" for char in git_sha):
        raise ValueError("HAL_GIT_SHA must identify the committed source")
    return git_sha


def _gpu_record() -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("059 gameplay evaluation requires a CUDA device")
    properties = torch.cuda.get_device_properties(0)
    return {
        "name": torch.cuda.get_device_name(0),
        "total_memory_bytes": properties.total_memory,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def main(args: Args) -> None:
    git_sha = _git_sha()
    hardware = _gpu_record()
    configure_inference_process()
    profile = profile_for(args.profile)
    baseline = _baseline()
    print(f"[eval] historical 059 W&B baseline NSM={baseline['net_stock_per_min']:.4f}", flush=True)

    checkpoint = resolve_checkpoint(CHECKPOINT_URI)
    if _sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("downloaded checkpoint differs from the pinned 059 checkpoint")
    bundle = Path("/tmp/hal-kv-cache-eval.hal")
    manifest = export_action_sequence_policy(checkpoint, bundle)
    if manifest.source_sha256 != CHECKPOINT_SHA256:
        raise ValueError("exported bundle refers to a different checkpoint")
    runtime = RuntimeConfig(max_batch_size=1, transport_delays=(0,), replan_interval_frames=2)
    policy = load_action_sequence_policy(
        bundle,
        device="cuda",
        seed=EVAL_SEED,
        compiled=True,
        history_mode=profile.history_mode,
        kv_update_frames=2,
    )
    p90 = policy.return_p90
    if not math.isfinite(p90) or not 0.0 <= p90 <= 40.0:
        raise ValueError("checkpoint p90 return target is invalid")
    timing = profile.timing
    policy.prepare_prediction(runtime, timing.prediction_horizon_frames, timing.fixed_prefix_frames)
    freeze_inference_runtime()
    print(f"[eval] prepared {args.profile}; p90 return={p90:.6f}", flush=True)

    schedule = [(int(ego.value), int(cpu.value)) for ego, cpu in matchups_for_vs_cpu(MATCHUPS)]
    schedule_hash = hashlib.sha256(json.dumps(schedule, separators=(",", ":")).encode()).hexdigest()
    run_name = f"o59-eval-{args.profile}-{git_sha[:8]}"
    output = Path("runs") / run_name / "eval96-p90"
    output.mkdir(parents=True, exist_ok=False)
    telemetry = ProcessVecTelemetry()
    factory = EvaluationPolicyFactory(policy, runtime, timing, p90, profile.observed_actions)

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
    metrics["historical_baseline_net_stock_per_min"] = baseline["net_stock_per_min"]
    if "net_stock_per_min" in metrics:
        metrics["delta_from_historical_baseline"] = metrics["net_stock_per_min"] - baseline["net_stock_per_min"]
        metrics["above_historical_0p2_nsm_floor"] = float(
            metrics["delta_from_historical_baseline"] >= -BASELINE_TOLERANCE_NSM
        )

    protocol = {
        "schema_version": 2,
        "name": args.profile,
        "git_sha": git_sha,
        "checkpoint_uri": CHECKPOINT_URI,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "bundle_sha256": _sha256(bundle),
        "bundle_capability_version": policy.capability_version,
        "historical_baseline": {
            "wandb_run": BASELINE_RUN,
            "history_step": BASELINE_HISTORY_STEP,
            "net_stock_per_min": baseline["net_stock_per_min"],
            "recorded_delay_frames": 2,
            "comparison_kind": "historical descriptive baseline; matched same-hardware control is separate",
        },
        "matchup_schedule_sha256": schedule_hash,
        "n_matchups": MATCHUPS,
        "matchup_unit": "instant-restart boot; each boot can contain multiple games",
        "max_frames_per_boot": MAX_FRAMES,
        "max_parallel": 1,
        "cpu_level": 9,
        "ego_port": 1,
        "seed_stage": int(PRIOR_SWEEP_SEED_STAGE.value),
        "eval_seed": EVAL_SEED,
        "physical_delay_frames": timing.physical_delay_frames,
        "inference_allowance_frames": timing.inference_allowance_frames,
        "fixed_prefix_frames": timing.fixed_prefix_frames,
        "replan_interval_frames": timing.replan_interval_frames,
        "prediction_horizon_frames": timing.prediction_horizon_frames,
        "return_target": "p90",
        "desired_return": p90,
        "player_identity": None,
        "temperature": 1.0,
        "history_mode": profile.history_mode,
        "action_history": "observed" if profile.observed_actions else "intended",
        "kv_update_frames": 2 if profile.history_mode == "kv_cache" else None,
        "hardware": hardware,
        "melee_version": importlib.metadata.version("melee"),
        "iso_fixture_sha256": ISO.sha256,
        "dolphin_fixture_sha256": DOLPHIN_EXIAI.sha256,
    }
    _save_json(output / "protocol.json", protocol)
    _save_json(output / "historical_baseline_metrics.json", baseline)
    _save_json(output / "match_rows.json", {"schema_version": 1, "rows": [row.as_dict() for row in rows]})
    _save_json(output / "metrics.json", metrics)
    artifacts = {str(path.relative_to(output)): _sha256(path) for path in sorted(output.rglob("*")) if path.is_file()}
    _save_json(output / "artifact_sha256.json", {"schema_version": 1, "files": artifacts})
    uploader = BackgroundUploader(run_name)
    try:
        uploader.upload_tree(output, base=output.parent)
    finally:
        uploader.close()
    print(json.dumps({"run_name": run_name, "protocol": protocol, "metrics": metrics}, sort_keys=True), flush=True)
    if any(int(metrics.get(key, 0.0)) != MATCHUPS for key in ("scheduled_boots", "completed_boots", "boots")):
        raise RuntimeError("059 evaluation did not complete all 96 matchup boots")


if __name__ == "__main__":
    main(tyro.cli(Args))
