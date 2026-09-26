"""One peer of a nonblocking netplay comparison with an explicit prediction shape."""

import argparse
import gc
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import melee
import torch

from hal.eval.netplay import run_netplay_match
from hal.eval.qualification import check_realtime_budget
from hal.eval.replays import require_completed_replay
from hal.eval.scheduling import ActionScheduler
from hal.eval.scheduling import FrameTiming
from hal.inference.api import RuntimeConfig
from hal.inference.backends.history_decoder.policy import load_o59_policy
from hal.inference.worker import start_inference_worker
from hal.paths import ISO_PATH
from hal.paths import NETPLAY_EMULATOR_PATH
from hal.sim.netplay import NetplaySession
from hal.sim.netplay import NetplaySetup
from hal.sim.trajectory import Trajectory


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ScheduleMeasurements:
    def __init__(self) -> None:
        self.counters: dict[str, int] = {}

    def observe_schedule(self, schedule: ActionScheduler) -> None:
        self.counters = {
            "deadline_misses": schedule.deadline_misses,
            "prefix_mismatches": schedule.prefix_mismatches,
            "exhausted_chunks": schedule.exhausted_chunks,
            "neutral_fallback_frames": schedule.neutral_fallback_frames,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("opponent_code")
    parser.add_argument("output", type=Path)
    parser.add_argument("--user-json", type=Path, required=True)
    parser.add_argument("--history-mode", choices=("window", "kv_cache"), required=True)
    parser.add_argument("--update-frames", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--slippi-port", type=int, required=True)
    parser.add_argument("--prediction-horizon", type=int, default=8)
    parser.add_argument("--thinking-allowance-frames", type=int, default=1)
    parser.add_argument("--replan-interval-frames", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    policy = load_o59_policy(
        args.bundle,
        device="cuda",
        seed=args.seed,
        compiled=True,
        history_mode=args.history_mode,
        kv_update_frames=args.update_frames,
    )
    runtime = RuntimeConfig(1, (2,), replan_interval_frames=args.replan_interval_frames)
    timing = FrameTiming(2, args.thinking_allowance_frames, args.replan_interval_frames, args.prediction_horizon)
    check_realtime_budget(
        policy, runtime, 0.0005, shape=(timing.prediction_horizon_frames, timing.fixed_prefix_frames)
    )
    gc.collect()
    gc.freeze()
    measurements = ScheduleMeasurements()
    replay_dir = args.output / "replays"
    replay_dir.mkdir()
    with (
        start_inference_worker(policy, timing.prediction_horizon_frames, 0.0005) as client,
        NetplaySession(
            ISO_PATH,
            dolphin_path=NETPLAY_EMULATOR_PATH,
            user_json_path=args.user_json,
            online_delay=2,
            replay_dir=replay_dir,
            slippi_port=args.slippi_port,
            realtime=True,
        ) as session,
        torch.compiler.set_stance("fail_on_recompile"),
    ):
        result = run_netplay_match(
            session,
            NetplaySetup(melee.Character.FOX, args.opponent_code),
            client,
            runtime,
            timing,
            player_identity="IBDW#0",
            max_frames=28800,
            schedule_observer=measurements,
        )
    replay = require_completed_replay(replay_dir, ())
    completed = Trajectory.from_slp(replay)
    summary = {
        "config": {
            **vars(args),
            "batch_size": 1,
            "timing": asdict(timing),
            "temperature": 1.0,
            "desired_return": 20.0,
            "character": "FOX",
            "identity": "IBDW#0",
            "stage": result.stage,
            "realtime": True,
        },
        "frames": len(result.trajectory),
        "gc_frozen": True,
        "schedule": measurements.counters,
        "inference_seconds": result.inference_seconds,
        "frame_interval_seconds": result.frame_interval_seconds,
        "dolphin_step_seconds": result.dolphin_step_seconds,
        "fps": result.game_fps,
        "wall_seconds": result.wall_seconds,
        "frame_p95_ms": result.frame_interval_p95_ms,
        "policy_p95_ms": result.inference_p95_ms,
        "dolphin_p95_ms": result.dolphin_step_p95_ms,
        "transport_corrections": result.transport_correction_frames,
        "ego_port": result.ego_port,
        "opponent_port": result.opponent_port,
        "final_stocks": {str(port): int(state["stock"][-1]) for port, state in completed.post.items()},
        "replay": str(replay),
        "replay_sha256": sha256(replay),
        "bundle_sha256": sha256(args.bundle),
        "user_json_sha256": sha256(args.user_json),
        "iso_sha256": sha256(Path(ISO_PATH)),
        "dolphin_sha256": sha256(Path(NETPLAY_EMULATOR_PATH)),
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "source_sha256": {
            str(path): sha256(path)
            for path in (
                Path(__file__),
                *sorted(Path("hal/inference").rglob("*.py")),
                Path("hal/eval/netplay.py"),
                Path("hal/eval/scheduling.py"),
                Path("hal/sim/netplay.py"),
            )
        },
    }
    (args.output / "results.json").write_text(json.dumps(summary, indent=2, default=str))
    print(
        json.dumps(
            {key: value for key, value in summary.items() if not key.endswith("_seconds")}, indent=2, default=str
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
