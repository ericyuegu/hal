"""One peer of a controlled, blocking 4/2/2 O59 netplay comparison."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import melee
import torch

from hal.eval.play import require_completed_replay
from hal.eval.play import run_netplay_match
from hal.inference.api import RuntimeConfig
from hal.inference.backends.history_decoder.policy import load_o59_policy
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("opponent_code")
    parser.add_argument("output", type=Path)
    parser.add_argument("--user-json", type=Path, required=True)
    parser.add_argument("--history-mode", choices=("window", "kv_cache"), required=True)
    parser.add_argument("--update-frames", type=int, choices=(1, 2), default=2)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--slippi-port", type=int, required=True)
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
    runtime = RuntimeConfig(1, (2,), replan_interval_frames=2)
    policy.prepare(runtime)
    replay_dir = args.output / "replays"
    replay_dir.mkdir()
    with (
        NetplaySession(
            ISO_PATH,
            dolphin_path=NETPLAY_EMULATOR_PATH,
            user_json_path=args.user_json,
            online_delay=2,
            replay_dir=replay_dir,
            slippi_port=args.slippi_port,
            realtime=False,
        ) as session,
        torch.compiler.set_stance("fail_on_recompile"),
    ):
        result = run_netplay_match(
            session,
            NetplaySetup(melee.Character.FOX, args.opponent_code),
            policy,
            runtime,
            player_identity="IBDW#0",
            max_frames=28800,
        )
    replay = require_completed_replay(replay_dir, ())
    completed = Trajectory.from_slp(replay)
    summary = {
        "config": {
            **vars(args),
            "batch_size": 1,
            "transport_frames": 2,
            "replan_frames": 2,
            "horizon": 4,
            "temperature": 1.0,
            "desired_return": 20.0,
            "character": "FOX",
            "identity": "IBDW#0",
            "stage": "FINAL_DESTINATION",
            "realtime": False,
        },
        "frames": len(result.trajectory),
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
                *sorted(Path("hal/inference").glob("*.py")),
                Path("hal/eval/play.py"),
                Path("hal/sim/netplay.py"),
            )
        },
    }
    (args.output / "results.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
