"""Profile synchronous O65 evaluation on a local checkpoint."""

import cProfile
import io
import json
import platform
import pstats
import subprocess
import sys
import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro

from hal.eval.conditioned_matchups import EvaluationStart
from hal.eval.conditioned_matchups import ScheduledMatch
from hal.eval.conditioned_matchups import build_batched_policy
from hal.eval.conditioned_matchups import build_policy
from hal.eval.conditioned_matchups import run_dolphin_match
from hal.eval.conditioned_matchups import run_dolphin_matches
from hal.eval.conditioned_matchups import run_msl_match
from hal.eval.conditioned_matchups import run_msl_matches
from hal.eval.harness import default_session_cfg
from hal.eval.self_play_checkpoint import load_evaluation_checkpoint
from hal.eval.self_play_checkpoint import sha256
from hal.eval.self_play_policy import PlayerCondition
from hal.sim.native import NativeMatch
from hal.sim.native import NativePlayer
from hal.sim.native import NativeRolloutBatch
from hal.sim.native_bridge import library_sha256


@dataclass(frozen=True, slots=True)
class Args:
    checkpoint: Path
    output: Path
    data_dir: Path
    simulator: Literal["msl", "dolphin"] = "msl"
    max_frames: int = 512
    delay: int = 8
    replan: int = 4
    execution: Literal["serial", "batched"] = "serial"
    matches: int = 1


def main(args: Args) -> None:
    if args.max_frames < 125:
        raise ValueError("measurement must include active frames")
    if args.matches < 1 or (args.execution == "serial" and args.matches != 1):
        raise ValueError("serial profiling requires one match; batched profiling requires a positive count")
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = load_evaluation_checkpoint(args.checkpoint, device=torch.device("cuda"))
    condition = PlayerCondition(delay=args.delay, replan=args.replan)
    schedule = tuple(
        ScheduledMatch(EvaluationStart("fd-fox-falco", 25, (1, 22)), 17 + index // 2, 1 + index % 2, str(index // 2))
        for index in range(args.matches)
    )
    inputs = {
        "format": "hal.o65.evaluation-profile.v1",
        "arguments": asdict(args),
        "condition": asdict(condition),
        "schedule": [asdict(game) for game in schedule],
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "checkpoint_provenance": checkpoint.provenance,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "sources": {
            name: sha256(Path(name))
            for name in (
                "hal/eval/batched_self_play.py",
                "hal/eval/conditioned_matchups.py",
                "hal/eval/self_play_policy.py",
                "hal/inference/column_context.py",
                "hal/inference/decision_decoder.py",
                "hal/inference/kv_cache.py",
                "scripts/profile_o65_evaluation.py",
            )
        },
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
    }
    (args.output / "inputs.json").write_text(json.dumps(inputs, indent=2, default=str) + "\n")
    torch.cuda.reset_peak_memory_stats()
    profile = cProfile.Profile()
    started = time.perf_counter()
    with profile:
        if args.simulator == "msl":
            with NativeRolloutBatch(args.data_dir, args.matches) as simulator:
                inputs["msl"] = {
                    "library_sha256": simulator.library_sha256,
                    "manifest_sha256": simulator.manifest_sha256,
                    "adapter_sha256": library_sha256(),
                }
                (args.output / "inputs.json").write_text(json.dumps(inputs, indent=2, default=str) + "\n")
                if args.execution == "batched":
                    policy = build_batched_policy((checkpoint, checkpoint), (condition, condition), schedule)
                    results = run_msl_matches(
                        simulator,
                        policy,
                        tuple(
                            NativeMatch(
                                25,
                                (NativePlayer(1, game.characters[0]), NativePlayer(2, game.characters[1])),
                                game.seed,
                            )
                            for game in schedule
                        ),
                        max_frames=args.max_frames,
                    )
                else:
                    serial_policy = build_policy(
                        (checkpoint, checkpoint), (condition, condition), seed=17, focal_port=1
                    )
                    results = [
                        run_msl_match(
                            simulator,
                            serial_policy,
                            NativeMatch(25, (NativePlayer(1, 1), NativePlayer(2, 22)), 17),
                            max_frames=args.max_frames,
                        )
                    ]
        else:
            if args.execution == "batched":
                live_results = run_dolphin_matches(
                    build_batched_policy((checkpoint, checkpoint), (condition, condition), schedule),
                    schedule,
                    max_frames=args.max_frames,
                    replay_dir=args.output / "replays",
                    labels=[{1: "O65-profile", 2: "O65-reference"} for _ in schedule],
                    session=default_session_cfg(),
                )
            else:
                serial_policy = build_policy((checkpoint, checkpoint), (condition, condition), seed=17, focal_port=1)
                live_results = [
                    run_dolphin_match(
                        serial_policy,
                        schedule[0].start,
                        (1, 22),
                        max_frames=args.max_frames,
                        replay_dir=args.output / "replays",
                        labels={1: "O65-profile", 2: "O65-reference"},
                        session=default_session_cfg(),
                    )
                ]
            results = [(trajectory, reason, seconds) for trajectory, reason, seconds, _ in live_results]
    torch.cuda.synchronize()
    total = time.perf_counter() - started
    seconds = sum(seconds for _, _, seconds in results)
    frames = sum(len(trajectory) for trajectory, _, _ in results)
    profile.dump_stats(args.output / "profile.pstats")
    stream = io.StringIO()
    pstats.Stats(profile, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(60)
    (args.output / "profile.txt").write_text(stream.getvalue())
    result = {
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "simulator": args.simulator,
        "delay": args.delay,
        "replan": args.replan,
        "execution": args.execution,
        "matches": args.matches,
        "frames": frames,
        "active_frames": sum(int((trajectory.frame_id >= 0).sum()) for trajectory, _, _ in results),
        "match_seconds": seconds,
        "total_seconds": total,
        "frames_per_second": frames / seconds,
        "stop_reasons": [reason for _, reason, _ in results],
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    (args.output / "measurement.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Args))
