"""Bounded CPU self-play of the pinned O59 policy in the native simulator."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import melee
import torch

from hal.eval.native_rollout import first_difference
from hal.eval.native_rollout import replay_inputs
from hal.eval.native_rollout import run_episode
from hal.eval.policy import PolicyBatchAdapter
from hal.eval.scheduling import FrameTiming
from hal.inference.action_sequence_artifact import load_action_sequence_policy
from hal.inference.api import RuntimeConfig
from hal.inference.bundle import read_policy_manifest
from hal.sim.native import NativeMatch
from hal.sim.native import NativePlayer
from hal.sim.native import NativeRolloutBatch

DEFAULT_BUNDLE = Path(__file__).parents[1] / "runs/netplay/o59-vywk3cih.hal"
CHECKPOINT_SHA256 = "52b5233ed506f59f514f7e90a6a6111206152db7413f451dc35be5c30d1e671b"
TIMING = FrameTiming(2, 0, 2, 2, 4)
RUNTIME = RuntimeConfig(2, (2,), replan_interval_frames=2)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Extracted melee-sim-light assets")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--native-seed", type=int, default=0)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> dict:
    if args.episodes < 1 or args.max_frames < 1 or args.torch_threads < 1:
        raise ValueError("episodes, max-frames, and torch-threads must be positive")
    if args.output.exists():
        raise FileExistsError(f"output directory already exists: {args.output}")
    manifest = read_policy_manifest(args.bundle)
    if manifest.source_sha256 != CHECKPOINT_SHA256 or manifest.supported_transport_delays != (2,):
        raise ValueError("bundle is not the pinned two-frame-delay O59 checkpoint")
    if len(manifest.required_observation_fields) != 71:
        raise ValueError("O59 bundle does not declare 71 observation fields")
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    loaded_at = time.perf_counter()
    policy = load_action_sequence_policy(
        args.bundle,
        device="cpu",
        seed=args.policy_seed,
        compiled=False,
        history_mode="kv_cache",
        kv_update_frames=2,
    )
    policy.prepare_prediction(RUNTIME, 4, 2)
    preparation_seconds = time.perf_counter() - loaded_at

    args.output.mkdir(parents=True, exist_ok=False)
    summaries = []
    with NativeRolloutBatch(args.data_dir, 1) as simulator:
        native_manifest_sha256 = simulator.manifest_sha256
        native_library_sha256 = simulator.library_sha256
        for index in range(args.episodes):
            native_seed = args.native_seed + index
            policy_seed = args.policy_seed + index
            match = NativeMatch(
                stage=int(melee.Stage.FINAL_DESTINATION.value),
                players=(
                    NativePlayer(port=1, character=int(melee.Character.FOX.value)),
                    NativePlayer(port=2, character=int(melee.Character.FALCO.value)),
                ),
                seed=native_seed,
            )

            def new_adapter(seed: int) -> PolicyBatchAdapter:
                policy.reset_prediction(seed=seed)
                return PolicyBatchAdapter(
                    policy,
                    RUNTIME,
                    TIMING,
                    player_identity=None,
                    desired_return=policy.return_p90,
                    temperature=1.0,
                    observed_actions=False,
                )

            primary = run_episode(simulator, new_adapter(policy_seed), match, max_frames=args.max_frames)
            trace_path = args.output / f"episode-{index:02d}.npz"
            primary.save(trace_path)
            sim_difference, replay_seconds = replay_inputs(simulator, match, primary)
            if sim_difference is not None:
                raise RuntimeError(f"native input replay differs in {sim_difference[0]} row {sim_difference[1]}")
            policy_difference = None
            policy_repeat_seconds = None
            if index == 0:
                repeat = run_episode(simulator, new_adapter(policy_seed), match, max_frames=args.max_frames)
                repeat.save(args.output / "episode-00-repeat.npz")
                policy_difference = first_difference(primary, repeat)
                policy_repeat_seconds = repeat.wall_seconds
                if policy_difference is not None:
                    raise RuntimeError(
                        f"closed-loop policy repeat differs in {policy_difference[0]} row {policy_difference[1]}"
                    )
            summary = {
                "schema_version": 1,
                "trace": trace_path.name,
                "bundle": str(args.bundle.resolve()),
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "data_dir": str(args.data_dir.resolve()),
                "native_manifest_sha256": native_manifest_sha256,
                "native_library_sha256": native_library_sha256,
                "native_seed": native_seed,
                "policy_seed": policy_seed,
                "native_match": asdict(match),
                "frame_budget": args.max_frames,
                "stage": int(melee.Stage.FINAL_DESTINATION.value),
                "characters_by_port": [int(melee.Character.FOX.value), int(melee.Character.FALCO.value)],
                "timing": asdict(TIMING),
                "observed_actions": False,
                "desired_return": policy.return_p90,
                "steps": primary.steps,
                "first_frame_id": int(primary.arrays["frame_id"][0]),
                "last_frame_id": int(primary.arrays["frame_id"][-1]),
                "stop_reason": primary.stop_reason,
                "wall_seconds": primary.wall_seconds,
                "simulator_seconds": primary.simulator_seconds,
                "policy_seconds": primary.policy_seconds,
                "full_frames_per_second": primary.steps / primary.wall_seconds,
                "simulator_frames_per_second": primary.steps / primary.simulator_seconds,
                "input_replay_seconds": replay_seconds,
                "input_replay_exact": True,
                "closed_loop_repeat_seconds": policy_repeat_seconds,
                "closed_loop_repeat_exact": None if index else True,
            }
            (args.output / f"episode-{index:02d}.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n"
            )
            summaries.append(summary)
            print(
                f"episode {index}: {primary.steps} steps, {primary.stop_reason}, "
                f"{summary['full_frames_per_second']:.1f} full FPS",
                flush=True,
            )
    run_summary = {
        "schema_version": 1,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "native_manifest_sha256": native_manifest_sha256,
        "native_library_sha256": native_library_sha256,
        "preparation_seconds": preparation_seconds,
        "torch_threads": args.torch_threads,
        "episodes": summaries,
    }
    (args.output / "summary.json").write_text(json.dumps(run_summary, indent=2, sort_keys=True) + "\n")
    return run_summary


if __name__ == "__main__":
    evaluate(_arguments())
