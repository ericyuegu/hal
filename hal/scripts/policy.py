"""Export current policy artifacts and run declared local or netplay profiles."""

import argparse
import os
from collections.abc import Sequence
from functools import partial
from pathlib import Path

import melee
import torch

from hal.data.schema import Rank
from hal.eval.harness import default_session_cfg
from hal.eval.harness import resolve_parallelism
from hal.eval.harness import run_matches_vec
from hal.eval.matchups import matchups_for_vs_cpu
from hal.eval.netplay import run_netplay_match
from hal.eval.policy import PolicyBatchAdapter
from hal.eval.qualification import check_realtime_budget
from hal.eval.replays import require_completed_replay
from hal.eval.scheduling import FrameTiming
from hal.inference.action_sequence_artifact import export_action_sequence_policy
from hal.inference.action_sequence_artifact import load_action_sequence_policy
from hal.inference.api import PreparedInferenceProfile
from hal.inference.api import RuntimeConfig
from hal.inference.checkpoints import resolve_checkpoint
from hal.inference.engine import configure_inference_process
from hal.inference.engine import freeze_inference_runtime
from hal.inference.engine import start_inference_worker
from hal.paths import ISO_PATH
from hal.paths import NETPLAY_EMULATOR_PATH
from hal.sim.netplay import NetplaySession
from hal.sim.netplay import NetplaySetup
from hal.sim.rollout import VecMatch
from hal.sim.session import Matchup
from hal.sim.session import PlayerSetup

_RANK_IDENTITIES = tuple(rank.name for rank in (Rank.PLATINUM, Rank.DIAMOND, Rank.MASTER))


def _character(name: str) -> melee.Character:
    try:
        return melee.Character[name.upper()]
    except KeyError as error:
        raise argparse.ArgumentTypeError(f"unknown Melee character {name!r}") from error


def _user_json(override: str | None) -> Path:
    value = override if override is not None else os.environ.get("HAL_SLIPPI_USER_JSON")
    if not value:
        raise ValueError("set HAL_SLIPPI_USER_JSON or pass --user-json")
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Slippi user JSON does not exist: {path}")
    return path.resolve()


def _normalize_imitate(value: str) -> str:
    rank = value.upper()
    return rank if rank in _RANK_IDENTITIES else value


def _netplay_policy_settings(desired_return: float) -> tuple[float, float]:
    return desired_return, 1.0


def _policy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hal-policy")
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="export a supported 059 checkpoint")
    export.add_argument("checkpoint")
    export.add_argument("output", type=Path)
    evaluate = commands.add_parser("eval", help="evaluate an artifact against local CPUs")
    evaluate.add_argument("policy")
    evaluate.add_argument("--profile", choices=("local", "official-059", "local-stride-one"), default="local")
    evaluate.add_argument("--imitate", type=_normalize_imitate, default="IBDW#0")
    evaluate.add_argument("--n-matches", type=int, default=1)
    evaluate.add_argument("--max-parallel", type=int)
    evaluate.add_argument("--max-frames", type=int, default=7200)
    evaluate.add_argument("--cpu-level", type=int, choices=range(1, 10), default=9)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--compiled", action=argparse.BooleanOptionalAction, default=True)
    evaluate.add_argument("--replay-dir", type=Path)
    return parser


def _play_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hal-play")
    parser.add_argument("policy")
    parser.add_argument("opponent_code")
    parser.add_argument("--character", type=_character, default=melee.Character.FOX)
    parser.add_argument("--imitate", type=_normalize_imitate, default="IBDW#0")
    parser.add_argument("--online-delay", type=int, choices=(2, 3), default=2)
    parser.add_argument("--user-json")
    parser.add_argument("--replay-dir", type=Path, default=Path("replays/human-play"))
    parser.add_argument("--dolphin-path", default=NETPLAY_EMULATOR_PATH)
    parser.add_argument("--iso-path", default=ISO_PATH)
    parser.add_argument("--slippi-port", type=int, default=51441)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--compiled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-frames", type=int, default=54_000)
    return parser


def _eval(args: argparse.Namespace) -> None:
    if args.n_matches < 1:
        raise ValueError("n-matches must be positive")
    configure_inference_process()
    parallel = resolve_parallelism(args.n_matches, args.max_parallel)
    is_official = args.profile == "official-059"
    stride = 1 if args.profile == "local-stride-one" else 2
    timing = FrameTiming(0, 0, 2 if is_official else 0, stride, 4)
    policy = load_action_sequence_policy(
        resolve_checkpoint(args.policy),
        device=args.device,
        seed=args.seed,
        compiled=args.compiled,
        history_mode="window" if is_official else "kv_cache",
        kv_update_frames=2,
    )
    runtime = RuntimeConfig(parallel, (0,), replan_interval_frames=stride)
    policy.prepare_prediction(runtime, 4, timing.fixed_prefix_frames)
    matches = [
        VecMatch(
            Matchup(
                melee.Stage.FINAL_DESTINATION,
                (PlayerSetup(1, ego, cpu_level=0), PlayerSetup(2, cpu, cpu_level=args.cpu_level)),
            ),
            model_ports=(1,),
        )
        for ego, cpu in matchups_for_vs_cpu(args.n_matches)
    ]
    factory = partial(
        PolicyBatchAdapter,
        policy,
        runtime,
        timing,
        player_identity=args.imitate,
        desired_return=policy.return_p90,
        observed_actions=not is_official,
    )
    freeze_inference_runtime()
    boots = run_matches_vec(
        default_session_cfg(args.replay_dir),
        matches,
        factory,
        max_frames=args.max_frames,
        max_parallel=parallel,
    )
    completed = sum(len(boot) for boot in boots)
    if completed != args.n_matches:
        raise RuntimeError(f"completed {completed} of {args.n_matches} requested matches")
    print(f"completed {completed} matches with profile {args.profile}")


def _play(args: argparse.Namespace) -> None:
    configure_inference_process()
    runtime = RuntimeConfig(1, (args.online_delay,), replan_interval_frames=4)
    policy = load_action_sequence_policy(
        resolve_checkpoint(args.policy),
        device=args.device,
        seed=args.seed,
        compiled=args.compiled,
        history_mode="kv_cache",
        kv_update_frames=4,
    )
    user_json = _user_json(args.user_json)
    replay_dir = args.replay_dir.resolve()
    replay_dir.mkdir(parents=True, exist_ok=True)
    previous_replays = frozenset(replay_dir.rglob("*.slp"))
    budget = check_realtime_budget(policy, runtime, 0.0005, shape=(8, args.online_delay + 1))
    timing = next(item for item in budget.timings if item.physical_delay_frames == args.online_delay)
    profile = PreparedInferenceProfile(
        f"netplay-delay-{args.online_delay}",
        policy.checkpoint_sha256,
        "kv_cache",
        timing.prediction_horizon_frames,
        timing.fixed_prefix_frames,
        (1, 2, 4),
        1,
    )
    freeze_inference_runtime()
    with (
        start_inference_worker(policy, profile, 0.0005) as client,
        NetplaySession(
            args.iso_path,
            dolphin_path=args.dolphin_path,
            user_json_path=user_json,
            online_delay=args.online_delay,
            replay_dir=replay_dir,
            slippi_port=args.slippi_port,
            realtime=True,
        ) as session,
        torch.compiler.set_stance("fail_on_recompile"),
    ):
        result = run_netplay_match(
            session,
            NetplaySetup(character=args.character, opponent_code=args.opponent_code),
            client,
            runtime,
            timing,
            player_identity=args.imitate,
            policy_settings=partial(_netplay_policy_settings, policy.return_p90),
            max_frames=args.max_frames,
        )
    replay = require_completed_replay(replay_dir, previous_replays)
    print(
        f"completed {len(result.trajectory)} frames at {result.game_fps:.1f} FPS; "
        f"frame p95={result.frame_interval_p95_ms:.1f} ms; inference p95={result.inference_p95_ms:.1f} ms; "
        f"replay={replay}"
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = _policy_parser().parse_args(argv)
    if args.command == "export":
        export_action_sequence_policy(args.checkpoint, args.output)
        print(args.output.resolve())
        return
    _eval(args)


def play_main(argv: Sequence[str] | None = None) -> None:
    _play(_play_parser().parse_args(argv))


if __name__ == "__main__":
    main()
