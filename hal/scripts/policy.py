"""Export, evaluate, and play portable HAL policies."""

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

import melee
import torch

from hal.data.schema import Rank
from hal.eval.harness import default_session_cfg
from hal.eval.harness import resolve_parallelism
from hal.eval.harness import run_matches_vec
from hal.eval.matchups import matchups_for_vs_cpu
from hal.eval.play import require_completed_replay
from hal.eval.play import run_netplay_match
from hal.eval.policy import PolicyBatchAdapter
from hal.inference.api import RuntimeConfig
from hal.inference.checkpoints import resolve_checkpoint
from hal.inference.loader import load_policy
from hal.paths import ISO_PATH
from hal.paths import NETPLAY_EMULATOR_PATH
from hal.sim.netplay import NetplaySession
from hal.sim.netplay import NetplaySetup
from hal.sim.session import Matchup
from hal.sim.session import PlayerSetup
from hal.sim.vec import VecMatch

_RANK_IDENTITIES = tuple(rank.name for rank in (Rank.PLATINUM, Rank.DIAMOND, Rank.MASTER))


def _character(name: str) -> melee.Character:
    try:
        return melee.Character[name]
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


def _player_identity(value: str) -> str:
    rank = value.upper()
    return rank if rank in _RANK_IDENTITIES else value


def _policy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hal-policy")
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="convert one trusted O50 checkpoint to a portable bundle")
    export.add_argument("checkpoint")
    export.add_argument("output", type=Path)
    export.add_argument("--cache-root", type=Path, default=Path("runs"))

    evaluate = commands.add_parser("eval", help="run a portable policy against local CPUs")
    evaluate.add_argument("policy")
    evaluate.add_argument(
        "--imitate",
        type=_player_identity,
        default="IBDW#0",
        help="exact connect code or PLATINUM, DIAMOND, or MASTER",
    )
    evaluate.add_argument("--transport-delay", type=int, choices=(0, 2, 3), default=2)
    evaluate.add_argument("--replan-interval", type=int)
    evaluate.add_argument("--n-matches", type=int, default=1)
    evaluate.add_argument("--max-parallel", type=int)
    evaluate.add_argument("--max-frames", type=int, default=28_800)
    evaluate.add_argument("--cpu-level", type=int, choices=range(1, 10), default=9)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--seed", type=int)
    evaluate.add_argument("--compiled", action=argparse.BooleanOptionalAction, default=True)
    evaluate.add_argument("--replay-dir", type=Path)
    return parser


def _play_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hal-play")
    parser.add_argument("policy")
    parser.add_argument("opponent_code")
    parser.add_argument("--character", type=_character, default=melee.Character.FOX)
    parser.add_argument(
        "--imitate",
        type=_player_identity,
        default="IBDW#0",
        help="exact connect code or PLATINUM, DIAMOND, or MASTER",
    )
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


def _export(args: argparse.Namespace) -> None:
    from hal.inference.o50 import export_o50_policy

    export_o50_policy(args.checkpoint, args.output, cache_root=args.cache_root)
    print(args.output.resolve())


def _eval(args: argparse.Namespace) -> None:
    if args.n_matches < 1:
        raise ValueError("n-matches must be positive")
    parallel = resolve_parallelism(args.n_matches, args.max_parallel)
    runtime = RuntimeConfig(
        max_batch_size=parallel,
        transport_delay_frames=args.transport_delay,
        replan_interval_frames=args.replan_interval,
    )
    bundle = resolve_checkpoint(args.policy)
    policy = load_policy(bundle, device=args.device, seed=args.seed, compiled=args.compiled)
    policy.prepare(runtime)
    matches = [
        VecMatch(
            Matchup(
                melee.Stage.FINAL_DESTINATION,
                (
                    PlayerSetup(1, ego, cpu_level=0),
                    PlayerSetup(2, cpu, cpu_level=args.cpu_level),
                ),
            ),
            model_ports=(1,),
        )
        for ego, cpu in matchups_for_vs_cpu(args.n_matches)
    ]
    adapter = PolicyBatchAdapter(policy, runtime, player_identity=args.imitate)
    boots = run_matches_vec(
        default_session_cfg(args.replay_dir),
        matches,
        lambda: adapter,
        max_frames=args.max_frames,
        max_parallel=parallel,
    )
    trajectories = [trajectory for boot in boots for trajectory in boot]
    if len(trajectories) != args.n_matches:
        raise RuntimeError(f"completed {len(trajectories)} of {args.n_matches} requested matches")
    print(f"completed {len(trajectories)} matches")


def _play(args: argparse.Namespace) -> None:
    runtime = RuntimeConfig(max_batch_size=1, transport_delay_frames=args.online_delay)
    bundle = resolve_checkpoint(args.policy)
    policy = load_policy(bundle, device=args.device, seed=args.seed, compiled=args.compiled)
    policy.prepare(runtime)
    user_json = _user_json(args.user_json)
    replay_dir = args.replay_dir.resolve()
    replay_dir.mkdir(parents=True, exist_ok=True)
    previous_replays = frozenset(replay_dir.rglob("*.slp"))
    session = NetplaySession(
        args.iso_path,
        dolphin_path=args.dolphin_path,
        user_json_path=user_json,
        online_delay=args.online_delay,
        replay_dir=replay_dir,
        slippi_port=args.slippi_port,
    )
    setup = NetplaySetup(character=args.character, opponent_code=args.opponent_code)
    with session, torch.compiler.set_stance("fail_on_recompile"):
        result = run_netplay_match(
            session,
            setup,
            policy,
            runtime,
            player_identity=args.imitate,
            max_frames=args.max_frames,
        )
    replay = require_completed_replay(replay_dir, previous_replays)
    limit_ms = 33.3 if args.online_delay == 2 else 16.7
    if result.inference_p95_ms >= limit_ms:
        raise RuntimeError(f"policy p95 {result.inference_p95_ms:.1f} ms must stay below {limit_ms:.1f} ms")
    print(
        f"completed {len(result.trajectory)} frames; policy p95={result.inference_p95_ms:.1f} ms; "
        f"Slippi transport corrections={result.transport_correction_frames}; replay={replay}"
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = _policy_parser().parse_args(argv)
    if args.command == "export":
        _export(args)
    elif args.command == "eval":
        _eval(args)
    else:
        raise AssertionError(args.command)


def play_main(argv: Sequence[str] | None = None) -> None:
    _play(_play_parser().parse_args(argv))
