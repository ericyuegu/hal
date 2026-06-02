"""Stress the closed-loop eval lifecycle at parallelism before trusting it on vast.

Repeats N eval rounds vs the lvl-9 CPU with a neutral policy (no model/streaming —
isolates the hal/sim + hal/eval emulator path that's been flaky headless), watching
for the things that bite in a long cloud run: leaked Dolphin processes between rounds,
RSS / open-fd growth (leakage), aggregate stepping throughput, and clean teardown.

Run in the cloud image via compose so Xvfb + the exi-ai Dolphin match vast:

    docker compose -f docker/compose.yaml run --rm hal \
        uv run notebooks/eval_stress.py --rounds 3 --replicas 8 --max-parallel 8

A healthy result: post-round dolphin count returns to 0 every round, and RSS/fds
plateau across rounds rather than climbing.
"""

import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import melee
import tyro
from loguru import logger

from hal.eval.cross_stage import sweep_vs_cpu
from hal.eval.harness import default_session_cfg
from hal.sim.sources import _NEUTRAL_INPUTS
from hal.sim.vec import Slot


class NeutralPolicy:
    """BatchPolicy holding neutral on every model slot — drives the emulator lifecycle
    without a torch model."""

    def __call__(self, frame_index: int, obs: Mapping[Slot, dict]) -> Mapping[Slot, object]:
        return {slot: _NEUTRAL_INPUTS for slot in obs}


def _dolphin_pids() -> list[int]:
    pids = []
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            comm = Path(f"/proc/{p}/comm").read_text().strip()
        except OSError:
            continue
        if "dolphin" in comm.lower():
            pids.append(int(p))
    return pids


def _rss_mb() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    return -1.0


@dataclass(frozen=True)
class Args:
    rounds: int = 3
    replicas: int = 8
    """matches per stage (1 stage here), all run in waves of max_parallel."""
    max_parallel: int = 8
    max_frames: int = 2400
    keep_replays: bool = False


def main(args: Args) -> None:
    logger.info(
        f"eval stress: {args.rounds} rounds x {args.replicas} matches | max_parallel={args.max_parallel} "
        f"| max_frames={args.max_frames} | starting dolphins={len(_dolphin_pids())}"
    )
    for rnd in range(args.rounds):
        pre_dol, pre_rss, pre_fd = len(_dolphin_pids()), _rss_mb(), len(os.listdir("/proc/self/fd"))
        replay_dir = Path(f"/opt/hal/data/evalstress/round{rnd}") if args.keep_replays else None
        t0 = time.monotonic()
        results = sweep_vs_cpu(
            lambda: NeutralPolicy(),
            session_cfg=default_session_cfg(replay_dir),
            stages=(melee.Stage.FINAL_DESTINATION,),
            replicas=args.replicas,
            max_parallel=args.max_parallel,
            max_frames=args.max_frames,
        )
        dt = time.monotonic() - t0
        ok = sum(1 for _, _, s in results if s is not None)
        time.sleep(3)  # let __exit__/atexit teardown settle before counting leftovers
        post_dol, post_rss, post_fd = len(_dolphin_pids()), _rss_mb(), len(os.listdir("/proc/self/fd"))
        logger.info(
            f"round {rnd}: {ok}/{len(results)} matches ok in {dt:.1f}s "
            f"| ~{ok * args.max_frames / dt:.0f} frame-steps/s "
            f"| dolphins {pre_dol}->[run]->{post_dol} (want 0) "
            f"| RSS {pre_rss:.0f}->{post_rss:.0f}MB | fds {pre_fd}->{post_fd}"
        )
    leftover = len(_dolphin_pids())
    logger.info(f"done | leftover dolphins={leftover} (want 0) | final RSS {_rss_mb():.0f}MB")


if __name__ == "__main__":
    main(tyro.cli(Args))
