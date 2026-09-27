"""Profile 012 closed-loop eval throughput vs parallelism on this box.

Loads a checkpoint and runs the prior-distribution instant-restart vs-CPU sweep at
several `max_parallel` values, timing wall-clock and reporting eval frames/s plus the
pooled vs-CPU metrics (damage/stocks rates). Used to pick `eval_parallel_per_cpu` and
to confirm a past checkpoint still loads and produces comparable numbers.

Run:  uv run notebooks/profile_eval_parallelism.py --ckpt runs/<run>/latest.pt
"""

import argparse
import importlib.util
import os
import time
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "m012", Path(__file__).resolve().parent.parent / "experiments" / "012_multi_token.py"
)
m012 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m012)

from hal.eval.cross_stage import sweep_vs_cpu_prior  # noqa: E402
from hal.eval.cross_stage import vs_cpu_metrics  # noqa: E402
from hal.eval.harness import default_session_cfg  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--parallel", type=int, nargs="+", default=[3, 6, 12, 24])
    ap.add_argument("--max-frames", type=int, default=7200)
    ap.add_argument("--temp", type=float, default=None)
    args = ap.parse_args()

    model, cfg, stats, state = m012._load_ckpt(args.ckpt)
    print(
        f"[profile] loaded {args.ckpt}  step={state['step']}  device={m012.DEVICE}  cpus={os.cpu_count()}", flush=True
    )

    replay_dir = Path(args.ckpt).resolve().parent / "profile_eval_replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    session_cfg = default_session_cfg(replay_dir, instant_match_restart=True)

    def policy_factory():
        return m012.make_policy(model, stats, cfg, decode_temp=args.temp)

    rows = []
    for n in args.parallel:
        print(f"\n[profile] ===== max_parallel={n}, n_matchups={n}, max_frames={args.max_frames} =====", flush=True)
        t0 = time.monotonic()
        result = sweep_vs_cpu_prior(
            policy_factory,
            session_cfg=session_cfg,
            n_matchups=n,
            max_parallel=n,
            max_frames=args.max_frames,
        )
        wall = time.monotonic() - t0
        metrics = vs_cpu_metrics(result)
        total_frames = sum(s.frames for _, _, s in result if s is not None)
        fps = total_frames / wall if wall > 0 else 0.0
        rows.append((n, wall, total_frames, fps, metrics))
        print(
            f"[profile] n={n}: wall={wall:.1f}s  frames={total_frames}  eval_fps={fps:.1f}  "
            f"matches={metrics.get('matches', 0)}  crashed={metrics.get('crashed', 1.0):.2f}",
            flush=True,
        )
        print(f"           metrics={metrics}", flush=True)

    print("\n[profile] ================ SUMMARY ================", flush=True)
    print(
        f"{'n':>4} {'wall_s':>8} {'frames':>9} {'eval_fps':>9} {'x_realtime':>10} {'dmg_dealt/min':>14} {'dmg_taken/min':>14} {'matches':>8} {'crashed':>8}",
        flush=True,
    )
    for n, wall, frames, fps, met in rows:
        print(
            f"{n:>4} {wall:>8.1f} {frames:>9} {fps:>9.1f} {fps / 60:>10.2f} "
            f"{met.get('damage_dealt_per_min', 0):>14.1f} {met.get('damage_taken_per_min', 0):>14.1f} "
            f"{met.get('matches', 0):>8.0f} {met.get('crashed', 1.0):>8.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
