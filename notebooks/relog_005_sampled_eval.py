"""Throwaway: re-run 005's closed-loop eval with SAMPLE decode and re-log to W&B.

Run 260605-004423 (wandb id n8ylc0ln) trained with cfg.decode="argmax", so its
``eval/*`` closed-loop curve is the do-nothing fixed-point collapse, not the real
controller. This loads a coarse subset of its checkpoints, re-evaluates each with
decode="sample" @ temp 1.0 (the deployed controller), and logs the corrected curve
back into the SAME run under ``eval_sample/*`` against a custom ``ckpt_step`` x-axis
(W&B history is append-only, so the old argmax points stay; the new namespace plots
at the identical numeric step for direct comparison).

    uv run notebooks/relog_005_sampled_eval.py
"""

# %%
import importlib.util
import tempfile
from dataclasses import replace
from pathlib import Path

import wandb

RUN = "260605-004423_cls-multi_label-naive_bins-b21-d256-L6-Lc256-Lk16-tp64_ranked-anon-1_cls-131k-b256"
WANDB_ID = "n8ylc0ln"
EVERY_NTH = 4  # of the 38 step ckpts -> ~10 points across the run

# 005's filename starts with a digit, so it can't be imported by name.
_spec = importlib.util.spec_from_file_location("cls005", "experiments/005_classification.py")
cls005 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cls005)


# %%
# Everything below MUST be guarded: the closed-loop harness spawns Dolphin sessions via a
# forkserver that re-imports THIS file. Without the guard each worker would re-run
# wandb.init(id=...) -> "run ID in use" -> the worker dies -> every match crashes.
def main() -> None:
    ckpts = sorted((Path("runs") / RUN).glob("step_*.pt"))
    subset = ckpts[::EVERY_NTH]
    if ckpts[-1] not in subset:
        subset.append(ckpts[-1])
    print(f"[relog] {len(ckpts)} ckpts -> evaluating {len(subset)}: {[p.stem for p in subset]}", flush=True)

    wandb.init(project="hal", id=WANDB_ID, resume="must")
    wandb.define_metric("ckpt_step")
    wandb.define_metric("eval_sample/*", step_metric="ckpt_step")

    tmp = Path(tempfile.mkdtemp(prefix="relog005_"))
    for ckpt in subset:
        model, cfg, stats, state = cls005._load_ckpt(str(ckpt))
        cfg = replace(cfg, decode="sample", decode_temp=1.0)
        step = state["step"]
        metrics = cls005.eval_vs_cpu(
            model, stats, cfg, max_frames=cfg.eval_max_frames, replay_dir=tmp / f"step_{step:06d}"
        )
        wandb.log({**{f"eval_sample/{k}": v for k, v in metrics.items()}, "ckpt_step": step})
        print(f"[relog] step {step}: {metrics}", flush=True)

    wandb.finish()
    print(f"[relog] done; replays in {tmp}", flush=True)


if __name__ == "__main__":
    main()
