"""Run identity + lightweight instrumentation shared across experiments.

Keeps the W&B run conventions — naming, the experiment-number identity, the
``samples``/``tokens`` x-axes — and the ``time.monotonic()`` timing boilerplate
out of the experiment files. Experiments call ``init_wandb`` once (run-specific
tags stay at the call site) and route every metric through the returned
``RunLog``.
"""

import time
from collections.abc import Iterator
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import wandb


def experiment_number(file: str) -> str:
    """The leading file-number identity of an experiment module
    (``experiments/007_factorized_ar.py`` → ``"007"``)."""
    return Path(file).stem.split("_", 1)[0]


@dataclass(frozen=True, slots=True)
class RunLog:
    """``wandb.log`` wrapper that stamps cumulative-progress x-axes onto every row.

    ``samples`` = optimizer steps × effective batch; ``tokens`` = samples × frames per
    training window. Both land in every history row, so any W&B chart can plot against
    data seen — comparable across batch-size / grad-accum / window-length changes,
    unlike the step index."""

    samples_per_step: int  # batch_size * grad_accum_steps
    tokens_per_sample: int  # frames per training window (L_ctx + L_chunk)

    def log(self, metrics: dict[str, float], *, step: int) -> None:
        samples = step * self.samples_per_step
        wandb.log({"samples": samples, "tokens": samples * self.tokens_per_sample, **metrics}, step=step)


def init_wandb(
    *,
    experiment: str,
    run_name: str,
    tags: Sequence[str],
    cfg: dict,
    samples_per_step: int,
    tokens_per_sample: int,
    resume_state: dict | None = None,
) -> RunLog:
    """Start the project W&B run under the shared conventions — the experiment number as
    both a tag and a ``config`` key (groupable/filterable in the runs table), resume wired
    off the checkpoint's ``wandb_id`` — and return the ``RunLog`` all metrics go through."""
    wandb.init(
        project="hal",
        name=run_name,
        id=resume_state["wandb_id"] if resume_state else None,
        resume="allow" if resume_state else None,
        tags=[experiment, *tags],
        config={"experiment": experiment, **cfg},
    )
    return RunLog(samples_per_step=samples_per_step, tokens_per_sample=tokens_per_sample)


def make_run_name(model_tag: str, data_root: str, comment: str = "") -> str:
    """``YYMMDD-HHMMSS_<model_tag>_<data_tag>[_comment]``.

    ``model_tag`` is the experiment's own arch/hparam signature (e.g.
    ``fm-d256-L6-H8-Lc256-Lk16-fs8``); ``data_tag`` is derived from the dataset
    path so runs over the same data sort together.
    """
    stamp = datetime.now().strftime("%y%m%d-%H%M%S")
    data_tag = Path(data_root).parent.name.replace("anonymized", "anon")
    parts = [stamp, model_tag, data_tag]
    if comment:
        parts.append(comment)
    return "_".join(parts)


def setup_run_dir(run_name: str) -> tuple[Path, Path]:
    """Create ``runs/<run_name>/`` and its ``replays/`` subdir; return both."""
    ckpt_dir = Path("runs") / run_name
    replay_dir = ckpt_dir / "replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ckpt] writing checkpoints to {ckpt_dir}", flush=True)
    print(f"[ckpt] writing eval replays to {replay_dir}", flush=True)
    return ckpt_dir, replay_dir


@dataclass
class Stopwatch:
    """Elapsed wall-clock seconds for the ``profile`` block; set on exit."""

    elapsed: float = 0.0


@contextmanager
def profile(label: str) -> Iterator[Stopwatch]:
    """Time a block. Read ``.elapsed`` (seconds) after the ``with`` exits:

    with profile("step") as sw:
        ...
    wandb.log({"throughput/step_s": sw.elapsed})
    """
    t0 = time.monotonic()
    sw = Stopwatch()
    try:
        yield sw
    finally:
        sw.elapsed = time.monotonic() - t0
