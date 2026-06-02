"""Run identity + lightweight instrumentation shared across experiments.

Keeps the wandb-run-naming convention and the ``time.monotonic()`` timing
boilerplate out of the experiment files. ``wandb.init`` itself stays in the
experiment (it's a one-liner and the tags are run-specific).
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


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
