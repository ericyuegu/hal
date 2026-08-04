"""Run a mirrored head-to-head between two checkpoints and print the paired summary.

This is the thin torch-carrying layer over ``hal.eval.h2h``. It resolves each model spec
(name, checkpoint, experiment module) into a ``PolicyBuilder`` and hands the builders to
the library, which stays torch-free.

An experiment loads through the convention every checkpoint-carrying experiment exposes:
``_load_ckpt`` returns ``(model, cfg, stats, state)``, ``_decode_settings`` reads the
decode knobs out of the saved cfg, and ``make_policy`` builds one closed-loop policy per
eval wave. The settings therefore come from each checkpoint's own cfg, which reproduces
that run's final-eval decode.

Run:
    python -m hal.scripts.h2h \\
        --model-a.name 019-factored --model-a.checkpoint <path> \\
        --model-a.experiment experiments.019_factored_frame \\
        --model-b.name 016-base --model-b.checkpoint <path> \\
        --model-b.experiment experiments.016_spatial_features \\
        --n-configs 64 --out-dir <dir>
"""

import importlib
import importlib.util
import os

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any
from typing import Literal

import melee
import tyro
from loguru import logger

from hal.eval.h2h import DEFAULT_MAX_FRAMES
from hal.eval.h2h import DEFAULT_START_RETRIES
from hal.eval.h2h import PolicyBuilder
from hal.eval.h2h import run_h2h
from hal.eval.paired import summarize_paired
from hal.policy import INCLUDED_STAGES

# Decode families. A receding-horizon experiment (013 / 016 / 019 / 020) takes the model
# in ``_decode_settings`` and per-knob keywords in ``make_policy``; an RLE token
# experiment (018) takes only the cfg and passes the settings object straight through.
Family = Literal["receding_horizon", "rle_token"]


@dataclass(frozen=True)
class ModelArgs:
    """One side of the head-to-head."""

    name: str
    """Label for the model. It names the replay directories and is stamped into every .slp."""
    checkpoint: str
    """Path to the .pt checkpoint."""
    experiment: str
    """Experiment module: a dotted name (experiments.016_spatial_features) or a .py path."""
    family: Family = "receding_horizon"
    """Decode family of the experiment. Only 018-style token policies use rle_token."""
    exec_cadence: int = 1
    """rle_token only: frames between replans. 1 replans every frame; 0 is token-native,
    which collapses in closed loop, so a token-native run must ask for it explicitly."""


@dataclass(frozen=True)
class Args:
    """Mirrored model-vs-model head-to-head."""

    model_a: ModelArgs
    model_b: ModelArgs
    out_dir: str
    """Directory for meta.json, matches.jsonl and the replays."""
    n_configs: int = 64
    """Prior-drawn configs. Each plays twice, once per orientation."""
    max_frames: int = DEFAULT_MAX_FRAMES
    """Frame budget per match."""
    max_parallel: int = 0
    """Concurrent Dolphin boots. 0 means one per CPU."""
    start_retries: int = DEFAULT_START_RETRIES
    """Retries for a boot that never reaches the first in-game frame."""
    seed: int = 0
    """Base decode seed."""
    stages: tuple[str, ...] = ()
    """Stage names. Empty means hal.policy.INCLUDED_STAGES."""
    verify_inputs: bool = True
    """Read each .slp back for per-port controller activity (the dead-policy tripwire)."""


def import_experiment(spec: str) -> ModuleType:
    """Import an experiment by dotted name or by file path.

    Experiment filenames start with a digit, so the ``import`` statement cannot name one;
    importlib can, both through the ``experiments`` namespace package and from a path.
    """
    if spec.endswith(".py"):
        path = Path(spec).resolve()
        module_spec = importlib.util.spec_from_file_location(path.stem, path)
        if module_spec is None or module_spec.loader is None:
            raise ValueError(f"cannot load an experiment module from {path}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        return module
    return importlib.import_module(spec)


def load_policy_builder(args: ModelArgs) -> tuple[PolicyBuilder, dict[str, Any]]:
    """Resolve one model spec into a per-wave policy builder plus its eval protocol.

    The protocol is what goes into ``meta.json``: enough to reconstruct which weights and
    which decode settings produced the replays.
    """
    module = import_experiment(args.experiment)
    for symbol in ("_load_ckpt", "_decode_settings", "make_policy"):
        if not hasattr(module, symbol):
            raise ValueError(f"{args.experiment} has no {symbol}; it does not follow the checkpoint convention")
    model, cfg, stats, state = module._load_ckpt(args.checkpoint)
    protocol: dict[str, Any] = {
        "name": args.name,
        "experiment": args.experiment,
        "checkpoint": args.checkpoint,
        "family": args.family,
        "step": int(state["step"]),
        "L_ctx": cfg.L_ctx,
    }

    if args.family == "rle_token":
        settings = module._decode_settings(cfg, exec_cadence=args.exec_cadence)
        protocol.update({"decode_settings": _settings_dict(settings)})

        def build(seed: int) -> Any:
            return module.make_policy(model, stats, cfg, settings=settings, decode_seed=seed)

        return build, protocol

    settings = module._decode_settings(model, cfg)
    protocol.update(
        {
            "decode_settings": _settings_dict(settings),
            "exec_horizon": cfg.exec_horizon,
            "head_offsets": list(cfg.head_offsets),
        }
    )

    def build_receding_horizon(seed: int) -> Any:
        return module.make_policy(
            model,
            stats,
            cfg,
            exec_horizon=cfg.exec_horizon,
            decode_temp=settings.temp,
            decode_temps=settings.temps,
            decode_btn_support_min=settings.btn_support_min,
            decode_min_p=settings.min_p,
            decode_click_trigger_fix=settings.click_trigger_fix,
            decode_seed=seed,
        )

    return build_receding_horizon, protocol


def _settings_dict(settings: Any) -> dict[str, Any]:
    """Experiment-local ``DecodeSettings`` as plain JSON-able fields."""
    return {f: getattr(settings, f) for f in settings.__dataclass_fields__}


def _git_revision() -> str:
    result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip()


def main(args: Args) -> None:
    stages = INCLUDED_STAGES if not args.stages else tuple(getattr(melee.Stage, s.upper()) for s in args.stages)
    logger.info(f"loading {args.model_a.name} and {args.model_b.name}")
    build_a, protocol_a = load_policy_builder(args.model_a)
    build_b, protocol_b = load_policy_builder(args.model_b)
    for protocol in (protocol_a, protocol_b):
        logger.info(f"  {protocol}")

    records = run_h2h(
        build_a,
        build_b,
        name_a=args.model_a.name,
        name_b=args.model_b.name,
        n_configs=args.n_configs,
        out_dir=args.out_dir,
        stages=stages,
        max_frames=args.max_frames,
        max_parallel=args.max_parallel,
        start_retries=args.start_retries,
        seed=args.seed,
        verify_inputs=args.verify_inputs,
        meta={"models": {args.model_a.name: protocol_a, args.model_b.name: protocol_b}, "git": _git_revision()},
    )
    print(summarize_paired(records, focal_model=args.model_a.name).format_table())


if __name__ == "__main__":
    main(tyro.cli(Args))
