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

Historical interleaved-token 013 (use the exact commit recorded by W&B):
    python -m hal.scripts.h2h \\
        --model-a.name 013-interleaved --model-a.checkpoint <path> \\
        --model-a.experiment git:<commit>:experiments/013_interleaved_groups.py \\
        --model-a.family historical_interleaved_groups \\
        --model-b.name 026 --model-b.checkpoint <path> \\
        --model-b.experiment experiments.026_temporal_mtp \\
        --n-configs 128 --out-dir <dir>
"""

import hashlib
import importlib
import importlib.util
import inspect
import math
import os
import re
import sys
from contextlib import contextmanager

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import subprocess
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any
from typing import Literal

import melee
import tyro
from loguru import logger

from hal import r2
from hal.data.schema import Rank
from hal.eval.action_trace import ACTION_TRACE_SCHEMA_VERSION
from hal.eval.action_trace import ActionTraceWriter
from hal.eval.h2h import DEFAULT_MAX_FRAMES
from hal.eval.h2h import DEFAULT_START_RETRIES
from hal.eval.h2h import PolicyBuilder
from hal.eval.h2h import orientation_replay_dir
from hal.eval.h2h import path_safe
from hal.eval.h2h import run_h2h
from hal.eval.paired import summarize_paired
from hal.inference.checkpoints import resolve_checkpoint
from hal.policy import INCLUDED_STAGES
from hal.training.checkpoints import BackgroundUploader
from hal.training.checkpoints import checkpoint_sha256
from hal.training.player_identity import PlayerVocabulary
from hal.training.player_identity import decode_player_codes

# Decode families. A receding-horizon experiment (013 / 016 / 019 / 020) takes the model
# in ``_decode_settings`` and per-knob keywords in ``make_policy``; an RLE token
# experiment (018) takes only the cfg and passes the settings object straight through.
Family = Literal["receding_horizon", "rle_token", "historical_interleaved_groups", "temporal_mtp"]


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
    """Decode family. ``historical_interleaved_groups`` is the June 2026 013
    four-full-trunk-pass decoder; its source may be loaded straight from Git."""
    exec_cadence: int = 1
    """rle_token only: frames between replans. 1 replans every frame; 0 is token-native,
    which collapses in closed loop, so a token-native run must ask for it explicitly."""
    player_identity: str | None = None
    """temporal_mtp only: exact connect code or PLATINUM, DIAMOND, or MASTER."""
    prediction_frames: int | None = None
    """temporal_mtp only: dense action heads to decode."""
    delay_frames: int | None = None
    """temporal_mtp only: transport-delay frames discarded from each plan."""
    replan_interval_frames: int | None = None
    """temporal_mtp only: actions executed before decoding another plan."""
    character: str | None = None
    """Fixed Melee character for this model. None keeps the prior draw."""
    temperature: float | None = None
    """temporal_mtp only: categorical sampling temperature. None keeps the experiment default."""


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
    upload_run: str | None = None
    """Unique evaluation name below the R2 h2h-evals prefix. None keeps artifacts local."""
    upload_prefix: str = "h2h-evals"
    """R2 prefix containing upload_run. Each path component must be alphanumeric, dot, underscore, or hyphen."""
    trace_actions: bool = False
    """Persist the exact logits and likelihood of each executed temporal-MTP action."""


def import_experiment(spec: str) -> ModuleType:
    """Import an experiment by dotted name or by file path.

    Experiment filenames start with a digit, so the ``import`` statement cannot name one;
    importlib can, both through the ``experiments`` namespace package and from a path.
    """
    if spec.startswith("git:"):
        return _import_git_experiment(spec)
    if spec.endswith(".py"):
        path = Path(spec).resolve()
        module_spec = importlib.util.spec_from_file_location(path.stem, path)
        if module_spec is None or module_spec.loader is None:
            raise ValueError(f"cannot load an experiment module from {path}")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = module
        try:
            module_spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_spec.name, None)
            raise
        return module
    return importlib.import_module(spec)


@contextmanager
def _historical_import_aliases():
    """Temporarily expose the two module names used by the June 2026 source.

    Both modules were moves, not semantic rewrites. Keeping the aliases scoped to
    execution of the old source avoids adding permanent compatibility modules to
    the training package.
    """
    from hal.data import feature_stats
    from hal.training import ego_stats

    aliases = {
        "hal.data.stats": feature_stats,
        "hal.training.stats": ego_stats,
    }
    previous = {name: sys.modules.get(name) for name in aliases}
    sys.modules.update(aliases)
    try:
        yield
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def _import_git_experiment(spec: str) -> ModuleType:
    """Import ``git:<revision>:<repo-relative.py>`` without changing the worktree.

    The resolved commit and blob hashes are attached to the module and later
    written to ``meta.json``. An evaluator can therefore use the exact W&B source
    revision instead of checking out or copying a historical branch tip.
    """
    try:
        revision, source_path = spec[4:].split(":", 1)
    except ValueError as exc:
        raise ValueError("git experiment must be git:<revision>:<repo-relative.py>") from exc
    if not revision or not source_path or not source_path.endswith(".py"):
        raise ValueError("git experiment must be git:<revision>:<repo-relative.py>")
    commit = subprocess.run(["git", "rev-parse", "--verify", f"{revision}^{{commit}}"], capture_output=True, text=True)
    if commit.returncode:
        raise ValueError(f"cannot resolve Git revision {revision!r}: {commit.stderr.strip()}")
    commit_sha = commit.stdout.strip()
    shown = subprocess.run(["git", "show", f"{commit_sha}:{source_path}"], capture_output=True, text=True)
    if shown.returncode:
        raise ValueError(f"cannot read {source_path!r} at {commit_sha}: {shown.stderr.strip()}")
    source = shown.stdout
    blob = subprocess.run(
        ["git", "rev-parse", f"{commit_sha}:{source_path}"], capture_output=True, text=True, check=True
    ).stdout.strip()
    module_name = f"_hal_historical_{hashlib.sha256(f'{commit_sha}:{source_path}'.encode()).hexdigest()[:16]}"
    module = ModuleType(module_name)
    module.__file__ = f"git:{commit_sha}:{source_path}"
    module.__package__ = ""
    module.__dict__["__hal_source__"] = {"commit": commit_sha, "blob": blob, "path": source_path}
    sys.modules[module_name] = module
    try:
        with _historical_import_aliases():
            exec(compile(source, module.__file__, "exec"), module.__dict__)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def load_policy_builder(
    args: ModelArgs,
    *,
    action_trace: ActionTraceWriter | None = None,
) -> tuple[PolicyBuilder, dict[str, Any]]:
    """Resolve one model spec into a per-wave policy builder plus its eval protocol.

    The protocol is what goes into ``meta.json``: enough to reconstruct which weights and
    which decode settings produced the replays.
    """
    module = import_experiment(args.experiment)
    if action_trace is not None and args.family != "temporal_mtp":
        raise ValueError("action tracing requires family='temporal_mtp'")
    if args.temperature is not None and args.family != "temporal_mtp":
        raise ValueError("temperature overrides require family='temporal_mtp'")
    if args.family == "historical_interleaved_groups":
        required = ("_load_ckpt", "decode", "RecedingHorizon")
    elif args.family == "temporal_mtp":
        required = ("load_checkpoint", "make_policy")
    else:
        required = ("_load_ckpt", "_decode_settings", "make_policy")
    for symbol in required:
        if not hasattr(module, symbol):
            raise ValueError(f"{args.experiment} has no {symbol}; it does not follow the checkpoint convention")
    load_checkpoint = module.load_checkpoint if args.family == "temporal_mtp" else module._load_ckpt
    checkpoint = resolve_checkpoint(args.checkpoint) if args.checkpoint.startswith("r2://") else Path(args.checkpoint)
    model, cfg, stats, state = load_checkpoint(str(checkpoint))
    architecture: Any = getattr(cfg, "arch", None)
    context_frames = cfg.L_ctx if hasattr(cfg, "L_ctx") else architecture.L_ctx
    protocol: dict[str, Any] = {
        "name": args.name,
        "experiment": args.experiment,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": checkpoint_sha256(checkpoint),
        "family": args.family,
        "step": int(state["step"]),
        "L_ctx": context_frames,
        "model_dtype": str(next(model.parameters()).dtype),
    }
    source = getattr(module, "__hal_source__", None)
    if source is not None:
        protocol["source"] = source

    if args.family == "historical_interleaved_groups":
        temp = float(cfg.decode_temp)
        protocol.update(
            {
                "decode_settings": {"temp": temp},
                "exec_horizon": 1,
                "head_offsets": [1],
                "trunk_passes_per_action": 4,
            }
        )

        def build_historical_interleaved(seed: int) -> Any:
            return _build_historical_interleaved_policy(module, model, stats, cfg, temp=temp, seed=seed)

        return build_historical_interleaved, protocol

    if args.family == "temporal_mtp":
        if action_trace is not None and "action_trace" not in inspect.signature(module.make_policy).parameters:
            raise ValueError(f"{args.experiment} does not support action tracing")
        head_offsets = tuple(cfg.head_offsets if hasattr(cfg, "head_offsets") else cfg.arch.head_offsets)
        decode_temp = getattr(cfg, "decode_temp", 1.0) if args.temperature is None else args.temperature
        if (
            not isinstance(decode_temp, int | float)
            or isinstance(decode_temp, bool)
            or not math.isfinite(decode_temp)
            or decode_temp <= 0
        ):
            raise ValueError(f"sampling temperature must be finite and positive, got {decode_temp!r}")
        decode_temp = float(decode_temp)
        if (
            args.temperature is not None
            and "decode_temperature" not in inspect.signature(module.make_policy).parameters
        ):
            raise ValueError(f"{args.experiment} does not support temperature overrides")
        if args.prediction_frames is not None:
            dense_offsets = tuple(range(1, args.prediction_frames + 1))
            if head_offsets[: args.prediction_frames] != dense_offsets:
                raise ValueError(
                    f"prediction_frames={args.prediction_frames} requires dense heads {dense_offsets}, "
                    f"got {head_offsets[: args.prediction_frames]}"
                )
            if not hasattr(cfg, "delay_frames"):
                raise ValueError(f"{args.experiment} does not support prediction_frames overrides")
            cfg = replace(cfg, prediction_frames=args.prediction_frames)
            model.cfg = cfg
            model.temporal.live_horizons = (args.prediction_frames,)
        deployment_override = any(
            value is not None
            for value in (
                args.player_identity,
                args.prediction_frames,
                args.delay_frames,
                args.replan_interval_frames,
            )
        )
        if deployment_override and not hasattr(cfg, "delay_frames"):
            raise ValueError(f"{args.experiment} does not support player or deployment-timing overrides")
        if deployment_override:
            delay = cfg.delay_frames if args.delay_frames is None else args.delay_frames
            replan = cfg.replan_interval_frames if args.replan_interval_frames is None else args.replan_interval_frames
            if delay + replan > cfg.prediction_frames:
                raise ValueError(
                    "delay_frames + replan_interval_frames must not exceed prediction_frames: "
                    f"{delay} + {replan} > {cfg.prediction_frames}"
                )
            identity = args.player_identity
            if identity is None:
                player_id = 0
            else:
                rank = Rank.__members__.get(identity)
                if rank is not None:
                    player_id = int(rank)
                else:
                    encoded = model.get_buffer("player_code_bytes")
                    vocabulary = PlayerVocabulary(decode_player_codes(encoded.detach().cpu().numpy().tobytes()))
                    player_id = vocabulary.id_for_code(identity)
        else:
            identity = None
            player_id = 0
        protocol.update(
            {
                "decode_settings": {"temp": decode_temp},
                "player_identity": identity,
                "player_id": int(player_id),
                "head_offsets": list(head_offsets),
            }
        )
        if action_trace is not None:
            protocol["action_trace_schema_version"] = ACTION_TRACE_SCHEMA_VERSION
        trace_kwargs = {} if action_trace is None else {"action_trace": action_trace}
        if deployment_override:
            protocol.update(
                {
                    "prediction_frames": int(cfg.prediction_frames),
                    "delay_frames": int(delay),
                    "replan_interval_frames": int(replan),
                }
            )
        else:
            protocol["exec_horizon"] = int(cfg.exec_horizon)

        def build_temporal_mtp(seed: int) -> Any:
            if not deployment_override:
                if args.temperature is None:
                    return module.make_policy(
                        model,
                        stats,
                        cfg,
                        exec_horizon=cfg.exec_horizon,
                        decode_seed=seed,
                        **trace_kwargs,
                    )
                return module.make_policy(
                    model,
                    stats,
                    cfg,
                    exec_horizon=cfg.exec_horizon,
                    decode_seed=seed,
                    decode_temperature=decode_temp,
                    **trace_kwargs,
                )
            if args.temperature is None:
                return module.make_policy(
                    model,
                    stats,
                    cfg,
                    decode_seed=seed,
                    ego_player_id=player_id,
                    delay_frames=delay,
                    replan_interval_frames=replan,
                    **trace_kwargs,
                )
            return module.make_policy(
                model,
                stats,
                cfg,
                decode_seed=seed,
                ego_player_id=player_id,
                delay_frames=delay,
                replan_interval_frames=replan,
                decode_temperature=decode_temp,
                **trace_kwargs,
            )

        return build_temporal_mtp, protocol

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


def _build_historical_interleaved_policy(
    module: ModuleType, model: Any, stats: dict[str, Any], cfg: Any, *, temp: float, seed: int
) -> Any:
    """Put the old 013 decoder behind today's spawned-broker policy contract."""
    # The original decoder used torch's global generator. A private generator
    # produces the identical multinomial sequence from a fixed seed while
    # preventing the opponent policy's calls from perturbing that sequence.
    torch = module.torch
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)

    @torch.no_grad()
    def predict_chunk(ctx: Any, committed: Any) -> Any:
        assert committed is None
        return module.decode(model, ctx, temp=temp, gen=generator).cpu().numpy()

    return module.RecedingHorizon(
        predict_chunk=predict_chunk,
        stats=stats,
        L_ctx=cfg.L_ctx,
        L_chunk=1,
        s=1,
        d=0,
        device=str(device),
    )


def _settings_dict(settings: Any) -> dict[str, Any]:
    """Experiment-local ``DecodeSettings`` as plain JSON-able fields."""
    return {f: getattr(settings, f) for f in settings.__dataclass_fields__}


def _git_revision() -> str:
    result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip()


def _require_empty_upload_prefix(name: str, prefix_root: str) -> None:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) is None:
        raise ValueError(f"upload_run must be one safe path component, got {name!r}")
    components = prefix_root.split("/")
    if not components or any(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", part) is None for part in components):
        raise ValueError(f"upload_prefix must contain only safe path components, got {prefix_root!r}")
    prefix = f"{prefix_root}/{name}/"
    client = r2.client()
    result = client.list_objects_v2(Bucket=r2.bucket(), Prefix=prefix, MaxKeys=1)
    if result.get("KeyCount", 0):
        raise FileExistsError(f"refusing to overwrite existing R2 prefix r2://{r2.bucket()}/{prefix}")


def main(args: Args) -> None:
    stages = INCLUDED_STAGES if not args.stages else tuple(getattr(melee.Stage, s.upper()) for s in args.stages)
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory {out_dir}")
    if args.upload_run is not None:
        _require_empty_upload_prefix(args.upload_run, args.upload_prefix)
    uploader = None if args.upload_run is None else BackgroundUploader(args.upload_run, prefix=args.upload_prefix)
    trace_writers: list[ActionTraceWriter] = []

    def upload_orientation(orientation: int) -> None:
        if uploader is not None:
            replay_dir = orientation_replay_dir(out_dir, args.model_a.name, orientation)
            uploader.upload_tree(replay_dir, base=out_dir.parent)
            uploader.wait()

    try:
        traces: tuple[ActionTraceWriter | None, ActionTraceWriter | None] = (None, None)
        if args.trace_actions:
            trace_root = out_dir / "action-traces"
            trace_writers.append(
                ActionTraceWriter(trace_root / f"model-a-{path_safe(args.model_a.name)}", model=args.model_a.name)
            )
            trace_writers.append(
                ActionTraceWriter(trace_root / f"model-b-{path_safe(args.model_b.name)}", model=args.model_b.name)
            )
            traces = (trace_writers[0], trace_writers[1])
        logger.info(f"loading {args.model_a.name} and {args.model_b.name}")
        build_a, protocol_a = load_policy_builder(args.model_a, action_trace=traces[0])
        build_b, protocol_b = load_policy_builder(args.model_b, action_trace=traces[1])
        for protocol in (protocol_a, protocol_b):
            logger.info(f"  {protocol}")

        fixed_characters = {
            model.name: getattr(melee.Character, model.character.upper())
            for model in (args.model_a, args.model_b)
            if model.character is not None
        }
        records = run_h2h(
            build_a,
            build_b,
            name_a=args.model_a.name,
            name_b=args.model_b.name,
            n_configs=args.n_configs,
            out_dir=out_dir,
            stages=stages,
            max_frames=args.max_frames,
            max_parallel=args.max_parallel,
            start_retries=args.start_retries,
            seed=args.seed,
            verify_inputs=args.verify_inputs,
            fixed_characters=fixed_characters,
            meta={"models": {args.model_a.name: protocol_a, args.model_b.name: protocol_b}, "git": _git_revision()},
            on_orientation_done=upload_orientation,
        )
    finally:
        for writer in trace_writers:
            writer.close()
        if uploader is not None:
            uploader.upload_tree(out_dir, base=out_dir.parent)
            uploader.close()
    print(summarize_paired(records, focal_model=args.model_a.name).format_table())


if __name__ == "__main__":
    main(tyro.cli(Args))
