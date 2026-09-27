"""Evaluate two policy artifacts with mirrored ports and shared checkpoint weights."""

import re
from dataclasses import asdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal

import melee
import tyro

from hal import r2
from hal.eval.h2h import DEFAULT_MAX_FRAMES
from hal.eval.h2h import DEFAULT_START_RETRIES
from hal.eval.h2h import H2HModel
from hal.eval.h2h import orientation_replay_dir
from hal.eval.h2h import prepare_h2h_policies
from hal.eval.h2h import run_h2h
from hal.eval.harness import resolve_parallelism
from hal.eval.paired import summarize_paired
from hal.inference.engine import configure_inference_process
from hal.inference.engine import freeze_inference_runtime
from hal.policy import INCLUDED_STAGES
from hal.training.checkpoints import BackgroundUploader
from hal.training.runs import source_git_sha


@dataclass(frozen=True, slots=True)
class Args:
    model_a: H2HModel
    model_b: H2HModel
    out_dir: Path
    profile: Literal["local", "official-059"] = "local"
    n_configs: int = 64
    max_frames: int = DEFAULT_MAX_FRAMES
    max_parallel: int | None = None
    start_retries: int = DEFAULT_START_RETRIES
    seed: int = 0
    stages: tuple[str, ...] = ()
    character_a: str | None = None
    character_b: str | None = None
    device: str = "cuda"
    compiled: bool = True
    verify_inputs: bool = True
    upload_run: str | None = None
    upload_prefix: str = "h2h-evals"


def _validate_empty_upload_prefix(name: str, prefix_root: str) -> None:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) is None:
        raise ValueError("upload_run must be one safe path component")
    components = prefix_root.split("/")
    if any(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", part) is None for part in components):
        raise ValueError("upload_prefix must contain only safe path components")
    prefix = f"{prefix_root}/{name}/"
    result = r2.client().list_objects_v2(Bucket=r2.bucket(), Prefix=prefix, MaxKeys=1)
    if result.get("KeyCount", 0):
        raise FileExistsError(f"refusing to overwrite existing R2 prefix r2://{r2.bucket()}/{prefix}")


def _upload_orientation(uploader: BackgroundUploader, out_dir: Path, name: str, orientation: int) -> None:
    uploader.upload_tree(orientation_replay_dir(out_dir, name, orientation), base=out_dir.parent)
    uploader.wait()


def main(args: Args) -> None:
    if args.n_configs < 1:
        raise ValueError("n_configs must be positive")
    stages = INCLUDED_STAGES if not args.stages else tuple(melee.Stage[name.upper()] for name in args.stages)
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory {out_dir}")
    parallel = resolve_parallelism(args.n_configs, args.max_parallel)
    configure_inference_process()
    builder = prepare_h2h_policies(
        (args.model_a, args.model_b),
        max_parallel=parallel,
        profile=args.profile,
        device=args.device,
        compiled=args.compiled,
    )
    freeze_inference_runtime()
    fixed_characters = {
        model.name: melee.Character[character.upper()]
        for model, character in ((args.model_a, args.character_a), (args.model_b, args.character_b))
        if character is not None
    }
    if args.upload_run is not None:
        _validate_empty_upload_prefix(args.upload_run, args.upload_prefix)
    uploader = None if args.upload_run is None else BackgroundUploader(args.upload_run, prefix=args.upload_prefix)
    try:
        records = run_h2h(
            builder,
            name_a=args.model_a.name,
            name_b=args.model_b.name,
            n_configs=args.n_configs,
            out_dir=out_dir,
            stages=stages,
            max_frames=args.max_frames,
            max_parallel=parallel,
            start_retries=args.start_retries,
            seed=args.seed,
            verify_inputs=args.verify_inputs,
            fixed_characters=fixed_characters,
            meta={
                "profile": args.profile,
                "timing": asdict(builder.timing),
                "models": {
                    model.name: {
                        **asdict(model),
                        "checkpoint_sha256": builder.policies_by_name[model.name].checkpoint_sha256,
                    }
                    for model in (args.model_a, args.model_b)
                },
                "git": source_git_sha(),
            },
            on_orientation_done=None
            if uploader is None
            else partial(_upload_orientation, uploader, out_dir, args.model_a.name),
        )
    finally:
        if uploader is not None:
            try:
                uploader.upload_tree(out_dir, base=out_dir.parent)
            finally:
                uploader.close()
    print(summarize_paired(records, focal_model=args.model_a.name).format_table())


if __name__ == "__main__":
    main(tyro.cli(Args))
