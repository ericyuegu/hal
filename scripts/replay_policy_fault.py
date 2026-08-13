"""Replay an experiment-029 parent-policy crash capsule on one CUDA process."""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("capsule", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--cuda-sync-debug", action="store_true")
    parser.add_argument("--eager", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse()
    if args.cuda_sync_debug and os.environ.get("CUDA_LAUNCH_BLOCKING") != "1":
        env = dict(os.environ)
        env["CUDA_LAUNCH_BLOCKING"] = "1"
        os.execvpe(sys.executable, [sys.executable, *sys.argv], env)

    import numpy as np
    import torch

    from hal.training.features import Context

    root = Path(__file__).resolve().parents[1]
    experiment_path = root / "experiments" / "029_game_state_flow.py"
    spec = importlib.util.spec_from_file_location("hal_replay_exp029", experiment_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {experiment_path}")
    experiment = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = experiment
    spec.loader.exec_module(experiment)

    payload = json.loads(args.capsule.read_text())
    arrays = np.load(args.capsule.with_suffix(".npz"))
    meta = payload["policy"]
    floats = arrays["floats"]
    cats = arrays["cats"]
    value_names = meta["value_names"]
    mask_names = meta["mask_names"]
    emitted = meta["emitted_masks"]
    features = {name: torch.from_numpy(floats[index]) for index, name in enumerate(value_names)}
    value_count = len(value_names)
    features.update(
        {
            name: torch.from_numpy(floats[value_count + index])
            for index, name in enumerate(mask_names)
            if emitted[index]
        }
    )
    features.update({name: torch.from_numpy(cats[index]) for index, name in enumerate(meta["cat_names"])})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    context = Context(
        features={name: value.to(device) for name, value in features.items()},
        ctx_pad=torch.tensor(meta["ctx_pad"], dtype=torch.long, device=device),
        slot_ids=torch.tensor(
            [slot["match"] * 8 + slot["port"] for slot in meta["slots"]], dtype=torch.long, device=device
        ),
        reset=torch.tensor(meta["reset"], dtype=torch.bool, device=device),
    )
    model, cfg, _stats, _state = experiment.load_checkpoint(str(args.checkpoint))
    if args.eager:
        cfg = experiment.replace(cfg, inference_mode="eager")
    bucket = int(meta.get("inference_bucket") or len(meta["slots"]))
    engine = experiment.BF16Inference(model, cfg, bucket=bucket)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    out = engine.decode(context, int(meta["exec_horizon"]))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    print(f"replay completed: output={tuple(out.shape)} finite={bool(torch.isfinite(out).all())}")


if __name__ == "__main__":
    main()
