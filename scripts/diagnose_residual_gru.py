"""Read-only full/x-only/h-only diagnostics for experiment 036 checkpoints."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("hal_diagnostic_exp036", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _readout(module, mode, values, gradients):
    def apply(token, state):
        values["token_squared"] += float(token.detach().float().square().sum())
        values["state_squared"] += float(state.detach().float().square().sum())
        values["elements"] += token.numel()
        if token.requires_grad:
            token.register_hook(lambda grad: gradients.append(("token", float(grad.float().norm()))))
        if state.requires_grad:
            state.register_hook(lambda grad: gradients.append(("state", float(grad.float().norm()))))
        if mode == "full":
            return module.decoder_rmsnorm(token + state)
        if mode == "x_only":
            return module.decoder_rmsnorm(token)
        if mode == "h_only":
            return module.decoder_rmsnorm(state)
        raise ValueError(f"unknown mode {mode}")

    return apply


def _diagnose_mode(module, model, batches, cfg, mode):
    values = {"token_squared": 0.0, "state_squared": 0.0, "elements": 0}
    gradients = []
    original = model.temporal.readout
    model.temporal.readout = _readout(module, mode, values, gradients)
    nll_sum = torch.zeros(len(cfg.head_offsets), module.N_GROUPS, dtype=torch.float64)
    count = 0
    try:
        with torch.no_grad():
            for cpu_batch in batches:
                batch = cpu_batch.to(next(model.parameters()).device)
                history, targets, valid = module.prepared_targets(model, batch)
                with module.amp_context(cfg, next(model.parameters()).device):
                    hidden = model(batch.context.features, batch.context.ctx_pad, history)
                    nll = model.temporal.teacher_forced_nll(hidden, history, targets)[valid]
                nll_sum += nll.double().sum(dim=0).cpu()
                count += nll.shape[0]
        model.zero_grad(set_to_none=True)
        parts = module.action_loss(model, batches[0].to(next(model.parameters()).device))
        objective = module.objective(parts)
        objective.backward()
    finally:
        model.temporal.readout = original
    mean = nll_sum / count / module._LN2
    token_grad = sum(value * value for name, value in gradients if name == "token") ** 0.5
    state_grad = sum(value * value for name, value in gradients if name == "state") ** 0.5
    return {
        "aggregate_nll_bits": float(mean.sum(dim=-1)[:4].mean() + mean.sum(dim=-1)[4:].mean()),
        "joint_nll_bits_by_offset": {
            f"o{offset:02d}": float(mean[depth].sum()) for depth, offset in enumerate(cfg.head_offsets)
        },
        "nll_bits_by_offset_group": {
            f"o{offset:02d}": {name: float(mean[depth, group]) for group, name in enumerate(module.GROUP_NAMES)}
            for depth, offset in enumerate(cfg.head_offsets)
        },
        "token_rms": (values["token_squared"] / values["elements"]) ** 0.5,
        "state_rms": (values["state_squared"] / values["elements"]) ** 0.5,
        "objective_bits": float(objective.detach()),
        "token_total_grad_norm": token_grad,
        "state_total_grad_norm": state_grad,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=128)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    module = _load_module(root / "experiments/036_residual_recursive_gru.py")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, stats, _ = module.load_checkpoint(str(args.checkpoint), device=device)
    _, cache = module._make_loaders(cfg, stats)
    batches = []
    rows = 0
    for batch in cache:
        batches.append(batch)
        rows += batch.context.ctx_pad.shape[0]
        if rows >= args.samples:
            break
    payload = {
        "checkpoint": str(args.checkpoint.resolve()),
        "validation_rows": rows,
        "modes": {mode: _diagnose_mode(module, model, batches, cfg, mode) for mode in ("full", "x_only", "h_only")},
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
