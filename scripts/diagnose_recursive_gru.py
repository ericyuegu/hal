"""Read-only matched-checkpoint diagnostics for experiment 035.

This script does not alter training or checkpoints. It compares teacher-forced
validation NLL and inspects the trained GRU's gates, state geometry, and action
input gradient on the same cached validation rows as a matched 026 checkpoint.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _gate_and_state_metrics(module, model, batches, cfg) -> dict[str, object]:
    reset_values = []
    update_values = []
    states_by_depth = [[] for _ in cfg.head_offsets]
    relative_updates = [[] for _ in cfg.head_offsets]
    cell = model.temporal.cell
    weight_ir, weight_iz, weight_in = cell.weight_ih.chunk(3)
    weight_hr, weight_hz, weight_hn = cell.weight_hh.chunk(3)
    bias_ir, bias_iz, bias_in = cell.bias_ih.chunk(3)
    bias_hr, bias_hz, bias_hn = cell.bias_hh.chunk(3)

    with torch.no_grad():
        for cpu_batch in batches:
            batch = cpu_batch.to(next(model.parameters()).device)
            history, targets, valid = module.prepared_targets(model, batch)
            with module.amp_context(cfg, next(model.parameters()).device):
                hidden = model(batch.context.features, batch.context.ctx_pad, history)
                previous = torch.cat((history[:, :, None], targets[..., :-1, :]), dim=2)
                tokens = model.temporal._tokens(hidden, previous)[valid].float()
            state = tokens.new_zeros(tokens.shape[0], cfg.temporal_d_model)
            for depth, token in enumerate(tokens.unbind(1)):
                reset = torch.sigmoid(F.linear(token, weight_ir, bias_ir) + F.linear(state, weight_hr, bias_hr))
                update = torch.sigmoid(F.linear(token, weight_iz, bias_iz) + F.linear(state, weight_hz, bias_hz))
                candidate = torch.tanh(
                    F.linear(token, weight_in, bias_in) + reset * F.linear(state, weight_hn, bias_hn)
                )
                next_state = (1.0 - update) * candidate + update * state
                torch.testing.assert_close(next_state, cell(token, state))
                reset_values.append(reset.float().cpu())
                update_values.append(update.float().cpu())
                states_by_depth[depth].append(next_state.float().cpu())
                denominator = state.float().norm(dim=-1).clamp_min(1e-8)
                relative_updates[depth].append(((next_state.float() - state.float()).norm(dim=-1) / denominator).cpu())
                state = next_state

    reset = torch.cat(reset_values)
    update = torch.cat(update_values)
    out: dict[str, object] = {
        "reset_mean": float(reset.mean()),
        "reset_below_0.01": float((reset < 0.01).float().mean()),
        "reset_above_0.99": float((reset > 0.99).float().mean()),
        "update_mean": float(update.mean()),
        "update_below_0.01": float((update < 0.01).float().mean()),
        "update_above_0.99": float((update > 0.99).float().mean()),
    }
    depth_metrics = {}
    for depth, offset in enumerate(cfg.head_offsets):
        states = torch.cat(states_by_depth[depth])
        centered = states - states.mean(dim=0)
        singular_values = torch.linalg.svdvals(centered)
        eigenvalues = singular_values.square().clamp_min(1e-20)
        probabilities = eigenvalues / eigenvalues.sum()
        effective_rank = torch.exp(-(probabilities * probabilities.log()).sum())
        participation_ratio = eigenvalues.sum().square() / eigenvalues.square().sum()
        depth_metrics[f"o{offset:02d}"] = {
            "state_rms": float(states.square().mean().sqrt()),
            "relative_update_norm": None if depth == 0 else float(torch.cat(relative_updates[depth]).mean()),
            "effective_rank": float(effective_rank),
            "participation_ratio": float(participation_ratio),
        }
    out["by_offset"] = depth_metrics
    return out


def _nll_metrics(module, model, batches, cfg) -> torch.Tensor:
    total = torch.zeros(len(cfg.head_offsets), module.N_GROUPS, dtype=torch.float64)
    count = 0
    with torch.no_grad():
        for cpu_batch in batches:
            batch = cpu_batch.to(next(model.parameters()).device)
            history, targets, valid = module.prepared_targets(model, batch)
            with module.amp_context(cfg, next(model.parameters()).device):
                hidden = model(batch.context.features, batch.context.ctx_pad, history)
                nll = model.temporal.teacher_forced_nll(hidden, history, targets)[valid]
            total += nll.double().sum(dim=0).cpu()
            count += nll.shape[0]
    return total / count / module._LN2


def _action_gradient_metrics(module, model, batch, cfg) -> dict[str, float]:
    model.zero_grad(set_to_none=True)
    device_batch = batch.to(next(model.parameters()).device)
    loss = module.objective(module.action_loss(model, device_batch))
    loss.backward()
    gradient = model.temporal.token_projection.weight.grad.float()
    action_start = cfg.d_model
    action_stop = action_start + module.N_GROUPS * cfg.action_embed_dim
    action_gradient = gradient[:, action_start:action_stop]
    class_gradient = torch.cat(
        [embedding.weight.grad.float().reshape(-1) for embedding in model.codec.class_embeddings.values()]
    )
    return {
        "objective_bits": float(loss.detach().cpu()),
        "token_projection_action_grad_norm": float(action_gradient.norm()),
        "token_projection_full_grad_norm": float(gradient.norm()),
        "token_projection_action_grad_fraction": float(action_gradient.norm() / gradient.norm().clamp_min(1e-20)),
        "action_class_embedding_grad_norm": float(class_gradient.norm()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gru-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    gru_module = _load_module("hal_diagnostic_exp035", root / "experiments/035_recursive_gru.py")
    baseline_module = _load_module("hal_diagnostic_exp026", root / "experiments/026_temporal_mtp.py")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gru_model, gru_cfg, stats, _ = gru_module.load_checkpoint(str(args.gru_checkpoint), device=device)
    baseline_model, baseline_cfg, _, _ = baseline_module.load_checkpoint(str(args.baseline_checkpoint), device=device)
    if gru_cfg.head_offsets != baseline_cfg.head_offsets:
        raise ValueError("checkpoints do not use the same offsets")
    _, batches = gru_module._make_loaders(gru_cfg, stats)
    rows = []
    count = 0
    for batch in batches:
        rows.append(batch)
        count += batch.context.ctx_pad.shape[0]
        if count >= args.samples:
            break

    gru_nll = _nll_metrics(gru_module, gru_model, rows, gru_cfg)
    baseline_nll = _nll_metrics(baseline_module, baseline_model, rows, baseline_cfg)
    deltas = gru_nll - baseline_nll
    per_offset_group = {
        f"o{offset:02d}": {
            name: {
                "gru_bits": float(gru_nll[depth, group]),
                "baseline_bits": float(baseline_nll[depth, group]),
                "delta_bits": float(deltas[depth, group]),
            }
            for group, name in enumerate(gru_module.GROUP_NAMES)
        }
        for depth, offset in enumerate(gru_cfg.head_offsets)
    }
    payload = {
        "gru_checkpoint": str(args.gru_checkpoint.resolve()),
        "baseline_checkpoint": str(args.baseline_checkpoint.resolve()),
        "validation_rows": count,
        "mean_group_nll_delta_bits": float(deltas.mean()),
        "joint_nll_delta_bits_by_offset": {
            f"o{offset:02d}": float(deltas[depth].sum()) for depth, offset in enumerate(gru_cfg.head_offsets)
        },
        "per_offset_group": per_offset_group,
        "gru_dynamics": _gate_and_state_metrics(gru_module, gru_model, rows, gru_cfg),
        "action_gradient": _action_gradient_metrics(gru_module, gru_model, rows[0], gru_cfg),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
