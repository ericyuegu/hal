"""Emit the current experiment-051 contract from an isolated interpreter."""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import sys
import tempfile
from dataclasses import asdict
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import torch
from torch.optim.lr_scheduler import LambdaLR

from hal.training.checkpoints import save_checkpoint
from hal.training.o51_replay_loader import GenerationDescriptor
from hal.training.o51_replay_loader import PhysicalRow

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments" / "051_muon_parameterization.py"
RESULT_PREFIX = "O51_V5_CHARACTERIZATION="


def _load_experiment() -> ModuleType:
    spec = importlib.util.spec_from_file_location("o51_v5_characterization", EXPERIMENT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {EXPERIMENT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _tensor_schema(module: ModuleType, level: str) -> dict[str, object]:
    with torch.device("meta"):
        model = module.Policy(module.config_for(level))
    state_schema = [[name, list(tensor.shape), str(tensor.dtype)] for name, tensor in model.state_dict().items()]
    result = {
        "state_entries": len(state_schema),
        "state_schema_sha256": _digest(state_schema),
        "subsystem_parameters": module.subsystem_parameter_counts(model),
    }
    del model
    gc.collect()
    return result


def _tiny_config(module: ModuleType):
    architecture = replace(
        module.ARCHITECTURE,
        d_model=32,
        n_layers=2,
        n_heads=4,
        L_ctx=8,
        temporal_d_model=32,
        temporal_layers=2,
        temporal_heads=4,
        temporal_ff_dim=64,
        group_head_dim=16,
        value_hidden_dim=16,
        item_hidden_dim=8,
        item_dim=5,
    )
    return module.TrainConfig(
        arch=architecture,
        target_positions=256,
        batch_size=2,
        compile_trunk=False,
        compile_temporal=False,
        inference_mode="eager",
        num_workers=0,
        push_to_r2=False,
    )


def _numeric_contract(module: ModuleType) -> dict[str, object]:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(2_026_051)
    cfg = _tiny_config(module)
    model = module.Policy(cfg).eval()
    batch = module.synthetic_awr_batch(cfg, torch.device("cpu"))
    hidden = model.forward_unpadded(batch.context.features, batch.context.ctx_pad)
    loss, nll, metrics = module.microbatch_loss(
        model,
        batch,
        cfg,
        step=cfg.warmup_steps,
        valid_prefixes=cfg.batch_size * (cfg.arch.L_ctx // 2),
        trunk_fn=model.forward_unpadded,
        temporal_fn=model.temporal.teacher_forced_nll,
    )
    return {
        "hidden_shape": list(hidden.shape),
        "hidden_mean": float(hidden.detach().mean()),
        "hidden_l2": float(torch.linalg.vector_norm(hidden.detach())),
        "loss": float(loss.detach()),
        "nll_mean": float(nll.mean()),
        "metric_keys": sorted(metrics),
        "policy_loss_bits": float(metrics["train/loss"]),
        "value_loss": float(metrics["value/loss"]),
    }


def _optimizer_contract(module: ModuleType) -> dict[str, object]:
    cfg = module.config_for("base")
    model = module.Policy(cfg)
    optimizer = module.make_optimizer(model, cfg)
    name_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    groups: list[dict[str, object]] = []
    for group in optimizer.param_groups:
        names = [name_by_id[id(parameter)] for parameter in group["params"]]
        values = {
            name: group[name]
            for name in (
                "use_muon",
                "lr",
                "weight_decay",
                "momentum",
                "logical_splits",
                "betas",
                "eps",
                "update_clip_threshold",
            )
            if name in group
        }
        if group["use_muon"]:
            values["muon_scale_clamp_min_one"] = bool(
                group.get(
                    "muon_scale_clamp_min_one",
                    group.get("muon_scale_mode") != "o51",
                )
            )
        groups.append(
            {
                **values,
                "parameter_count": len(names),
                "parameter_names_sha256": _digest(names),
                "first_parameter": names[0],
                "last_parameter": names[-1],
            }
        )
    return {"groups": groups, "matrix_scale": {"wide_2_by_8": 0.5, "tall_8_by_2": 2.0}}


def _checkpoint_contract(module: ModuleType) -> dict[str, object]:
    cfg = module.config_for("base")
    config = module._checkpoint_config(cfg)
    model = torch.nn.Linear(2, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.25)
    scheduler = LambdaLR(optimizer, lambda _step: 1.0)
    loader = {
        "schema": 2,
        "data_protocol": "o51-balanced-replay-v3",
        "source_selection_sha256": "a" * 64,
        "source_manifest_sha256": {"source": "b" * 64},
        "cursor": (3, 4, 5),
        "batch_sampler_state": {"batch_index": 9, "schedule": {}},
        "slots": (
            GenerationDescriptor(
                slot=0,
                locator=PhysicalRow(source="source", shard=7, row=11),
                epoch=3,
                replay_checksum=13,
                next_window=2,
            ),
        ),
        "buffer_geometry": {
            "replay_slots": 1,
            "windows_per_generation": 4,
            "batch_size": 1,
            "window_length": 276,
        },
    }
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "checkpoint.pt"
        save_checkpoint(
            checkpoint,
            step=6,
            model=model,
            opt=optimizer,
            sched=scheduler,
            cfg=config,
            wandb_id="characterization",
            extra_state={
                "actual_loss_positions": 896,
                "loader": loader,
                "identity_masker": {"forced": 0, "masked": 0, "total": 0},
            },
        )
        restored = torch.load(checkpoint, map_location="cpu", weights_only=False)
    descriptor = restored["loader"]["slots"][0]
    return {
        "top_level_keys": sorted(restored),
        "boundary_step": restored["step"],
        "config_keys": sorted(config),
        "architecture_keys": sorted(config["architecture"]),
        "awr_calibration_keys": sorted(config["awr_calibration"]),
        "config_sha256": _digest(config),
        "config_round_trip": module.config_from_state(config) == cfg,
        "loader_keys": sorted(restored["loader"]),
        "loader_descriptor": asdict(descriptor),
    }


def _sweep_contract(module: ModuleType) -> dict[str, object]:
    sweep = module

    center = sweep.Treatment()
    factories = {
        "initialization-screen": lambda: sweep.initialization_screen_arms(center),
        "initialization-extension": lambda: sweep.initialization_extension_arms(center),
        "lr": lambda: sweep.lr_arms(center),
        "decay": lambda: sweep.decay_arms(center),
        "batch": lambda: sweep.batch_arms(center),
        "proxy-transfer": lambda: sweep.proxy_transfer_arms(center),
        "mid-search": lambda: sweep.mid_search_arms(center),
        "seed-repeat": lambda: sweep.seed_repeat_arms(center),
        "duration": lambda: sweep.duration_arms(center),
    }
    arms_by_stage = {stage: tuple(factory()) for stage, factory in factories.items()}
    arm_records = {stage: [asdict(arm) for arm in arms] for stage, arms in arms_by_stage.items()}

    arms = arms_by_stage["lr"][:3]
    final_update = arms[0].target_positions // (arms[0].treatment.batch_size * 128)
    validation_values = (
        (0.4, 0.3, 0.2),
        (0.4, 0.2, 0.3),
        (0.5, 0.1, 0.1),
    )
    validation = sweep.select_validation_winner(
        arms,
        {
            arm.arm_id: sweep.ValidationOutcome(
                arm_id=arm.arm_id,
                run_path=f"entity/project/{index}",
                state="finished",
                processed_positions=arm.target_positions,
                final_update=final_update,
                val_nll=values[0],
                val_far_nll=values[1],
                val_rollout_nll=values[2],
            )
            for index, (arm, values) in enumerate(zip(arms, validation_values, strict=True))
        },
    )
    finalists = validation.ranking[:2]
    closed_loop = sweep.select_closed_loop_winner(
        arms,
        validation,
        {
            outcome.arm_id: sweep.ClosedLoopOutcome(
                arm_id=outcome.arm_id,
                run_path=outcome.run_path,
                state="finished",
                final_update=final_update,
                boots=96,
                crashed=0,
                net_stock_per_min=0.1 + index,
                net_stock_lcb=0.2 + index,
                net_dmg_per_min=1.0 + index,
            )
            for index, outcome in enumerate(finalists)
        },
    )
    return {
        "arm_counts": {stage: len(arms) for stage, arms in arms_by_stage.items()},
        "arms_sha256": _digest(arm_records),
        "validation_ranking": [outcome.arm_id for outcome in validation.ranking],
        "validation_winner": validation.winner.arm_id,
        "closed_loop_ranking": [outcome.arm_id for outcome in closed_loop.ranking],
        "closed_loop_winner": closed_loop.winner.arm_id,
    }


def main() -> None:
    module = _load_experiment()
    contract = {
        "describe": module.describe(),
        "checkpoint": _checkpoint_contract(module),
        "models": {level: _tensor_schema(module, level) for level in module.MODEL_LEVELS},
        "numeric": _numeric_contract(module),
        "optimizer": _optimizer_contract(module),
        "sweep": _sweep_contract(module),
    }
    print(RESULT_PREFIX + json.dumps(contract, sort_keys=True))


if __name__ == "__main__":
    main()
