"""Capture a production resume update in the checkpoint's original environment.

Run control and candidate in separate processes and checkouts. This calls each
checkout's existing loader, prefetcher, loss, optimizer, and compile functions.
An optional throughput window continues from the captured update. Neither path
runs evaluations, creates W&B runs, or publishes a training checkpoint.
"""

import argparse
import hashlib
import importlib
import json
import os
import random
import sys
from collections.abc import Callable
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _cpu_snapshot(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: _cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_snapshot(item) for item in value)
    if isinstance(value, list):
        return [_cpu_snapshot(item) for item in value]
    return value


def _column_digest(columns: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, array in sorted(columns.items()):
        digest.update(json.dumps((name, array.dtype.str, array.shape)).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


class _RawBatchCapture:
    def __init__(self, loader: Any, transform: Callable, batch_record: Callable) -> None:
        self.loader = loader
        self.transform = transform
        self.batch_record = batch_record
        self.record: dict[str, Any] | None = None

    def __call__(self, replay_ids: tuple[str, ...], columns: Mapping[str, np.ndarray]) -> Any:
        if self.record is not None:
            raise RuntimeError("the capture fetched more than the single boundary batch")
        ring = self.loader._ring
        slots = ring.schedule.selected_slots(ring.fifo_head)
        identities = []
        for replay_id, slot, ordinal in zip(replay_ids, slots, ring.schedule.window_ordinals, strict=True):
            locator = ring.locators[int(slot)]
            if locator is None or ring.replay_ids[int(slot)] != replay_id:
                raise RuntimeError("raw capture does not match the selected replay slot")
            identities.append(
                {
                    "replay_id": replay_id,
                    "locator": asdict(locator),
                    "epoch": int(ring.epochs[int(slot)]),
                    "window_ordinal": int(ordinal),
                }
            )
        batch = self.transform(replay_ids, columns)
        self.record = {
            "seed": self.loader.seed,
            "window_identities": identities,
            "raw_columns_sha256": _column_digest(columns),
            "transformed_batch": self.batch_record(batch),
        }
        return batch


class _GradientCapture:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.gradients: dict[str, torch.Tensor | None] | None = None

    def __call__(self, optimizer: torch.optim.Optimizer, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        del optimizer, args, kwargs
        if self.gradients is not None:
            raise RuntimeError("the capture ran more than one optimizer update")
        self.gradients = {
            name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
            for name, parameter in self.model.named_parameters()
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("control", "candidate"))
    parser.add_argument("root", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--source-transition", type=Path)
    parser.add_argument("--measure-training", action="store_true")
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    if args.source_transition is not None:
        args.source_transition = args.source_transition.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    if _sha256(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError("checkpoint content differs from the pinned fixture")
    root = args.root
    os.chdir(root)
    sys.path[:0] = [str(root), str(root / "experiments")]
    name = "059_muon_history_decoder" if args.mode == "control" else "059_muon_action_sequence"
    experiment = importlib.import_module(name)
    state = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    cfg = experiment.config_from_state(state["cfg"])
    experiment.validate_config(cfg)
    experiment.validate_conditioning_state(state, cfg)
    current_provenance = experiment.run_provenance(cfg)
    if args.mode == "control":
        if args.source_transition is not None:
            raise ValueError("the control uses the checkpoint's exact source without a transition")
        if current_provenance != state["provenance"]:
            raise ValueError("the control provenance differs from the saved production run")
    else:
        from hal.training.checkpoints import read_resume_lineage
        from hal.training.checkpoints import validate_resume_provenance

        if args.source_transition is None:
            raise ValueError("the candidate requires an explicit source transition")
        transition = read_resume_lineage(args.source_transition)
        validate_resume_provenance(
            state["provenance"],
            current_provenance,
            transition=transition,
            parent_checkpoint_sha256=args.checkpoint_sha256,
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("production resume requires the single pinned training GPU")
    from hal.training.system_metrics import read_cgroup_memory

    memory = read_cgroup_memory()
    limit = memory.get("system/cgroup/limit_gib")
    if limit is None or not 100 <= limit <= 256:
        raise RuntimeError(f"the production replay buffer needs an isolated 100–256 GiB host; found {memory}")
    if cfg.batch_size != 512 or cfg.replay_slots != 131072:
        raise ValueError("this capture requires the production 059 replay-buffer geometry")
    args.output.mkdir(parents=True)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high" if cfg.allow_tf32 else "highest")
    stats = experiment.load_stats(cfg)
    sidecar = experiment.load_identity_sidecar(cfg)

    with ExitStack() as resources:
        loader = experiment._make_train_loader(cfg, stats, experiment.ReplayPlayerLookup(sidecar.by_replay))
        resources.callback(loader.close)
        raw_capture = _RawBatchCapture(loader, loader.batch_transform, experiment._return_batch_state)
        loader.batch_transform = raw_capture
        loader.load_state_dict(state["loader"])
        if args.mode == "control":
            model = experiment.GPT(cfg, sidecar.vocabulary).cuda()
            calibration = model.return_calibration
            scheduler_factory = torch.optim.lr_scheduler.LambdaLR
        else:
            model = experiment.make_model(cfg, sidecar.vocabulary).cuda()
            calibration = experiment.ReturnCalibration(window_count=experiment.CALIBRATION_WINDOWS)
            scheduler_factory = experiment.LearningRateScheduler
        optimizer = experiment.make_optimizer(model, cfg)
        scheduler = scheduler_factory(optimizer, experiment.lr_schedule(cfg))
        identity = experiment.IdentityMasker(cfg.seed ^ 0x0501D, cfg.identity_dropout)
        return_masker = experiment.ReturnMasker(
            cfg.seed ^ 0x059C0D, cfg.return_dropout, enabled=cfg.return_conditioning
        )
        prefix = experiment.PrefixSampler(cfg.seed ^ 0x059A11, "cuda")
        calibration.load_state_dict(state["return_calibration"])
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["opt"])
        scheduler.load_state_dict(state["sched"])
        identity.load_state_dict(state["identity_masker"])
        return_masker.load_state_dict(state["return_masker"])
        prefix.load_state_dict(state["prefix_sampler"])
        experiment.restore_rng(state["rng"])
        step = int(state["step"]) + 1
        trunk_fn, temporal_fn = experiment._training_functions(model, cfg)
        experiment._compile_synthetic_forward_backward(
            model,
            cfg,
            step=step,
            trunk_fn=trunk_fn,
            temporal_fn=temporal_fn,
        )
        prefetch = experiment.DeviceBatchPrefetcher(
            loader,
            cfg,
            "cuda",
            identity,
            return_masker=return_masker,
            calibration=calibration,
        )
        resources.callback(prefetch.close)
        batch, valid_prefixes = prefetch.next()
        prefetch.fill_lookahead(0)
        if not prefetch.drained or raw_capture.record is None:
            raise RuntimeError("resume capture did not drain exactly one batch")
        expected_prefix = experiment.PrefixSampler(0, "cuda")
        expected_prefix.load_state_dict(prefix.state_dict())
        positions = expected_prefix.sample(
            batch.context.ctx_pad,
            length=cfg.arch.L_ctx,
            suffix_start=cfg.arch.direct_loss_start,
            validated_on_cpu=True,
        )
        gradients = _GradientCapture(model)
        hook = optimizer.register_step_pre_hook(gradients)
        resources.callback(hook.remove)
        result = experiment.train_step(
            model,
            batch,
            cfg,
            step=step,
            update=step + 1,
            valid_prefixes=valid_prefixes,
            trunk_fn=trunk_fn,
            temporal_fn=temporal_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            prefix_sampler=prefix,
            prefix_validated_on_cpu=True,
        )
        if not torch.equal(prefix.generator.get_state(), expected_prefix.generator.get_state()):
            raise RuntimeError("prefix capture consumed different RNG draws from the actual update")
        if gradients.gradients is None:
            raise RuntimeError("optimizer gradient capture did not run")
        if float(result.metrics["awr/active"]) != 1 or float(result.metrics["awr/weight_max"]) <= 1:
            raise RuntimeError("the production batch did not exercise non-unit AWR weights")
        snapshot = {
            "checkpoint_sha256": args.checkpoint_sha256,
            "update": step + 1,
            "raw_batch": raw_capture.record,
            "masked_batch": experiment._return_batch_state(batch),
            "prefix_positions": positions,
            "valid_prefixes": valid_prefixes,
            "parameter_order": tuple(name for name, _ in model.named_parameters()),
            "result": {
                "nll_sum": result.nll_sum,
                "gradient_norm": result.gradient_norm,
                "metrics": result.metrics,
                "muon_lr": result.muon_lr,
                "adam_lr": result.adam_lr,
            },
            "gradients": gradients.gradients,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "loader": loader.state_dict(),
            "rng": experiment.rng_state(),
            "prefix_sampler": prefix.state_dict(),
            "identity_masker": identity.state_dict(),
            "return_masker": return_masker.state_dict(),
            "calibration": calibration.state_dict(),
        }
        torch.save(_cpu_snapshot(snapshot), args.output / "next-update.pt")
        report = {
            "mode": args.mode,
            "source_sha": current_provenance["git_sha"],
            "checkpoint_sha256": args.checkpoint_sha256,
            "update": step + 1,
            "capture_sha256": _sha256(Path(__file__)),
            "environment": current_provenance["environment"],
            "snapshot_sha256": _sha256(args.output / "next-update.pt"),
            "raw_columns_sha256": raw_capture.record["raw_columns_sha256"],
            "awr_weight_max": float(result.metrics["awr/weight_max"]),
            "cgroup_memory": read_cgroup_memory(),
        }
        (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        if args.measure_training:
            from measure_training_updates import measure_training_updates

            hook.remove()
            loader.batch_transform = raw_capture.transform
            del state, snapshot, gradients, raw_capture, batch, positions, expected_prefix
            performance = measure_training_updates(
                prefetch,
                partial(
                    experiment.train_step,
                    model=model,
                    cfg=cfg,
                    trunk_fn=trunk_fn,
                    temporal_fn=temporal_fn,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    prefix_sampler=prefix,
                ),
                next_step=step + 1,
                batch_size=cfg.batch_size,
                warm_updates=100,
                measured_updates=200,
            )
            (args.output / "performance.json").write_text(json.dumps(performance, indent=2, sort_keys=True) + "\n")
            print(json.dumps(performance, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
