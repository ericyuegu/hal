"""Common-input qualification of the complete dense model boundary on the proxy."""

import argparse
import contextlib
import dataclasses
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any
from typing import Literal

import numpy as np
import pytest
import torch
from test_065_loop import ROOT
from test_065_loop import exp
from test_window_replay_proxy import ProxyBatch
from test_window_replay_proxy import error
from test_window_replay_proxy import exp as control
from test_window_replay_proxy import flattened_gradients
from test_window_replay_proxy import old_forward
from test_window_replay_proxy import quantities
from test_window_replay_proxy import targets
from test_window_replay_proxy import values

from hal.inference.cuda_graph import prepared_shape_compilation
from hal.models.replay_actor import actor_loss
from hal.models.window_replay import ReplayCriticFeatures
from hal.models.window_replay import ReplayOutput

CONTROL_SOURCE = control.__file__
if CONTROL_SOURCE is None:
    raise ValueError("control trainer has no source")
CONTROL_PATH = Path(CONTROL_SOURCE)
if CONTROL_PATH.read_bytes() != subprocess.check_output(
    ["git", "show", "aacdfee0:experiments/065_selfplay_ppo.py"], cwd=ROOT
):
    raise ValueError("dense qualification requires the recorded aacdfee0 control")


def frozen_rows(table: Any) -> Any:
    store = control.FrameStore(1, table.store.records.shape[1], table.device, control.native.OBSERVATION_FIELDS)
    for name in ("records", "actions", "rewards", "flags"):
        setattr(store, name, getattr(table.store, name))
    old = control.DecisionTable(len(table.values), store)
    old.count = len(table.values)
    for name in ("integers", "conditions", "coefficients", "prefix", "opponent", "actions", "behavior", "rewards"):
        getattr(old, name).copy_(getattr(table, name))
    old.meta[:] = table.integers.cpu().numpy()
    old.log_lambda[:] = table.conditions.cpu().numpy()
    return old


class ForcedControl(control.TransformerPolicy):
    """Score common actions through the recorded cached rollout kernels."""

    table: Any
    players: int

    def _action_uniforms(self, batch: Any, rng: Any) -> torch.Tensor:
        return torch.full((4, 4, len(batch)), 0.5, device=self.device, dtype=torch.float64)

    def _decode_tensors(
        self,
        integers: Any,
        conditions: Any,
        prefix: Any,
        opponent: Any,
        hidden: Any,
        observed: Any,
        uniforms: Any,
        actions: Any,
        *,
        history: Any,
        count: int,
        decoder: Any,
    ) -> Any:
        rows = (
            integers[:, control.CLOCK] // 2 * self.players + integers[:, control.LANE] * 2 + integers[:, control.SIDE]
        )
        return super()._decode_tensors(
            integers,
            conditions,
            prefix,
            opponent,
            hidden,
            observed,
            uniforms,
            self.table.actions[rows],
            history=history,
            count=count,
            decoder=decoder,
        )


def qualify(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("inputs.pt", "precision.json")):
        raise FileExistsError("precision evidence already exists")
    print("dense precision: initialize", flush=True)
    random.seed(65)
    np.random.seed(65)
    torch.set_num_threads(2)
    torch.manual_seed(65)
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = exp.load_evaluation_checkpoint(ROOT / "runs/proxy/final.pt", device=torch.device("cuda"))
    model = checkpoint.model.requires_grad_(True)
    model.percentile_head.requires_grad_(False)
    schedule = exp.development_config(model.head_offsets)
    policy = exp.TransformerPolicy(model, schedule, checkpoint.statistics)
    reference = policy.frozen_copy()
    with torch.no_grad():
        for head in model.temporal.outputs.values():
            head.down.bias.add_(torch.linspace(-0.1, 0.1, head.down.out_features, device=policy.device))
        torch.nn.init.normal_(model.critic_residual.residual.down.weight, std=0.005)
    source = Path(os.environ.get("HAL_MSL_SOURCE", "~/src/melee-sim-light")).expanduser()
    with contextlib.ExitStack() as resources:
        shards = []
        for index, cpu in enumerate(sorted(os.sched_getaffinity(0))[:4]):
            shard = exp.NativeSimulatorShard(
                source / "data", tuple(range(4 * index, 4 * index + 4)), cpu=cpu, device=policy.device, max_frame=28800
            )
            resources.callback(shard.close)
            shards.append(shard)
        sampling = exp.EpisodeSampler(
            exp.read_native_sampler(ROOT / "experiments/o65/pilot-sampler.json"), schedule, exp.CounterRng(65)
        )
        engine = exp.RolloutEngine((tuple(shards),), policy, sampling, {0: reference}, fixed_anchor_lanes=True)
        engine.initialize()
        run = exp.Trainer(engine, exp.TrainConfig(critic_warmup_updates=0))
        print("dense precision: rollout", flush=True)
        rollout = engine.collect(0)
        print("dense precision: prepare", flush=True)
        prepared = run.prepare(rollout)
        table = rollout.table
        old_table = frozen_rows(table)
        windows = prepared.windows.take(prepared.steps[0], prepared.placement.positions, prepared.steps[0] >= 0)
        valid = windows.slots.valid_WK
        chosen = windows.rows[valid]
        states = old_table.rows(chosen.cpu().numpy())
        features, observed = policy._packed_context(windows.inputs()[:5])
        dense_advantage = torch.zeros_like(table.values)
        dense_targets = torch.zeros_like(table.values)
        dense_selected = torch.zeros_like(table.valid)
        dense_advantage[rollout.ids] = prepared.advantages
        dense_targets[rollout.ids] = prepared.targets
        dense_selected[rollout.ids] = prepared.selected
        batch = ProxyBatch(
            features,
            observed,
            windows.slots,
            states,
            table.actions[chosen],
            chosen,
            dense_advantage[windows.rows],
            dense_targets[windows.rows],
            dense_selected[windows.rows] & valid,
            table.coefficients[windows.rows],
        )
        behavior = table.behavior[windows.rows]
        default_policy = ForcedControl(model, schedule, checkpoint.statistics, compiled=True)
        default_policy.table, default_policy.players = old_table, table.players
        default_policy.tick_decoder.replay_variants = default_policy.tick_decoder.variants.copy()
        default_ref = control.TransformerPolicy(reference.model, schedule, checkpoint.statistics, compiled=True)
        print("dense precision: control actor", flush=True)
        default = quantities(default_policy, default_ref, batch, behavior, window=False, fp32=False)
        print("dense precision: production actor", flush=True)
        full_reference = reference.evaluate_reference(
            prepared.windows, prepared.placement.positions, prepared.placement.windows
        )
        indices = exp.window_indices(windows.rows, table.players)

        def take(value: torch.Tensor) -> torch.Tensor:
            return value.flatten(0, 1)[indices]

        ref_output = ReplayOutput(
            take(full_reference.log_probs_WKF),
            tuple(take(p) for p in full_reference.distributions),
            take(full_reference.factor_mask_WKF),
            ReplayCriticFeatures(
                *(
                    take(getattr(full_reference.features, field.name))
                    for field in dataclasses.fields(full_reference.features)
                )
            ),
        )
        target = targets(ref_output, ref_output, behavior, batch)
        _, actor_output = policy.actor_step(windows, target, clip_epsilon=run.config.clip_epsilon)
        candidate = {
            "log_probs": actor_output.log_probs_WKF[actor_output.factor_mask_WKF].detach().clone(),
            "kl": actor_loss(
                actor_output, dataclasses.replace(target, selected_WK=actor_output.factor_mask_WKF.any(-1))
            )
            .factor_kl_WKF[actor_output.factor_mask_WKF]
            .detach()
            .clone(),
            "actor_gradients": torch.cat(
                [
                    (torch.zeros_like(p) if p.grad is None else p.grad).detach().flatten()
                    for p in policy.actor_parameters()
                ]
            ).clone(),
        }
        repeated = {name: [value] for name, value in candidate.items()}
        for _ in range(2):
            again = reference.evaluate_reference(
                prepared.windows, prepared.placement.positions, prepared.placement.windows
            )
            repeated_target = dataclasses.replace(
                target, reference_distributions=tuple(take(p) for p in again.distributions)
            )
            _, output_again = policy.actor_step(windows, repeated_target, clip_epsilon=run.config.clip_epsilon)
            mask_again = output_again.factor_mask_WKF
            repeated["log_probs"].append(output_again.log_probs_WKF[mask_again].detach().clone())
            repeated["kl"].append(
                actor_loss(output_again, dataclasses.replace(repeated_target, selected_WK=mask_again.any(-1)))
                .factor_kl_WKF[mask_again]
                .detach()
                .clone()
            )
            repeated["actor_gradients"].append(
                torch.cat(
                    [
                        (torch.zeros_like(p) if p.grad is None else p.grad).detach().flatten()
                        for p in policy.actor_parameters()
                    ]
                ).clone()
            )
        scoring = reference.reference_forward_model
        if (
            scoring is None
            or any(p.grad is not None for p in scoring.parameters())
            or any(p.requires_grad or p.grad is not None for p in reference.model.parameters())
        ):
            raise ValueError("reference scoring changed frozen parameters or accumulated gradients")
        print("dense precision: FP32 reference", flush=True)
        high_models = []
        for weights in (model.state_dict(), reference.model.state_dict()):
            high = exp.load_evaluation_checkpoint(
                ROOT / "runs/proxy/final.pt", device=policy.device, decoder_precision="fp32"
            ).model
            high.load_state_dict(weights)
            high.requires_grad_(True)
            high.percentile_head.requires_grad_(False)
            high_models.append(control.TransformerPolicy(high, schedule, checkpoint.statistics))
        high = quantities(high_models[0], high_models[1], batch, behavior, window=False, fp32=True)
        high_output = old_forward(high_models[0], batch, grad=False, fp32=True)
        high_ref = old_forward(high_models[1], batch, grad=False, fp32=True)
        baseline_ref = old_forward(default_ref, batch, grad=False)
        baseline_table = exp.DenseRollout(table.ticks, table.store)
        baseline_table.allocate_evaluation(model)
        ring = control.ActionRing(table.players // 2, engine.rings[0].capacity, policy.device)
        ring.controllers = torch.zeros_like(engine.rings[0].controllers)
        student_slots = torch.nonzero(engine.owners[0] < 0).flatten()
        meta = old_table.meta[student_slots.cpu().numpy()]
        layout = control.TickLayout(
            tuple(int(x) for x in meta[:, control.STREAM]), tuple(int(x) for x in meta[:, control.LANE])
        )
        print("dense precision: cached control", flush=True)
        with torch.no_grad():
            for tick in range(table.ticks):
                ids = tick * table.players + student_slots
                ids = ids[table.active[ids]]
                if not len(ids):
                    continue
                ring.length_device.zero_()
                ring.length.fill(0)
                output_tick = default_policy.sample_tick(
                    old_table.rows(ids.cpu().numpy()), control.CounterRng(65), layout, ring
                )
                baseline_table.record_evaluation(ids, output_tick, default_policy.values(output_tick.features))
        assert baseline_table.evaluation is not None and table.evaluation is not None
        for name, source_table, destination in (("default", baseline_table, default), ("candidate", table, candidate)):
            evaluation = source_table.evaluation
            assert evaluation is not None
            mu = exp._take_evaluation(evaluation, chosen)
            flat_ref = exp.window_evaluation(
                baseline_ref if name == "default" else ref_output, mu.actions, torch.nonzero(valid.flatten()).flatten()
            )
            mask = windows.slots.executed_WKF[valid]
            destination["rollout_log_probs"] = mu.log_probs[mask].detach().clone()
            destination["rollout_kl"] = exp.factor_kl(mu, flat_ref, mask)[mask].detach().clone()
            destination["rollout_values"] = source_table.values[chosen].detach().clone()
            loss = exp.critic_loss(
                policy.values(mu.features),
                dense_targets[chosen],
                torch.ones_like(chosen, dtype=torch.bool),
                value_scale=run.config.value_scale,
            )
            destination["rollout_critic_gradients"] = flattened_gradients(loss, policy.critic_parameters())
        high_mu = exp.window_evaluation(high_output, table.actions[chosen], torch.nonzero(valid.flatten()).flatten())
        high_ref_flat = exp.window_evaluation(
            high_ref, table.actions[chosen], torch.nonzero(valid.flatten()).flatten()
        )
        mask = windows.slots.executed_WKF[valid]
        high["rollout_log_probs"] = high_mu.log_probs[mask]
        high["rollout_kl"] = exp.factor_kl(high_mu, high_ref_flat, mask)[mask]
        high["rollout_values"] = values(high_models[0].model, high_output)[valid].detach()
        loss = exp.critic_loss(
            high_models[0].values(high_mu.features),
            dense_targets[chosen],
            torch.ones_like(chosen, dtype=torch.bool),
            value_scale=run.config.value_scale,
        )
        high["rollout_critic_gradients"] = flattened_gradients(loss, high_models[0].critic_parameters())
        precision = {}
        for name, value in candidate.items():
            baseline_error, candidate_error = error(default[name], high[name]), error(value, high[name])
            floor = 1e-6 if "gradients" in name else 1e-5
            replay_errors = [error(item, high[name]) for item in repeated.get(name, [value])]
            passed = all(
                errors[key] <= 1.5 * baseline_error[key] + (1e-5 if key == "relative_l2" else floor)
                for errors in replay_errors
                for key in errors
            )
            precision[name] = {
                "default": baseline_error,
                "candidate": candidate_error,
                "candidate_replays": replay_errors,
                "absolute_floor": floor,
                "relative_l2_floor": 1e-5,
                "passed": passed,
            }
        report = {
            "format": "hal.o65.dense-precision.v1",
            "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "trainer_sha256": exp.sha256(ROOT / "experiments/065_selfplay_ppo.py"),
            "test_sha256": exp.sha256(Path(__file__)),
            "checkpoint_sha256": checkpoint.checkpoint_sha256,
            "environment": exp._environment(),
            "control_commit": "aacdfee0",
            "configuration": {
                "matches": 16,
                "actor_shape": list(valid.shape),
                "tf32": False,
                "bias_shift": [-0.1, 0.1],
                "critic_residual_std": 0.005,
                "rollout_control": "recorded student TickLayout with active rows and common teacher-forced actions",
                "rollout_control_slots": len(student_slots),
            },
            "precision": precision,
        }
        torch.save(
            {
                "windows": dataclasses.asdict(windows),
                "reference_windows": dataclasses.asdict(prepared.windows),
                "reference_positions": prepared.placement.positions,
                "window_order": prepared.placement.windows,
                "actor_features": dataclasses.asdict(actor_output.features),
                "reference_features": dataclasses.asdict(ref_output.features),
                "actor_distributions": tuple(p.detach() for p in actor_output.distributions),
                "reference_distributions": tuple(p.detach() for p in ref_output.distributions),
                "student": model.state_dict(),
                "reference": reference.model.state_dict(),
                "behavior": behavior,
                "advantages": batch.advantages_WK,
                "targets": batch.value_targets_WK,
            },
            output / "inputs.pt",
        )
        report["inputs_sha256"] = exp.sha256(output / "inputs.pt")
        report["control_sha256"] = exp.sha256(CONTROL_PATH)
        report["sources"] = {
            name: exp.sha256(ROOT / name)
            for name in (
                "hal/models/action_sequence.py",
                "hal/models/window_replay.py",
                "hal/models/replay_actor.py",
                "hal/inference/graphed_actor.py",
                "tests/o65_reference.py",
                "tests/test_window_replay_proxy.py",
            )
        }
        report["configuration"]["reference"] = (
            "one CUDA graph; actor window order, padding and decision slots; read-only autograd scoring copy"
        )
        (output / "precision.json").write_text(json.dumps(report, indent=2))
        if not all(row["passed"] for row in precision.values()):
            raise ValueError(f"dense precision gate failed; see {output / 'precision.json'}")
        return report


def trace_statistics(output: Path, mode: Literal["dense", "control"], *, verify_resume: bool) -> dict[str, Any]:
    if mode not in ("dense", "control"):
        raise ValueError("statistics mode must be dense or control")
    if verify_resume and mode != "dense":
        raise ValueError("the recorded control is not a supported checkpoint writer")
    output.mkdir(parents=True, exist_ok=False)
    module = exp if mode == "dense" else control
    torch.set_num_threads(2)
    torch.manual_seed(65)
    random.seed(65)
    np.random.seed(65)
    torch.use_deterministic_algorithms(False)
    checkpoint = module.load_evaluation_checkpoint(
        ROOT / "runs/proxy/final.pt", device=torch.device("cuda"), decoder_precision="bf16"
    )
    model = checkpoint.model.requires_grad_(True)
    model.percentile_head.requires_grad_(False)
    schedule = module.development_config(model.head_offsets)
    policy = (
        module.TransformerPolicy(model, schedule, checkpoint.statistics)
        if mode == "dense"
        else module.TransformerPolicy(model, schedule, checkpoint.statistics, compiled=True)
    )
    sampling = module.read_native_sampler(ROOT / "experiments/o65/pilot-sampler.json")
    source = Path(os.environ.get("HAL_MSL_SOURCE", "~/src/melee-sim-light")).expanduser()
    updates = []
    with contextlib.ExitStack() as resources:
        shards = []
        for index, cpu in enumerate(sorted(os.sched_getaffinity(0))[:4]):
            shard = module.NativeSimulatorShard(
                source / "data",
                tuple(range(16 * index, 16 * index + 16)),
                cpu=cpu,
                device=policy.device,
                max_frame=512,
            )
            resources.callback(shard.close)
            shards.append(shard)
        engine = module.RolloutEngine(
            (tuple(shards),),
            policy,
            module.EpisodeSampler(sampling, schedule, module.CounterRng(65)),
            {0: policy.frozen_copy()},
            fixed_anchor_lanes=True,
        )
        engine.initialize()
        run = module.Trainer(engine, module.TrainConfig(critic_warmup_updates=0))
        for update in range(3):
            batch, metrics = run.run_update()
            if mode == "dense":
                matches = batch.rollout.table.integers[batch.rollout.ids, module.MATCH]
                frame_reward = batch.rollout.table.rewards[batch.rollout.ids].sum(-1)
            else:
                matches = torch.tensor([row.state.match for row in batch.rollout.records], device=policy.device)
                frame_reward = torch.tensor([sum(row.rewards) for row in batch.rollout.records], device=policy.device)
            kl = module.factor_kl(batch.old, batch.reference, batch.decisions.executed_factors_JF)
            entropy = torch.zeros_like(batch.values)
            for p in batch.old.distributions:
                entropy += torch.where(p > 0, -p * torch.where(p > 0, p, 1).log(), 0).sum(-1).sum(-1)
            rows = []
            for match in range(64):
                at = matches == match
                completed = [result.duration for result in batch.rollout.results if result.match == match]
                rows.append(
                    {
                        "match": match,
                        "reward": float(frame_reward[at].sum()),
                        "kl": float(kl[at].sum() / batch.decisions.replan_J[at].sum()),
                        "entropy": float(entropy[at].sum() / batch.decisions.replan_J[at].sum()),
                        "advantages": batch.advantages[at].cpu().tolist(),
                        "targets": batch.targets[at].cpu().tolist(),
                        "episode_lengths": completed,
                        "resets": len(completed),
                    }
                )
            updates.append({"update": update, "matches": rows, "metrics": metrics})
            print(mode, update, metrics["systems/update_seconds"], flush=True)
        result = {
            "format": "hal.o65.dense-statistical-raw.v1",
            "mode": mode,
            "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "trainer_sha256": exp.sha256(ROOT / "experiments/065_selfplay_ppo.py")
            if mode == "dense"
            else exp.sha256(CONTROL_PATH),
            "test_sha256": exp.sha256(Path(__file__)),
            "environment": exp._environment(),
            "checkpoint": checkpoint.checkpoint_sha256,
            "data_manifests": checkpoint.provenance["data_manifests"],
            "sampling": dataclasses.asdict(sampling),
            "schedule": dataclasses.asdict(schedule),
            "training": dataclasses.asdict(run.config),
            "max_frame": 512,
            "seed": 65,
            "simulator": shards[0].shard.library_sha256,
            "adapter": module.native_bridge.library_sha256(),
            "simulator_manifest": exp.sha256(source / "data/raw/manifest.json"),
            "updates": updates,
        }
        (output / "statistics.json").write_text(json.dumps(result, indent=2))
        if verify_resume:
            provenance = {
                "git_sha": result["git_sha"],
                "checkpoint_sha256": checkpoint.checkpoint_sha256,
                "data_manifest_sha256": checkpoint.provenance["data_manifests"]["ranked-anonymized-1-policy-world-v8"],
                "simulator_sha256": shards[0].shard.library_sha256,
                "adapter_sha256": module.native_bridge.library_sha256(),
            }
            exp.verify_checkpoint_resume(run, output / "resume", provenance)
        return result


@pytest.mark.parametrize("name", ["inputs.pt", "precision.json"])
def test_precision_rejects_existing_evidence_before_cuda(name: str, tmp_path: Path) -> None:
    (tmp_path / name).write_bytes(b"recorded evidence")
    with pytest.raises(FileExistsError, match="evidence already exists"):
        qualify(tmp_path)


def test_statistics_rejects_control_checkpoint_writes_before_cuda(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a supported checkpoint writer"):
        trace_statistics(tmp_path / "new", "control", verify_resume=True)
    assert not (tmp_path / "new").exists()


@pytest.mark.integration
def test_dense_proxy_precision(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.fail("dense proxy qualification requires CUDA")
    with prepared_shape_compilation(64):
        qualify(tmp_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("precision", "stat-dense", "stat-control"), default="precision")
    parser.add_argument("--verify-resume", action="store_true")
    args = parser.parse_args()
    with prepared_shape_compilation(64):
        if args.mode == "precision":
            qualify(args.output)
        else:
            trace_statistics(
                args.output, "dense" if args.mode == "stat-dense" else "control", verify_resume=args.verify_resume
            )
