"""Common-input BF16/FP32 qualification on the pinned proxy and native simulator."""

import contextlib
import dataclasses
import io
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from test_065_columnar import ROOT
from test_065_columnar import exp
from test_065_vectorize import actor_gradients
from test_065_vectorize import errors
from test_065_vectorize import executed_kl
from test_065_vectorize import gradient_comparison
from test_065_vectorize import grouped_score
from test_065_vectorize import sequential_replay
from torch import Tensor


def test_resume_rejects_a_changed_decoder_precision(tmp_path: Path) -> None:
    from test_065_loop import PROVENANCE
    from test_065_loop import exp as loop_exp
    from test_065_loop import transformer_trainer

    run = transformer_trainer("cpu", decoder_precision="bf16")
    manifest = loop_exp.save_checkpoint(run, tmp_path, PROVENANCE)
    other = transformer_trainer("cpu", decoder_precision="fp32")
    assert loop_exp._configuration_digest(run) != loop_exp._configuration_digest(other)
    with pytest.raises(ValueError, match="configuration mismatch"):
        loop_exp.restore_checkpoint(other, manifest, PROVENANCE)
    assert run.policy.frozen_copy().model.decoder_precision == "bf16"


def test_native_precision_is_validated_and_profile_preserves_it(tmp_path: Path) -> None:
    cpu = min(os.sched_getaffinity(0))
    profile = exp.GpuProfileCli(tmp_path, tmp_path, tmp_path, cpus=(cpu,), decoder_precision="fp32")
    assert profile.native().decoder_precision == "fp32"
    assert dataclasses.replace(profile, decoder_precision="bf16").native().decoder_precision == "bf16"
    with pytest.raises(ValueError, match="decoder precision"):
        dataclasses.replace(profile.native(), decoder_precision="fp16")


def test_resume_rejects_a_mismatched_actor_precision_before_model_mutation(tmp_path: Path) -> None:
    from test_065_loop import PROVENANCE
    from test_065_loop import exp as loop_exp
    from test_065_loop import transformer_trainer

    run = transformer_trainer("cpu", decoder_precision="bf16")
    manifest = loop_exp.save_checkpoint(run, tmp_path, PROVENANCE)
    raw = json.loads(manifest.read_text())
    shard = next(shard for shard in raw["shards"] if shard["role"] == "actor")
    actor = torch.load(tmp_path / shard["name"], weights_only=False)
    actor["decoder_precision"] = "fp32"
    payload = io.BytesIO()
    torch.save(actor, payload)
    shard["name"], shard["sha256"] = loop_exp._durable_file(tmp_path, payload.getvalue(), "actor.pt")
    manifest.write_text(json.dumps(raw))
    before = tuple(parameter.detach().clone() for parameter in run.policy.model.parameters())
    with pytest.raises(ValueError, match="actor precision"):
        loop_exp.restore_checkpoint(run, manifest, PROVENANCE)
    for actual, expected in zip(run.policy.model.parameters(), before, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def gap_distribution(old: Tensor, behavior: Tensor, mask: Tensor) -> dict[str, float]:
    gap = torch.where(mask, behavior - old, 0).sum(-1).detach().float()
    absolute = gap.abs()
    return {
        "mean": float(gap.mean()),
        "std": float(gap.std()),
        "p50": float(absolute.quantile(0.5)),
        "p95": float(absolute.quantile(0.95)),
        "p99": float(absolute.quantile(0.99)),
        "max": float(absolute.max()),
        "truncated_fraction": float(((-gap).exp() > 2).float().mean()),
    }


class CommonInputPolicy(exp.TransformerPolicy):
    def __init__(self, control: Any, treatment: Any) -> None:
        super().__init__(control.model, control.schedule, control.statistics, compiled=True)
        self.comparisons = tuple(
            exp.TransformerPolicy(policy.model, policy.schedule, policy.statistics) for policy in (control, treatment)
        )
        self.outputs: tuple[list[Any], list[Any]] = ([], [])
        for policy in self.comparisons:
            sequential_replay(policy)

    def clear_caches(self) -> None:
        super().clear_caches()
        for policy in self.comparisons:
            policy.clear_caches()

    @torch.no_grad()
    def sample_tick(self, states: Any, rng: Any, layout: Any, ring: Any) -> Any:
        result = super().sample_tick(states, rng, layout, ring)
        due = np.flatnonzero(states.meta[:, exp.CLOCK] % states.meta[:, exp.REPLAN] == 0)
        if len(due):
            actions = result.actions[torch.from_numpy(due).to(self.device)]
            for policy, outputs in zip(self.comparisons, self.outputs, strict=True):
                outputs.append(grouped_score(policy, states.take(due), actions))
        return result


@pytest.mark.integration
def test_proxy_decoder_precision_on_common_inputs(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.fail("proxy decoder qualification requires CUDA")
    torch.set_num_threads(2)
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    torch.manual_seed(65)
    try:
        policies = []
        for precision in ("fp32", "bf16"):
            checkpoint = exp.load_evaluation_checkpoint(
                ROOT / "runs/proxy/final.pt", device=torch.device("cuda"), decoder_precision=precision
            )
            checkpoint.model.requires_grad_(True)
            checkpoint.model.percentile_head.requires_grad_(False)
            schedule = exp.development_config(checkpoint.model.head_offsets)
            policies.append(exp.TransformerPolicy(checkpoint.model, schedule, checkpoint.statistics, compiled=True))
        control, treatment = policies
        references = [policy.frozen_copy() for policy in policies]
        # Exercise a nonzero KL and its GAE contribution at common actor weights.
        with torch.no_grad():
            for head in control.model.temporal.outputs.values():
                head.down.bias.add_(torch.linspace(-0.1, 0.1, head.down.out_features, device=control.device))
        treatment.model.load_state_dict(control.model.state_dict())
        references[1].model.load_state_dict(references[0].model.state_dict())
        rollout_policy = CommonInputPolicy(control, treatment)
        source = Path(os.environ.get("HAL_MSL_SOURCE", "~/src/melee-sim-light")).expanduser()
        sampling = exp.read_native_sampler(ROOT / "experiments/o65/pilot-sampler.json")
        with contextlib.ExitStack() as resources:
            shards = []
            for index, cpu in enumerate(sorted(os.sched_getaffinity(0))[:4]):
                shard = exp.NativeSimulatorShard(
                    source / "data",
                    tuple(range(4 * index, 4 * index + 4)),
                    cpu=cpu,
                    device=control.device,
                    max_frame=28800,
                )
                resources.callback(shard.close)
                shards.append(shard)
            engine = exp.RolloutEngine(
                (tuple(shards),),
                rollout_policy,
                exp.EpisodeSampler(sampling, schedule, exp.CounterRng(65)),
                {0: references[0]},
                fixed_anchor_lanes=True,
            )
            engine.initialize()
            rollout = engine.collect(0)
            batches = []
            runs = []
            for policy, reference in zip(policies, references, strict=True):
                scoring = exp.RolloutEngine(
                    engine.groups, policy, engine.sampler, {0: reference}, fixed_anchor_lanes=True
                )
                scoring.restore_pool(engine.snapshot())
                run = exp.Trainer(scoring, exp.TrainConfig(critic_warmup_updates=0))
                runs.append(run)
                batches.append(run.prepare(rollout))
            expected, actual = batches
            expected_mu, actual_mu = (exp._concatenate_evaluations(rows) for rows in rollout_policy.outputs)
            mask = expected.decisions.executed_factors_JF
            assert bool((executed_kl(expected.old, expected.reference, mask) > 0).any())
            for index, behavior in enumerate((expected_mu, actual_mu)):
                batches[index] = dataclasses.replace(
                    batches[index],
                    decisions=dataclasses.replace(
                        batches[index].decisions,
                        behavior_log_probs_JF=behavior.log_probs,
                    ),
                )
            gaps = {}
            gaps_by_cadence = {}
            for precision, batch, behavior in zip(("fp32", "bf16"), batches, (expected_mu, actual_mu), strict=True):
                gaps[precision] = gap_distribution(batch.old.log_probs, behavior.log_probs, mask)
                for timing in schedule.configurations:
                    selected = (batch.decisions.delay_J == timing.delay) & (batch.decisions.replan_J == timing.replan)
                    gaps_by_cadence[f"{precision}_d{timing.delay}_r{timing.replan}"] = gap_distribution(
                        batch.old.log_probs[selected], behavior.log_probs[selected], mask[selected]
                    )
            report: dict[str, Any] = {
                "format": "hal.o65.decoder-precision-numerics.v1",
                "source": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "source_sha256": exp.sha256(ROOT / "experiments/065_selfplay_ppo.py"),
                "test_source_sha256": exp.sha256(Path(__file__)),
                "environment": exp._environment(),
                "checkpoint": checkpoint.checkpoint_sha256,
                "data_manifests": checkpoint.provenance["data_manifests"],
                "sampler_sha256": exp.sha256(ROOT / "experiments/o65/pilot-sampler.json"),
                "simulator_manifest_sha256": exp.sha256(source / "data/raw/manifest.json"),
                "configuration": {
                    "schedule": dataclasses.asdict(schedule),
                    "training": dataclasses.asdict(runs[0].config),
                    "sampling": dataclasses.asdict(sampling),
                    "actor_bias_shift": "linspace(-0.1, 0.1) on each temporal factor's final bias",
                    "fixed_anchor_lanes": True,
                    "matches": 16,
                },
                "simulator": shards[0].shard.library_sha256,
                "adapter": exp.native_bridge.library_sha256(),
                "training_log_prob": errors(actual.old.log_probs[mask], expected.old.log_probs[mask]),
                "rollout_log_prob": errors(actual_mu.log_probs[mask], expected_mu.log_probs[mask]),
                "kl": errors(
                    executed_kl(actual.old, actual.reference, mask),
                    executed_kl(expected.old, expected.reference, mask),
                ),
                "kl_fp32_mean": float(executed_kl(expected.old, expected.reference, mask).mean()),
                "mu_old_gap": gaps,
                "mu_old_gap_by_cadence": gaps_by_cadence,
                "values": errors(actual.values, expected.values),
                "targets": errors(actual.targets, expected.targets),
                "actor_gradient": gradient_comparison(
                    actor_gradients(runs[1], batches[1]),
                    actor_gradients(runs[0], batches[0]),
                ),
            }
            gradients = []
            for run, batch in zip(runs, batches, strict=True):
                loss = exp.critic_loss(
                    run.policy.values(batch.old.features),
                    batch.targets,
                    torch.ones_like(batch.selected),
                    value_scale=run.config.value_scale,
                )
                gradients.append(torch.autograd.grad(loss, run.policy.critic_parameters()))
            report["critic_gradient"] = gradient_comparison(gradients[1], gradients[0])
            (tmp_path / "decoder-precision.json").write_text(json.dumps(report, indent=2, sort_keys=True))
            print(json.dumps(report, sort_keys=True), flush=True)
            for name in ("training_log_prob", "rollout_log_prob"):
                assert report[name]["max"] < 0.2
                assert report[name]["p99"] < 0.06
            assert report["kl"]["max"] < 0.02
            assert report["kl"]["p99"] < 0.005
            for name in ("actor_gradient", "critic_gradient"):
                assert report[name]["cosine"] > 0.999
                assert report[name]["relative_norm"] < 0.03
    finally:
        torch.use_deterministic_algorithms(previous)
