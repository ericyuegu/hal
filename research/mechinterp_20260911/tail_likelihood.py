"""Score the executed suffix of O50's frozen latent-prefix policy.

The historical policy samples four actions and executes offsets +3/+4. Its
likelihood marginalizes the first two sampled actions: average probabilities
before taking the logarithm. Averaging conditional NLLs is a different score.
"""

import argparse
import gc
import hashlib
import json
import math
import time
from collections.abc import Callable
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor

from hal.inference import o50_model
from hal.training import features
from research.mechinterp_20260911 import o50_probe

SEED = 20_260_913
LATENT_FRAMES = 2


def log_mean_likelihood(log_weights: Tensor) -> tuple[Tensor, Tensor]:
    """Return log mean probability and likelihood-weight ESS for each replay.

    Input is [replay, independently sampled latent prefix]. Zero conditional
    probabilities are allowed, but an all-zero Monte Carlo estimate is rejected.
    """
    if log_weights.ndim != 2 or min(log_weights.shape) < 1:
        raise ValueError("log weights must be a nonempty [replay, particle] matrix")
    if torch.isnan(log_weights).any() or (log_weights > 0).any():
        raise ValueError("log probabilities must be non-positive and not NaN")
    values = log_weights.double()
    total = torch.logsumexp(values, dim=1)
    if not torch.isfinite(total).all():
        raise ValueError("all sampled prefixes give zero target likelihood for a replay")
    normalized = torch.softmax(values, dim=1)
    ess = normalized.square().sum(dim=1).reciprocal()
    return total - math.log(values.shape[1]), ess


def sequential_teacher_logits(
    model: o50_model.O50Model,
    hidden: Tensor,
    observed: Tensor,
    forced: Tensor,
) -> dict[str, Tensor]:
    """Evaluate joint-action density with the public decoder's cached steps.

    This preserves the sampler's addition order, temporal cache, and group
    factorization. It does not sample a replacement target or change world state.
    """
    if forced.shape != (hidden.shape[0], 4, o50_model.CONTROLLER_GROUP_COUNT):
        raise ValueError("forced actions must have shape [replay, 4, controller group]")
    if observed.shape != forced.shape[:1] + (o50_model.CONTROLLER_GROUP_COUNT,):
        raise ValueError("observed actions do not match the replay batch")
    temporal = model.temporal
    raw_trunk = hidden[:, -1]
    normalized_trunk = F.rms_norm(raw_trunk, (raw_trunk.shape[-1],), eps=1e-6)
    weight = temporal.token_projection.weight
    action_stop = temporal.trunk_width + temporal.controller_width
    state_bias = F.linear(normalized_trunk, weight[:, : temporal.trunk_width], temporal.token_projection.bias)
    previous = observed
    caches: list[tuple[Tensor, Tensor] | None] = [None] * len(temporal.blocks)
    collected: dict[str, list[Tensor]] = {name: [] for name in o50_model.CONTROLLER_GROUP_NAMES}
    for depth, offset in enumerate(temporal.head_offsets[:4]):
        offsets = torch.full((hidden.shape[0],), offset, dtype=torch.long, device=hidden.device)
        action = F.linear(model.codec.embed_frame(previous), weight[:, temporal.trunk_width : action_stop])
        offset_features = F.linear(temporal.offset_embedding(offsets), weight[:, action_stop:])
        state = F.rms_norm(state_bias + (action + offset_features), (temporal.d_model,), eps=1e-6)
        for index, block in enumerate(temporal.blocks):
            forward_step = cast(
                Callable[[Tensor, tuple[Tensor, Tensor] | None], tuple[Tensor, tuple[Tensor, Tensor]]],
                block.forward_step,
            )
            state, caches[index] = forward_step(state, caches[index])
        state = F.rms_norm(state, (temporal.d_model,), eps=1e-6)
        embedded: dict[str, Tensor] = {}
        for position, name in enumerate(o50_model.CONTROLLER_DECODE_ORDER):
            if position:
                prefix = torch.cat([embedded[group] for group in o50_model.CONTROLLER_DECODE_ORDER[:position]], dim=-1)
                scale, shift = temporal.group_condition[name](prefix).chunk(2, dim=-1)
                conditioned = state * (1 + scale.tanh()) + shift
            else:
                conditioned = state
            logits = temporal.outputs[name](conditioned) + temporal.trunk_outputs[name](raw_trunk)
            logits = logits - logits.mean(dim=-1, keepdim=True)
            if name == "buttons":
                trigger = forced[:, depth, o50_model.CONTROLLER_GROUP_INDEX["triggers"]]
                logits = logits.masked_fill(model.codec.button_mask(trigger), float("-inf"))
            collected[name].append(logits)
            group_index = o50_model.CONTROLLER_GROUP_INDEX[name]
            embedded[name] = model.codec.group_embedding(name, forced[:, depth, group_index])
        previous = forced[:, depth]
    return {name: torch.stack(values, dim=1) for name, values in collected.items()}


def sampled_prefix_log_weights(
    model: o50_model.O50Model,
    hidden: Tensor,
    observed: Tensor,
    targets: Tensor,
    uniforms: Tensor,
) -> tuple[Tensor, bool]:
    """Integrand log p(expert suffix | sampled latent prefix, replay history)."""
    batch = targets.shape[0]
    if uniforms.ndim != 4 or uniforms.shape[:3] != (4, 4, batch):
        raise ValueError("uniforms must have shape [4, 4, replay, particle]")
    particles = uniforms.shape[-1]
    hidden_particles = hidden[:, -1:].repeat_interleave(particles, dim=0)
    observed_particles = observed.repeat_interleave(particles, dim=0)
    targets_particles = targets.repeat_interleave(particles, dim=0)
    flat_uniforms = uniforms.reshape(4, 4, batch * particles)
    force_mask = torch.arange(4, device=hidden.device)[None].expand(batch * particles, -1) >= LATENT_FRAMES
    sampled = model.temporal.sample_conditioned(
        hidden_particles,
        observed_particles,
        targets_particles,
        force_mask,
        flat_uniforms,
        argmax=False,
        sample_start=0,
    )
    if not torch.equal(sampled[:, LATENT_FRAMES:], targets_particles[:, LATENT_FRAMES:]):
        raise RuntimeError("suffix forcing did not retain the expert target")
    logits = sequential_teacher_logits(model, hidden_particles, observed_particles, sampled)
    reconstructed = torch.stack(
        [
            o50_model.sample_categorical(
                logits[name],
                argmax=False,
                uniform=flat_uniforms[:, group].transpose(0, 1),
            )
            for group, name in enumerate(o50_model.CONTROLLER_GROUP_NAMES)
        ],
        dim=-1,
    )
    parity = bool(torch.equal(reconstructed[:, :LATENT_FRAMES], sampled[:, :LATENT_FRAMES]))
    if not parity:
        raise RuntimeError("sequential density does not reproduce the public sampler's latent actions")
    loss = o50_probe.horizon_group_nll(logits, sampled)
    return -loss[:, LATENT_FRAMES:].sum(dim=(1, 2)).reshape(batch, particles), parity


def object_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError("expected a mapping with string keys")
    return cast(Mapping[str, object], value)


def tensor_bytes(value: object) -> bytes:
    if not isinstance(value, Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
        raise ValueError("encoded package member must be a one-dimensional uint8 tensor")
    return value.numpy().tobytes()


def historical_context(context: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Frozen gameplay supplies the unknown-player ID, not a replay identity."""
    return {**context, "ego_player_id": torch.zeros_like(context["ego_player_id"])}


def run_checkpoint(
    cohort: Path,
    checkpoint: Path,
    *,
    device: torch.device,
    particle_counts: tuple[int, ...],
    batch_size: int,
    script_sha256: str,
    source_probe_sha256: str,
) -> dict[str, object]:
    started = time.monotonic()
    package = object_mapping(torch.load(cohort, weights_only=True, map_location="cpu"))
    expected = {
        "bundle_reference_json",
        "context_features",
        "data_json",
        "player_codes",
        "reference_config",
        "stats_payload",
        "target",
    }
    if package.keys() != expected:
        raise ValueError("cohort package does not match the fixed O50 probe contract")
    reference_config = o50_model.O50Config.from_json(tensor_bytes(package["reference_config"]))
    stats = object_mapping(json.loads(tensor_bytes(package["stats_payload"])))
    context_raw = object_mapping(package["context_features"])
    if any(not isinstance(value, Tensor) for value in context_raw.values()):
        raise ValueError("context feature values must be tensors")
    contexts = cast(Mapping[str, Tensor], context_raw)
    targets_native = package["target"]
    if not isinstance(targets_native, Tensor) or targets_native.shape != (256, 4, 14):
        raise ValueError("tail probe requires the fixed 256-replay, four-frame native-action cohort")
    if any(value.shape[:2] != (256, reference_config.architecture.L_ctx) for value in contexts.values()):
        raise ValueError("context shape differs from the cohort reference config")
    data = object_mapping(json.loads(cast(str, package["data_json"])))
    windows = data["windows"]
    if not isinstance(windows, list) or len({object_mapping(row)["replay_id"] for row in windows}) != 256:
        raise ValueError("cohort must record 256 distinct replay identities")
    model, config, provenance = o50_probe.load_checkpoint_model(
        checkpoint,
        reference_config=reference_config,
        stats_payload=stats,
        player_codes=tensor_bytes(package["player_codes"]) or None,
        device=device,
    )
    if config.architecture.L_ctx != reference_config.architecture.L_ctx or config.prediction_frames != 4:
        raise ValueError("checkpoint temporal representation differs from the cohort")
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    uniforms = torch.rand(4, 4, 256, max(particle_counts), generator=generator)
    weights = []
    teacher_losses = []
    recorded_identity_teacher_losses = []
    parallel_loss_differences = []
    for start in range(0, 256, batch_size):
        stop = min(start + batch_size, 256)
        context = {name: value[start:stop].to(device) for name, value in contexts.items()}
        with torch.inference_mode(), o50_model.amp_context(config, device):
            observed_history = model.codec.quantize(features.stack_actions(context))
            observed = observed_history[:, -1]
            target = model.codec.quantize(targets_native[start:stop].to(device))
            padding = torch.zeros(stop - start, device=device, dtype=torch.long)
            recorded_identity_hidden = model.forward_dense(context, padding, observed_history)
            recorded_identity_logits = sequential_teacher_logits(model, recorded_identity_hidden, observed, target)
            recorded_identity_loss = o50_probe.horizon_group_nll(recorded_identity_logits, target)
            context = historical_context(context)
            hidden = model.forward_dense(context, padding, observed_history)
            sequential = sequential_teacher_logits(model, hidden, observed, target)
            teacher_loss = o50_probe.horizon_group_nll(sequential, target)
            parallel = o50_probe.teacher_forced_logits(model, hidden, observed, target)
            parallel_loss = o50_probe.horizon_group_nll(parallel, target)
            log_weights, _ = sampled_prefix_log_weights(
                model, hidden, observed, target, uniforms[:, :, start:stop].to(device)
            )
        weights.append(log_weights.cpu())
        teacher_losses.append(teacher_loss.cpu())
        recorded_identity_teacher_losses.append(recorded_identity_loss.cpu())
        parallel_loss_differences.append((parallel_loss - teacher_loss).cpu())
    log_weights = torch.cat(weights).double()
    teacher_loss = torch.cat(teacher_losses).double()
    recorded_identity_loss = torch.cat(recorded_identity_teacher_losses).double()
    per_count: dict[str, object] = {}
    bits_denominator = (4 - LATENT_FRAMES) * math.log(2)
    for particles in particle_counts:
        marginal, ess = log_mean_likelihood(log_weights[:, :particles])
        nll_bits = -marginal / bits_denominator
        per_count[str(particles)] = {
            "mean_joint_bits_per_executed_frame": float(nll_bits.mean()),
            "per_replay_joint_bits_per_executed_frame": nll_bits.tolist(),
            "ess_quantiles": torch.quantile(ess, torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0], dtype=ess.dtype)).tolist(),
            "per_replay_likelihood_weight_ess": ess.tolist(),
            "fraction_ess_below_four": float((ess < 4).double().mean()),
        }
    half = log_weights.shape[1] // 2
    left, _ = log_mean_likelihood(log_weights[:, :half])
    right, _ = log_mean_likelihood(log_weights[:, half:])
    difference = torch.cat(parallel_loss_differences)
    return {
        "schema_version": 2,
        "checkpoint": provenance,
        "cohort_sha256": hashlib.sha256(cohort.read_bytes()).hexdigest(),
        "source_probe_sha256": source_probe_sha256,
        "script_sha256": script_sha256,
        "analysis_base_git_sha": o50_probe.ANALYSIS_BASE_GIT_SHA,
        "protocol": {
            "name": "frozen-o50-latent-prefix-tail-likelihood-v1",
            "latent_offsets": [1, 2],
            "executed_offsets": [3, 4],
            "random_seed": SEED,
            "particle_counts": particle_counts,
            "nested_particles": True,
            "sampling": "model prefix; expert suffix forced only for density evaluation",
            "player_identity": "ego_player_id forced to zero, as in frozen gameplay",
            "score": "negative log of averaged likelihood, not averaged negative log likelihood",
            "monte_carlo_bias": "probability estimate is unbiased; its negative logarithm has upward finite-particle bias",
        },
        "estimates": per_count,
        "teacher_forced": {
            "full_joint_bits_per_predicted_frame": float(teacher_loss.sum(dim=2).mean() / math.log(2)),
            "tail_given_expert_prefix_bits_per_executed_frame": float(
                teacher_loss[:, 2:].sum(dim=(1, 2)).mean() / bits_denominator
            ),
            "per_replay_tail_given_expert_prefix_bits_per_executed_frame": (
                teacher_loss[:, 2:].sum(dim=(1, 2)) / bits_denominator
            ).tolist(),
        },
        "mixed_prefix_score": {
            "mean_conditional_nll_bits_per_executed_frame": float(-log_weights.mean() / bits_denominator),
            "proper_marginal_log_score": False,
        },
        "controls": {
            "public_sampler_latent_action_parity": True,
            "recorded_identity_teacher_full_joint_bits_per_predicted_frame": float(
                recorded_identity_loss.sum(dim=2).mean() / math.log(2)
            ),
            "masked_minus_recorded_identity_teacher_bits_per_predicted_frame": float(
                (teacher_loss - recorded_identity_loss).sum(dim=2).mean() / math.log(2)
            ),
            "parallel_minus_sequential_teacher_nll_mean_nats_per_group_frame": float(difference.mean()),
            "parallel_minus_sequential_teacher_nll_max_abs_nats": float(difference.abs().max()),
            "independent_half_estimate_mean_difference_bits_per_executed_frame": float(
                (left - right).mean() / bits_denominator
            ),
            "independent_half_estimate_mean_absolute_difference_bits_per_executed_frame": float(
                (left - right).abs().mean() / bits_denominator
            ),
            "first_half_per_replay_bits_per_executed_frame": (-left / bits_denominator).tolist(),
            "second_half_per_replay_bits_per_executed_frame": (-right / bits_denominator).tolist(),
        },
        "device": {
            "name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "batch_size": batch_size,
        },
        "elapsed_seconds": time.monotonic() - started,
        "limitations": [
            "This is the frozen decoder's executed-tail marginal log score on identity-masked replay contexts.",
            "It is a proper conditional action target, not a validated return-ranking metric.",
            "Replay histories and opponents differ from learner-visited states and CPU evaluation.",
            "The current committed-prefix portable policy has a different likelihood contract.",
            "Low likelihood-weight ESS or particle-count drift makes a finite-particle score unreliable.",
        ],
    }


def particle_schedule(counts: list[int]) -> tuple[int, ...]:
    """Use nested counts and two equal-size independent halves at the maximum."""
    if not counts or any(count < 2 for count in counts):
        raise ValueError("every particle count must be at least two")
    if len(set(counts)) != len(counts):
        raise ValueError("particle counts must be distinct")
    if max(counts) % 2:
        raise ValueError("the maximum particle count must be even for the equal-size half controls")
    return tuple(sorted(counts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--particles", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    counts = particle_schedule(args.particles)
    torch.set_num_threads(4)
    script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    source_probe_sha256 = hashlib.sha256(Path(o50_probe.__file__).read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    for checkpoint in args.checkpoint:
        output = args.output / f"{checkpoint.name.removesuffix('.model.pt')}.json"
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
        result = run_checkpoint(
            args.cohort,
            checkpoint,
            device=torch.device(args.device),
            particle_counts=counts,
            batch_size=args.batch_size,
            script_sha256=script_sha256,
            source_probe_sha256=source_probe_sha256,
        )
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(f"Saved {output}; {result['elapsed_seconds']:.1f} seconds", flush=True)
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
