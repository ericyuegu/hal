"""Run against the exact R2 proxy with HAL_O65_PROXY_CHECKPOINT set to final.pt."""

import hashlib
import json
import math
import os
from pathlib import Path

import pytest
import torch

from hal.inference.decision_decoder import DecisionDecoder
from hal.inference.kv_cache import BlockRestartCache
from hal.models.action_sequence import ActionSequenceConfig
from hal.models.action_sequence import ActionSequenceTransformer
from hal.models.conditioning import ConditioningRanges
from hal.models.self_play import SelfPlayTransformer
from hal.representation.player_identity import vocabulary_from_checkpoint_buffer
from hal.wire import ACTION_DIM


@pytest.mark.skipif(not torch.cuda.is_available(), reason="proxy parity requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_real_bootstrap30_checkpoint_conditioning_and_cache_parity(dtype: torch.dtype) -> None:
    source = os.environ.get("HAL_O65_PROXY_CHECKPOINT")
    if source is None:
        pytest.skip("set HAL_O65_PROXY_CHECKPOINT to the O63 bootstrap30 final.pt")
    path = Path(source)
    with path.open("rb") as source_file:
        digest = hashlib.file_digest(source_file, "sha256").hexdigest()
    assert digest == "65e7e43e5cbb18a365a005922ec1cfd026ad807c045eae13b331f487b4ec82e6"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    saved = checkpoint["cfg"]
    assert (saved["experiment_id"], saved["checkpoint_format_version"], saved["arm"], checkpoint["step"]) == (
        "063_percentile_conditioning_v1",
        5,
        "bootstrap30",
        16383,
    )
    cfg = ActionSequenceConfig(
        d_model=256,
        n_layers=12,
        n_heads=4,
        temporal_d_model=256,
        temporal_layers=6,
        temporal_heads=4,
        temporal_ff_dim=1024,
        group_head_dim=256,
        value_hidden_dim=128,
    )
    assert saved["architecture"] == {name: getattr(cfg, name) for name in saved["architecture"]}
    state = checkpoint["model"]
    vocabulary = vocabulary_from_checkpoint_buffer(state["player_code_bytes"].numpy())
    bc = ActionSequenceTransformer(cfg, vocabulary).eval().cuda()
    bc.load_state_dict({name: value for name, value in state.items() if not name.startswith("percentile_head.")})
    ranges = ConditioningRanges((math.log(0.01), math.log(100)), (8, 10), (2, 4))
    policy = (
        SelfPlayTransformer(
            cfg, ranges, vocabulary, max_opponent_queue=14, critic_hidden_dim=128, decoder_precision="fp32"
        )
        .eval()
        .cuda()
    )
    policy.load_bc_state_dict(state)
    decoder = DecisionDecoder(policy, ((8, 4), (10, 2)), compiled=os.environ.get("HAL_O65_COMPILE_CUDA") == "1")
    torch.manual_seed(101)
    tokens = torch.randn(2, 140, 256, device="cuda")
    observed = policy.codec.quantize(torch.rand(2, ACTION_DIM, device="cuda"))
    queue = policy.codec.quantize(torch.rand(2, 10, ACTION_DIM, device="cuda"))
    delay = torch.tensor([8, 10], device="cuda")
    replan = torch.tensor([4, 2], device="cuda")
    uniforms = torch.rand(4, 4, 2, device="cuda")
    cache = BlockRestartCache(policy, 2, torch.device("cuda"), dtype=dtype)
    errors: list[float] = []
    conditioner_errors: list[float] = []
    divergences: list[torch.Tensor] = []
    joint_errors: list[float] = []
    with torch.autocast("cuda", dtype=dtype, enabled=dtype == torch.bfloat16):
        padding = torch.zeros(2, device="cuda", dtype=torch.long)
        full = (
            policy.trunk(tokens, padding)
            if dtype == torch.bfloat16
            else policy.trunk.forward_dense(tokens, padding, force_dense=True)
        )
        cache.restart(tokens[:, :128], torch.tensor([128, 128], device="cuda"))
        for at in range(128, 140, 4):
            cached = cache.append(tokens[:, at : at + 4])
        for log_lambda in torch.linspace(*ranges.log_lambda, 9, device="cuda"):
            lambdas = log_lambda.expand(2)
            output = decoder(full, observed, queue, lambdas, delay, replan, uniforms, sampled_offsets=4)
            cached_output = decoder(
                cached[:, -1:],
                observed,
                queue,
                lambdas,
                delay,
                replan,
                uniforms,
                sampled_offsets=4,
                history=cache.memory(),
                sampled_actions=output.actions,
            )
            error = (cached_output.log_probs - output.log_probs).abs().max().item()
            errors.append(error)
            joint_errors.append((cached_output.log_probs - output.log_probs).sum((1, 2)).abs().max().item())
            for cached_probs, probs in zip(cached_output.distributions, output.distributions, strict=True):
                kl = (probs * (probs.clamp_min(1e-30).log() - cached_probs.clamp_min(1e-30).log())).sum(-1)
                divergences.append(kl[output.valid])
            tolerance = 1e-4 if dtype == torch.float32 else 0.1
            torch.testing.assert_close(cached_output.log_probs, output.log_probs, atol=tolerance, rtol=tolerance)
            for row, (d, r) in enumerate(((8, 4), (10, 2))):
                bc.temporal.configure_live_horizons((d + r,))
                draws = torch.full((d + r, 4, 1), 0.5, device="cuda")
                draws[d:] = uniforms[:r, :, row : row + 1]
                # This parity control selects FP32 decoding under either trunk dtype.
                with torch.autocast("cuda", enabled=False):
                    indices, logits = bc.temporal.sample_indices_with_logits(
                        full[row : row + 1],
                        observed[row : row + 1],
                        tuple(range(1, d + r + 1)),
                        torch.tensor([108.0], device="cuda"),
                        torch.tensor([True], device="cuda"),
                        argmax=False,
                        uniforms=draws,
                        forced_prefix=queue[row : row + 1, :d],
                    )
                single = decoder(
                    full[row : row + 1],
                    observed[row : row + 1],
                    queue[row : row + 1],
                    lambdas[row : row + 1],
                    delay[row : row + 1],
                    replan[row : row + 1],
                    uniforms[..., row : row + 1],
                    sampled_offsets=4,
                )
                torch.testing.assert_close(single.actions[0, :r], indices[0, d:])
                for group, value in enumerate(logits):
                    probs = value[0, d:].float().softmax(-1)
                    conditioner_errors.append((probs - single.distributions[group][0, :r]).abs().max().item())
                    torch.testing.assert_close(single.distributions[group][0, :r], probs, atol=1e-5, rtol=1e-4)
        # Compile/run the 2-offset program as a separate variant too.
        only_two = decoder(
            full,
            observed,
            queue,
            torch.zeros(2, device="cuda"),
            torch.full_like(delay, 10),
            torch.full_like(replan, 2),
            uniforms[:2],
            sampled_offsets=2,
        )
        assert only_two.actions.shape == (2, 2, 4)
    kl_values = torch.cat(divergences).float()
    assert kl_values.mean() < 5e-4
    assert kl_values.quantile(0.99) < 5e-3
    assert max(joint_errors) < (1e-4 if dtype == torch.float32 else 0.04)
    print(
        json.dumps(
            {
                "checkpoint_sha256": digest,
                "dtype": str(dtype),
                "mean_cache_kl": kl_values.mean().item(),
                "p99_cache_kl": kl_values.quantile(0.99).item(),
                "max_joint_log_prob_error": max(joint_errors),
                "max_cache_factor_log_prob_error": max(errors),
                "max_conditioner_probability_error": max(conditioner_errors),
            }
        )
    )
