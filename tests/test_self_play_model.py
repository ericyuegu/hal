"""Check conditioning parity, detached critic gradients, and mixed-delay actions."""

import math
from dataclasses import replace
from functools import partial
from typing import cast

import pytest
import torch
from torch import nn

from hal.inference.decision_decoder import DecisionDecoder
from hal.models.action_sequence import ActionSequenceConfig
from hal.models.action_sequence import ActionSequenceTransformer
from hal.models.action_sequence import NonlinearActionHead
from hal.models.action_sequence import ReturnConditioner
from hal.models.action_sequence import SwiGLU
from hal.models.action_sequence import TemporalBlock
from hal.models.conditioning import ConditioningRanges
from hal.models.self_play import DecoderPrecision
from hal.models.self_play import SelfPlayTransformer
from hal.models.self_play import validate_executed_offsets
from hal.wire import ACTION_DIM

RANGES = ConditioningRanges((math.log(0.01), math.log(100)), (8, 10), (2, 4))


def config(*, proxy: bool = False) -> ActionSequenceConfig:
    if proxy:
        return ActionSequenceConfig(
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
    return ActionSequenceConfig(
        d_model=32,
        n_layers=2,
        n_heads=4,
        temporal_d_model=32,
        temporal_layers=2,
        temporal_heads=4,
        temporal_ff_dim=64,
        group_head_dim=16,
        value_hidden_dim=16,
        return_embed_dim=8,
    )


def model(precision: DecoderPrecision = "fp32") -> SelfPlayTransformer:
    return SelfPlayTransformer(
        config(), RANGES, max_opponent_queue=14, critic_hidden_dim=16, decoder_precision=precision
    ).eval()


def test_widened_conditioner_preserves_learned_bc_modulation() -> None:
    torch.manual_seed(5)
    cfg = config()
    bc = ReturnConditioner(cfg)
    for projection in bc.projections:
        assert isinstance(projection, nn.Linear)
        nn.init.normal_(projection.weight, std=0.2)
        nn.init.normal_(projection.bias, std=0.2)
    widened = ReturnConditioner(cfg, RANGES)
    state = bc.state_dict()
    state["embedding.weight"] = torch.cat((state["embedding.weight"], torch.zeros(cfg.return_embed_dim, 3)), dim=1)
    widened.load_state_dict(state)
    log_lambda = torch.linspace(*RANGES.log_lambda, 101)
    for delay, replan in ((8, 4), (10, 2)):
        condition = RANGES.normalize(
            log_lambda, torch.full_like(log_lambda, delay), torch.full_like(log_lambda, replan)
        )
        returns = torch.linspace(-120, 120, 101)
        present = torch.arange(101) % 3 != 0
        expected, actual = bc(returns, present), widened(returns, present, condition)
        assert expected is not None and actual is not None
        for old, new in zip(expected, actual, strict=True):
            torch.testing.assert_close(new, old, rtol=0, atol=0)
    with pytest.raises(ValueError, match="requires normalized"):
        widened(returns, present)
    with pytest.raises(ValueError, match="outside"):
        RANGES.normalize(torch.tensor([float("nan")]), torch.tensor([8]), torch.tensor([4]))
    fixed = ConditioningRanges((0, 0), (8, 8), (4, 4))
    assert torch.equal(
        fixed.normalize(torch.tensor([0.0]), torch.tensor([8]), torch.tensor([4])),
        torch.tensor([[0.0, 0.0, 0.0, 0.9]]),
    )


@pytest.mark.parametrize("dense_offsets", [False, True])
def test_bc_load_is_strict_and_critic_starts_at_value_head(dense_offsets: bool) -> None:
    cfg = (
        replace(config(), sample_chunk_length=30, head_offsets=(*range(1, 25), 27, 30)) if dense_offsets else config()
    )
    bc = ActionSequenceTransformer(cfg)
    bc.percentile_head = SwiGLU(32, 16, 65, output_bias=True)
    policy = SelfPlayTransformer(cfg, RANGES, max_opponent_queue=14, critic_hidden_dim=16).eval()
    policy.load_bc_state_dict(bc.state_dict())
    trunk = torch.randn(3, 32, requires_grad=True)
    decoder = torch.randn(3, 32, requires_grad=True)
    queue = policy.codec.quantize(torch.rand(3, 14, ACTION_DIM))
    condition = torch.randn(3, 4)
    value = policy.self_play_value(trunk, decoder, condition, condition, queue, torch.tensor([0, 10, 14]))
    torch.testing.assert_close(value, policy.estimate_value(trunk[:, None]), rtol=0, atol=0)
    # A trained residual must still stop both paths into the actor.
    nn.init.normal_(policy.critic_residual.residual.down.weight)
    policy.self_play_value(trunk, decoder, condition, condition, queue, torch.tensor([0, 10, 14])).sum().backward()
    assert trunk.grad is decoder.grad is None
    assert all(p.grad is None for p in policy.actor_parameters())
    assert all(not p.requires_grad for p in policy.percentile_head.parameters())
    assert set(map(id, policy.actor_parameters())).isdisjoint(map(id, policy.critic_parameters()))
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.critic_residual.parameters())
    actor_before = tuple(p.detach().clone() for p in policy.actor_parameters())
    torch.optim.SGD(policy.critic_parameters(), lr=0.01).step()
    for before, after in zip(actor_before, policy.actor_parameters(), strict=True):
        torch.testing.assert_close(after, before, rtol=0, atol=0)
    bad = dict(bc.state_dict())
    del bad["percentile_head.down.bias"]
    with pytest.raises(ValueError, match="keys mismatch"):
        policy.load_bc_state_dict(bad)
    bad = dict(bc.state_dict())
    bad["temporal.return_conditioner.embedding.weight"] = torch.zeros(8, 4)
    with pytest.raises(ValueError, match="one input"):
        policy.load_bc_state_dict(bad)


def test_queue_padding_is_invisible_even_for_nan_bytes() -> None:
    policy = model()
    nn.init.normal_(policy.critic_residual.residual.down.weight)
    inputs = (torch.randn(2, 32), torch.randn(2, 4), torch.randn(2, 4))
    queue = torch.randn(2, 14, 128)
    lengths = torch.tensor([0, 7])
    expected = policy.critic_residual(*inputs, queue, lengths)
    queue[0] = float("nan")
    queue[1, 7:] = float("nan")
    torch.testing.assert_close(policy.critic_residual(*inputs, queue, lengths), expected, rtol=0, atol=0)


def test_executed_offsets_validate_proxy_and_production() -> None:
    validate_executed_offsets(config().head_offsets, ((8, 4), (10, 2)))
    production = tuple((d, r) for d in range(16, 22) for r in (2, 4) if d + r <= 24)
    assert len(production) == 11
    validate_executed_offsets((*range(1, 25), 27, 30), production)
    with pytest.raises(ValueError, match="unsupervised"):
        validate_executed_offsets(config().head_offsets, ((10, 4),))


def decision_inputs(policy: SelfPlayTransformer) -> tuple[torch.Tensor, ...]:
    return (
        torch.randn(2, 9, 32),
        policy.codec.quantize(torch.rand(2, ACTION_DIM)),
        policy.codec.quantize(torch.rand(2, 10, ACTION_DIM)),
        torch.tensor([-2.0, 1.0]),
        torch.tensor([8, 10]),
        torch.tensor([4, 2]),
        torch.rand(4, 4, 2),
    )


@torch.no_grad()
def test_mixed_delay_sampling_matches_individual_bc_and_replay() -> None:
    torch.manual_seed(7)
    policy = model()
    decoder = DecisionDecoder(policy, ((8, 4), (10, 2)), compiled=False)
    args = decision_inputs(policy)
    output = decoder(*args, sampled_offsets=4)
    assert torch.equal(output.valid, torch.tensor([[True] * 4, [True, True, False, False]]))
    assert torch.equal(output.log_probs[1, 2:], torch.zeros(2, 4))
    # The original one-column decoder is an independent reference at init.
    bc = ActionSequenceTransformer(config()).eval()
    state = {
        name: value
        for name, value in policy.state_dict().items()
        if not name.startswith(("critic_residual.", "percentile_head."))
    }
    state["temporal.return_conditioner.embedding.weight"] = state["temporal.return_conditioner.embedding.weight"][
        :, :1
    ]
    bc.load_state_dict(state)
    for row, (delay, replan) in enumerate(((8, 4), (10, 2))):
        hidden, observed, queue, _, _, _, draws = args
        uniforms = torch.full((delay + replan, 4, 1), 0.5)
        uniforms[delay:] = draws[:replan, :, row : row + 1]
        bc.temporal.configure_live_horizons((delay + replan,))
        indices, logits = bc.temporal.sample_indices_with_logits(
            hidden[row : row + 1],
            observed[row : row + 1],
            tuple(range(1, delay + replan + 1)),
            torch.tensor([108.0]),
            torch.tensor([True]),
            argmax=False,
            uniforms=uniforms,
            forced_prefix=queue[row : row + 1, :delay],
        )
        torch.testing.assert_close(output.actions[row, :replan], indices[0, delay:])
        for group, reference in enumerate(logits):
            expected = reference[:, delay:].float().softmax(-1)[0]
            torch.testing.assert_close(output.distributions[group][row, :replan], expected, atol=2e-6, rtol=2e-5)
            logs = expected.gather(-1, output.actions[row, :replan, group, None]).squeeze(-1).log()
            torch.testing.assert_close(output.log_probs[row, :replan, group], logs, atol=2e-6, rtol=2e-5)
    replay = decoder(*args, sampled_offsets=4, sampled_actions=output.actions)
    torch.testing.assert_close(output.log_probs, replay.log_probs, rtol=2e-5, atol=2e-6)
    changed = output.actions.clone()
    changed[..., 2] = (changed[..., 2] + 1) % 9
    other = decoder(*args, sampled_offsets=4, sampled_actions=changed)
    torch.testing.assert_close(replay.decoder_state, other.decoder_state, rtol=0, atol=0)


@pytest.mark.parametrize("sampled_offsets", [2, 4])
@pytest.mark.parametrize("with_history", [False, True])
def test_parallel_decision_replay_matches_sequential_outputs_and_gradients(
    sampled_offsets: int, with_history: bool
) -> None:
    from hal.models.attention import KVMemory

    torch.manual_seed(67)
    policy = model()
    for projection in policy.temporal.return_conditioner.projections:
        assert isinstance(projection, nn.Linear)
        nn.init.normal_(projection.weight, std=0.03)
    hidden, observed, queue, log_lambda, delay, replan, draws = decision_inputs(policy)
    if sampled_offsets == 2:
        delay = torch.full_like(delay, 10)
        replan = torch.full_like(replan, 2)
    condition = RANGES.normalize(log_lambda, delay, replan)
    with torch.no_grad():
        actions = policy.codec.quantize(torch.rand(2, sampled_offsets, ACTION_DIM))

    def evaluate(parallel: bool, prefix: int) -> tuple[torch.Tensor, ...]:
        policy.zero_grad(set_to_none=True)
        history = None
        if with_history:
            key, value = policy.temporal.history_attention.project_memory(hidden)
            history = KVMemory(
                torch.stack((key, value)), torch.arange(9)[None].expand(2, -1), torch.tensor([8, 5]), 256
            )
        decode = (
            policy.temporal.replay_decisions
            if parallel
            else partial(policy.temporal.decode_decisions, forced_prefix_length=prefix)
        )
        output = decode(
            hidden,
            observed,
            queue,
            delay,
            replan,
            condition,
            draws[:sampled_offsets],
            max_delay=10,
            sampled_offsets=sampled_offsets,
            sampled_actions=actions,
            history=history,
        )
        (-output.log_probs.sum()).backward()
        gradients = torch.cat([p.grad.flatten() for p in policy.parameters() if p.grad is not None])
        return output.log_probs.detach(), output.decoder_state.detach(), gradients

    expected = evaluate(False, 0)
    for parallel, prefix in ((False, 8), (True, 0)):
        actual = evaluate(parallel, prefix)
        for new, old in zip(actual, expected, strict=True):
            torch.testing.assert_close(new, old, rtol=2e-4, atol=3e-6)


def test_replay_has_policy_gradient_and_decoder_state_is_before_draw() -> None:
    policy = model()
    decoder = DecisionDecoder(policy, ((8, 4), (10, 2)), compiled=False)
    args = decision_inputs(policy)
    with torch.no_grad():
        output = decoder(*args, sampled_offsets=4)
    replay = decoder(*args, sampled_offsets=4, sampled_actions=output.actions)
    (-replay.log_probs.sum()).backward()
    assert policy.temporal.return_conditioner.embedding.weight.grad is not None
    assert policy.temporal.token_projection.weight.grad is not None


@pytest.mark.parametrize("sampled_offsets", [2, 4])
def test_compiled_decision_variant_matches_eager(sampled_offsets: int) -> None:
    policy = model()
    decoder = DecisionDecoder(policy, ((8, 4), (10, 2)), compiled=False)
    args = list(decision_inputs(policy))
    args[5] = torch.full((2,), 2) if sampled_offsets == 2 else args[5]
    if sampled_offsets == 2:
        args[4] = torch.full((2,), 10)
    args[6] = args[6][:sampled_offsets]
    expected = decoder(*args, sampled_offsets=sampled_offsets)
    # Dynamo's eager backend checks full-graph capture without a costly compiler build.
    decoder.variants[sampled_offsets] = torch.compile(
        decoder.variants[sampled_offsets], backend="eager", fullgraph=True
    )
    actual = decoder(*args, sampled_offsets=sampled_offsets)
    torch.testing.assert_close(actual.log_probs, expected.log_probs)
    torch.testing.assert_close(actual.decoder_state, expected.decoder_state)


@pytest.mark.parametrize(
    "case",
    [
        "fractional_replan",
        "boolean_offset",
        "nan_draw",
        "out_of_range_factor",
        "illegal_buttons",
        "configuration",
        "shape",
    ],
)
def test_invalid_decision_data_is_rejected(case: str) -> None:
    policy = model()
    if case == "fractional_replan":
        with pytest.raises(ValueError, match="replan"):
            validate_executed_offsets(config().head_offsets, cast(tuple[tuple[int, int], ...], ((8, 2.0),)))
        return
    if case == "boolean_offset":
        with pytest.raises(ValueError, match="offsets"):
            validate_executed_offsets((True, 2), ((0, 2),))
        return
    decoder = DecisionDecoder(policy, ((8, 4), (10, 2)), compiled=False)
    args = list(decision_inputs(policy))
    expected = ""
    if case == "nan_draw":
        args[-1][0, 0, 0] = torch.nan
        expected = "uniforms"
    elif case == "out_of_range_factor":
        args[1][0, 0] = 256
        expected = "invalid controller factor"
    elif case == "illegal_buttons":
        invalid = (~policy.codec.button_valid_for_trigger).nonzero()[0]
        args[1][0, 3] = invalid[0]
        args[1][0, 0] = invalid[1]
        expected = "incompatible"
    elif case == "configuration":
        args[5][0] = 2
        expected = "unsupported configuration"
    else:
        args[1] = args[1][:1]
        expected = "shape"
    with pytest.raises(ValueError, match=expected):
        decoder(*args, sampled_offsets=4)


def test_decision_decoder_rejects_reduced_precision_parameters() -> None:
    with pytest.raises(ValueError, match="FP32 parameters"):
        DecisionDecoder(model().bfloat16(), ((8, 4), (10, 2)), compiled=False)


def test_decoder_precision_defaults_to_bf16_and_rejects_other_values() -> None:
    assert (
        SelfPlayTransformer(config(), RANGES, max_opponent_queue=14, critic_hidden_dim=16).decoder_precision == "bf16"
    )
    for invalid in ("fp16", "BF16", "", None):
        with pytest.raises(ValueError, match="decoder precision"):
            model(cast(DecoderPrecision, invalid))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@torch.no_grad()
def test_bf16_readouts_preserve_small_bias_changes(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    policy = model("bf16").to(device)
    head = cast(NonlinearActionHead, policy.temporal.outputs["c_stick"])
    assert head.down.bias is not None
    head.down.bias.fill_(1)
    features = torch.randn(2, 32, device=device)
    delta = torch.linspace(-0.001, 0.001, head.down.out_features, device=device)
    with torch.autocast(device, dtype=torch.bfloat16):
        before = policy.temporal._project_action_head(head, features)
        head.down.bias.add_(delta)
        after = policy.temporal._project_action_head(head, features)
    assert before.dtype == after.dtype == torch.float32
    torch.testing.assert_close(after - before, delta.expand_as(after), rtol=0, atol=2e-7)


@pytest.mark.parametrize("precision", ["bf16", "fp32"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_decoder_owns_compute_precision_and_keeps_probabilities_and_critic_in_fp32(
    precision: DecoderPrecision, device: str
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(79)
    policy = model(precision).to(device)
    decoder = DecisionDecoder(policy, ((8, 4), (10, 2)), compiled=False)
    args = tuple(value.to(device) for value in decision_inputs(policy))
    dtypes: list[torch.dtype] = []

    def record_dtype(_module: nn.Module, _args: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        dtypes.append(output.dtype)

    handle = cast(TemporalBlock, policy.temporal.blocks[0]).qkv.register_forward_hook(record_dtype)
    try:
        with torch.no_grad():
            sampled = decoder(*args, sampled_offsets=4)
        expected = decoder(*args, sampled_offsets=4, sampled_actions=sampled.actions)
        with torch.autocast(device, dtype=torch.bfloat16):
            actual = decoder(*args, sampled_offsets=4, sampled_actions=sampled.actions)
        assert dtypes and set(dtypes) == {torch.bfloat16 if precision == "bf16" else torch.float32}
    finally:
        handle.remove()
    assert actual.log_probs.dtype == torch.float32
    assert actual.decoder_state.dtype == torch.float32
    assert all(probability.dtype == torch.float32 for probability in actual.distributions)
    torch.testing.assert_close(actual.log_probs, expected.log_probs, rtol=0, atol=0)
    (-actual.log_probs.sum()).backward()
    grad = policy.temporal.token_projection.weight.grad
    assert grad is not None and grad.dtype == torch.float32 and bool(grad.any())
    hidden, _, _, log_lambda, delay, replan, _ = args
    condition = RANGES.normalize(log_lambda, delay, replan)
    queue = policy.codec.quantize(torch.zeros(2, 14, ACTION_DIM, device=device))
    with torch.autocast(device, dtype=torch.bfloat16):
        values = policy.self_play_value(hidden[:, -1], actual.decoder_state, condition, condition, queue, delay)
    assert values.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_cuda_compiled_two_and_four_offset_variants_match_eager() -> None:
    torch.manual_seed(23)
    cfg = replace(config(), temporal_heads=2)
    policy = (
        SelfPlayTransformer(cfg, RANGES, max_opponent_queue=14, critic_hidden_dim=16, decoder_precision="fp32")
        .cuda()
        .eval()
    )
    args = tuple(value.cuda() for value in decision_inputs(policy))
    eager = DecisionDecoder(policy, ((8, 4), (10, 2)), compiled=False)
    compiled = DecisionDecoder(policy, ((8, 4), (10, 2)), compiled=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for count in (2, 4):
            hidden, observed, forced, log_lambda, delay, replan, draws = args
            if count == 2:
                delay = torch.full_like(delay, 10)
                replan = torch.full_like(replan, 2)
            inputs = (hidden, observed, forced, log_lambda, delay, replan, draws[:count])
            actual = compiled(*inputs, sampled_offsets=count)
            expected = eager(*inputs, sampled_offsets=count, sampled_actions=actual.actions)
            repeated = compiled(*inputs, sampled_offsets=count)
            torch.testing.assert_close(repeated.actions, actual.actions, rtol=0, atol=0)
            torch.testing.assert_close(actual.log_probs, expected.log_probs, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(actual.decoder_state, expected.decoder_state, atol=2e-6, rtol=2e-5)
            assert torch.equal(actual.valid, expected.valid)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("precision", ["bf16", "fp32"])
@torch.no_grad()
def test_cuda_graph_buckets_keep_outputs_and_read_updated_weights(precision: DecoderPrecision) -> None:
    from hal.models.attention import KVMemory

    torch.manual_seed(71)
    policy = model(precision).cuda()
    compiled = DecisionDecoder(policy, ((8, 4), (10, 2)), compiled=True)
    eager = DecisionDecoder(policy, ((8, 4), (10, 2)), compiled=False)
    for count in (2, 4):
        for batch in (3, 4, 5, 3):
            hidden = torch.randn(batch, 13, 32, device="cuda")
            observed = policy.codec.quantize(torch.rand(batch, ACTION_DIM, device="cuda"))
            queue = policy.codec.quantize(torch.rand(batch, 10, ACTION_DIM, device="cuda"))
            delay = torch.full((batch,), 10, device="cuda")
            replan = torch.full_like(delay, 2)
            if count == 4:
                delay[::2], replan[::2] = 8, 4
            args = (
                observed,
                queue,
                torch.zeros(batch, device="cuda"),
                delay,
                replan,
                torch.rand(count, 4, batch, device="cuda"),
            )
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                key, value = policy.temporal.history_attention.project_memory(hidden)
                history = KVMemory(
                    torch.stack((key, value)),
                    torch.arange(13, device="cuda")[None].expand(batch, -1),
                    torch.full_like(delay, 12),
                    256,
                )
                actual = compiled(hidden[:, -1:], *args, sampled_offsets=count, history=history)
                expected = eager(
                    hidden[:, -1:], *args, sampled_offsets=count, history=history, sampled_actions=actual.actions
                )
            if precision == "fp32":
                torch.testing.assert_close(actual.log_probs, expected.log_probs, atol=2e-6, rtol=2e-5)
            else:
                difference = (actual.log_probs - expected.log_probs).abs().flatten()
                assert difference.max() < 0.2 and difference.quantile(0.99) < 0.06
            saved = actual.log_probs.clone()
            head = policy.temporal.outputs["c_stick"]
            assert isinstance(head, torch.nn.Module)
            assert isinstance(head.down, torch.nn.Linear)
            head.down.bias.add_(torch.linspace(-0.5, 0.5, head.down.out_features, device="cuda"))
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                updated = compiled(hidden[:, -1:], *args, sampled_offsets=count, history=history)
                updated_expected = eager(
                    hidden[:, -1:], *args, sampled_offsets=count, history=history, sampled_actions=updated.actions
                )
            torch.testing.assert_close(actual.log_probs, saved, rtol=0, atol=0)
            if precision == "fp32":
                torch.testing.assert_close(updated.log_probs, updated_expected.log_probs, atol=2e-6, rtol=2e-5)
            else:
                difference = (updated.log_probs - updated_expected.log_probs).abs().flatten()
                assert difference.max() < 0.2 and difference.quantile(0.99) < 0.06
            assert not torch.equal(updated.log_probs, saved)
    assert len(compiled.graphs) == 4
    with torch.inference_mode():
        compiled(hidden[:, -1:], *args, sampled_offsets=4, history=history)
    compiled(hidden[:, -1:], *args, sampled_offsets=4, history=history)
    assert len(compiled.graphs) == 5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bf16_autocast_keeps_decision_compute_and_gradients_in_fp32() -> None:
    from hal.models.attention import KVMemory

    torch.manual_seed(73)
    policy = model().cuda()
    decoder = DecisionDecoder(policy, ((8, 4), (10, 2)), compiled=False)
    hidden, observed, queue, log_lambda, delay, replan, draws = (value.cuda() for value in decision_inputs(policy))
    key, value = policy.temporal.history_attention.project_memory(hidden)
    history = KVMemory(
        torch.stack((key, value)), torch.arange(9, device="cuda")[None].expand(2, -1), torch.full_like(delay, 8), 256
    )
    expected = decoder(
        hidden[:, -1:], observed, queue, log_lambda, delay, replan, draws, sampled_offsets=4, history=history
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = decoder(
            hidden[:, -1:],
            observed,
            queue,
            log_lambda,
            delay,
            replan,
            draws,
            sampled_offsets=4,
            history=history,
            sampled_actions=expected.actions,
        )
    assert actual.decoder_state.dtype == torch.float32
    torch.testing.assert_close(actual.log_probs, expected.log_probs, rtol=2e-5, atol=2e-6)
    (-actual.log_probs.sum()).backward()
    grad = policy.temporal.token_projection.weight.grad
    assert grad is not None and grad.dtype == torch.float32 and bool(grad.any())


@pytest.mark.parametrize("field", ["trunk", "decoder"])
def test_critic_rejects_a_time_dimension_that_would_broadcast_values(field: str) -> None:
    policy = model()
    with pytest.raises(ValueError, match=f"{field} state"):
        policy.self_play_value(
            torch.zeros(2, 1, 32) if field == "trunk" else torch.zeros(2, 32),
            torch.zeros(2, 1, 32) if field == "decoder" else torch.zeros(2, 32),
            torch.zeros(2, 4),
            torch.zeros(2, 4),
            torch.zeros(2, 14, 4, dtype=torch.long),
            torch.tensor([10, 10]),
        )


@pytest.mark.parametrize("factors,dtype", [(5, torch.long), (4, torch.float32)])
def test_critic_rejects_changed_controller_schema(factors: int, dtype: torch.dtype) -> None:
    policy = model()
    with pytest.raises(ValueError, match="opponent queue"):
        policy.self_play_value(
            torch.zeros(2, 32),
            torch.zeros(2, 32),
            torch.zeros(2, 4),
            torch.zeros(2, 4),
            torch.zeros(2, 14, factors, dtype=dtype),
            torch.tensor([10, 10]),
        )
