"""Exercise diagnostic target construction against a scalar PPO/KL oracle."""

import dataclasses
import math

import pytest
import torch
from test_self_play_model import model
from test_window_replay_actor import targets_for
from test_window_replay_model import inputs
from torch import Tensor

from hal.models.replay_actor import actor_loss
from hal.models.replay_actor import validate_actor_targets
from hal.models.window_replay import ReplayOutput
from hal.models.window_replay import ReplaySlots
from hal.models.window_replay import replay_windows
from scripts.o65_actor_trace import recorded_actor_targets
from scripts.o65_replay_probe import CapturedCohort
from scripts.o65_replay_probe import capture_reference_log_probs


@pytest.mark.parametrize("diagnostic", ["probe", "trace"])
@pytest.mark.parametrize("reference_shift", [0.0, 0.7])
def test_diagnostic_targets_match_scalar_objective_and_gradient(diagnostic: str, reference_shift: float) -> None:
    policy = model()
    hidden, observed, slots = inputs(policy)
    base = replay_windows(policy, hidden, observed, slots, grad=False)
    logits = torch.tensor([0.2, -0.4, 0.1], requires_grad=True)
    head_logs = logits.log_softmax(-1)
    output = dataclasses.replace(
        base,
        log_probs_WKF=head_logs[0].expand_as(base.log_probs_WKF),
        log_distributions=(head_logs.expand(2, 3, 4, 3),) * 4,
    )
    reference_logs = (logits.detach() + torch.tensor([-reference_shift, 0.0, reference_shift])).log_softmax(-1)
    reference = dataclasses.replace(output, log_distributions=(reference_logs.expand(2, 3, 4, 3),) * 4)
    recorded = targets_for(output)

    def forward(_slots: ReplaySlots) -> ReplayOutput:
        return reference

    if diagnostic == "probe":
        targets = dataclasses.replace(
            recorded, reference_log_distributions=capture_reference_log_probs(forward, slots)
        )
    else:
        cohort: CapturedCohort = {
            "inputs": (),
            "context": {},
            "observed": observed,
            "behavior": recorded.behavior_log_probs_WKF,
            "advantages": recorded.advantages_WK,
            "selected": recorded.selected_WK,
            "coefficients": recorded.kl_coefficient_WK,
            "advantage_mean": recorded.advantage_mean,
            "advantage_sigma": recorded.advantage_sigma,
            "denominator": recorded.denominator,
        }
        targets = recorded_actor_targets(cohort, reference, torch.arange(2), recorded.selected_WK)
    validate_actor_targets(output, targets)
    result = actor_loss(output, targets)

    # Scalar double arithmetic is independent of the tensor loss and its reductions.
    def expected(values: list[float]) -> tuple[float, float]:
        log_normalizer = math.log(sum(math.exp(value) for value in values))
        logs = [value - log_normalizer for value in values]
        kl_per_factor = sum(
            math.exp(log_prob) * (log_prob - float(reference_log_prob))
            for log_prob, reference_log_prob in zip(logs, reference_logs, strict=True)
        )
        policy_loss, kl_loss = 0.0, 0.0
        for window in range(2):
            for slot in range(3):
                if not bool(targets.selected_WK[window, slot]):
                    continue
                factors = int(output.factor_mask_WKF[window, slot].sum())
                old_log_prob = float(targets.behavior_log_probs_WKF[window, slot, 0])
                ratio = math.exp(factors * (logs[0] - old_log_prob))
                advantage = (float(targets.advantages_WK[window, slot]) - float(targets.advantage_mean)) / float(
                    targets.advantage_sigma
                )
                policy_loss -= min(ratio * advantage, min(max(ratio, 0.8), 1.2) * advantage)
                kl_loss += (
                    float(targets.kl_coefficient_WK[window, slot])
                    / float(targets.advantage_sigma)
                    * factors
                    * kl_per_factor
                )
        return policy_loss / float(targets.denominator), kl_loss / float(targets.denominator)

    values = logits.detach().tolist()
    expected_policy, expected_kl = expected(values)
    assert float(result.policy_loss.detach()) == pytest.approx(expected_policy, abs=2e-6)
    assert float(result.kl_loss.detach()) == pytest.approx(expected_kl, abs=2e-6)
    if reference_shift == 0:
        assert float(result.kl_loss.detach()) == 0
    else:
        assert float(result.kl_loss.detach()) > 0
    gradient = torch.autograd.grad(result.loss, logits)[0]
    delta = 1e-5
    for category in range(3):
        high, low = values.copy(), values.copy()
        high[category] += delta
        low[category] -= delta
        finite_difference = (sum(expected(high)) - sum(expected(low))) / (2 * delta)
        assert float(gradient[category]) == pytest.approx(finite_difference, abs=5e-6)

    wrong_units = dataclasses.replace(
        targets, reference_log_distributions=tuple(logs.exp() for logs in targets.reference_log_distributions)
    )
    with pytest.raises(ValueError, match="normalized"):
        validate_actor_targets(output, wrong_units)
    assert float(actor_loss(output, wrong_units).kl_loss.detach()) < -1
    owned_reference = tuple(logs.clone() for logs in targets.reference_log_distributions)
    reference_logs.fill_(123)
    for actual, expected_logs in zip(targets.reference_log_distributions, owned_reference, strict=True):
        torch.testing.assert_close(actual, expected_logs, rtol=0, atol=0)
        assert not actual.requires_grad


def test_probe_copies_each_reference_chunk_before_forward_reuses_storage() -> None:
    policy = model()
    hidden, observed, slots = inputs(policy)
    base = replay_windows(policy, hidden, observed, slots, grad=False)
    slots = dataclasses.replace(
        slots,
        **{
            field.name: getattr(slots, field.name)[:, :1].expand(2, 33, *getattr(slots, field.name).shape[2:])
            for field in dataclasses.fields(slots)
            if field.name != "history_starts_WL"
        },
    )
    buffer = torch.empty(2, 16, 4, 3)
    sizes: list[int] = []
    expected: list[Tensor] = []

    def forward(chunk: ReplaySlots) -> ReplayOutput:
        count = chunk.valid_WK.shape[1]
        sizes.append(count)
        logs = torch.tensor([float(len(sizes)), 0.0, -1.0]).log_softmax(-1)
        buffer.copy_(logs)
        expected.append(buffer[:, :count].clone())
        return dataclasses.replace(base, log_distributions=(buffer[:, :count],) * 4)

    actual = capture_reference_log_probs(forward, slots)
    assert sizes == [16, 16, 1]
    buffer.fill_(torch.nan)
    for head in actual:
        torch.testing.assert_close(head, torch.cat(expected, 1), rtol=0, atol=0)
