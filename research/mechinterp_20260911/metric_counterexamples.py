"""Exact small examples of behavior-cloning metric failures."""

import argparse
import itertools
import json
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def correlated_actions(stay_probability: float, horizon: int) -> dict[str, float]:
    """Two equally likely expert plans are all-zero and all-one action strings.

    The model samples its first action uniformly, then repeats the last sampled
    action with the supplied probability. Either constant string succeeds.
    The rollout-prefix score compares an independent expert suffix against the
    model's sampled prefix, as an open-loop exposure diagnostic would.
    """
    if not 0 < stay_probability < 1 or horizon < 2:
        raise ValueError("need 0 < stay_probability < 1 and horizon >= 2")
    teacher = math.log(2) - (horizon - 1) * math.log(stay_probability)
    mixed_prefix = math.log(2) - 0.5 * (horizon - 1) * math.log(stay_probability * (1 - stay_probability))
    return {
        "stay_probability": stay_probability,
        "teacher_nll": teacher,
        "mixed_prefix_nll": mixed_prefix,
        "gap": mixed_prefix - teacher,
        "success_probability": stay_probability ** (horizon - 1),
    }


def enumerate_correlated_actions(stay_probability: float, horizon: int) -> dict[str, float]:
    """Independent finite enumeration checks the closed-form metric calculation."""
    if not 0 < stay_probability < 1 or not 2 <= horizon <= 12:
        raise ValueError("enumeration needs 0 < stay_probability < 1 and 2 <= horizon <= 12")
    success = 0.0
    mixed_prefix = 0.0
    teacher = 0.0
    for sequence in itertools.product((0, 1), repeat=horizon):
        probability = 0.5
        for previous, current in itertools.pairwise(sequence):
            probability *= stay_probability if previous == current else 1 - stay_probability
        if len(set(sequence)) == 1:
            success += probability
            teacher -= 0.5 * math.log(probability)
        for expert_action in (0, 1):
            score = math.log(2)
            for previous in sequence[:-1]:
                target_probability = stay_probability if previous == expert_action else 1 - stay_probability
                score -= math.log(target_probability)
            mixed_prefix += 0.5 * probability * score
    return {
        "teacher_nll": teacher,
        "mixed_prefix_nll": mixed_prefix,
        "success_probability": success,
    }


def unequal_consequences(
    common_correct: float,
    critical_correct: float,
    critical_frequency: float,
    expert_preference: float = 0.99,
) -> dict[str, float]:
    """Both states and both actions have positive demonstration probability.

    The expert chooses the preferred action with the same probability in either
    state. Both actions succeed in the common state; only the preferred action
    succeeds in the critical state. Thus loss also measures harmless style.
    """
    if any(not 0 < value < 1 for value in (common_correct, critical_correct, critical_frequency, expert_preference)):
        raise ValueError("probabilities must be strictly between zero and one")
    common_nll = -expert_preference * math.log(common_correct) - (1 - expert_preference) * math.log1p(-common_correct)
    critical_nll = -expert_preference * math.log(critical_correct) - (1 - expert_preference) * math.log1p(
        -critical_correct
    )
    return {
        "nll": (1 - critical_frequency) * common_nll + critical_frequency * critical_nll,
        "success": 1 - critical_frequency * (1 - critical_correct),
    }


def imperfect_demonstrator(policy_correct: float, expert_correct: float = 0.99) -> dict[str, float]:
    """A one-state task has one successful action and a fallible demonstrator."""
    if any(not 0 < value < 1 for value in (policy_correct, expert_correct)):
        raise ValueError("probabilities must be strictly between zero and one")
    return {
        "nll": -expert_correct * math.log(policy_correct) - (1 - expert_correct) * math.log1p(-policy_correct),
        "success": policy_correct,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    horizon = 6
    probabilities = np.linspace(0.5, 0.999, 400)
    rows = [correlated_actions(float(value), horizon) for value in probabilities]
    evidence = {
        "schema_version": 1,
        "kind": "analytical_counterexample_not_a_HAL_experiment",
        "horizon": horizon,
        "rows": rows,
        "unequal_consequences": {
            "critical_state_probability": 0.05,
            "expert_preferred_action_probability_in_both_states": 0.99,
            "all_state_action_pairs_have_positive_population_probability": True,
            "policy_A": unequal_consequences(0.7, 0.99, 0.05),
            "policy_B": unequal_consequences(0.99, 0.1, 0.05),
        },
        "imperfect_demonstrator": {
            "expert_success_probability": 0.99,
            "all_actions_have_positive_demonstration_probability": True,
            "population_nll_minimizer": imperfect_demonstrator(0.99),
            "better_control_worse_nll": imperfect_demonstrator(0.9999),
        },
    }
    (args.output / "counterexamples.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):
        figure, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, layout="constrained")
        axes[0].plot(probabilities, [row["teacher_nll"] for row in rows], label="True trajectory NLL", color="#2166ac")
        axes[0].plot(
            probabilities,
            [row["mixed_prefix_nll"] for row in rows],
            label="Expert suffix scored after an independent sampled prefix",
            color="#b2182b",
        )
        axes[0].set_ylabel("NLL per six-action plan (nats)")
        axes[0].legend(fontsize=9, loc="upper left")
        axes[0].set_title("A perfect coherent policy can have an infinite rollout-prefix loss")
        axes[1].plot(probabilities, [row["success_probability"] for row in rows], color="#1b7837")
        axes[1].set_ylabel("Task success probability")
        axes[1].set_xlabel("Probability of staying with the previous action")
        axes[1].set_ylim(0, 1.03)
        for axis in axes:
            axis.grid(alpha=0.2)
        figure.savefig(args.output / "mixed_prefix_counterexample.png", dpi=180)
        figure.savefig(args.output / "mixed_prefix_counterexample.pdf")
        plt.close(figure)
    for probability in (0.5, 0.9, 0.99):
        print(correlated_actions(probability, horizon))
    print(evidence["unequal_consequences"])
    print(evidence["imperfect_demonstrator"])


if __name__ == "__main__":
    main()
