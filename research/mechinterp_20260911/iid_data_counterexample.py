"""An exact finite-sum example where more IID BC data lowers task success."""

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXPERT_CORRECT = 0.9
PSEUDOCOUNT = 0.01
HORIZON = 20
SAMPLE_COUNTS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


@dataclass(frozen=True, slots=True)
class ExpectedMetrics:
    examples: int
    nll_nats_per_action: float
    task_success: float
    action_correct_probability: float
    training_count_probability_mass: float


def fitted_probability(correct_labels: int, examples: int) -> float:
    """Fit one stationary Bernoulli policy with a fixed symmetric pseudocount."""
    if not 1 <= examples <= 512 or not 0 <= correct_labels <= examples:
        raise ValueError("need 1 <= examples <= 512 and 0 <= correct_labels <= examples")
    return (correct_labels + PSEUDOCOUNT) / (examples + 2 * PSEUDOCOUNT)


def population_nll(policy_correct: float) -> float:
    if not 0 < policy_correct < 1:
        raise ValueError("policy_correct must be strictly between zero and one")
    return -EXPERT_CORRECT * math.log(policy_correct) - (1 - EXPERT_CORRECT) * math.log1p(-policy_correct)


def expected_metrics(examples: int) -> ExpectedMetrics:
    """Average over every binomial training count, with no Monte Carlo sampling.

    Training labels are IID expert decisions at the only nonterminal state.
    A rollout needs HORIZON correct decisions; an incorrect action terminates
    it unsuccessfully. The same fitted stochastic policy is used throughout.
    """
    if not 1 <= examples <= 512:
        raise ValueError("the bounded example supports 1 through 512 labels")
    weights = [
        math.comb(examples, count) * EXPERT_CORRECT**count * (1 - EXPERT_CORRECT) ** (examples - count)
        for count in range(examples + 1)
    ]
    probabilities = [fitted_probability(count, examples) for count in range(examples + 1)]
    mass = math.fsum(weights)
    if not math.isclose(mass, 1, rel_tol=0, abs_tol=1e-12):
        raise ArithmeticError(f"binomial probability mass is {mass!r}, not one")
    return ExpectedMetrics(
        examples=examples,
        nll_nats_per_action=math.fsum(
            weight * population_nll(probability) for weight, probability in zip(weights, probabilities, strict=True)
        ),
        task_success=math.fsum(
            weight * probability**HORIZON for weight, probability in zip(weights, probabilities, strict=True)
        ),
        action_correct_probability=math.fsum(
            weight * probability for weight, probability in zip(weights, probabilities, strict=True)
        ),
        training_count_probability_mass=mass,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [expected_metrics(count) for count in SAMPLE_COUNTS]
    evidence = {
        "schema_version": 1,
        "kind": "analytical_counterexample_not_a_HAL_experiment",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "expert_correct": EXPERT_CORRECT,
        "symmetric_pseudocount": PSEUDOCOUNT,
        "required_consecutive_correct_actions": HORIZON,
        "all_actions_have_positive_expert_probability_at_the_only_decision_state": True,
        "training_distribution": "IID Bernoulli expert labels at the nonterminal state",
        "fitted_policy": "q=(correct_labels+alpha)/(examples+2*alpha), stationary and stochastic",
        "expectation_unit": "independently drawn training datasets; rollout success given q is q**H",
        "population_bc_optimum": {
            "nll_nats_per_action": population_nll(EXPERT_CORRECT),
            "task_success": EXPERT_CORRECT**HORIZON,
        },
        "rows": [asdict(row) for row in rows],
    }
    (args.output / "results.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):
        figure, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, layout="constrained")
        axes[0].plot(SAMPLE_COUNTS, [row.nll_nats_per_action for row in rows], "o-", color="#2166ac")
        axes[0].axhline(population_nll(EXPERT_CORRECT), color="0.4", linestyle="--", label="Population BC optimum")
        axes[0].set_ylabel("Expected validation NLL\n(nats / action)")
        axes[0].legend(fontsize=9)
        axes[0].set_title("More IID demonstrations can improve NLL and worsen task success")
        axes[1].plot(SAMPLE_COUNTS, [row.task_success for row in rows], "o-", color="#b2182b")
        axes[1].axhline(EXPERT_CORRECT**HORIZON, color="0.4", linestyle="--", label="Matching the fallible expert")
        axes[1].set_ylabel("Expected task success")
        axes[1].set_xlabel("IID training labels, with the fitting rule held fixed")
        axes[1].set_ylim(0, 1)
        axes[1].set_xscale("log", base=2)
        axes[1].set_xticks(SAMPLE_COUNTS, labels=[str(count) for count in SAMPLE_COUNTS])
        axes[1].legend(fontsize=9)
        for axis in axes:
            axis.grid(alpha=0.2)
        figure.savefig(args.output / "iid_data_counterexample.png", dpi=180)
        figure.savefig(args.output / "iid_data_counterexample.pdf")
        plt.close(figure)
    for row in rows:
        print(asdict(row))


if __name__ == "__main__":
    main()
