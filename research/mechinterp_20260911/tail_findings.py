"""Summarize the frozen v1 executed-tail likelihood experiment."""

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

SEED = 20_260_914
EXPECTED_COHORT = "ec61e97e332672a4e4e2fdc438e208a281a4609e10f7eded72368eccac323437"
EXPECTED_PROBE = "bae7012998b0ac1ddbe701690b0ec4b0fd9138dd15648b6252d0a8a24a85c977"
EXPECTED_SCORER = "f46195c36a097cfbe97a3997ff2c12f7b4488e8f9eb1b054379a439cb5b63bd2"


def mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError("expected a mapping with string keys")
    return cast(Mapping[str, object], value)


def scalar(value: object) -> float:
    if not isinstance(value, (float, int)) or isinstance(value, bool) or not np.isfinite(value):
        raise ValueError("expected a finite number")
    return float(value)


def replay_vector(value: object) -> NDArray[np.float64]:
    if not isinstance(value, list) or len(value) != 256:
        raise ValueError("expected one likelihood value for each of 256 replays")
    return np.asarray([scalar(item) for item in value], dtype=np.float64)


def paired_interval(delta: NDArray[np.float64]) -> dict[str, object]:
    """Replay bootstrap conditions on the already sampled Monte Carlo particles."""
    if delta.shape != (256,) or not np.isfinite(delta).all():
        raise ValueError("paired differences must contain 256 finite values")
    generator = np.random.default_rng(SEED)
    samples = generator.integers(0, len(delta), size=(20_000, len(delta)))
    means = delta[samples].mean(axis=1)
    return {
        "mean_bits_per_executed_frame": float(delta.mean()),
        "replay_bootstrap_95": np.quantile(means, (0.025, 0.975)).tolist(),
        "includes_monte_carlo_error": False,
    }


def load_result(path: Path) -> Mapping[str, object]:
    result = mapping(json.loads(path.read_bytes()))
    if result["schema_version"] != 1:
        raise ValueError("this analysis is pinned to the completed v1 tail experiment")
    for name, expected in (
        ("cohort_sha256", EXPECTED_COHORT),
        ("source_probe_sha256", EXPECTED_PROBE),
        ("script_sha256", EXPECTED_SCORER),
    ):
        if result[name] != expected:
            raise ValueError(f"{path}: mismatched {name}")
    protocol = mapping(result["protocol"])
    if protocol["particle_counts"] != [64, 256, 1024] or protocol["random_seed"] != 20_260_913:
        raise ValueError("particle design differs from the common-random-number comparison")
    if protocol["player_identity"] != "ego_player_id forced to zero, as in frozen gameplay":
        raise ValueError("the likelihood context must mask player identity")
    if mapping(result["controls"])["public_sampler_latent_action_parity"] is not True:
        raise ValueError("the cached public sampler parity gate did not pass")
    return result


def summarize(directory: Path) -> dict[str, object]:
    files = sorted(directory.glob("*.json"))
    records: dict[tuple[str, int], Mapping[str, object]] = {}
    rows: list[dict[str, object]] = []
    for path in files:
        result = load_result(path)
        checkpoint = mapping(result["checkpoint"])
        run = checkpoint["wandb_id"]
        step = checkpoint["checkpoint_step"]
        if not isinstance(run, str) or not isinstance(step, int) or isinstance(step, bool):
            raise ValueError("checkpoint identity is invalid")
        identity = (run, step + 1)
        if identity in records:
            raise ValueError("duplicate checkpoint identity")
        records[identity] = result
        estimates = mapping(result["estimates"])
        final = mapping(estimates["1024"])
        teacher = mapping(result["teacher_forced"])
        controls = mapping(result["controls"])
        scores = replay_vector(final["per_replay_joint_bits_per_executed_frame"])
        if not np.isclose(scores.mean(), scalar(final["mean_joint_bits_per_executed_frame"]), atol=1e-12, rtol=0):
            raise ValueError("reported score differs from the replay mean")
        rows.append(
            {
                "run": run,
                "update_boundary": step + 1,
                "artifact": str(path),
                "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "teacher_tail_bits_per_frame": teacher["tail_given_expert_prefix_bits_per_executed_frame"],
                "marginal_tail_bits_per_frame": float(scores.mean()),
                "improper_mixed_prefix_bits_per_frame": mapping(result["mixed_prefix_score"])[
                    "mean_conditional_nll_bits_per_executed_frame"
                ],
                "marginal_bits_by_particle_count": {
                    key: mapping(value)["mean_joint_bits_per_executed_frame"] for key, value in estimates.items()
                },
                "ess_quantiles_at_1024": final["ess_quantiles"],
                "fraction_ess_below_four": final["fraction_ess_below_four"],
                "half_estimate_mean_absolute_difference_bits_per_frame": controls[
                    "independent_half_estimate_mean_absolute_difference_bits"
                ],
            }
        )
    expected = {
        ("p1fyyp1z", 24_576),
        ("p1fyyp1z", 57_344),
        ("p1fyyp1z", 98_304),
        ("p1fyyp1z", 131_072),
        ("tgpuo1be", 57_344),
        ("tgpuo1be", 131_072),
    }
    if records.keys() != expected:
        raise ValueError("the analysis requires all six preregistered checkpoint identities")
    contrasts: dict[str, object] = {}
    for label, first, second in (
        ("muon_slump_minus_early", ("p1fyyp1z", 24_576), ("p1fyyp1z", 57_344)),
        ("muon_final_minus_early", ("p1fyyp1z", 24_576), ("p1fyyp1z", 131_072)),
        ("adam_final_minus_middle", ("tgpuo1be", 57_344), ("tgpuo1be", 131_072)),
    ):
        baseline, treatment = records[first], records[second]
        estimate_before = mapping(mapping(baseline["estimates"])["1024"])
        estimate_after = mapping(mapping(treatment["estimates"])["1024"])
        marginal_difference = replay_vector(
            estimate_after["per_replay_joint_bits_per_executed_frame"]
        ) - replay_vector(estimate_before["per_replay_joint_bits_per_executed_frame"])
        # V1's short key is per frame, despite omitting that unit from its name.
        teacher_difference = replay_vector(
            mapping(treatment["teacher_forced"])["per_replay_tail_given_expert_prefix_bits"]
        ) - replay_vector(mapping(baseline["teacher_forced"])["per_replay_tail_given_expert_prefix_bits"])
        half_gap_before = scalar(mapping(baseline["controls"])["independent_half_estimate_mean_difference_bits"])
        half_gap_after = scalar(mapping(treatment["controls"])["independent_half_estimate_mean_difference_bits"])
        contrasts[label] = {
            "marginal": paired_interval(marginal_difference),
            "teacher_given_expert_prefix": paired_interval(teacher_difference),
            "first_minus_second_half_checkpoint_contrast_bits_per_frame": half_gap_before - half_gap_after,
        }
    return {
        "schema_version": 1,
        "cohort_sha256": EXPECTED_COHORT,
        "rows": rows,
        "contrasts": contrasts,
        "bootstrap_seed": SEED,
        "limitations": [
            "The proper probability target does not remove replay-versus-learner occupancy mismatch.",
            "The replay bootstrap conditions on Monte Carlo draws and does not bound their bias.",
            "Observed likelihood-weight ESS can miss an unsampled rare high-weight prefix.",
            "No checkpoint was selected for gameplay using these offline values.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["contrasts"], indent=2))


if __name__ == "__main__":
    main()
