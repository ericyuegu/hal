"""Pair-bootstrap match-level trajectory summaries; no equivalence margin is assumed."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def summaries(run: dict[str, Any]) -> dict[str, NDArray[np.float64]]:
    result = {}
    rows = {match: [u["matches"][match] for u in run["updates"]] for match in range(64)}
    for field in ("kl", "entropy", "resets"):
        result[field] = np.array(
            [sum(r[field] for r in values) / (3 if field in ("kl", "entropy") else 1) for values in rows.values()]
        )
    trajectories = {
        field: [np.concatenate([np.array(r[field], dtype=np.float64) for r in items]) for items in rows.values()]
        for field in ("advantages", "targets")
    }
    trajectories["reward"] = [
        np.array([sum(r["player_rewards"][side] for r in items) for side in items[0]["player_rewards"]])
        for items in rows.values()
    ]
    for field, values in trajectories.items():
        for statistic in ("mean", "std", "q05", "q50", "q95"):
            result["reward" if field == "reward" and statistic == "mean" else f"{field}/{statistic}"] = np.array(
                [
                    float(v.mean())
                    if statistic == "mean"
                    else float(v.std())
                    if statistic == "std"
                    else float(np.quantile(v, float(statistic[1:]) / 100))
                    for v in values
                ]
            )
    lengths = [
        np.concatenate([np.array(r["episode_lengths"], dtype=np.float64) for r in values]) for values in rows.values()
    ]
    if any(len(v) == 0 for v in lengths):
        raise ValueError("512-frame truncations must complete an episode for every match")
    result["episode_length"] = np.array([v.mean() for v in lengths])
    if any(not np.isfinite(v).all() for v in result.values()):
        raise ValueError("comparison statistics must be finite")
    return result


def compare(control: dict[str, Any], dense: dict[str, Any]) -> dict[str, Any]:
    if (
        control["format"] != "hal.o65.dense-statistical-raw.v2"
        or dense["format"] != control["format"]
        or (control["mode"], dense["mode"]) != ("control", "dense")
    ):
        raise ValueError("unsupported comparison inputs")
    for name in (
        "seed",
        "max_frame",
        "checkpoint",
        "data_manifests",
        "environment",
        "sampling",
        "schedule",
        "simulator",
        "adapter",
        "simulator_manifest",
    ):
        if control[name] != dense[name]:
            raise ValueError(f"comparison invariant changed: {name}")
    training = dict(control["training"])
    if training.pop("importance_cap", None) != 2 or training != dense["training"]:
        raise ValueError("training settings changed beyond the approved importance correction")
    for run in (control, dense):
        if [u["update"] for u in run["updates"]] != [0, 1, 2] or any(
            [r["match"] for r in u["matches"]] != list(range(64)) for u in run["updates"]
        ):
            raise ValueError("comparison requires three complete 64-match updates")

    for match in range(64):
        sides = set(control["updates"][0]["matches"][match]["player_rewards"])
        if (
            not sides
            or sides - {"0", "1"}
            or any(
                set(update["matches"][match]["player_rewards"]) != sides
                for run in (control, dense)
                for update in run["updates"]
            )
        ):
            raise ValueError("comparison student sides changed")

    old, new = summaries(control), summaries(dense)
    rng = np.random.default_rng(65)
    indices = rng.integers(0, 64, size=(10000, 64))
    rows = {}
    for name in old:
        a, b = old[name], new[name]
        difference = b - a
        interval = np.quantile(difference[indices].mean(-1), [0.025, 0.975])
        rows[name] = {
            "control_mean": float(a.mean()),
            "dense_mean": float(b.mean()),
            "mean_difference": float(difference.mean()),
            "paired_match_bootstrap_95ci": interval.tolist(),
            "includes_zero": bool(interval[0] <= 0 <= interval[1]),
            "control_match_quantiles": np.quantile(a, [0.05, 0.5, 0.95]).tolist(),
            "dense_match_quantiles": np.quantile(b, [0.05, 0.5, 0.95]).tolist(),
        }
    result = {
        "format": "hal.o65.dense-statistical-comparison.v2",
        "configuration": {
            "matches": 64,
            "updates": 3,
            "seed": 65,
            "max_frame": 512,
            "bootstrap_replicates": 10000,
            "confidence": 0.95,
            "resampling_unit": "paired logical match, retaining all three updates",
            "trajectory_statistics": "per-match pooled decisions; means of match-level mean/std/quantiles",
            "reward": "per-match student returns across three updates; mean/std/quantiles before match clustering",
            "kl": "mean per planned frame across the three updates",
            "entropy": "mean per planned action frame across the three updates",
            "resets": "completed episodes across three updates",
            "acceptance": "descriptive confidence intervals; the plan supplies no equivalence margin",
        },
        "invariants": {
            k: control[k]
            for k in (
                "checkpoint",
                "seed",
                "sampling",
                "schedule",
                "max_frame",
                "environment",
                "simulator",
                "adapter",
                "simulator_manifest",
            )
        },
        "control_git_sha": control["git_sha"],
        "dense_git_sha": dense["git_sha"],
        "control_trainer_sha256": control["trainer_sha256"],
        "dense_trainer_sha256": dense["trainer_sha256"],
        "statistics": rows,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("dense", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = compare(json.loads(args.control.read_text()), json.loads(args.dense.read_text()))
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    for name, row in result["statistics"].items():
        print(name, row["control_mean"], row["dense_mean"], row["paired_match_bootstrap_95ci"])


if __name__ == "__main__":
    main()
