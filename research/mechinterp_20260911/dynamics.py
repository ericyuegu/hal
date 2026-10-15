"""Align exact O50 histories and expose disagreements between logged metrics."""

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


@dataclass(frozen=True, slots=True)
class Evaluation:
    step: int
    stock_rate: float
    damage_rate: float
    stock_lower_bound: float
    nll_bits: float
    near_nll_bits: float
    far_nll_bits: float
    change_f1: float
    exact_frame_accuracy: float
    sequence_accuracy: float


def history(path: Path) -> dict[int, dict[str, float]]:
    """Collapse exact duplicate events, but reject conflicting values at a step."""
    rows: dict[int, dict[str, float]] = {}
    with path.open() as source:
        for line in source:
            raw = json.loads(line)
            if not isinstance(raw, dict) or "global_step" not in raw:
                raise ValueError(f"{path}: history row has no global_step")
            row: dict[str, float] = {}
            for key, value in raw.items():
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"{path}: non-finite or non-numeric {key}")
                row[key] = float(value)
            step = int(row["global_step"])
            if row["global_step"] != step:
                raise ValueError(f"{path}: non-integral global_step")
            if step in rows and rows[step] != row:
                raise ValueError(f"{path}: conflicting history at step {step}")
            rows[step] = row
    return dict(sorted(rows.items()))


def align_evaluations(
    evaluations: Mapping[int, Mapping[str, float]],
    validation: Mapping[int, Mapping[str, float]],
) -> list[Evaluation]:
    aligned = []
    for step, row in sorted(evaluations.items()):
        if step not in validation:
            raise ValueError(f"no validation at evaluation step {step}; interpolation is not allowed")
        if row["eval/checkpoint_step"] != step:
            raise ValueError(f"checkpoint/global-step mismatch at {step}")
        if row["eval/boots"] != 96 or row["eval/crashed"] != 0:
            raise ValueError(f"incomplete or unexpected evaluation protocol at {step}")
        val = validation[step]
        total = (val["val/near_nll"] + 0.5 * val["val/far_nll"]) / 1.5
        if not math.isclose(total, val["val/nll"], rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError(f"near/far NLL do not reproduce the logged total at {step}")
        aligned.append(
            Evaluation(
                step=step,
                stock_rate=row["eval/net_stock_per_min"],
                damage_rate=row["eval/net_dmg_per_min"],
                stock_lower_bound=row["eval/net_stock_lcb"],
                nll_bits=val["val/nll"],
                near_nll_bits=val["val/near_nll"],
                far_nll_bits=val["val/far_nll"],
                change_f1=val["val/change_f1"],
                exact_frame_accuracy=val["val/exact_frame_acc"],
                sequence_accuracy=val["val/sequence_acc"],
            )
        )
    return aligned


def correlations(rows: list[Evaluation]) -> list[dict[str, str | int | float]]:
    """Report descriptive ranks, without treating successive checkpoints as IID."""
    result = []
    for target in ("stock_rate", "damage_rate"):
        for metric in ("nll_bits", "near_nll_bits", "far_nll_bits", "change_f1", "exact_frame_accuracy"):
            x = np.array([asdict(row)[metric] for row in rows], dtype=np.float64)
            y = np.array([asdict(row)[target] for row in rows], dtype=np.float64)
            for difference in (False, True):
                left, right = (np.diff(x), np.diff(y)) if difference else (x, y)
                value = float(stats.spearmanr(left, right).statistic)
                result.append(
                    {
                        "metric": metric,
                        "target": target,
                        "series": "successive_changes" if difference else "levels",
                        "count": len(left),
                        "spearman": value,
                    }
                )
    return result


def clipped_windows(training: Mapping[int, Mapping[str, float]], warmup: int) -> list[Mapping[str, float]]:
    """A mean gradient below threshold can still contain a clipped update."""
    return [row for step, row in sorted(training.items()) if step > warmup and row["optimizer/clip_fraction"] > 0]


def figures(
    records: Mapping[str, list[Evaluation]],
    training: Mapping[str, Mapping[int, Mapping[str, float]]],
    output: Path,
) -> None:
    colors = {"Muon + AdamW": "#c95737", "AdamW": "#23799c"}
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), sharex=True, layout="constrained")
    panels = (
        (axes[0, 0], "stock_rate", "Net stocks / minute"),
        (axes[0, 1], "damage_rate", "Net damage / minute"),
        (axes[1, 0], "near_nll_bits", "Near-action NLL (bits / controller frame)"),
        (axes[1, 1], "far_nll_bits", "Far-action NLL (bits / controller frame)"),
        (axes[2, 0], "change_f1", "Action-change F1"),
    )
    for label, rows in records.items():
        steps = np.array([row.step for row in rows]) / 1000
        for axis, field, title in panels:
            axis.plot(steps, [asdict(row)[field] for row in rows], "o-", ms=3, label=label, color=colors[label])
            axis.set_title(title, fontsize=10)
        samples = [row for step, row in training[label].items() if step >= 8192]
        axes[2, 1].plot(
            [row["global_step"] / 1000 for row in samples],
            [row["optimizer/grad_norm"] for row in samples],
            lw=0.7,
            alpha=0.75,
            color=colors[label],
        )
    axes[2, 1].set_yscale("log")
    axes[2, 1].set_title("Pre-clip gradient norm (25-update window means)", fontsize=10)
    axes[2, 1].axhline(1, color="#444444", ls="--", lw=1)
    for axis in axes.flat:
        axis.axvspan(24.576, 57.344, color="#aaaaaa", alpha=0.13)
        axis.grid(alpha=0.2)
    for axis in axes[2]:
        axis.set_xlabel("Optimizer updates (thousands)")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("The Muon gameplay slump is not one isolated gradient spike", fontsize=14)
    fig.savefig(output / "named_run_dynamics.png", dpi=180)
    fig.savefig(output / "named_run_dynamics.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for label, rows in records.items():
        steps = np.array([row.step for row in rows]) / 1000
        for axis, field, title in (
            (axes[0], "nll_bits", "Aggregate validation NLL (bits)"),
            (axes[1], "change_f1", "Action-change F1"),
        ):
            x = np.array([asdict(row)[field] for row in rows])
            y = np.array([row.stock_rate for row in rows])
            axis.plot(x, y, "o-", color=colors[label], ms=4, lw=1, label=label)
            for index in (2, 6, 15):
                axis.annotate(
                    f"{steps[index]:.0f}k", (x[index], y[index]), xytext=(4, 3), textcoords="offset points", fontsize=8
                )
            axis.set_xlabel(title)
            axis.set_ylabel("Net stocks / minute")
            axis.grid(alpha=0.2)
    axes[0].invert_xaxis()
    axes[0].legend(frameon=False)
    fig.suptitle("A metric can track the broad trend and miss an important interval", fontsize=13)
    fig.savefig(output / "metric_trajectories.png", dpi=180)
    fig.savefig(output / "metric_trajectories.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("research/mechinterp_20260911/artifacts/forensics"))
    parser.add_argument("--output", type=Path, default=Path("research/mechinterp_20260911/artifacts/dynamics"))
    args = parser.parse_args()
    records: dict[str, list[Evaluation]] = {}
    training: dict[str, dict[int, dict[str, float]]] = {}
    report: dict[str, object] = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "units": "Logged NLL is joint controller-frame bits; total=(near+0.5*far)/1.5.",
        "limitations": [
            "Two near-matched runs, one training seed each; not a randomized optimizer experiment.",
            "Checkpoint correlations are descriptive; checkpoints share training and evaluation seeds.",
            "The shaded interval was selected retrospectively from gameplay for mechanism diagnosis.",
            "Gradient norms and clip fractions are 25-update window means, not individual-update samples.",
            "Lower confidence bounds are recorded but are not symmetric confidence intervals.",
        ],
    }
    for run_id, label in (("p1fyyp1z", "Muon + AdamW"), ("tgpuo1be", "AdamW")):
        root = args.input / run_id
        validation = history(root / "validation.jsonl")
        rows = align_evaluations(history(root / "evaluation.jsonl"), validation)
        records[label] = rows
        training[label] = history(root / "training.jsonl")
        clipped = clipped_windows(training[label], warmup=4096)
        report[run_id] = {
            "label": label,
            "source_sha256": {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in ("run.json", "training.jsonl", "validation.jsonl", "evaluation.jsonl")
            },
            "aligned_evaluations": [asdict(row) for row in rows],
            "correlations": correlations(rows),
            "first_post_warmup_logged_clip": clipped[0] if clipped else None,
            "logged_post_warmup_windows_with_clips": len(clipped),
        }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    figures(records, training, args.output)
    print(f"Wrote exact-step dynamics for {sum(len(rows) for rows in records.values())} evaluations.")


if __name__ == "__main__":
    main()
