"""Write MSL viewer files, value plots, and stock-reward checks for O65 traces.

Run with ``uv run --group analysis python scripts/o65_trace_report.py TRACE_DIR OUT_DIR``.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import cast

import matplotlib
import numpy as np
from numpy.typing import NDArray

from hal.sim import msl_trace
from hal.sim import native
from hal.wire import MASK_INT32

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_trace(path: Path) -> tuple[dict[str, object], dict[str, NDArray]]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    header = json.loads(str(arrays.pop("header").item()))
    if (
        not isinstance(header, dict)
        or header.get("format") != "hal.o65.match-trace.v1"
        or header.get("native_columns") != list(native.OBSERVATION_FIELDS)
        or not isinstance(header.get("resolved"), dict)
    ):
        raise ValueError(f"{path}: match trace identity or schema mismatch")
    count = len(arrays["columns"])
    if count < 2 or not np.array_equal(arrays["frame"], np.arange(count)):
        raise ValueError(f"{path}: match trace frames must be contiguous from reset")
    for name, shape, dtype in (
        ("columns", (count, len(native.OBSERVATION_FIELDS)), np.uint32),
        ("applied_actions", (count, 2, 14), np.float32),
        ("rewards", (count, 2), np.float32),
        ("terminal", (count,), np.bool_),
        ("timeout", (count,), np.bool_),
    ):
        if arrays[name].shape != shape or arrays[name].dtype != dtype:
            raise ValueError(f"{path}: invalid trace array {name}")
    ended = arrays["terminal"] | arrays["timeout"]
    if not ended[-1] or ended[:-1].any() or (arrays["terminal"] & arrays["timeout"]).any():
        raise ValueError(f"{path}: expected one complete match")
    decisions = len(arrays["decision_frame"])
    for name in ("side", "value_mean", "advantage", "lambda"):
        if arrays[f"decision_{name}"].shape != (decisions,) or not np.isfinite(arrays[f"decision_{name}"]).all():
            raise ValueError(f"{path}: invalid decision {name}")
    if (
        (arrays["decision_frame"] < 0).any()
        or (arrays["decision_frame"] >= count - 1).any()
        or not np.isin(arrays["decision_side"], (0, 1)).all()
    ):
        raise ValueError(f"{path}: invalid decision frame or side")
    return cast(dict[str, object], header), arrays


def stock_rewards(columns: NDArray[np.uint32]) -> NDArray[np.float32]:
    """Match native_adapter.c, including masks and the deciding-stock bonus."""
    stocks = np.stack([msl_trace.column_values(columns, f"p{side}_stock") for side in (1, 2)], -1)
    losses = (stocks[1:] < stocks[:-1]) & (stocks[1:] != MASK_INT32) & (stocks[:-1] != MASK_INT32)
    finals = losses & (stocks[1:] == 0)
    base = 120 * (losses[:, 1].astype(np.int32) - losses[:, 0]) + 50 * (finals[:, 1].astype(np.int32) - finals[:, 0])
    result = np.zeros((len(columns), 2), dtype=np.float32)
    result[1:, 0] = base
    result[1:, 1] = -base
    return result


def write_match(path: Path, output: Path) -> bool:
    _, arrays = load_trace(path)
    columns = arrays["columns"]
    viewer = msl_trace.to_msltrace(columns, arrays["applied_actions"])
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{path.stem}.msltrace.json").write_text(json.dumps(viewer, allow_nan=False))
    expected = stock_rewards(columns)
    matches = np.isclose(arrays["rewards"], expected, rtol=0, atol=1e-6)
    with (output / f"{path.stem}.rewards.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("frame", "side", "logged_stock_reward", "recomputed_stock_reward", "difference", "match"))
        for frame in range(len(columns)):
            for side in range(2):
                logged = float(arrays["rewards"][frame, side])
                reward = float(expected[frame, side])
                writer.writerow((frame, side, logged, reward, logged - reward, bool(matches[frame, side])))
    figure, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, layout="constrained")
    try:
        for side, color in ((0, "tab:blue"), (1, "tab:orange")):
            student = arrays["decision_side"] == side
            if student.any():
                frames = arrays["decision_frame"][student]
                order = np.argsort(frames, kind="stable")
                lambdas = np.unique(arrays["decision_lambda"][student])
                label = f"P{side + 1}, lambda=" + ", ".join(f"{value:.3g}" for value in lambdas)
                axes[0].plot(frames[order], arrays["decision_value_mean"][student][order], color=color, label=label)
                axes[1].plot(frames[order], arrays["decision_advantage"][student][order], color=color, label=label)
            stock = msl_trace.column_values(columns, f"p{side + 1}_stock")
            losses = (
                np.flatnonzero((stock[1:] < stock[:-1]) & (stock[1:] != MASK_INT32) & (stock[:-1] != MASK_INT32)) + 1
            )
            for index, frame in enumerate(losses):
                axes[0].axvline(
                    frame,
                    color=color,
                    linestyle="--",
                    alpha=0.6,
                    label=f"P{side + 1} stock loss" if index == 0 else None,
                )
                axes[1].axvline(frame, color=color, linestyle="--", alpha=0.6)
        axes[0].set_title(path.stem)
        axes[0].set_ylabel("Student value mean")
        axes[0].legend(loc="best", fontsize="small")
        axes[1].set_ylabel("Advantage")
        axes[1].set_xlabel("Frame since reset (60 frames/s)")
        axes[1].axhline(0, color="gray", linewidth=0.5)
        figure.savefig(output / f"{path.stem}.png", dpi=150)
    finally:
        plt.close(figure)
    return bool(matches.all())


def report(trace_dir: Path, output: Path) -> int:
    paths = sorted(trace_dir.glob("*.npz"))
    if not paths:
        raise ValueError(f"no match traces in {trace_dir}")
    mismatches = [path.name for path in paths if not write_match(path, output)]
    if mismatches:
        print("Stock reward mismatch: " + ", ".join(mismatches))
        return 1
    print(f"Checked {len(paths)} match traces; stock rewards match.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    return report(args.trace_dir, args.out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
