"""Build experiment-039 capacity, exposure, and delay artifacts from W&B.

The primary matrices contain stock LCB and mask cells whose terminal run is
incomplete or whose measured evaluation path missed its deadline. Local RTX
3060 latency evidence is joined separately for the native deployment curve.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Final

import matplotlib
import numpy as np
import pandas as pd
import tyro

import wandb

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

WANDB_GROUP: Final[str] = "026-capacity-scaling-data-delay"
DELAYS: Final[tuple[int, ...]] = (1, 2, 4, 6, 8, 10, 12, 14, 16)
D_EXPONENTS: Final[tuple[int, ...]] = (26, 27, 28, 29, 30)
MODEL_ORDER: Final[tuple[str, ...]] = ("026base", "L5", "L7", "L10", "L13", "L16", "L18")
PRIMARY_METRIC: Final[str] = "stock_lcb"


@dataclass(frozen=True, slots=True)
class Args:
    wandb_path: str = "ericyuegu/hal"
    """W&B entity/project path."""
    group: str = WANDB_GROUP
    """W&B group to aggregate."""
    output_dir: str = "results/039_capacity_scaling"
    """Destination for CSV, JSON, and PNG artifacts."""
    latency_dir: str = "results/039_capacity_latency"
    """Directory containing local RTX 3060 benchmark JSON files."""
    unique_data_divisor: int = 1
    """Nested replay subset to include: 1 is U, 2 is U/2, and 4 is U/4."""
    fixed_d_exp: int = 30
    """Processed-exposure column used by the delay matrix and fixed-D plots."""
    fixed_delay: int = 1
    """Delay used by fixed-delay plots and wave-decision support."""
    allow_incomplete: bool = True
    """Emit partial-wave artifacts instead of failing when cells are missing."""


def _model_label(config: dict[str, Any]) -> str:
    if config.get("model_family") == "026-baseline":
        return "026base"
    return f"L{int(config['n_layers'])}"


def _summary_number(summary: dict[str, Any], name: str) -> float:
    value = summary.get(name, math.nan)
    return float(value) if isinstance(value, (int, float)) else math.nan


def _metric(summary: dict[str, Any], delay: int, name: str) -> float:
    return _summary_number(summary, f"eval_d{delay}/{name}")


def _terminal_row(run: Any) -> dict[str, Any] | None:
    config = dict(run.config)
    if config.get("phase") != "cooldown":
        return None
    summary = dict(run.summary)
    target = int(config["target_processed_positions"])
    processed = int(summary.get("training/processed_positions", -1))
    complete = run.state == "finished" and processed == target
    label = _model_label(config)
    row: dict[str, Any] = {
        "run_id": run.id,
        "run_name": run.name,
        "run_url": run.url,
        "run_state": run.state,
        "endpoint_complete": complete,
        "model": label,
        "model_order": MODEL_ORDER.index(label),
        "model_family": config["model_family"],
        "L": int(config["n_layers"]),
        "d_model": int(config["d_model"]),
        "trunk_parameters": int(config["trunk_parameters"]),
        "decoder_parameters": int(config["decoder_parameters"]),
        "total_parameters": int(config["total_parameters"]),
        "D": target,
        "D_exp": int(math.log2(target)),
        "U_divisor": int(config["unique_data_divisor"]),
        "unique_replays": int(config["unique_replays"]),
        "episode_hash": config["episode_hash"],
        "unique_loss_positions": int(config["unique_loss_positions"]),
        "D_over_U_positions": _summary_number(summary, "training/D_over_U_positions"),
        "D_over_N": _summary_number(summary, "training/D_over_N"),
        "optimizer_mode": config["adam_tau_scaling"],
        "muon_lr": float(config["muon_lr"]),
        "adam_lr": float(config["adam_lr"]),
        "muon_weight_decay": float(config["muon_weight_decay"]),
        "adam_weight_decay": float(config["adam_weight_decay"]),
        "adam_tau": _summary_number(summary, "training/adam_tau"),
        "flops_approx": _summary_number(summary, "training/flops"),
        "flops_formula": summary.get("training/flops_formula", ""),
        "training_time_seconds": _summary_number(summary, "training/wall_seconds"),
        "cumulative_training_time_seconds": _summary_number(summary, "training/cumulative_wall_seconds"),
        "incremental_branch_time_seconds": _summary_number(summary, "training/incremental_run_wall_seconds"),
        "shared_prefix_time_seconds": _summary_number(summary, "training/prefix_wall_seconds"),
        "validation_nll": _summary_number(summary, "val/nll"),
        "created_at": str(run.created_at),
    }
    for delay in DELAYS:
        valid = _metric(summary, delay, "valid_latency_bucket") == 1.0
        row[f"d{delay}_valid"] = valid
        for name in (
            "win_rate",
            "mean_stock_difference",
            "stock_lcb",
            "mean_damage_difference",
            "damage_lcb",
            "latency_p50_ms",
            "latency_p95_ms",
            "sustained_inference_rows_per_s",
            "deadline_misses",
            "deadline_miss_fraction",
            "observation_age_mean",
            "observation_age_p95",
        ):
            value = _metric(summary, delay, name)
            row[f"d{delay}_{name}"] = value if valid or name.startswith(("latency_", "deadline_")) else math.nan
    return row


def _select_unique_endpoints(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).sort_values("created_at")
    key = ["model", "D", "U_divisor"]
    selected: list[pd.Series] = []
    for _, candidates in frame.groupby(key, sort=False):
        completed = candidates[candidates["endpoint_complete"]]
        selected.append((completed if not completed.empty else candidates).iloc[-1])
    return pd.DataFrame(selected).sort_values(["U_divisor", "model_order", "D"]).reset_index(drop=True)


def _load_latency(directory: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not directory.is_dir():
        return out
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text())
        if payload.get("schema_version") != 2:
            raise ValueError(f"unsupported latency artifact {path}")
        label = "026base" if payload["model_family"] == "026-baseline" else f"L{payload['L']}"
        if label in out:
            raise ValueError(f"duplicate local latency artifact for {label}")
        out[label] = payload
    return out


def _join_native_latency(frame: pd.DataFrame, latency: dict[str, dict[str, Any]]) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    native_delays: list[float] = []
    p50_values: list[float] = []
    p95_values: list[float] = []
    throughput_values: list[float] = []
    native_performance: list[float] = []
    for _, row in frame.iterrows():
        evidence = latency.get(str(row["model"]))
        native = None if evidence is None else evidence.get("native_delay")
        bucket = None if native is None else evidence["latency"][str(native)]
        native_delays.append(math.nan if native is None else float(native))
        p50_values.append(math.nan if bucket is None else float(bucket["latency_p50_ms"]))
        p95_values.append(math.nan if bucket is None else float(bucket["latency_p95_ms"]))
        throughput_values.append(math.nan if bucket is None else float(bucket["sustained_inference_rows_per_s"]))
        native_performance.append(math.nan if native is None else float(row[f"d{native}_{PRIMARY_METRIC}"]))
    frame["native_delay"] = native_delays
    frame["native_latency_p50_ms"] = p50_values
    frame["native_latency_p95_ms"] = p95_values
    frame["native_sustained_inference_rows_per_s"] = throughput_values
    frame["native_stock_lcb"] = native_performance
    return frame


def _matrix(
    frame: pd.DataFrame,
    *,
    row_name: str,
    row_values: tuple[Any, ...],
    column_name: str,
    column_values: tuple[Any, ...],
    value: str,
) -> pd.DataFrame:
    table = frame.pivot(index=row_name, columns=column_name, values=value) if not frame.empty else pd.DataFrame()
    return table.reindex(index=row_values, columns=column_values)


def _save_matrix_artifacts(frame: pd.DataFrame, args: Args, output: Path) -> dict[str, list[str]]:
    selected = frame[frame["U_divisor"] == args.unique_data_divisor] if not frame.empty else frame
    missing: dict[str, list[str]] = {}
    at_d = selected[selected["D_exp"] == args.fixed_d_exp] if not selected.empty else selected
    delay_values = pd.DataFrame(index=MODEL_ORDER, columns=DELAYS, dtype=float)
    delay_valid = pd.DataFrame(index=MODEL_ORDER, columns=DELAYS, dtype=float)
    for _, row in at_d.iterrows():
        for delay in DELAYS:
            delay_values.loc[row["model"], delay] = row[f"d{delay}_{PRIMARY_METRIC}"]
            delay_valid.loc[row["model"], delay] = float(row[f"d{delay}_valid"])
    delay_values.to_csv(output / f"delay_matrix_D2p{args.fixed_d_exp}.csv", index_label="model")
    delay_valid.to_csv(output / f"delay_validity_D2p{args.fixed_d_exp}.csv", index_label="model")

    for delay in DELAYS:
        values = pd.DataFrame(index=MODEL_ORDER, columns=D_EXPONENTS, dtype=float)
        validity = pd.DataFrame(index=MODEL_ORDER, columns=D_EXPONENTS, dtype=float)
        for _, row in selected.iterrows():
            values.loc[row["model"], row["D_exp"]] = row[f"d{delay}_{PRIMARY_METRIC}"]
            validity.loc[row["model"], row["D_exp"]] = float(row[f"d{delay}_valid"])
        values.columns = [f"2^{exponent}" for exponent in values.columns]
        validity.columns = values.columns
        values.to_csv(output / f"nx_d_d{delay}.csv", index_label="model")
        validity.to_csv(output / f"nx_d_validity_d{delay}.csv", index_label="model")
        missing[f"d{delay}"] = [
            f"{model}/D2p{exponent}"
            for model in MODEL_ORDER
            for exponent in D_EXPONENTS
            if pd.isna(values.loc[model, f"2^{exponent}"])
        ]
    return missing


def _save_latency_frontier(latency: dict[str, dict[str, Any]], output: Path) -> pd.DataFrame:
    """Record the largest locally deployable capacity in each delay bucket."""
    rows: list[dict[str, Any]] = []
    for delay in DELAYS:
        eligible = []
        for model, evidence in latency.items():
            bucket = evidence["latency"][str(delay)]
            if bucket["valid_bucket"] == 1.0:
                eligible.append((int(evidence["total_parameters"]), model, bucket))
        eligible.sort()
        largest = eligible[-1] if eligible else None
        rows.append(
            {
                "delay": delay,
                "frame_budget_ms": delay * 1_000 / 60,
                "eligible_models": ",".join(model for _, model, _ in eligible),
                "largest_model": None if largest is None else largest[1],
                "largest_model_parameters": math.nan if largest is None else largest[0],
                "largest_model_p95_ms": math.nan if largest is None else largest[2]["latency_p95_ms"],
                "largest_model_deadline_misses": math.nan if largest is None else largest[2]["deadline_misses"],
                "valid": largest is not None,
            }
        )
    frontier = pd.DataFrame(rows)
    frontier.to_csv(output / "latency_capacity_frontier.csv", index=False)
    (output / "latency_capacity_frontier.json").write_text(
        json.dumps(frontier.replace({np.nan: None}).to_dict(orient="records"), indent=2) + "\n"
    )
    return frontier


def _plot_lines(
    groups: list[tuple[str, np.ndarray, np.ndarray]],
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path,
    xscale: str = "linear",
) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    plotted = False
    for label, x, y in groups:
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.any():
            axis.plot(x[finite], y[finite], marker="o", label=label)
            plotted = True
    if plotted and len(groups) > 1:
        axis.legend(fontsize=8)
    if not plotted:
        axis.text(0.5, 0.5, "No complete valid cells", ha="center", va="center", transform=axis.transAxes)
    axis.set_xscale(xscale)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _save_plots(frame: pd.DataFrame, latency: dict[str, dict[str, Any]], args: Args, output: Path) -> None:
    selected = frame[frame["U_divisor"] == args.unique_data_divisor] if not frame.empty else frame
    fixed = selected[selected["D_exp"] == args.fixed_d_exp] if not selected.empty else selected
    ordered = fixed.sort_values("total_parameters") if not fixed.empty else fixed
    _plot_lines(
        [
            (
                f"D=2^{args.fixed_d_exp}, d={args.fixed_delay}",
                ordered["total_parameters"].to_numpy(float),
                ordered[f"d{args.fixed_delay}_{PRIMARY_METRIC}"].to_numpy(float),
            )
        ]
        if not ordered.empty
        else [],
        xlabel="Parameters",
        ylabel="Stock LCB",
        title="Performance vs parameters at fixed exposure and delay",
        path=output / "performance_vs_parameters_fixed_D_delay.png",
        xscale="log",
    )
    exposure_groups = []
    nll_groups = []
    for model in MODEL_ORDER:
        rows = selected[selected["model"] == model].sort_values("D") if not selected.empty else selected
        if rows.empty:
            continue
        exposure_groups.append(
            (model, rows["D"].to_numpy(float), rows[f"d{args.fixed_delay}_{PRIMARY_METRIC}"].to_numpy(float))
        )
    for exponent in D_EXPONENTS:
        rows = (
            selected[selected["D_exp"] == exponent].sort_values("total_parameters") if not selected.empty else selected
        )
        if not rows.empty:
            nll_groups.append(
                (f"D=2^{exponent}", rows["total_parameters"].to_numpy(float), rows["validation_nll"].to_numpy(float))
            )
    _plot_lines(
        exposure_groups,
        xlabel="Processed loss positions D",
        ylabel=f"Stock LCB at d={args.fixed_delay}",
        title="Performance vs processed exposure at fixed capacity",
        path=output / "performance_vs_D_fixed_capacity.png",
        xscale="log",
    )
    _plot_lines(
        nll_groups,
        xlabel="Parameters",
        ylabel="Validation NLL (bits objective)",
        title="Validation NLL vs parameters",
        path=output / "validation_nll_vs_parameters.png",
        xscale="log",
    )

    latency_groups = []
    for delay in DELAYS:
        xs = []
        ys = []
        for model in MODEL_ORDER:
            evidence = latency.get(model)
            if evidence is None:
                continue
            xs.append(float(evidence["total_parameters"]))
            ys.append(float(evidence["latency"][str(delay)]["latency_p95_ms"]))
        latency_groups.append((f"d={delay}", np.asarray(xs), np.asarray(ys)))
    _plot_lines(
        latency_groups,
        xlabel="Parameters",
        ylabel="RTX 3060 p95 end-to-end latency (ms)",
        title="Production-path latency vs parameters",
        path=output / "latency_vs_parameters.png",
        xscale="log",
    )
    _plot_lines(
        [
            (
                "native delay",
                ordered["total_parameters"].to_numpy(float),
                ordered["native_stock_lcb"].to_numpy(float),
            )
        ]
        if not ordered.empty
        else [],
        xlabel="Parameters",
        ylabel="Stock LCB at RTX 3060 native delay",
        title="Native deployment curve J(N, d_native(N))",
        path=output / "native_deployment_curve.png",
        xscale="log",
    )

    gain_x: list[float] = []
    gain_y: list[float] = []
    if not ordered.empty:
        values = ordered[f"d{args.fixed_delay}_{PRIMARY_METRIC}"].to_numpy(float)
        parameters = ordered["total_parameters"].to_numpy(float)
        gain_x = parameters[1:].tolist()
        gain_y = np.diff(values).tolist()
    _plot_lines(
        [("gain over previous capacity", np.asarray(gain_x), np.asarray(gain_y))],
        xlabel="Parameters",
        ylabel="Change in stock LCB",
        title=f"Capacity gain at D=2^{args.fixed_d_exp}, d={args.fixed_delay}",
        path=output / "capacity_gain_fixed_delay.png",
        xscale="log",
    )
    penalty_groups = []
    for _, row in ordered.iterrows():
        values = np.asarray([row[f"d{delay}_{PRIMARY_METRIC}"] for delay in DELAYS], dtype=float)
        baseline = values[0]
        penalty_groups.append((str(row["model"]), np.asarray(DELAYS, dtype=float), values - baseline))
    _plot_lines(
        penalty_groups,
        xlabel="Observation-to-action delay d (frames), R=d",
        ylabel="Stock LCB change from d=1",
        title=f"Delay penalty at fixed model size and D=2^{args.fixed_d_exp}",
        path=output / "delay_penalty_fixed_model.png",
    )


def _decision_support(frame: pd.DataFrame, args: Args) -> pd.DataFrame:
    selected = frame[
        (frame["U_divisor"] == args.unique_data_divisor) & (frame["D_exp"] == args.fixed_d_exp)
    ].sort_values("total_parameters")
    rows: list[dict[str, Any]] = []
    previous: pd.Series | None = None
    for _, current in selected.iterrows():
        if previous is not None:
            delta_mean = (
                current[f"d{args.fixed_delay}_mean_stock_difference"]
                - previous[f"d{args.fixed_delay}_mean_stock_difference"]
            )
            delta_lcb = current[f"d{args.fixed_delay}_stock_lcb"] - previous[f"d{args.fixed_delay}_stock_lcb"]
            rows.append(
                {
                    "smaller_model": previous["model"],
                    "larger_model": current["model"],
                    "D_exp": args.fixed_d_exp,
                    "delay": args.fixed_delay,
                    "delta_mean_stock": delta_mean,
                    "delta_stock_lcb": delta_lcb,
                    "meaningful_0p2_or_more": bool(np.isfinite(delta_mean) and delta_mean >= 0.2),
                    "small_below_0p05": bool(np.isfinite(delta_mean) and abs(delta_mean) < 0.05),
                    "conservative_lcb_improves": bool(np.isfinite(delta_lcb) and delta_lcb > 0.0),
                }
            )
        previous = current
    return pd.DataFrame(rows)


def main(args: Args) -> None:
    if args.unique_data_divisor not in (1, 2, 4):
        raise SystemExit("--unique-data-divisor must be 1, 2, or 4")
    if args.fixed_d_exp not in D_EXPONENTS or args.fixed_delay not in DELAYS:
        raise SystemExit(f"fixed D/d must be in {D_EXPONENTS}/{DELAYS}")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=120)
    runs = api.runs(args.wandb_path, filters={"group": args.group})
    frame = _select_unique_endpoints([row for run in runs if (row := _terminal_row(run)) is not None])
    latency = _load_latency(Path(args.latency_dir))
    frame = _join_native_latency(frame, latency)
    capacity_path = output / "capacity_table.csv"
    frame.drop(columns=["model_order"], errors="ignore").to_csv(capacity_path, index=False)
    missing = _save_matrix_artifacts(frame, args, output)
    frontier = _save_latency_frontier(latency, output)
    decision = _decision_support(frame, args)
    decision.to_csv(output / "wave_decision_support.csv", index=False)
    _save_plots(frame, latency, args, output)

    selected = frame[frame["U_divisor"] == args.unique_data_divisor] if not frame.empty else frame
    expected = len(MODEL_ORDER) * len(D_EXPONENTS)
    complete_endpoints = int(selected["endpoint_complete"].sum()) if not selected.empty else 0
    report = {
        "schema_version": 1,
        "wandb_path": args.wandb_path,
        "group": args.group,
        "unique_data_divisor": args.unique_data_divisor,
        "primary_metric": PRIMARY_METRIC,
        "invalid_latency_cells_are_masked": True,
        "expected_terminal_endpoints": expected,
        "complete_terminal_endpoints": complete_endpoints,
        "missing_or_invalid_cells": missing,
        "local_latency_models": sorted(latency, key=MODEL_ORDER.index),
        "delay_buckets_with_local_frontier": int(frontier["valid"].sum()),
    }
    (output / "completeness.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not args.allow_incomplete and (complete_endpoints != expected or any(missing.values())):
        raise SystemExit(f"wave is incomplete; see {output / 'completeness.json'}")
    print(f"[039 artifacts] wrote {capacity_path} and {len(DELAYS)} N x D matrices", flush=True)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
