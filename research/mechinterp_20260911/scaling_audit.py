"""Audit Experiment 039 scaling metrics and the extrapolated compute optimum.

The audit reads the persisted W&B/R2 result slab without loading ``config_json``.
It reproduces the notebook's endpoint and divergence rules, recomputes gameplay
rates from active match rows, and keeps deployment latency validity separate from
scientific gameplay availability.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import scipy.optimize

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SOURCE_CSV = Path("/home/ericgu/src/hal/results/039_capacity_scaling/experiment_039_raw_outcomes.csv")
DEFAULT_OUTPUT_DIR = Path("research/mechinterp_20260911/artifacts/scaling")
TARGET_COMPUTE_FLOPS = 4.2e19
DIVERGENCE_NLL_MARGIN = 0.01
FRAMES_PER_MINUTE = 3600.0
DEFAULT_FIT_BOOTSTRAPS = 200
DEFAULT_RHO_BOOTSTRAPS = 500

REQUIRED_COLUMNS = frozenset(
    {
        "record_type",
        "source",
        "run_id",
        "model",
        "model_family",
        "L",
        "phase",
        "endpoint_exact",
        "analysis_excluded",
        "analysis_exclusion_reason",
        "wandb_state",
        "target_processed_positions",
        "processed_positions",
        "validation_nll",
        "validation_primary_nll",
        "validation_auxiliary_nll",
        "training_flops",
        "total_parameters",
        "trunk_parameters",
        "decoder_parameters",
        "unique_data_divisor",
        "unique_loss_positions",
        "unique_replays",
        "episode_hash",
        "evaluation_key",
        "delay_frames",
        "eval_control_delay",
        "eval_replan_interval",
        "eval_decode_horizon",
        "eval_completed_boots",
        "eval_scheduled_boots",
        "eval_crashed",
        "eval_matches",
        "eval_zero_active",
        "eval_valid_latency_bucket",
        "eval_net_stock_per_min",
        "eval_net_stock_lcb",
        "eval_net_dmg_per_min",
        "protocol_matchup_schedule_sha256",
        "protocol_checkpoint_sha256",
        "local_latency_valid_bucket",
        "local_latency_latency_p95_ms",
        "local_latency_frame_budget_ms",
        "local_native_delay",
        "match_active_frames",
        "match_boot_index",
        "match_stock_difference",
        "match_damage_difference",
        "match_total_frames",
    }
)

FIT_MODELS = ("L3", "L4", "L5", "L7", "L10", "L13", "L16", "L18")
NLL_COLUMNS = ("validation_nll", "validation_primary_nll", "validation_auxiliary_nll")


@dataclass(frozen=True, slots=True)
class ScalingFit:
    """One robust fit of the five-parameter scaling law."""

    axis: str
    n_points: int
    models: tuple[str, ...]
    objective: float
    asymptote: float
    capacity_coefficient: float
    data_coefficient: float
    capacity_exponent: float
    data_exponent: float
    n_eff_min: float
    n_eff_max: float
    data_min: float
    data_max: float
    compute_min: float
    compute_max: float
    n_eff_opt: float
    processed_opt: float
    predicted_nll: float

    @property
    def parameter_vector(self) -> tuple[float, ...]:
        """Return the optimizer-space vector for a nearby refit."""
        return (
            math.log(self.asymptote),
            math.log(self.capacity_coefficient),
            math.log(self.data_coefficient),
            self.capacity_exponent,
            self.data_exponent,
        )


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _header(path: Path) -> tuple[str, ...]:
    with path.open(newline="") as source:
        values = next(csv.reader(source))
    return tuple(values)


def load_raw(path: Path) -> pd.DataFrame:
    """Load only the persisted columns needed by the audit.

    Raises:
      FileNotFoundError: If the result slab is absent.
      ValueError: If the persisted schema does not contain required columns.
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    columns = set(_header(path))
    missing = sorted(REQUIRED_COLUMNS - columns)
    if missing:
        raise ValueError(f"{path}: missing persisted columns {missing}")
    return pd.read_csv(path, usecols=lambda name: name in REQUIRED_COLUMNS, low_memory=False)


def distinct_positions(processed_positions: pd.Series | np.ndarray, corpus_positions: float) -> np.ndarray:
    """Estimate distinct positions covered using the notebook's replacement model."""
    values = np.asarray(processed_positions, dtype=float)
    if not np.isfinite(corpus_positions) or corpus_positions <= 0:
        raise ValueError(f"corpus position count must be positive, got {corpus_positions}")
    return corpus_positions * (1.0 - np.exp(-values / corpus_positions))


def _bool_mask(values: pd.Series) -> pd.Series:
    return values.eq(True)


def select_completed_runs(raw: pd.DataFrame) -> pd.DataFrame:
    """Select exact, finished cooldown endpoints and mark notebook divergence.

    The returned table includes ``diverged`` rows for audit reporting. The
    original notebook excludes those rows from scaling curves and fits.
    """
    runs = raw.loc[raw["record_type"].eq("run")].copy()
    if runs["run_id"].isna().any() or runs["run_id"].duplicated().any():
        raise ValueError("run records must have one unique run_id each")

    exact = _bool_mask(runs["endpoint_exact"])
    invalid_exact = exact & (
        runs["phase"].ne("cooldown")
        | runs["processed_positions"].isna()
        | runs["target_processed_positions"].isna()
        | runs["processed_positions"].ne(runs["target_processed_positions"])
    )
    if invalid_exact.any():
        ids = runs.loc[invalid_exact, "run_id"].tolist()
        raise ValueError(f"exact endpoint records do not reach their target: {ids}")

    completed = runs.loc[
        exact
        & runs["phase"].eq("cooldown")
        & runs["wandb_state"].eq("finished")
        & ~_bool_mask(runs["analysis_excluded"])
        & runs["validation_nll"].notna()
    ].copy()
    if completed.empty:
        raise ValueError("no exact completed cooldown endpoints")

    completed = completed.sort_values(["model", "processed_positions", "run_id"]).reset_index(drop=True)
    prior = completed.groupby("model", sort=False)["validation_nll"].transform(lambda values: values.cummin().shift(1))
    completed["best_prior_nll"] = prior
    completed["diverged"] = completed["best_prior_nll"].notna() & (
        completed["validation_nll"] - completed["best_prior_nll"] > DIVERGENCE_NLL_MARGIN
    )
    completed["n_eff"] = completed["training_flops"] / (6.0 * completed["processed_positions"])
    corpus_values = completed["unique_loss_positions"].dropna().unique()
    if len(corpus_values) != 1:
        raise ValueError(f"completed runs disagree on unique corpus size: {corpus_values.tolist()}")
    completed["distinct_positions"] = distinct_positions(completed["processed_positions"], float(corpus_values[0]))
    completed["compute_over_corpus"] = completed["processed_positions"] / float(corpus_values[0])
    return completed


def select_evaluations(raw: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """Select complete evaluations and validate their timing/protocol identity."""
    evaluations = raw.loc[raw["record_type"].eq("evaluation")].copy()
    if evaluations["evaluation_key"].isna().any() or evaluations["evaluation_key"].duplicated().any():
        raise ValueError("evaluation records must have unique non-null evaluation_key values")

    evaluations = evaluations.loc[
        evaluations["run_id"].isin(runs["run_id"])
        & evaluations["eval_completed_boots"].eq(evaluations["eval_scheduled_boots"])
        & evaluations["eval_crashed"].eq(0)
    ].copy()
    if evaluations.empty:
        raise ValueError("no complete evaluations belong to exact endpoints")

    timing = (
        evaluations["delay_frames"].eq(evaluations["eval_control_delay"])
        & evaluations["eval_control_delay"].eq(evaluations["eval_replan_interval"])
        & evaluations["eval_decode_horizon"].eq(
            evaluations["eval_control_delay"] + evaluations["eval_replan_interval"] - 1
        )
    )
    if not timing.all():
        raise ValueError("an evaluation has inconsistent delay, replan, or decode-horizon fields")
    evaluations["protocol_hash_present"] = evaluations["protocol_matchup_schedule_sha256"].notna()

    run_columns = [
        "run_id",
        "model_family",
        "L",
        "phase",
        "endpoint_exact",
        "unique_data_divisor",
        "unique_loss_positions",
        "unique_replays",
        "episode_hash",
        "validation_nll",
        "validation_primary_nll",
        "validation_auxiliary_nll",
        "training_flops",
        "total_parameters",
        "trunk_parameters",
        "decoder_parameters",
        "n_eff",
        "distinct_positions",
        "diverged",
    ]
    evaluations = evaluations.merge(
        runs[run_columns], on="run_id", how="left", validate="many_to_one", suffixes=("", "_run")
    )
    if evaluations["model_family"].isna().any():
        raise ValueError("an evaluation could not be aligned to a selected run")
    for left, right in (("model", "model_run"), ("processed_positions", "processed_positions_run")):
        if right in evaluations and not evaluations[left].eq(evaluations[right]).all():
            raise ValueError(f"evaluation/run identity mismatch in {left}")
    evaluations["matched_delay"] = True
    return evaluations


def _active_matches(raw: pd.DataFrame, evaluation_keys: set[str]) -> pd.DataFrame:
    matches = raw.loc[
        raw["record_type"].eq("match")
        & raw["evaluation_key"].notna()
        & raw["evaluation_key"].isin(evaluation_keys)
        & raw["match_active_frames"].gt(0)
    ].copy()
    required = ["match_boot_index", "match_stock_difference", "match_damage_difference"]
    if matches[required].isna().any().any():
        raise ValueError("active match rows must have boot, stock, and damage values")
    matches["active_minutes"] = matches["match_active_frames"] / FRAMES_PER_MINUTE
    return matches


def _seed_from_key(key: str, salt: int = 0) -> int:
    digest = hashlib.sha256(f"{salt}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def bootstrap_pooled_rate(
    matches: pd.DataFrame,
    *,
    value_column: str,
    seed: int,
    resamples: int,
) -> tuple[float, float]:
    """Bootstrap a rate over boot clusters, never over individual match rows."""
    if matches.empty:
        raise ValueError("cannot bootstrap an empty evaluation")
    grouped = matches.groupby("match_boot_index", sort=True).agg(
        value=(value_column, "sum"), minutes=("active_minutes", "sum")
    )
    if len(grouped) < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(grouped), size=(resamples, len(grouped)))
    values = grouped["value"].to_numpy()[draws].sum(axis=1)
    minutes = grouped["minutes"].to_numpy()[draws].sum(axis=1)
    rates = values / minutes
    return float(np.percentile(rates, 5.0)), float(np.percentile(rates, 95.0))


def attach_match_metrics(
    evaluations: pd.DataFrame,
    raw: pd.DataFrame,
    *,
    bootstrap_resamples: int = DEFAULT_RHO_BOOTSTRAPS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute stored gameplay rates and return active match rows.

    Evaluations with no active match rows remain in the first returned table,
    marked ``has_active_matches=False``. The second table contains only active
    rows, which is the notebook's scientific gameplay population.
    """
    evaluation_keys = set(evaluations["evaluation_key"].astype(str))
    matches = _active_matches(raw, evaluation_keys)
    aggregate = matches.groupby("evaluation_key", sort=False).agg(
        active_matches=("evaluation_key", "size"),
        active_boots=("match_boot_index", "nunique"),
        active_frames=("match_active_frames", "sum"),
        stock_difference=("match_stock_difference", "sum"),
        damage_difference=("match_damage_difference", "sum"),
    )
    aggregate["recomputed_net_stock_per_min"] = aggregate["stock_difference"] / (
        aggregate["active_frames"] / FRAMES_PER_MINUTE
    )
    aggregate["recomputed_net_dmg_per_min"] = aggregate["damage_difference"] / (
        aggregate["active_frames"] / FRAMES_PER_MINUTE
    )
    result = evaluations.join(aggregate, on="evaluation_key")
    result["has_active_matches"] = result["active_matches"].notna()
    if (result["has_active_matches"] & ~result["protocol_hash_present"]).any():
        raise ValueError("active match rows belong to an evaluation without a matchup schedule hash")

    active = result["has_active_matches"]
    stock_ok = np.isclose(
        result.loc[active, "recomputed_net_stock_per_min"],
        result.loc[active, "eval_net_stock_per_min"],
        rtol=1e-9,
        atol=1e-9,
    )
    damage_ok = np.isclose(
        result.loc[active, "recomputed_net_dmg_per_min"],
        result.loc[active, "eval_net_dmg_per_min"],
        rtol=1e-9,
        atol=1e-9,
    )
    if not stock_ok.all() or not damage_ok.all():
        raise ValueError("stored evaluation rates do not match active match rows")

    ci_rows: list[dict[str, float | str]] = []
    for key, group in matches.groupby("evaluation_key", sort=True):
        stock_lo, stock_hi = bootstrap_pooled_rate(
            group,
            value_column="match_stock_difference",
            seed=_seed_from_key(str(key), salt=1),
            resamples=bootstrap_resamples,
        )
        damage_lo, damage_hi = bootstrap_pooled_rate(
            group,
            value_column="match_damage_difference",
            seed=_seed_from_key(str(key), salt=2),
            resamples=bootstrap_resamples,
        )
        ci_rows.append(
            {
                "evaluation_key": str(key),
                "recomputed_stock_ci_lo": stock_lo,
                "recomputed_stock_ci_hi": stock_hi,
                "recomputed_damage_ci_lo": damage_lo,
                "recomputed_damage_ci_hi": damage_hi,
            }
        )
    result = result.merge(pd.DataFrame(ci_rows), on="evaluation_key", how="left", validate="one_to_one")
    return result, matches


def _spearman(x_values: Iterable[float], y_values: Iterable[float]) -> float:
    x = pd.Series(list(x_values), dtype=float)
    y = pd.Series(list(y_values), dtype=float)
    finite = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return math.nan
    return float(np.corrcoef(x.rank(method="average"), y.rank(method="average"))[0, 1])


def bootstrap_spearman(
    frame: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    cluster_column: str,
    seed: int,
    resamples: int,
) -> tuple[float, float]:
    """Bootstrap a rank correlation over the supplied cluster identity."""
    values = frame[[x_column, y_column, cluster_column]].dropna().copy()
    if len(values) < 3:
        return math.nan, math.nan
    clusters = values[cluster_column].astype(str).to_numpy()
    names = np.unique(clusters)
    if len(names) < 3:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    sampled: list[float] = []
    for _ in range(resamples):
        selected = rng.choice(names, size=len(names), replace=True)
        indices = np.concatenate([np.flatnonzero(clusters == name) for name in selected])
        value = _spearman(values.iloc[indices][x_column], values.iloc[indices][y_column])
        if np.isfinite(value):
            sampled.append(value)
    if not sampled:
        return math.nan, math.nan
    return float(np.percentile(sampled, 5.0)), float(np.percentile(sampled, 95.0))


def correlation_table(
    evaluations: pd.DataFrame,
    *,
    include_diverged: bool,
    bootstrap_resamples: int = DEFAULT_RHO_BOOTSTRAPS,
) -> pd.DataFrame:
    """Measure NLL/gameplay rank agreement at matched scopes.

    Fixed-delay rows are the primary comparison. The pooled-delay row is
    included only to show the delay confound; it is not a selection metric.
    """
    frame = evaluations.loc[
        evaluations["has_active_matches"]
        & evaluations["matched_delay"]
        & (~evaluations["diverged"] | include_diverged)
    ].copy()
    rows: list[dict[str, Any]] = []

    def add(scope: str, label: str, subset: pd.DataFrame, delay: float | None) -> None:
        if len(subset) < 3:
            return
        for metric in NLL_COLUMNS:
            rho = _spearman(subset[metric], subset["recomputed_net_stock_per_min"])
            lo, hi = bootstrap_spearman(
                subset,
                x_column=metric,
                y_column="recomputed_net_stock_per_min",
                cluster_column="run_id",
                seed=_seed_from_key(f"{scope}:{label}:{metric}:{include_diverged}"),
                resamples=bootstrap_resamples,
            )
            rows.append(
                {
                    "scope": scope,
                    "group": label,
                    "delay_frames": delay,
                    "metric": metric,
                    "n_evaluations": len(subset),
                    "n_runs": subset["run_id"].nunique(),
                    "n_models": subset["model"].nunique(),
                    "rho": rho,
                    "rho_ci_lo": lo,
                    "rho_ci_hi": hi,
                    "diverged_included": include_diverged,
                }
            )

    add("pooled_delays", "all", frame, None)
    for delay, subset in frame.groupby("delay_frames", sort=True):
        add("fixed_delay", "all", subset, float(delay))
    for (model, delay), subset in frame.groupby(["model", "delay_frames"], sort=True):
        add("within_model_across_exposure", str(model), subset, float(delay))
    for (processed, delay), subset in frame.groupby(["processed_positions", "delay_frames"], sort=True):
        add("within_exposure_across_models", f"{processed:.0f}", subset, float(delay))
    return pd.DataFrame(rows)


def _fit_objective(
    parameters: np.ndarray,
    n_eff: np.ndarray,
    data_axis: np.ndarray,
    target: np.ndarray,
) -> float:
    log_e, log_a, log_b, alpha, beta = parameters
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        prediction = np.exp(log_e) + np.exp(log_a) * n_eff**-alpha + np.exp(log_b) * data_axis**-beta
    residual = prediction - target
    if not np.isfinite(residual).all():
        return float("inf")
    delta = 1e-3
    return float(
        np.where(
            np.abs(residual) < delta,
            0.5 * residual**2,
            delta * (np.abs(residual) - 0.5 * delta),
        ).sum()
    )


def _initial_parameters(max_starts: int) -> list[tuple[float, ...]]:
    starts: list[tuple[float, ...]] = [
        (math.log(e0), log_a, log_b, alpha0, beta0)
        for e0, alpha0, beta0, log_a, log_b in itertools.product(
            (1.2, 1.5, 1.7),
            (0.1, 0.3, 0.5),
            (0.1, 0.3, 0.5),
            (0.0, 3.0, 6.0),
            (0.0, 3.0, 6.0),
        )
    ]
    return starts[:max_starts]


def _optimum_for_fit(
    *,
    axis: str,
    asymptote: float,
    capacity_coefficient: float,
    data_coefficient: float,
    capacity_exponent: float,
    data_exponent: float,
    n_eff_min: float,
    n_eff_max: float,
    compute: float,
    corpus_positions: float,
) -> tuple[float, float, float]:
    if min(asymptote, capacity_coefficient, data_coefficient, capacity_exponent, data_exponent) <= 0:
        raise ValueError("scaling fit cannot extrapolate with non-positive parameters")

    if axis == "processed_positions":
        n_eff = (capacity_exponent * capacity_coefficient / (data_exponent * data_coefficient)) ** (
            1.0 / (capacity_exponent + data_exponent)
        ) * (compute / 6.0) ** (data_exponent / (capacity_exponent + data_exponent))
    elif axis == "distinct_positions":
        lower = math.log(n_eff_min / 32.0)
        upper = math.log(n_eff_max * 32.0)

        def loss(log_n_eff: float) -> float:
            value = math.exp(log_n_eff)
            processed = compute / (6.0 * value)
            covered = float(distinct_positions(np.asarray([processed]), corpus_positions)[0])
            return (
                asymptote
                + capacity_coefficient * value**-capacity_exponent
                + data_coefficient * covered**-data_exponent
            )

        result = scipy.optimize.minimize_scalar(loss, bounds=(lower, upper), method="bounded")
        if not result.success:
            raise ValueError(f"coverage optimum failed: {result.message}")
        n_eff = math.exp(float(result.x))
    else:
        raise ValueError(f"unsupported scaling data axis {axis!r}")
    processed = compute / (6.0 * n_eff)
    data_value = (
        processed
        if axis == "processed_positions"
        else float(distinct_positions(np.asarray([processed]), corpus_positions)[0])
    )
    prediction = (
        asymptote + capacity_coefficient * n_eff**-capacity_exponent + data_coefficient * data_value**-data_exponent
    )
    return n_eff, processed, prediction


def fit_scaling_law(
    frame: pd.DataFrame,
    *,
    data_axis: str = "processed_positions",
    max_starts: int = 243,
    initial: Sequence[float] | None = None,
    target_compute: float = TARGET_COMPUTE_FLOPS,
) -> ScalingFit:
    """Fit the notebook's robust scaling law and extrapolate one compute point."""
    required = {"training_flops", "processed_positions", "validation_nll", "n_eff", data_axis}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"scaling fit is missing columns {missing}")
    selected = frame[list(required)].dropna().copy()
    if len(selected) < 5:
        raise ValueError(f"scaling fit needs at least five points, got {len(selected)}")
    n_eff = selected["n_eff"].to_numpy(float)
    data_values = selected[data_axis].to_numpy(float)
    target = selected["validation_nll"].to_numpy(float)
    if (
        not np.isfinite(np.column_stack((n_eff, data_values, target))).all()
        or (n_eff <= 0).any()
        or (data_values <= 0).any()
    ):
        raise ValueError("scaling fit inputs must be finite and positive")

    starts = [tuple(float(value) for value in initial)] if initial is not None else _initial_parameters(max_starts)
    best: scipy.optimize.OptimizeResult | None = None
    for start in starts:
        result = scipy.optimize.minimize(
            _fit_objective,
            np.asarray(start, dtype=float),
            args=(n_eff, data_values, target),
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-12},
        )
        if np.isfinite(result.fun) and (best is None or result.fun < best.fun):
            best = result
    if best is None:
        raise ValueError("all scaling-law fits failed")

    log_e, log_a, log_b, alpha, beta = (float(value) for value in best.x)
    asymptote, capacity_coefficient, data_coefficient = math.exp(log_e), math.exp(log_a), math.exp(log_b)
    corpus_values = frame["unique_loss_positions"].dropna().unique()
    if len(corpus_values) != 1:
        raise ValueError(f"fit frame has inconsistent corpus sizes: {corpus_values.tolist()}")
    n_opt, processed_opt, predicted_nll = _optimum_for_fit(
        axis=data_axis,
        asymptote=asymptote,
        capacity_coefficient=capacity_coefficient,
        data_coefficient=data_coefficient,
        capacity_exponent=alpha,
        data_exponent=beta,
        n_eff_min=float(n_eff.min()),
        n_eff_max=float(n_eff.max()),
        compute=target_compute,
        corpus_positions=float(corpus_values[0]),
    )
    return ScalingFit(
        axis=data_axis,
        n_points=len(selected),
        models=tuple(sorted(frame["model"].dropna().astype(str).unique())),
        objective=float(best.fun),
        asymptote=asymptote,
        capacity_coefficient=capacity_coefficient,
        data_coefficient=data_coefficient,
        capacity_exponent=alpha,
        data_exponent=beta,
        n_eff_min=float(n_eff.min()),
        n_eff_max=float(n_eff.max()),
        data_min=float(data_values.min()),
        data_max=float(data_values.max()),
        compute_min=float(selected["training_flops"].min()),
        compute_max=float(selected["training_flops"].max()),
        n_eff_opt=n_opt,
        processed_opt=processed_opt,
        predicted_nll=predicted_nll,
    )


def fit_sensitivity(
    runs: pd.DataFrame,
    *,
    target_compute: float = TARGET_COMPUTE_FLOPS,
) -> pd.DataFrame:
    """Fit the full scaled family and cheap width/axis sensitivity variants."""
    healthy = runs.loc[runs["model_family"].eq("scaled") & ~runs["diverged"]].copy()
    full = fit_scaling_law(healthy, max_starts=243, target_compute=target_compute)
    rows: list[dict[str, Any]] = []

    def add(label: str, subset: pd.DataFrame, fit: ScalingFit, status: str = "ok") -> None:
        row = asdict(fit)
        row.update({"variant": label, "status": status})
        row["models"] = ",".join(fit.models)
        rows.append(row)

    add("full_scaled", healthy, full)
    for model in FIT_MODELS:
        subset = healthy.loc[healthy["model"].ne(model)]
        if len(subset) < 5:
            continue
        try:
            fit = fit_scaling_law(
                subset,
                initial=full.parameter_vector,
                max_starts=1,
                target_compute=target_compute,
            )
        except ValueError as error:
            rows.append({"variant": f"leave_out_{model}", "status": str(error)})
        else:
            add(f"leave_out_{model}", subset, fit)

    powers = {2.0**exponent for exponent in (26, 27, 28, 29, 30)}
    standard = healthy.loc[healthy["processed_positions"].isin(powers)]
    if len(standard) >= 5:
        add(
            "standard_power_two_only",
            standard,
            fit_scaling_law(standard, initial=full.parameter_vector, max_starts=1, target_compute=target_compute),
        )
    add(
        "include_026base",
        runs.loc[~runs["diverged"]],
        fit_scaling_law(
            runs.loc[~runs["diverged"]],
            initial=full.parameter_vector,
            max_starts=1,
            target_compute=target_compute,
        ),
    )
    add(
        "distinct_coverage_axis",
        healthy,
        fit_scaling_law(
            healthy,
            data_axis="distinct_positions",
            initial=full.parameter_vector,
            max_starts=1,
            target_compute=target_compute,
        ),
    )
    all_scaled = runs.loc[runs["model_family"].eq("scaled")]
    add(
        "include_diverged_L10_endpoint",
        all_scaled,
        fit_scaling_law(
            all_scaled,
            initial=full.parameter_vector,
            max_starts=1,
            target_compute=target_compute,
        ),
    )
    return pd.DataFrame(rows)


def bootstrap_scaling_optimum(
    runs: pd.DataFrame,
    *,
    fit: ScalingFit,
    repetitions: int,
    seed: int,
    target_compute: float = TARGET_COMPUTE_FLOPS,
) -> pd.DataFrame:
    """Bootstrap the scaling fit by model curve, not by individual matches."""
    frame = runs.loc[runs["model_family"].eq("scaled") & ~runs["diverged"]].copy()
    models = np.array(sorted(frame["model"].unique()), dtype=object)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    for index in range(repetitions):
        sampled = rng.choice(models, size=len(models), replace=True)
        pieces = [frame.loc[frame["model"].eq(model)] for model in sampled]
        bootstrap_frame = pd.concat(pieces, ignore_index=True)
        try:
            result = fit_scaling_law(
                bootstrap_frame,
                initial=fit.parameter_vector,
                max_starts=1,
                target_compute=target_compute,
            )
        except ValueError:
            continue
        rows.append(
            {
                "replicate": index,
                "n_eff_opt": result.n_eff_opt,
                "processed_opt": result.processed_opt,
                "predicted_nll": result.predicted_nll,
            }
        )
    return pd.DataFrame(rows)


def _standardized_regression(
    frame: pd.DataFrame,
    *,
    response: str,
    predictors: Sequence[str],
) -> pd.DataFrame:
    columns = [response, *predictors]
    values = frame[columns].dropna().to_numpy(float)
    if len(values) < len(predictors) + 2:
        return pd.DataFrame()
    y = values[:, 0]
    x = values[:, 1:]
    scales = x.std(axis=0)
    if (scales == 0).any():
        raise ValueError(f"cannot standardize constant predictors {predictors}")
    x = (x - x.mean(axis=0)) / scales
    design = np.column_stack((np.ones(len(x)), x))
    coefficient = np.linalg.lstsq(design, y, rcond=None)[0]
    predicted = design @ coefficient
    total = ((y - y.mean()) ** 2).sum()
    r_squared = math.nan if total == 0 else 1.0 - ((y - predicted) ** 2).sum() / total
    rows = [
        {
            "response": response,
            "predictor": "intercept",
            "coefficient": float(coefficient[0]),
            "n": len(values),
            "condition_number": float(np.linalg.cond(design)),
            "r_squared": float(r_squared),
        }
    ]
    rows.extend(
        {
            "response": response,
            "predictor": predictor,
            "coefficient": float(value),
            "n": len(values),
            "condition_number": float(np.linalg.cond(design)),
            "r_squared": float(r_squared),
        }
        for predictor, value in zip(predictors, coefficient[1:], strict=True)
    )
    return pd.DataFrame(rows)


def confound_tables(
    runs: pd.DataFrame, evaluations: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Return rank confounds, fixed-delay regression, and objective checks."""
    healthy = runs.loc[~runs["diverged"]].copy()
    healthy["log2_n_eff"] = np.log2(healthy["n_eff"])
    healthy["log2_processed_positions"] = np.log2(healthy["processed_positions"])
    columns = [
        "validation_nll",
        "validation_primary_nll",
        "validation_auxiliary_nll",
        "L",
        "n_eff",
        "processed_positions",
        "training_flops",
    ]
    matrix = healthy[columns].corr(method="spearman")
    pair_rows = [
        {"left": left, "right": right, "rho": float(matrix.loc[left, right])}
        for left, right in itertools.combinations(columns, 2)
    ]
    regression_rows = _standardized_regression(
        healthy,
        response="validation_nll",
        predictors=("log2_n_eff", "log2_processed_positions"),
    )
    fixed_delay = evaluations.loc[
        evaluations["has_active_matches"] & evaluations["delay_frames"].eq(4) & ~evaluations["diverged"]
    ].copy()
    fixed_delay["log2_n_eff"] = np.log2(fixed_delay["n_eff"])
    fixed_delay["log2_processed_positions"] = np.log2(fixed_delay["processed_positions"])
    gameplay_regression = _standardized_regression(
        fixed_delay,
        response="recomputed_net_stock_per_min",
        predictors=(
            "validation_primary_nll",
            "validation_auxiliary_nll",
            "log2_n_eff",
            "log2_processed_positions",
        ),
    )
    regressions = pd.concat([regression_rows, gameplay_regression], ignore_index=True)

    objective_error = healthy["validation_nll"] - (
        healthy["validation_primary_nll"] + 0.5 * healthy["validation_auxiliary_nll"]
    )
    contribution = 0.5 * healthy["validation_auxiliary_nll"] / healthy["validation_nll"]
    checks = {
        "objective_max_abs_error": float(objective_error.abs().max()),
        "auxiliary_weight": 0.5,
        "auxiliary_contribution_mean": float(contribution.mean()),
        "auxiliary_contribution_min": float(contribution.min()),
        "auxiliary_contribution_max": float(contribution.max()),
        "fixed_delay_4_n": float(len(fixed_delay)),
        "fixed_delay_4_condition_number": float(
            regressions.loc[
                (regressions["response"] == "recomputed_net_stock_per_min") & regressions["predictor"].eq("intercept"),
                "condition_number",
            ].iloc[0]
        )
        if not regressions.loc[
            (regressions["response"] == "recomputed_net_stock_per_min") & regressions["predictor"].eq("intercept")
        ].empty
        else math.nan,
    }
    return pd.DataFrame(pair_rows), regressions, checks


def _finite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_finite(value), indent=2, sort_keys=True) + "\n")


def _plot_metric_comparison(evaluations: pd.DataFrame, path: Path) -> None:
    frame = evaluations.loc[
        evaluations["has_active_matches"] & evaluations["delay_frames"].eq(4) & ~evaluations["diverged"]
    ].copy()
    if frame.empty:
        return
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.5), sharey=True)
    colors = {
        model: color
        for model, color in zip(
            sorted(frame["model"].unique()),
            matplotlib.colormaps["viridis"](np.linspace(0.1, 0.9, frame["model"].nunique())),
            strict=True,
        )
    }
    for axis, metric in zip(axes, NLL_COLUMNS, strict=True):
        for model, subset in frame.groupby("model", sort=True):
            axis.scatter(
                subset[metric],
                subset["recomputed_net_stock_per_min"],
                color=colors[model],
                label=model,
                alpha=0.8,
                s=26,
            )
        axis.set_xlabel(metric.removeprefix("validation_").replace("_", " "))
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("net stocks/min")
    axes[-1].legend(fontsize=8, loc="best")
    figure.suptitle("O39 NLL metrics versus gameplay at fixed delay d=4")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_optimum_sensitivity(frame: pd.DataFrame, fit_bootstrap: pd.DataFrame, path: Path) -> None:
    usable = frame.loc[frame["status"].eq("ok") & frame["n_eff_opt"].notna()].copy()
    if usable.empty:
        return
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
    labels = list(usable["variant"])
    positions = np.arange(len(labels))
    axes[0].scatter(positions, usable["n_eff_opt"], color="#245a9b")
    axes[1].scatter(positions, usable["processed_opt"], color="#c85d2e")
    for axis, _values, ylabel in (
        (axes[0], usable["n_eff_opt"], "N_eff at C = 4.2e19"),
        (axes[1], usable["processed_opt"], "processed D at C = 4.2e19"),
    ):
        axis.set_yscale("log")
        axis.set_xticks(positions, labels, rotation=70, ha="right", fontsize=7)
        axis.set_ylabel(ylabel)
        axis.grid(True, axis="y", alpha=0.25)
    if not fit_bootstrap.empty:
        for axis, column in ((axes[0], "n_eff_opt"), (axes[1], "processed_opt")):
            axis.axhspan(
                fit_bootstrap[column].quantile(0.05),
                fit_bootstrap[column].quantile(0.95),
                color="#8aa6c5",
                alpha=0.2,
                label="model-cluster bootstrap 90%",
            )
    figure.suptitle("Sensitivity of the extrapolated O39 compute optimum")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _model_summary(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, group in runs.groupby("model", sort=True):
        rows.append(
            {
                "model": model,
                "model_family": group["model_family"].iloc[0],
                "L": int(group["L"].iloc[0]),
                "total_parameters": int(group["total_parameters"].iloc[0]),
                "n_eff": float(group["n_eff"].iloc[0]),
                "endpoint_count": len(group),
                "diverged_count": int(group["diverged"].sum()),
                "processed_min": float(group["processed_positions"].min()),
                "processed_max": float(group["processed_positions"].max()),
                "compute_min": float(group["training_flops"].min()),
                "compute_max": float(group["training_flops"].max()),
            }
        )
    return pd.DataFrame(rows)


def run_audit(
    csv_path: Path,
    output_dir: Path,
    *,
    fit_bootstraps: int = DEFAULT_FIT_BOOTSTRAPS,
    rho_bootstraps: int = DEFAULT_RHO_BOOTSTRAPS,
) -> dict[str, Any]:
    """Run the complete audit and write machine-readable artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_raw(csv_path)
    runs = select_completed_runs(raw)
    evaluations = select_evaluations(raw, runs)
    evaluations, matches = attach_match_metrics(
        evaluations,
        raw,
        bootstrap_resamples=rho_bootstraps,
    )
    correlations_healthy = correlation_table(
        evaluations,
        include_diverged=False,
        bootstrap_resamples=rho_bootstraps,
    )
    correlations_all = correlation_table(
        evaluations,
        include_diverged=True,
        bootstrap_resamples=rho_bootstraps,
    )
    confounds, regressions, objective_checks = confound_tables(runs, evaluations)
    fit_table = fit_sensitivity(runs)
    healthy_scaled = runs.loc[runs["model_family"].eq("scaled") & ~runs["diverged"]]
    full_fit = fit_scaling_law(healthy_scaled)
    fit_bootstrap = bootstrap_scaling_optimum(
        runs,
        fit=full_fit,
        repetitions=fit_bootstraps,
        seed=20260911,
    )

    run_columns = [
        "run_id",
        "model",
        "model_family",
        "L",
        "phase",
        "endpoint_exact",
        "analysis_excluded",
        "analysis_exclusion_reason",
        "wandb_state",
        "processed_positions",
        "target_processed_positions",
        "training_flops",
        "total_parameters",
        "trunk_parameters",
        "decoder_parameters",
        "unique_data_divisor",
        "unique_loss_positions",
        "unique_replays",
        "episode_hash",
        "validation_nll",
        "validation_primary_nll",
        "validation_auxiliary_nll",
        "n_eff",
        "distinct_positions",
        "best_prior_nll",
        "diverged",
    ]
    runs[run_columns].to_csv(output_dir / "completed_endpoints.csv", index=False)
    eval_columns = [
        "evaluation_key",
        "run_id",
        "model",
        "model_family",
        "L",
        "processed_positions",
        "training_flops",
        "n_eff",
        "distinct_positions",
        "delay_frames",
        "eval_control_delay",
        "eval_replan_interval",
        "eval_decode_horizon",
        "eval_completed_boots",
        "eval_scheduled_boots",
        "eval_crashed",
        "eval_matches",
        "eval_zero_active",
        "eval_valid_latency_bucket",
        "protocol_hash_present",
        "protocol_matchup_schedule_sha256",
        "protocol_checkpoint_sha256",
        "validation_nll",
        "validation_primary_nll",
        "validation_auxiliary_nll",
        "diverged",
        "has_active_matches",
        "active_matches",
        "active_boots",
        "active_frames",
        "recomputed_net_stock_per_min",
        "recomputed_net_dmg_per_min",
        "recomputed_stock_ci_lo",
        "recomputed_stock_ci_hi",
        "recomputed_damage_ci_lo",
        "recomputed_damage_ci_hi",
    ]
    evaluations[eval_columns].to_csv(output_dir / "complete_evaluations.csv", index=False)
    matches[
        [
            "evaluation_key",
            "match_boot_index",
            "match_active_frames",
            "active_minutes",
            "match_stock_difference",
            "match_damage_difference",
        ]
    ].to_csv(output_dir / "active_match_rows.csv", index=False)
    correlations_healthy.to_csv(output_dir / "correlations_healthy.csv", index=False)
    correlations_all.to_csv(output_dir / "correlations_all.csv", index=False)
    confounds.to_csv(output_dir / "run_confound_correlations.csv", index=False)
    regressions.to_csv(output_dir / "regressions.csv", index=False)
    fit_table.to_csv(output_dir / "fit_sensitivity.csv", index=False)
    fit_bootstrap.to_csv(output_dir / "fit_model_cluster_bootstrap.csv", index=False)
    _model_summary(runs).to_csv(output_dir / "model_summary.csv", index=False)

    _plot_metric_comparison(evaluations, output_dir / "nll_metric_comparison_d4.png")
    _plot_optimum_sensitivity(fit_table, fit_bootstrap, output_dir / "optimum_sensitivity.png")

    schedule_counts = evaluations.groupby("delay_frames")["protocol_matchup_schedule_sha256"].nunique().to_dict()
    corpus_values = runs["unique_loss_positions"].dropna().unique()
    raw_counts = raw["record_type"].value_counts().to_dict()
    summary: dict[str, Any] = {
        "schema_version": 1,
        "source": {
            "path": str(csv_path.resolve()),
            "sha256": file_sha256(csv_path),
            "bytes": csv_path.stat().st_size,
            "rows_loaded": len(raw),
            "record_counts": {str(key): int(value) for key, value in raw_counts.items()},
        },
        "selection": {
            "completed_exact_runs": int(len(runs)),
            "diverged_runs": int(runs["diverged"].sum()),
            "healthy_fit_runs": int(len(healthy_scaled)),
            "excluded_run_records": int(raw.loc[raw["record_type"].eq("run"), "analysis_excluded"].eq(True).sum()),
            "complete_evaluations_before_active_join": int(len(evaluations)),
            "evaluations_with_active_match_rows": int(evaluations["has_active_matches"].sum()),
            "evaluations_without_active_match_rows": int((~evaluations["has_active_matches"]).sum()),
            "active_match_rows": int(len(matches)),
            "active_boot_clusters": int(matches["match_boot_index"].nunique()),
            "complete_evaluation_delays": sorted(float(value) for value in evaluations["delay_frames"].unique()),
            "schedule_hashes_by_delay": {str(key): int(value) for key, value in schedule_counts.items()},
            "schedule_hash_count": int(evaluations["protocol_matchup_schedule_sha256"].nunique()),
            "checkpoint_hash_count": int(evaluations["protocol_checkpoint_sha256"].nunique()),
            "unique_data_divisors": sorted(float(value) for value in runs["unique_data_divisor"].unique()),
            "episode_hash_count": int(runs["episode_hash"].nunique()),
            "corpus_positions": float(corpus_values[0]) if len(corpus_values) == 1 else None,
        },
        "metric_definition": {
            "gameplay_metric": "recomputed_net_stock_per_min",
            "match_population": "match_active_frames > 0",
            "bootstrap_cluster": "match_boot_index within evaluation_key",
            "validation_objective": "validation_primary_nll + 0.5 * validation_auxiliary_nll",
            "objective_checks": objective_checks,
            "deployment_latency_is_filter": False,
            "deployment_latency_column": "eval_valid_latency_bucket",
        },
        "fit": asdict(full_fit),
        "fit_bootstrap": {
            "cluster": "model curve",
            "requested_repetitions": fit_bootstraps,
            "successful_repetitions": int(len(fit_bootstrap)),
            "n_eff_opt_p05": float(fit_bootstrap["n_eff_opt"].quantile(0.05)) if not fit_bootstrap.empty else None,
            "n_eff_opt_p95": float(fit_bootstrap["n_eff_opt"].quantile(0.95)) if not fit_bootstrap.empty else None,
            "processed_opt_p05": float(fit_bootstrap["processed_opt"].quantile(0.05))
            if not fit_bootstrap.empty
            else None,
            "processed_opt_p95": float(fit_bootstrap["processed_opt"].quantile(0.95))
            if not fit_bootstrap.empty
            else None,
        },
    }
    _write_json(output_dir / "audit_summary.json", summary)
    print(
        f"O39 audit: {len(runs)} exact runs ({len(healthy_scaled)} healthy scaled fit rows), "
        f"{len(evaluations)} complete evaluations, {len(matches)} active match rows."
    )
    print(
        f"Fit at C={TARGET_COMPUTE_FLOPS:.2e}: N_eff={full_fit.n_eff_opt:.3e}, "
        f"D={full_fit.processed_opt:.3e}, NLL={full_fit.predicted_nll:.4f}."
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=SOURCE_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fit-bootstraps", type=int, default=DEFAULT_FIT_BOOTSTRAPS)
    parser.add_argument("--rho-bootstraps", type=int, default=DEFAULT_RHO_BOOTSTRAPS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.fit_bootstraps < 10 or args.rho_bootstraps < 10:
        raise ValueError("bootstrap counts must be at least 10")
    run_audit(
        args.csv,
        args.output_dir,
        fit_bootstraps=args.fit_bootstraps,
        rho_bootstraps=args.rho_bootstraps,
    )


if __name__ == "__main__":
    main()
