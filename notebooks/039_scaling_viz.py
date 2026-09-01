# %% [markdown]
# # Experiment 039 — capacity, data, and delay scaling
#
# Figures written to `results/039_capacity_scaling/figures/`:
#
# | file | view |
# |---|---|
# | `01_deployable_pareto.png` | capability per delay at fixed D, RTX-3060-feasible points only |
# | `02_iso_param_D.png` | iso-param: capability and NLL against processed positions D |
# | `03_iso_param_U.png` | iso-param: the same against distinct positions covered U |
# | `04_iso_data_capacity.png` | iso-data: capability and NLL against parameters |
# | `05_capability_grid.png` | model x delay grid with feasibility and coverage |
# | `06_nll_vs_closed_loop.png` | does validation NLL predict closed-loop capability |
# | `07_delay_sensitivity_latency.png` | per-model delay sensitivity, measured 3060 latency vs frame budget |
# | `08_compute_frontier.png` | compute frontier and iso-FLOP profiles |
# | `09_isoflop_fit.png` | parametric fit and compute-optimal capacity |
# | `10_wallclock_frontier.png` | capability and NLL against measured training wall time |
# | `11_match_distributions.png` | per-match outcome distributions and per-character breakdown |
#
# Closed-loop metric: the rate against the CPU per minute of active play, read three ways from
# the same matches, each with a 90% CI from a bootstrap over boot clusters.
#
# - `median_rate` — median over matches of the per-match net stock rate. Net stocks is THE
#   metric, and the median is the robust central estimate. It is also coarse: matches are
#   capped at 7077 active frames (1.97 min) and only end early on a 4-stock loss, so full-clock
#   matches hold the middle of the distribution and the median usually lands on an integer
#   stock difference over that clock — a multiple of 0.51/min. That holds for 56% of all
#   evaluations and for 95% of the best-scoring ones.
# - `damage_rate` — the same median over per-match net damage. Damage is continuous, so every
#   evaluation gets a distinct value. It is a diagnostic, never the ranking metric; it breaks
#   ties the stock median cannot.
# - `pooled_rate` — total stocks over total active minutes, the ratio estimator the run
#   summaries use. Full resolution, and what the scaling panels plot.
#
# Two data axes are kept apart. **D** is the number of positions the optimizer processed. **U**
# is the number of *distinct* positions covered: the corpus holds 2.41e9 loss positions and the
# window sampler re-draws windows on every pass, so a run at D = 2^30 touches fewer distinct
# positions than D suggests.

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import melee
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar

RESULTS = Path(__file__).resolve().parent.parent / "results" / "039_capacity_scaling"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(exist_ok=True)

FRAME_MS = 1000.0 / 60.0
FRAMES_PER_MINUTE = 3600.0
BOOTSTRAP_RESAMPLES = 2000

raw = pd.read_csv(RESULTS / "experiment_039_raw_outcomes.csv", low_memory=False)

# %% [markdown]
# ## Runs
#
# One row per (model, D) checkpoint. Cooldown rows carry the comparable NLL; prefix rows are
# pre-anneal and only contribute wall time. Rows without `processed_positions` are aborted
# launches and drop out. `D_exp` is absent on the exact-iso-FLOP endpoints (L3, L4, and the
# 1e18-FLOP L13), so every panel keys on `processed_positions` and treats D as continuous.

# %%
runs = raw[raw.record_type == "run"].copy()
runs = runs[runs.processed_positions.notna()]
runs = runs.drop_duplicates(subset=["model", "processed_positions", "phase"], keep="first")
runs["analysis_excluded"] = runs.analysis_excluded.eq(True)

U_TOTAL = float(runs.unique_loss_positions.iloc[0])
assert (runs.unique_loss_positions == U_TOTAL).all(), "runs disagree on corpus size"
assert (runs.unique_data_divisor == 1).all(), "a U-divisor run is present; the U axis needs rescaling"


def distinct_positions(d: np.ndarray | float) -> np.ndarray | float:
    """Distinct loss positions covered after processing ``d`` positions.

    Windows are re-drawn on every pass over the replay list, so coverage follows the
    coupon-collector curve rather than D itself. Exact only in the with-replacement limit;
    within a single pass the sampler draws non-overlapping windows, so this is a lower bound.
    """
    return U_TOTAL * (1.0 - np.exp(-np.asarray(d, dtype=float) / U_TOTAL))


runs["U_seen"] = distinct_positions(runs.processed_positions)

params_by_model = runs.groupby("model").total_parameters.first()
MODELS = list(params_by_model.sort_values().index)
# 026base is the frozen 026 architecture, not a member of the scaled family; it sits at the same
# size as L5 and enters every capacity panel as a reference marker instead of a curve point.
SCALED = [m for m in MODELS if m != "026base"]
DELAYS = sorted(raw[raw.record_type == "evaluation"].delay_frames.dropna().unique())
D_EXPS = [26, 27, 28, 29, 30]

# Cooldown checkpoints carry the comparable NLL. A cooldown whose NLL is worse than the same
# model's least-trained checkpoint diverged. A diverged run measures a failed optimization, not
# the capacity or data level it was trained at, so it leaves every scaling curve — closed-loop
# as much as NLL — and the fit. Failed cooldown checkpoints plot as rings; failed prefixes are
# reported in the compute-figure footers because they have no comparable terminal NLL.
failed_runs = runs[runs.analysis_excluded].copy()
ckpt = runs[(runs.phase == "cooldown") & runs.validation_nll.notna() & ~runs.analysis_excluded].copy()
ckpt = ckpt.drop_duplicates(subset=["model", "processed_positions"], keep="first")
floor = ckpt.loc[ckpt.groupby("model").processed_positions.idxmin()].set_index("model").validation_nll
ckpt["diverged"] = ckpt.validation_nll > ckpt.model.map(floor)
DIVERGED = {(row.model, row.processed_positions) for row in ckpt[ckpt.diverged].itertuples()} | {
    (row.model, row.processed_positions) for row in failed_runs.itertuples()
}
for model, positions in sorted(DIVERGED):
    print(f"diverged run (off every scaling curve): {model} stopped near D=2^{np.log2(positions):.2f}")

if not failed_runs.empty:
    print("unfinished exact endpoints:")
    for row in failed_runs.sort_values("total_parameters").itertuples():
        print(
            f"  {row.model}: stopped at D={row.processed_positions:,.0f}; "
            f"target D={row.target_processed_positions:,.0f}; {row.analysis_exclusion_reason}"
        )


def failed_endpoint_note() -> str:
    """Short figure note for exact endpoints that failed before completion."""
    notes = []
    for row in failed_runs.sort_values("total_parameters").itertuples():
        target = f"{row.target_processed_positions:.3g}"
        note = f"{row.model} target D={target} diverged at D={row.processed_positions:.3g}"
        if row.model == "L7" and np.isclose(row.target_processed_positions, 3_803_727_888):
            ratio = row.target_processed_positions / U_TOTAL
            coverage = distinct_positions(row.target_processed_positions) / U_TOTAL
            note += f" (target D/U={ratio:.3f}, estimated distinct coverage={coverage:.1%})"
        notes.append(note)
    return "; ".join(notes)


# Wall time for the time frontier: the cooldown row's cumulative seconds already includes its
# prefix branch.
ckpt["wall_hours"] = ckpt.training_cumulative_wall_seconds / 3600.0

# %% [markdown]
# ## Latency
#
# Feasible = the measured RTX 3060 p95 inference latency fits that delay's frame budget.
# A model with no benchmark is *unknown*, never assumed feasible.

# %%
latency = raw[raw.record_type == "latency_benchmark"].copy()
latency = latency.drop_duplicates(subset=["model", "delay_frames"], keep="first")
feasible = latency.set_index(["model", "delay_frames"]).local_latency_valid_bucket.astype(bool)
p95_ms = latency.set_index(["model", "delay_frames"]).local_latency_latency_p95_ms
BENCHMARKED = sorted(latency.model.unique(), key=lambda m: params_by_model[m])
UNBENCHMARKED = [m for m in MODELS if m not in BENCHMARKED]
print(f"latency benchmarked: {BENCHMARKED}")
print(f"no 3060 benchmark (excluded from the feasible frontier): {UNBENCHMARKED}")

# Cross-check the precomputed frontier file over the benchmarked models.
frontier_file = pd.read_csv(RESULTS / "latency_capacity_frontier.csv")
for row in frontier_file.itertuples():
    eligible = set(row.eligible_models.split(","))
    measured = {m for m in BENCHMARKED if feasible.get((m, float(row.delay)), False)}
    assert eligible == measured, f"delay {row.delay}: frontier file {eligible} != benchmark {measured}"

# %% [markdown]
# ## Closed-loop metrics

# %%
matches = raw[raw.record_type == "match"].copy()
matches = matches[matches.match_active_frames > 0]
active_minutes = matches.match_active_frames / FRAMES_PER_MINUTE
matches["rate"] = matches.match_stock_difference / active_minutes
matches["damage_rate"] = matches.match_damage_difference / active_minutes


def eval_location(group: pd.DataFrame, seed: int) -> dict[str, float]:
    """Central estimates of one evaluation's rates, each with a 90% CI.

    All three intervals resample boot clusters, so games from one Dolphin process stay together
    — the same clustering the run summaries' LCB uses.
    """
    cluster = group.match_boot_index.to_numpy()
    order = np.argsort(cluster, kind="stable")
    rate = group.rate.to_numpy()[order]
    damage_rate = group.damage_rate.to_numpy()[order]
    stocks = group.match_stock_difference.to_numpy()[order]
    minutes = (group.match_active_frames / FRAMES_PER_MINUTE).to_numpy()[order]
    _, starts, sizes = np.unique(cluster[order], return_index=True, return_counts=True)

    def pad(values: np.ndarray) -> np.ndarray:
        out = np.full((len(starts), int(sizes.max())), np.nan)
        for i, (start, size) in enumerate(zip(starts, sizes)):
            out[i, :size] = values[start : start + size]
        return out

    padded, padded_damage = pad(rate), pad(damage_rate)
    cluster_stocks = np.add.reduceat(stocks, starts)
    cluster_minutes = np.add.reduceat(minutes, starts)

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(starts), size=(BOOTSTRAP_RESAMPLES, len(starts)))
    median_draws = np.nanmedian(padded[draws].reshape(BOOTSTRAP_RESAMPLES, -1), axis=1)
    damage_draws = np.nanmedian(padded_damage[draws].reshape(BOOTSTRAP_RESAMPLES, -1), axis=1)
    pooled_draws = cluster_stocks[draws].sum(axis=1) / cluster_minutes[draws].sum(axis=1)
    return {
        "median_rate": float(np.median(rate)),
        "median_ci_lo": float(np.percentile(median_draws, 5.0)),
        "median_ci_hi": float(np.percentile(median_draws, 95.0)),
        "damage_rate": float(np.median(damage_rate)),
        "damage_ci_lo": float(np.percentile(damage_draws, 5.0)),
        "damage_ci_hi": float(np.percentile(damage_draws, 95.0)),
        "pooled_rate": float(stocks.sum() / minutes.sum()),
        "pooled_ci_lo": float(np.percentile(pooled_draws, 5.0)),
        "pooled_ci_hi": float(np.percentile(pooled_draws, 95.0)),
    }


rows = []
for i, (key, group) in enumerate(matches.groupby("evaluation_key", sort=True)):
    rows.append(
        {
            "evaluation_key": key,
            **eval_location(group, seed=i),
            "n_matches": len(group),
            "n_boots": int(group.match_boot_index.nunique()),
            "win_rate": float((group.match_stock_difference > 0).mean()),
            "frac_shutout": float((group.match_stock_difference <= -4).mean()),
        }
    )
location = pd.DataFrame(rows).set_index("evaluation_key")

evals = raw[raw.record_type == "evaluation"].copy()
evals = evals.drop_duplicates(subset=["evaluation_key"], keep="first")
evals = evals.join(location, on="evaluation_key")
for row in evals[evals.median_rate.isna()].itertuples():
    print(
        f"evaluation without match rows (dropped): {row.model} "
        f"D=2^{np.log2(row.processed_positions):.2f} d{row.delay_frames:.0f}"
    )
evals = evals[evals.median_rate.notna()]

evals["parameters"] = evals.model.map(params_by_model)
evals["U_seen"] = distinct_positions(evals.processed_positions)
evals["feasible"] = [feasible.get((m, d), False) for m, d in zip(evals.model, evals.delay_frames)]
evals["diverged"] = [(m, p) in DIVERGED for m, p in zip(evals.model, evals.processed_positions)]
for column in ("validation_nll", "training_flops", "wall_hours"):
    lookup = ckpt.set_index(["model", "processed_positions"])[column]
    evals[column] = [lookup.get((m, p), np.nan) for m, p in zip(evals.model, evals.processed_positions)]


def best_by(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Best row per group: highest stock median, ties broken by the damage median."""
    return frame.sort_values(["median_rate", "damage_rate"]).groupby(keys, sort=False).tail(1).copy()


best_over_delay = best_by(evals, ["model", "processed_positions"])
best_over_d = best_by(evals, ["model", "delay_frames"])
feasible_evals = evals[evals.feasible]
best_over_d_feasible = best_by(feasible_evals, ["model", "delay_frames"])

print(f"\n{len(evals)} evaluations, {len(ckpt)} cooldown checkpoints, {len(matches)} matches")
print(
    f"corpus U = {U_TOTAL:.3e} loss positions; "
    f"largest run covers {distinct_positions(runs.processed_positions.max()) / U_TOTAL:.1%}"
)

# %% [markdown]
# ## Coverage
#
# What exists and what is missing, so no panel can quietly hide a hole.

# %%
coverage = evals.pivot_table(
    index=["model", "processed_positions"], columns="delay_frames", values="median_rate", aggfunc="size"
)
coverage = coverage.reindex(sorted(coverage.index, key=lambda k: (params_by_model[k[0]], k[1])))
print("evaluated delays per checkpoint (1 = present, blank = missing):")
print(coverage.fillna("").to_string())
print("\ncooldown NLL per checkpoint:")
print(
    ckpt.assign(log2_D=np.log2(ckpt.processed_positions))
    .sort_values(["total_parameters", "processed_positions"])[
        ["model", "log2_D", "validation_nll", "training_flops", "wall_hours", "diverged"]
    ]
    .to_string(index=False, float_format=lambda v: f"{v:,.3f}")
)

# %% [markdown]
# ## Palette
#
# Model size is ordinal magnitude, so models take a single-hue blue ramp light→dark by size,
# generated with an even OKLCH lightness step so all nine are separable (validated with the
# dataviz ordinal checks: monotone L, adjacent dL >= 0.06, light end >= 2:1 on the surface).
# D levels take their own orange ordinal ramp. 026base is not a member of the scaled family,
# so it also carries a dashed line and a diamond marker — identity never rests on hue alone.

# %%
BLUE_RAMP = ["#88b5ef", "#72a1de", "#5b8dcd", "#457abc", "#2e67ab", "#225491", "#174377", "#0d325e", "#042246"]
ORANGE_RAMP = ["#f2a07c", "#e07d4e", "#c95b20", "#a24713", "#7e3307"]
MODEL_COLOR = dict(zip(MODELS, BLUE_RAMP))
DEXP_COLOR = dict(zip(D_EXPS, ORANGE_RAMP))
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"
ACCENT = "#eb6834"
SEQ_CMAP = LinearSegmentedColormap.from_list("hal_blue", ["#cde2fb"] + BLUE_RAMP)
# Diverging pair for signed match outcomes: cool = stocks taken, warm = stocks lost, gray zero.
RED_ARM = ["#f0a9a5", "#e07d78", "#c9524d", "#a13330"]
BLUE_ARM = ["#a9c8f2", "#6f9fe0", "#3d78c6", "#1d539c"]
NEUTRAL = "#e6e5e0"

METRIC_MEDIAN = "median net stock rate (stocks/min)"
METRIC_POOLED = "net stock rate (stocks/min)"
METRIC_DAMAGE = "median net damage rate (%/min)"
QUANTUM_NOTE = "full-clock matches pin the median to a 0.51 stocks/min grid"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "figure.dpi": 150,
    }
)


def model_label(model: str) -> str:
    n = params_by_model[model]
    return f"{model} ({n / 1e6:.1f}M)" if n < 1e7 else f"{model} ({n / 1e6:.0f}M)"


def model_style(model: str) -> dict:
    """026base is the frozen reference architecture: dashed, diamond-marked."""
    base = dict(color=MODEL_COLOR[model], marker="o", ms=6)
    if model == "026base":
        base |= dict(ls="--", marker="D", ms=5.5)
    return base


def line_style(model: str) -> dict:
    """Marker and dash style for a model, without its color."""
    return {k: v for k, v in model_style(model).items() if k != "color"}


def headroom(ax: plt.Axes, right: float = 0.3) -> None:
    """Widen the x axis to the right so end-of-line labels stay inside the panel."""
    x0, x1 = ax.get_xlim()
    if ax.get_xscale() == "linear":
        ax.set_xlim(x0, x1 + (x1 - x0) * right)
    else:
        ax.set_xlim(x0, x1 * (x1 / x0) ** right)


def direct_label_lines(ax: plt.Axes, right: float = 0.3, fontsize: float = 8.5) -> None:
    """Label each line at its right end; push near-coincident labels apart vertically."""
    headroom(ax, right)
    ends = []
    for line in ax.get_lines():
        label = line.get_label()
        if label.startswith("_"):
            continue
        x, y = np.asarray(line.get_xdata(), dtype=float), np.asarray(line.get_ydata(), dtype=float)
        ok = np.isfinite(y) & np.isfinite(x)
        if not ok.any():
            continue
        i = np.nonzero(ok)[0][-1]
        ends.append([x[i], y[i], label, line.get_color()])
    if not ends:
        return
    pts = ax.transData.transform(np.array([[e[0], e[1]] for e in ends]))
    order = np.argsort(pts[:, 1])
    # One text line occupies fontsize/72 inches; keep a little more than that between labels.
    min_gap_px = fontsize / 72.0 * ax.figure.dpi * 1.5
    near_px = 6 * min_gap_px
    for prev, cur in zip(order[:-1], order[1:]):
        if abs(pts[cur, 0] - pts[prev, 0]) < near_px and pts[cur, 1] - pts[prev, 1] < min_gap_px:
            pts[cur, 1] = pts[prev, 1] + min_gap_px
    placed = ax.transData.inverted().transform(pts)
    for (x, _, label, color), (_, y) in zip(ends, placed):
        ax.annotate(
            label, (x, y), textcoords="offset points", xytext=(7, 0), color=color, fontsize=fontsize, va="center"
        )


def ring_diverged(ax: plt.Axes, frame: pd.DataFrame, x: str, y: str, color: str) -> None:
    """Hollow ring over a point whose checkpoint diverged in training."""
    bad = frame[frame.diverged]
    if bad.empty:
        return
    ax.scatter(bad[x], bad[y], s=64, facecolor=SURFACE, edgecolor=color, lw=1.8, zorder=5)


def plot_scaling_line(ax: plt.Axes, frame: pd.DataFrame, x: str, y: str, color: str, **kwargs) -> None:
    """Draw a scaling curve over healthy checkpoints only, ringing any diverged point."""
    frame = frame.sort_values(x)
    healthy = frame[~frame.diverged]
    style = kwargs.pop("style", {})
    ax.plot(healthy[x], healthy[y], color=color, **style, **kwargs)
    ring_diverged(ax, frame, x, y, color)


def log2_data_axis(ax: plt.Axes, values: pd.Series, label: str) -> None:
    """Log2 x axis over a continuous data amount, ticked at the sweep's power-of-two endpoints."""
    ax.set_xscale("log", base=2)
    ticks = [2.0**e for e in D_EXPS]
    inside = [t for t in ticks if values.min() / 1.6 <= t <= values.max() * 1.6]
    ax.set_xticks(inside, [f"$2^{{{int(np.log2(t))}}}$" for t in inside])
    ax.set_xticks([], minor=True)
    ax.set_xlabel(label)


def u_axis(ax: plt.Axes) -> None:
    """Log x axis over distinct coverage, ticked at the U each power-of-two D reaches."""
    ax.set_xscale("log")
    ticks = [float(distinct_positions(2.0**e)) for e in D_EXPS]
    ax.set_xticks(ticks, [f"{t / 1e6:.0f}M" if t < 1e9 else f"{t / 1e9:.2f}B" for t in ticks], fontsize=9)
    ax.set_xticks([], minor=True)
    ax.set_xlabel("distinct positions covered U")


def param_axis_ticks(ax: plt.Axes, models: list[str]) -> None:
    """Model-name ticks on a log parameter axis; the two 7M models share one tick."""
    ticks, labels = [], []
    for model in sorted(set(models), key=lambda m: params_by_model[m]):
        if model in ("L5", "026base"):
            if "L5 / 026base (7M)" in labels:
                continue
            if {"L5", "026base"} <= set(models):
                ticks.append(float(np.sqrt(params_by_model["L5"] * params_by_model["026base"])))
                labels.append("L5 / 026base (7M)")
                continue
        ticks.append(float(params_by_model[model]))
        labels.append(model_label(model))
    ax.set_xticks(ticks, labels, rotation=45, ha="right", fontsize=8)
    ax.set_xticks([], minor=True)


def whiskers(
    ax: plt.Axes, frame: pd.DataFrame, x: str, color: str, metric: str = "pooled", alpha: float = 0.45
) -> None:
    ax.vlines(frame[x], frame[f"{metric}_ci_lo"], frame[f"{metric}_ci_hi"], color=color, lw=1.2, alpha=alpha, zorder=1)


def annotate_diverged(ax: plt.Axes, frame: pd.DataFrame, x: str, metric: str = "pooled_rate") -> None:
    for row in frame[frame.diverged].itertuples():
        ax.annotate(
            "diverged ckpt",
            (getattr(row, x), getattr(row, metric)),
            textcoords="offset points",
            xytext=(0, -16),
            ha="center",
            fontsize=7.5,
            color=INK2,
        )


# %% [markdown]
# ## 01 — deployable pareto
#
# One checkpoint per model — the fully-trained D = 2^30 endpoint — against imposed control
# delay, restricted to (model, delay) pairs whose measured RTX 3060 p95 latency fits the frame
# budget. Fixing D removes the best-over-D selection, so nothing here is picked for looking good.
#
# Left is the primary metric, the median net stock rate. Full-clock matches pin it to a
# 0.51/min grid, so it still ties. Right is the median net damage rate over the same matches:
# damage is continuous, so every evaluation gets a distinct value and the ordering is readable.
# Damage is a diagnostic, not the metric — the frontier is chosen on stock median and broken on
# damage median, and the right panel shows what that same choice looks like in damage.
#
# The cost of fixing D is coverage. The panel states which models are missing or partial there,
# rather than quietly dropping them.

# %%
D_ISO = 2.0**30
DELAY_POS = {d: i for i, d in enumerate(DELAYS)}

iso = feasible_evals[(feasible_evals.processed_positions == D_ISO) & ~feasible_evals.diverged].copy()
iso["pos"] = iso.delay_frames.map(DELAY_POS)
iso_models = [m for m in BENCHMARKED if m in set(iso.model)]

no_endpoint = [m for m in MODELS if D_ISO not in set(evals[evals.model == m].processed_positions)]
partial = {
    m: sorted(int(d) for d in iso[iso.model == m].delay_frames)
    for m in iso_models
    if len(iso[iso.model == m]) < sum(feasible.get((m, d), False) for d in DELAYS)
}
dropped_diverged = sorted({m for m, positions in DIVERGED if positions == D_ISO} & set(evals.model))

frontier = iso.sort_values(["median_rate", "damage_rate"]).groupby("delay_frames").tail(1)
frontier = frontier.sort_values("delay_frames")

fig, axes = plt.subplots(1, 2, figsize=(15, 6.6), sharex=True)

for ax, metric, ylabel, title in (
    (axes[0], "median", METRIC_MEDIAN, f"Median net stocks — {QUANTUM_NOTE}"),
    (axes[1], "damage", METRIC_DAMAGE, "Median net damage — continuous, so no grid"),
):
    column = f"{metric}_rate"
    for model in iso_models:
        sub = iso[iso.model == model].sort_values("pos")
        whiskers(ax, sub, "pos", MODEL_COLOR[model], metric=metric)
        ax.plot(sub.pos, sub[column], label=model_label(model), zorder=3, **model_style(model))
    ax.fill_between(
        frontier.pos, frontier[f"{metric}_ci_lo"], frontier[f"{metric}_ci_hi"], color=INK, alpha=0.07, zorder=2
    )
    ax.plot(
        frontier.pos,
        frontier[column],
        color=INK,
        lw=3,
        drawstyle="steps-mid",
        zorder=4,
        label="deployable frontier" if metric == "median" else "same selection, in damage",
    )
    for row in frontier.itertuples():
        ax.annotate(
            row.model,
            (row.pos, getattr(row, column)),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8.5,
            color=INK,
        )
    ax.set_xticks(range(len(DELAYS)), [f"{int(d)}\n{d * FRAME_MS:.0f} ms" for d in DELAYS])
    ax.set_xlim(-0.55, len(DELAYS) - 0.45)
    ax.margins(y=0.16)
    ax.set_xlabel("imposed control delay (frames / inference budget)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10.5)
    ax.legend(loc="lower left", ncols=2, fontsize=8.5)

gaps = []
if no_endpoint:
    gaps.append(f"no $2^{{30}}$ checkpoint: {', '.join(no_endpoint)}")
gaps += [f"{m} evaluated at delays {', '.join(str(d) for d in delays)} only" for m, delays in partial.items()]
gaps += [f"{m} @ $2^{{30}}$ diverged in training and is excluded" for m in dropped_diverged]
if UNBENCHMARKED:
    gaps.append(f"never latency-benchmarked: {', '.join(UNBENCHMARKED)}")

fig.suptitle("Deployable pareto — fully-trained $2^{30}$ checkpoints, RTX 3060 feasible points only")
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.text(0.008, 0.008, "coverage gaps — " + ("; ".join(gaps) if gaps else "none"), fontsize=8, color=MUTED)
fig.savefig(FIGURES / "01_deployable_pareto.png")

# %% [markdown]
# ## 02 — iso-param against processed positions D

# %%
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))

ax = axes[0]
for model in MODELS:
    sub = best_over_delay[best_over_delay.model == model].sort_values("processed_positions")
    if sub.empty:
        continue
    whiskers(ax, sub, "processed_positions", MODEL_COLOR[model])
    plot_scaling_line(
        ax,
        sub,
        "processed_positions",
        "pooled_rate",
        MODEL_COLOR[model],
        label=model_label(model),
        style=line_style(model),
    )
    annotate_diverged(ax, sub, "processed_positions")
log2_data_axis(ax, evals.processed_positions, "processed positions D")
ax.set_ylabel(METRIC_POOLED)
ax.set_title("Closed-loop (best over delay)")
direct_label_lines(ax)

ax = axes[1]
for model in MODELS:
    sub = ckpt[(ckpt.model == model) & ~ckpt.diverged].sort_values("processed_positions")
    if sub.empty:
        continue
    ax.plot(sub.processed_positions, sub.validation_nll, label=model_label(model), **model_style(model))
log2_data_axis(ax, ckpt.processed_positions, "processed positions D")
ax.set_ylabel("validation NLL")
ax.set_title("Validation NLL")
direct_label_lines(ax)

fig.suptitle("Iso-param — capability against processed positions D (D counts repeats; see figure 03)")
fig.tight_layout()
fig.savefig(FIGURES / "02_iso_param_D.png")

# %% [markdown]
# ## 03 — iso-param against distinct positions covered U
#
# Same curves against the number of *distinct* corpus positions each run covered. The right-hand
# panel shows the correction: at the top of the sweep D over-counts distinct coverage by ~20%,
# so the D curves are stretched at their right ends relative to what the models actually saw.

# %%
fig = plt.figure(figsize=(13.5, 5.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 0.8], wspace=0.32)
axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

ax = axes[0]
for model in MODELS:
    sub = best_over_delay[best_over_delay.model == model].sort_values("U_seen")
    if sub.empty:
        continue
    whiskers(ax, sub, "U_seen", MODEL_COLOR[model])
    plot_scaling_line(
        ax, sub, "U_seen", "pooled_rate", MODEL_COLOR[model], label=model_label(model), style=line_style(model)
    )
u_axis(ax)
ax.set_ylabel(METRIC_POOLED)
ax.set_title("Closed-loop (best over delay)")
direct_label_lines(ax)

ax = axes[1]
for model in MODELS:
    sub = ckpt[(ckpt.model == model) & ~ckpt.diverged].sort_values("U_seen")
    if sub.empty:
        continue
    ax.plot(sub.U_seen, sub.validation_nll, label=model_label(model), **model_style(model))
u_axis(ax)
ax.set_ylabel("validation NLL")
ax.set_title("Validation NLL")
direct_label_lines(ax)

ax = axes[2]
d_grid = np.geomspace(2.0**25, 2.0**31, 200)
ax.plot(d_grid, distinct_positions(d_grid) / d_grid, color=ACCENT, lw=2)
observed = np.sort(ckpt.processed_positions.unique())
ax.scatter(observed, distinct_positions(observed) / observed, s=26, color=ACCENT, zorder=3)
ax.set_xscale("log", base=2)
ax.set_xticks([2.0**e for e in D_EXPS], [f"$2^{{{e}}}$" for e in D_EXPS])
ax.set_xticks([], minor=True)
ax.set_ylim(0.7, 1.02)
ax.set_xlabel("processed positions D")
ax.set_ylabel("U / D")
ax.set_title("Coverage efficiency", fontsize=10.5)
ax.annotate(
    f"corpus U = {U_TOTAL / 1e9:.2f}e9 positions\nlargest run covers {distinct_positions(observed.max()) / U_TOTAL:.0%}",
    (0.04, 0.06),
    xycoords="axes fraction",
    fontsize=8,
    color=INK2,
)

fig.suptitle("Iso-param — capability against distinct positions covered U (windows are re-drawn each pass)")
fig.subplots_adjust(left=0.06, right=0.985, top=0.86, bottom=0.13)
fig.savefig(FIGURES / "03_iso_param_U.png")

# %% [markdown]
# ## 04 — iso-data: capacity scaling at fixed D

# %%
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))

ax = axes[0]
for d_exp in D_EXPS:
    target = 2.0**d_exp
    color = DEXP_COLOR[d_exp]
    sub = best_over_delay[(best_over_delay.processed_positions == target) & best_over_delay.model.isin(SCALED)]
    if not sub.empty:
        sub = sub.sort_values("parameters")
        whiskers(ax, sub, "parameters", color)
        plot_scaling_line(
            ax, sub, "parameters", "pooled_rate", color, label=f"D = $2^{{{d_exp}}}$", style={"marker": "o", "ms": 6}
        )
    ref = best_over_delay[(best_over_delay.processed_positions == target) & (best_over_delay.model == "026base")]
    ax.scatter(
        ref.parameters,
        ref.pooled_rate,
        marker="D",
        s=42,
        facecolor=SURFACE,
        edgecolor=color,
        lw=1.6,
        zorder=4,
        label="026base (reference)" if d_exp == D_EXPS[0] else "_",
    )
ax.set_xscale("log")
ax.set_ylabel(METRIC_POOLED)
ax.set_title("Closed-loop (best over delay)")
ax.legend(fontsize=8, ncols=2, loc="lower left")

ax = axes[1]
for d_exp in D_EXPS:
    target = 2.0**d_exp
    color = DEXP_COLOR[d_exp]
    sub = ckpt[(ckpt.processed_positions == target) & ckpt.model.isin(SCALED) & ~ckpt.diverged]
    if not sub.empty:
        sub = sub.sort_values("total_parameters")
        ax.plot(sub.total_parameters, sub.validation_nll, color=color, marker="o", ms=6, label=f"D = $2^{{{d_exp}}}$")
    ref = ckpt[(ckpt.processed_positions == target) & (ckpt.model == "026base") & ~ckpt.diverged]
    ax.scatter(
        ref.total_parameters,
        ref.validation_nll,
        marker="D",
        s=42,
        facecolor=SURFACE,
        edgecolor=color,
        lw=1.6,
        zorder=4,
        label="026base (reference)" if d_exp == D_EXPS[0] else "_",
    )
ax.set_xscale("log")
ax.set_ylabel("validation NLL")
ax.set_title("Validation NLL")
ax.legend(fontsize=8, ncols=2, loc="lower left")

ISO_D_MODELS = [m for m in MODELS if (ckpt.model == m).sum() >= len(D_EXPS) - 1]
for ax in axes:
    param_axis_ticks(ax, ISO_D_MODELS)
    ax.set_xlabel("total parameters")

fig.suptitle(
    "Iso-data — capacity scaling at fixed D (L3 and L4 hold only exact-iso-FLOP endpoints, which sit off these D levels)"
)
fig.tight_layout()
fig.savefig(FIGURES / "04_iso_data_capacity.png")

# %% [markdown]
# ## 05 — capability grid
#
# Best over checkpoints for every (model, delay): stock median, with the pooled rate beneath it.
# Hatch = the model misses the RTX 3060 frame budget at that delay; cross-hatch = never
# benchmarked; dash = not evaluated.

# %%
grid = best_over_d.pivot_table(index="model", columns="delay_frames", values="median_rate").reindex(MODELS)
pooled_grid = best_over_d.pivot_table(index="model", columns="delay_frames", values="pooled_rate").reindex(MODELS)

fig, ax = plt.subplots(figsize=(11.5, 6.2))
mesh = ax.imshow(grid.to_numpy(), cmap=SEQ_CMAP, aspect="auto")
ax.set_xticks(range(len(DELAYS)), [f"{int(d)}\n{d * FRAME_MS:.0f} ms" for d in DELAYS])
ax.set_yticks(range(len(MODELS)), [model_label(m) for m in MODELS])
ax.set_xlabel("imposed control delay (frames / inference budget)")
ax.grid(False)

values = grid.to_numpy()
vmin, vmax = np.nanmin(values), np.nanmax(values)
for i, model in enumerate(MODELS):
    for j, delay in enumerate(DELAYS):
        v = values[i, j]
        if np.isnan(v):
            ax.text(j, i, "—", ha="center", va="center", fontsize=9, color=MUTED)
            continue
        light = (v - vmin) / (vmax - vmin) < 0.55
        ax.text(j, i - 0.10, f"{v:.2f}", ha="center", va="center", fontsize=9, color=INK if light else "#ffffff")
        ax.text(
            j,
            i + 0.26,
            f"({pooled_grid.iloc[i, j]:.2f})",
            ha="center",
            va="center",
            fontsize=7,
            color=INK2 if light else "#e8e8e8",
        )
        if model in UNBENCHMARKED:
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="xxx", edgecolor=INK2, lw=0))
        elif not feasible.get((model, delay), False):
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="///", edgecolor=INK2, lw=0))

fig.colorbar(mesh, ax=ax, label=METRIC_MEDIAN, shrink=0.85)
ax.set_title(
    "Capability grid — median, (pooled) below it. /// misses the 3060 budget, xxx never benchmarked, — not evaluated",
    fontsize=10.5,
)
fig.tight_layout()
fig.savefig(FIGURES / "05_capability_grid.png")

# %% [markdown]
# ## 06 — does validation NLL predict closed-loop capability
#
# Left: every evaluated checkpoint, NLL against its closed-loop rate at delay 4 (the shortest
# delay every benchmarked model clears). Right: Spearman rank correlation between NLL and the
# closed-loop rate, computed within each delay over the checkpoints evaluated there. A cheap
# offline metric that ranked models would sit near -1.

# %%
LINK_DELAY = 4.0


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return np.nan
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))

ax = axes[0]
link = evals[(evals.delay_frames == LINK_DELAY) & evals.validation_nll.notna()]
for model in MODELS:
    sub = link[link.model == model].sort_values("processed_positions")
    if sub.empty:
        continue
    ax.plot(
        sub.validation_nll,
        sub.pooled_rate,
        color=MODEL_COLOR[model],
        lw=1.2,
        alpha=0.7,
        marker="o",
        ms=6,
        ls="--" if model == "026base" else "-",
        label=model_label(model),
        zorder=2,
    )
rho_all = spearman(link.validation_nll.to_numpy(), link.pooled_rate.to_numpy())
ax.set_xlabel("validation NLL")
ax.set_ylabel(METRIC_POOLED)
ax.set_title(f"NLL vs closed-loop at delay {LINK_DELAY:.0f} (Spearman {rho_all:+.2f}, n={len(link)})")
ax.legend(fontsize=8, ncols=2, loc="lower left")
ax.annotate(
    "each line is one model's D sweep",
    (0.98, 0.96),
    xycoords="axes fraction",
    ha="right",
    va="top",
    fontsize=8,
    color=INK2,
)

ax = axes[1]
rhos, ns = [], []
for delay in DELAYS:
    sub = evals[(evals.delay_frames == delay) & evals.validation_nll.notna()]
    rhos.append(spearman(sub.validation_nll.to_numpy(), sub.pooled_rate.to_numpy()))
    ns.append(len(sub))
ax.bar(range(len(DELAYS)), rhos, color=MODEL_COLOR[MODELS[4]], width=0.62, zorder=2)
for i, (rho, n) in enumerate(zip(rhos, ns)):
    ax.annotate(
        f"{rho:+.2f}\nn={n}",
        (i, rho),
        textcoords="offset points",
        xytext=(0, -30 if rho < 0 else 10),
        ha="center",
        va="top" if rho < 0 else "bottom",
        fontsize=8,
        color=INK2,
    )
ax.axhline(0, color=AXIS, lw=1)
ax.set_xticks(range(len(DELAYS)), [f"{int(d)}" for d in DELAYS])
ax.set_xlabel("imposed control delay (frames)")
ax.set_ylabel("Spearman rank correlation (NLL vs net stock rate)")
ax.set_ylim(-1.2, 1.05)
ax.set_title("Rank agreement per delay (−1 = NLL ranks perfectly)")

fig.suptitle("Validation NLL against closed-loop capability")
fig.tight_layout()
fig.savefig(FIGURES / "06_nll_vs_closed_loop.png")

# %% [markdown]
# ## 07 — delay sensitivity and measured latency
#
# Left: each model's capability relative to its own best delay — the shape of the delay
# response, with model size divided out. Right: measured RTX 3060 p95 inference latency against
# model size, with each delay's frame budget as a horizontal line. Where a model's curve crosses
# a budget line is where it stops being deployable at that delay.

# %%
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))

ax = axes[0]
for model in BENCHMARKED:
    sub = best_over_d[best_over_d.model == model].sort_values("delay_frames")
    if sub.empty:
        continue
    ax.plot(
        sub.delay_frames.map(DELAY_POS),
        sub.pooled_rate - sub.pooled_rate.max(),
        label=model_label(model),
        **model_style(model),
    )
for model in UNBENCHMARKED:
    sub = best_over_d[best_over_d.model == model].sort_values("delay_frames")
    if len(sub) < 2:
        continue
    ax.plot(
        sub.delay_frames.map(DELAY_POS),
        sub.pooled_rate - sub.pooled_rate.max(),
        color=MUTED,
        lw=1.4,
        ls=":",
        marker="s",
        ms=4,
        label=f"{model_label(model)} — no 3060 benchmark",
    )
ax.axhline(0, color=AXIS, lw=1, zorder=0)
ax.set_xticks(range(len(DELAYS)), [f"{int(d)}" for d in DELAYS])
ax.set_xlim(-0.4, len(DELAYS) - 0.6)
ax.set_xlabel("imposed control delay (frames)")
ax.set_ylabel("net stock rate − model's own best (stocks/min)")
ax.set_title("Delay sensitivity, normalized per model")
ax.legend(fontsize=8, ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))

ax = axes[1]
for delay in DELAYS:
    budget = delay * FRAME_MS
    ax.axhline(budget, color=GRID, lw=1, zorder=0)
    ax.annotate(
        f"d{int(delay)} — {budget:.0f} ms",
        (1.01, budget),
        xycoords=("axes fraction", "data"),
        fontsize=7.5,
        color=MUTED,
        va="center",
    )
for delay in DELAYS:
    sub = latency[latency.delay_frames == delay].copy()
    sub["parameters"] = sub.model.map(params_by_model)
    sub = sub.sort_values("parameters")
    ok = sub.local_latency_valid_bucket.astype(bool)
    ax.plot(sub.parameters, sub.local_latency_latency_p95_ms, color=GRID, lw=1, zorder=1)
    ax.scatter(sub.parameters[ok], sub.local_latency_latency_p95_ms[ok], s=34, color=MODEL_COLOR[MODELS[5]], zorder=3)
    ax.scatter(
        sub.parameters[~ok],
        sub.local_latency_latency_p95_ms[~ok],
        s=40,
        facecolor=SURFACE,
        edgecolor=ACCENT,
        lw=1.6,
        zorder=3,
    )
    for row in sub[~ok].itertuples():
        ax.annotate(
            f"d{int(delay)}",
            (row.parameters, row.local_latency_latency_p95_ms),
            textcoords="offset points",
            xytext=(8, -1),
            fontsize=7.5,
            color=ACCENT,
            va="center",
        )
ax.set_xscale("log")
ax.set_yscale("log")
param_axis_ticks(ax, BENCHMARKED)
ax.set_xlabel("total parameters")
ax.set_ylabel("RTX 3060 p95 inference latency (ms)")
ax.set_title("Measured latency vs frame budget")
ax.legend(
    handles=[
        Line2D([], [], ls="", marker="o", ms=7, color=MODEL_COLOR[MODELS[5]], label="fits its delay's budget"),
        Line2D([], [], ls="", marker="o", ms=7, mfc=SURFACE, mec=ACCENT, mew=1.6, color="none", label="misses budget"),
    ],
    fontsize=8.5,
    loc="upper left",
)

fig.suptitle("Delay response and the RTX 3060 deadline")
fig.tight_layout()
fig.savefig(FIGURES / "07_delay_sensitivity_latency.png")

# %% [markdown]
# ## 08 — compute frontier and iso-FLOP profiles

# %%
FLOP_LEVELS = [5e16, 1e17, 3e17, 1e18]
LEVEL_COLOR = dict(zip(FLOP_LEVELS, ["#88b5ef", "#457abc", "#225491", "#042246"]))


def isoflop_nll(model: str, c: float) -> float:
    """NLL of ``model`` at compute budget ``c``.

    L3, L4, and the 1e18-FLOP L13 were trained to land exactly on a level, so an exact hit is
    read straight off the run; anything else is interpolated inside the model's observed range.
    """
    sub = ckpt[(ckpt.model == model) & ~ckpt.diverged].sort_values("training_flops")
    exact = sub[np.isclose(sub.training_flops, c, rtol=0.01)]
    if not exact.empty:
        return float(exact.validation_nll.iloc[0])
    if len(sub) < 2 or not (sub.training_flops.min() <= c <= sub.training_flops.max()):
        return np.nan
    return float(np.interp(np.log10(c), np.log10(sub.training_flops), sub.validation_nll))


ISOFLOP = {c: {m: isoflop_nll(m, c) for m in SCALED if np.isfinite(isoflop_nll(m, c))} for c in FLOP_LEVELS}

fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6))

ax = axes[0]
for model in MODELS:
    sub = ckpt[(ckpt.model == model) & ~ckpt.diverged].sort_values("training_flops")
    if sub.empty:
        continue
    ax.plot(sub.training_flops, sub.validation_nll, label=model_label(model), **model_style(model))
for c in FLOP_LEVELS:
    ax.axvline(c, color=MUTED, lw=0.8, ls=":", zorder=0)
ax.set_xscale("log")
ax.set_xlabel("training FLOPs")
ax.set_ylabel("validation NLL")
ax.set_title("Compute frontier (dotted = iso-FLOP slices)")
direct_label_lines(ax, right=0.22)

ax = axes[1]
for c in FLOP_LEVELS:
    points = sorted(ISOFLOP[c].items(), key=lambda kv: params_by_model[kv[0]])
    ax.plot(
        [params_by_model[m] for m, _ in points],
        [v for _, v in points],
        color=LEVEL_COLOR[c],
        marker="o",
        ms=6,
        label=f"C = {c:.0e}",
    )
ax.set_xscale("log")
param_axis_ticks(ax, sorted({m for level in ISOFLOP.values() for m in level}))
ax.set_xlabel("total parameters")
ax.set_ylabel("validation NLL (interpolated)")
ax.set_title("Iso-FLOP profiles")
ax.legend(fontsize=8.5)

ax = axes[2]
for model in MODELS:
    sub = evals[(evals.model == model) & evals.training_flops.notna()]
    if sub.empty:
        continue
    sub = best_by(sub, ["processed_positions"]).sort_values("training_flops")
    plot_scaling_line(
        ax, sub, "training_flops", "pooled_rate", MODEL_COLOR[model], label=model_label(model), style=line_style(model)
    )
ax.set_xscale("log")
ax.set_xlabel("training FLOPs")
ax.set_ylabel(METRIC_POOLED)
ax.set_title("Closed-loop compute frontier (best over delay)")
direct_label_lines(ax, right=0.22)

fig.suptitle("Compute scaling — cooldown checkpoints, FLOPs from the run's own 6*D*(N_trunk + 36*N_decoder)")
fig.tight_layout(rect=(0, 0.06, 1, 1))
if failed_endpoint_note():
    fig.text(0.008, 0.008, "unfinished exact endpoints — " + failed_endpoint_note(), fontsize=8, color=MUTED)
fig.savefig(FIGURES / "08_compute_frontier.png")

# %% [markdown]
# ## 09 — parametric fit and compute-optimal capacity
#
# Fit L(N_eff, D) = E + A/N_eff^alpha + B/D^beta on the scaled-family cooldown runs, where
# N_eff = training_flops / (6 D) — the compute-relevant size implied by the FLOPs formula.
# The constraint C = 6 N_eff D gives N_opt^(alpha+beta) = (alpha A / (beta B)) (C/6)^beta.
# Every fit run is sub-epoch, so no data-repetition correction applies; predictions past
# D = U enter the repeated-data regime and are optimistic.

# %%
fit_df = ckpt[ckpt.model.isin(SCALED) & ~ckpt.diverged].copy()
fit_df["n_eff"] = fit_df.training_flops / (6 * fit_df.processed_positions)
n_fit = fit_df.n_eff.to_numpy()
d_fit = fit_df.processed_positions.to_numpy()
y_fit = fit_df.validation_nll.to_numpy()


def fit_scaling_law(data_axis: np.ndarray) -> tuple[float, float, float, float, float]:
    """Fit the robust five-parameter scaling law on one data axis."""

    def huber_objective(parameters: np.ndarray) -> float:
        log_e, log_a, log_b, alpha, beta = parameters
        with np.errstate(over="ignore", invalid="ignore"):
            prediction = np.exp(log_e) + np.exp(log_a) * n_fit**-alpha + np.exp(log_b) * data_axis**-beta
            residual = prediction - y_fit
            delta = 1e-3
            return float(
                np.sum(
                    np.where(
                        np.abs(residual) < delta,
                        0.5 * residual**2,
                        delta * (np.abs(residual) - 0.5 * delta),
                    )
                )
            )

    best = min(
        (
            minimize(
                huber_objective,
                [np.log(e0), log_a, log_b, alpha0, beta0],
                method="Nelder-Mead",
                options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-12},
            )
            for e0 in [1.2, 1.5, 1.7]
            for alpha0 in [0.1, 0.3, 0.5]
            for beta0 in [0.1, 0.3, 0.5]
            for log_a in [0.0, 3.0, 6.0]
            for log_b in [0.0, 3.0, 6.0]
        ),
        key=lambda result: result.fun,
    )
    e_fit, a_fit, b_fit = np.exp(best.x[:3])
    return float(e_fit), float(a_fit), float(b_fit), float(best.x[3]), float(best.x[4])


E_FIT, A_FIT, B_FIT, ALPHA, BETA = fit_scaling_law(d_fit)
resid = E_FIT + A_FIT * n_fit**-ALPHA + B_FIT * d_fit**-BETA - y_fit
print(f"\nL(N_eff, D) = {E_FIT:.4f} + {A_FIT:.4g}/N^{ALPHA:.3f} + {B_FIT:.4g}/D^{BETA:.3f}")
print(f"rms residual {np.sqrt(np.mean(resid**2)):.4f} NLL, max {np.max(np.abs(resid)):.4f}, {len(fit_df)} points")

u_fit = distinct_positions(d_fit)
E_U, A_U, B_U, ALPHA_U, BETA_U = fit_scaling_law(u_fit)
resid_u = E_U + A_U * n_fit**-ALPHA_U + B_U * u_fit**-BETA_U - y_fit
print(f"coverage sensitivity L(N_eff, U_seen) = {E_U:.4f} + {A_U:.4g}/N^{ALPHA_U:.3f} + {B_U:.4g}/U_seen^{BETA_U:.3f}")
print(f"coverage-sensitivity rms residual {np.sqrt(np.mean(resid_u**2)):.4f} NLL")


def loss_at(n_eff: np.ndarray, c: float) -> np.ndarray:
    return E_FIT + A_FIT * n_eff**-ALPHA + B_FIT * (c / (6 * n_eff)) ** -BETA


def n_eff_opt(c: float) -> float:
    return float((ALPHA * A_FIT / (BETA * B_FIT)) ** (1 / (ALPHA + BETA)) * (c / 6) ** (BETA / (ALPHA + BETA)))


def coverage_loss_at(n_eff: np.ndarray | float, compute: float) -> np.ndarray | float:
    processed = compute / (6 * np.asarray(n_eff))
    covered = distinct_positions(processed)
    return E_U + A_U * np.asarray(n_eff) ** -ALPHA_U + B_U * covered**-BETA_U


def coverage_n_eff_opt(compute: float) -> float:
    """Numerical optimum under the estimated distinct-coverage sensitivity fit."""
    lower = np.log(n_fit.min() / 32)
    upper = np.log(n_fit.max() * 32)
    result = minimize_scalar(
        lambda log_n: float(coverage_loss_at(np.exp(log_n), compute)),
        bounds=(lower, upper),
        method="bounded",
    )
    return float(np.exp(result.x))


n_eff_by_model = fit_df.groupby("model").n_eff.first()
C_PREDICT = 4.2e19

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))

ax = axes[0]
grid_n = np.geomspace(n_eff_by_model.min() / 8, n_eff_by_model.max() * 2, 240)
for c in FLOP_LEVELS:
    color = LEVEL_COLOR[c]
    ax.plot(grid_n, loss_at(grid_n, c), color=color, lw=2, label=f"C = {c:.0e}")
    for model, nll in ISOFLOP[c].items():
        ax.scatter(n_eff_by_model[model], nll, color=color, s=38, zorder=3)
    n_star = n_eff_opt(c)
    ax.scatter(
        n_star, loss_at(np.array([n_star]), c), marker="D", s=55, facecolor=SURFACE, edgecolor=color, lw=1.8, zorder=4
    )
ax.set_xscale("log")
for i, model in enumerate(sorted(SCALED, key=lambda m: n_eff_by_model[m])):
    ax.axvline(n_eff_by_model[model], color=GRID, lw=0.8, zorder=0)
    ax.annotate(
        model,
        (n_eff_by_model[model], 0.012 if i % 2 == 0 else 0.052),
        xycoords=ax.get_xaxis_transform(),
        fontsize=7.5,
        color=MUTED,
        ha="center",
        va="bottom",
    )
ax.set_ylim(min(y_fit.min(), 1.70) - 0.03, y_fit.max() + 0.06)
ax.set_xlabel("effective parameters N_eff = trunk + 36 x decoder")
ax.set_ylabel("validation NLL")
ax.set_title("Fitted iso-FLOP parabolas (diamonds = predicted minima)")
ax.legend(fontsize=8.5, loc="upper right")

ax = axes[1]
c_grid = np.geomspace(1e16, 1e20, 120)
ax.plot(
    c_grid,
    [n_eff_opt(c) for c in c_grid],
    color=MODEL_COLOR[MODELS[5]],
    lw=2,
    label=f"$N_{{opt}} \\propto C^{{{BETA / (ALPHA + BETA):.2f}}}$",
)
ax.plot(
    c_grid,
    [coverage_n_eff_opt(c) for c in c_grid],
    color=ACCENT,
    lw=1.8,
    ls="--",
    label="distinct-coverage sensitivity",
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.axhspan(n_eff_by_model.min(), n_eff_by_model.max(), color=GRID, alpha=0.6, zorder=0, label="scaled family range")
for c in FLOP_LEVELS:
    ax.scatter(c, n_eff_opt(c), color=MODEL_COLOR[MODELS[5]], s=38, zorder=3)
    ax.scatter(c, coverage_n_eff_opt(c), color=ACCENT, marker="x", s=38, zorder=3)
n_star = n_eff_opt(C_PREDICT)
d_star = C_PREDICT / (6 * n_star)
ax.scatter(C_PREDICT, n_star, marker="*", s=230, color=ACCENT, zorder=4, label=f"C = {C_PREDICT:.1e}")
ax.annotate(
    f"N_eff = {n_star:.2e}\nD = {d_star:.2e} ({d_star / U_TOTAL:.1f} epochs)\n"
    f"predicted NLL = {loss_at(np.array([n_star]), C_PREDICT)[0]:.3f}",
    (C_PREDICT, n_star),
    textcoords="offset points",
    xytext=(-12, -48),
    ha="right",
    fontsize=8.5,
)
ax.set_xlabel("training FLOPs")
ax.set_ylabel("compute-optimal effective parameters")
ax.set_title("Compute-optimal capacity, extrapolated")
ax.legend(fontsize=8.5, loc="upper left")

fig.suptitle("Iso-FLOP structure from the parametric fit (minima below the family range are extrapolations)")
fig.tight_layout(rect=(0, 0.06, 1, 1))
if failed_endpoint_note():
    fig.text(0.008, 0.008, "unfinished exact endpoints — " + failed_endpoint_note(), fontsize=8, color=MUTED)
fig.savefig(FIGURES / "09_isoflop_fit.png")

print("\nRuns that would bracket each iso-FLOP minimum:")
for c in FLOP_LEVELS + [C_PREDICT]:
    n_star = n_eff_opt(c)
    lo, hi = n_eff_by_model[n_eff_by_model <= n_star], n_eff_by_model[n_eff_by_model > n_star]
    for side, series in (("below", lo.nlargest(2)), ("above", hi.nsmallest(2))):
        for model, n_eff in series.items():
            d_need = c / (6 * n_eff)
            covered = bool(((ckpt.model == model) & (ckpt.processed_positions >= d_need * 0.999)).any())
            note = "covered" if covered else f"NEEDED: cooldown @ D=2^{np.log2(d_need):.1f}"
            print(f"  C={c:.1e}  {side:5s} minimum: {model:7s} D={d_need:.2e}  {note}")
    if lo.empty:
        print(f"  C={c:.1e}  below minimum: none — needs a model smaller than {n_eff_by_model.idxmin()}")

# %% [markdown]
# ## 10 — wall-clock frontier
#
# The same two frontiers against measured training wall time (cumulative, prefix branch
# included). Runs used different cloud GPUs, so this ranks configurations under the hardware
# each actually ran on, not a controlled hardware comparison.

# %%
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))

ax = axes[0]
for model in MODELS:
    sub = ckpt[(ckpt.model == model) & ~ckpt.diverged].sort_values("wall_hours")
    if sub.empty:
        continue
    ax.plot(sub.wall_hours, sub.validation_nll, label=model_label(model), **model_style(model))
ax.set_xscale("log")
ax.set_xlabel("cumulative training wall time (hours)")
ax.set_ylabel("validation NLL")
ax.set_title("NLL against training time")
direct_label_lines(ax, right=0.28)

ax = axes[1]
for model in MODELS:
    sub = evals[(evals.model == model) & evals.wall_hours.notna()]
    if sub.empty:
        continue
    sub = best_by(sub, ["processed_positions"]).sort_values("wall_hours")
    whiskers(ax, sub, "wall_hours", MODEL_COLOR[model])
    plot_scaling_line(
        ax, sub, "wall_hours", "pooled_rate", MODEL_COLOR[model], label=model_label(model), style=line_style(model)
    )
ax.set_xscale("log")
ax.set_xlabel("cumulative training wall time (hours)")
ax.set_ylabel(METRIC_POOLED)
ax.set_title("Closed-loop against training time (best over delay)")
direct_label_lines(ax, right=0.28)

fig.suptitle("Time frontier — mixed cloud hardware, indicative rather than a controlled comparison")
fig.tight_layout()
fig.savefig(FIGURES / "10_wallclock_frontier.png")

# %% [markdown]
# ## 11 — match-level distributions
#
# Left: where each model's matches actually land. One row per model at its best feasible
# configuration; the bar is the share of matches at each net stock difference, warm = stocks
# behind, gray = even, cool = stocks ahead. Right: the same configuration's rate broken out by
# the ego character, which is where most of the spread lives.

# %%
STOCK_VALUES = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
STOCK_COLOR = {
    -4: RED_ARM[3],
    -3: RED_ARM[2],
    -2: RED_ARM[1],
    -1: RED_ARM[0],
    0: NEUTRAL,
    1: BLUE_ARM[0],
    2: BLUE_ARM[1],
    3: BLUE_ARM[2],
    4: BLUE_ARM[3],
}

pick = best_by(best_over_d_feasible, ["model"]).set_index("model")
pick = pick.reindex([m for m in MODELS if m in set(best_over_d_feasible.model)])

fig, axes = plt.subplots(1, 2, figsize=(15, 6.6), gridspec_kw={"width_ratios": [1.2, 1.0]})

ax = axes[0]
for i, (model, row) in enumerate(pick.iterrows()):
    group = matches[matches.evaluation_key == row.evaluation_key]
    share = group.match_stock_difference.value_counts(normalize=True)
    left = 0.0
    for value in STOCK_VALUES:
        width = float(share.get(value, 0.0))
        if width <= 0:
            continue
        ax.barh(i, width, left=left, height=0.62, color=STOCK_COLOR[value], edgecolor=SURFACE, lw=2)
        if width > 0.07:
            dark = value in (-4, -3, 3, 4)
            ax.text(
                left + width / 2,
                i,
                f"{value:+d}" if value else "0",
                ha="center",
                va="center",
                fontsize=8,
                color="#ffffff" if dark else INK,
            )
        left += width
    ax.annotate(
        f"d{int(row.delay_frames)} · $2^{{{np.log2(row.processed_positions):.1f}}}$ · {row.pooled_rate:+.2f}/min",
        (1.01, i),
        xycoords=("axes fraction", "data"),
        fontsize=7.5,
        color=INK2,
        va="center",
    )
ax.set_yticks(range(len(pick)), [model_label(m) for m in pick.index])
ax.set_xlim(0, 1)
ax.set_xticks(np.arange(0, 1.01, 0.25), [f"{v:.0%}" for v in np.arange(0, 1.01, 0.25)])
ax.set_xlabel("share of matches")
ax.invert_yaxis()
ax.grid(False)
ax.set_title("Per-match net stock difference at each model's best feasible configuration")
ax.legend(
    handles=[
        Line2D([], [], ls="", marker="s", ms=9, color=RED_ARM[3], label="−4 (shut out)"),
        Line2D([], [], ls="", marker="s", ms=9, color=RED_ARM[0], label="−1"),
        Line2D([], [], ls="", marker="s", ms=9, color=NEUTRAL, label="0"),
        Line2D([], [], ls="", marker="s", ms=9, color=BLUE_ARM[2], label="+1 or better"),
    ],
    fontsize=8,
    ncols=4,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.24),
)

ax = axes[1]
best_model, worst_model = pick.pooled_rate.idxmax(), pick.pooled_rate.idxmin()


def by_character(model: str) -> pd.Series:
    """Pooled net stock rate per ego character for one evaluation."""
    group = matches[matches.evaluation_key == pick.loc[model].evaluation_key]
    agg = group.groupby("match_ego_character").agg(
        stocks=("match_stock_difference", "sum"), frames=("match_active_frames", "sum"), n=("rate", "size")
    )
    agg = agg[agg.n >= 5]
    return agg.stocks / (agg.frames / FRAMES_PER_MINUTE)


series = {model: by_character(model) for model in (best_model, worst_model)}
characters = sorted(set().union(*(s.index for s in series.values())), key=lambda c: series[best_model].get(c, np.inf))
for model, marker, offset in ((best_model, "o", -0.16), (worst_model, "s", 0.16)):
    values = series[model].reindex(characters)
    ax.scatter(
        values.to_numpy(),
        np.arange(len(characters)) + offset,
        s=46,
        marker=marker,
        color=MODEL_COLOR[model],
        zorder=3,
        label=f"{model_label(model)} @ d{int(pick.loc[model].delay_frames)}",
    )
ax.set_yticks(range(len(characters)), [melee.Character(int(c)).name.title() for c in characters], fontsize=8.5)
ax.set_ylim(-0.6, len(characters) - 0.4)
ax.axvline(0, color=AXIS, lw=1, zorder=0)
ax.set_xlabel(METRIC_POOLED)
ax.set_title("Net stock rate by ego character (>= 5 matches)")
ax.legend(fontsize=8.5, loc="upper left")

fig.suptitle("Match-level structure behind the medians")
fig.tight_layout()
fig.savefig(FIGURES / "11_match_distributions.png")

# %% [markdown]
# ## Frontier summary

# %%
summary = frontier[
    [
        "delay_frames",
        "model",
        "processed_positions",
        "median_rate",
        "median_ci_lo",
        "median_ci_hi",
        "damage_rate",
        "damage_ci_lo",
        "damage_ci_hi",
        "pooled_rate",
        "n_matches",
        "n_boots",
    ]
].copy()
summary["budget_ms"] = summary.delay_frames * FRAME_MS
summary["parameters"] = summary.model.map(params_by_model)
summary["p95_ms"] = [p95_ms.get((m, d), np.nan) for m, d in zip(summary.model, summary.delay_frames)]
print("\nDeployable frontier at D = 2^30 (90% cluster-bootstrap CIs, ranked on stock median):")
print(
    summary[
        [
            "delay_frames",
            "budget_ms",
            "model",
            "parameters",
            "median_rate",
            "median_ci_lo",
            "median_ci_hi",
            "damage_rate",
            "damage_ci_lo",
            "damage_ci_hi",
            "p95_ms",
            "n_matches",
        ]
    ].to_string(index=False, float_format=lambda v: f"{v:,.2f}")
)
summary.to_csv(RESULTS / "deployable_frontier_median.csv", index=False)
print("\n" + json.dumps({"figures": sorted(p.name for p in FIGURES.glob("*.png"))}, indent=2))
