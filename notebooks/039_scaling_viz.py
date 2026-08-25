# %% [markdown]
# # Experiment 039 — capacity / exposure / delay scaling
#
# Panels:
# - A: iso-latency pareto (headline) — capability at each imposed delay + the RTX 3060 deployable frontier
# - B: iso-param — exposure scaling per model
# - C: iso-data — capacity scaling per exposure level
# - D: compute frontier + Chinchilla-style iso-FLOP profiles (val NLL)
# - E: model x delay heatmap with deployability and coverage
#
# Primary metric: eval_net_stock_lcb (net stocks is THE metric; win_rate has no resolution here).
# Secondary: validation_nll (delay-independent, cooldown checkpoints only).

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

RESULTS = Path(__file__).resolve().parent.parent / "results" / "039_capacity_scaling"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(exist_ok=True)

FRAME_MS = 1000.0 / 60.0

raw = pd.read_csv(RESULTS / "experiment_039_raw_outcomes.csv", low_memory=False)

runs = raw[raw.record_type == "run"][
    [
        "model",
        "phase",
        "D_exp",
        "total_parameters",
        "processed_positions",
        "training_flops",
        "validation_nll",
        "unique_loss_positions",
    ]
].copy()
evals = raw[raw.record_type == "evaluation"][
    ["model", "D_exp", "delay_frames", "eval_net_stock_lcb", "eval_net_stock_per_min", "eval_matches"]
].copy()
latency = raw[raw.record_type == "latency_benchmark"][
    ["model", "delay_frames", "local_latency_valid_bucket", "local_latency_latency_p95_ms"]
].copy()

params_by_model = runs.groupby("model").total_parameters.first()
# Evaluated models, ordered by size: L5, 026base, L7, L10, L13, L16, L18. Runs may contain extra
# probe models (L3/L4) that have no evaluations or latency benchmarks yet.
MODELS = list(params_by_model.loc[evals.model.unique()].sort_values().index)
# Capacity-scaling curves trace the homogeneous scaled family only; 026base (7.08M, the frozen
# 026 architecture) sits at the same size as L5 and stacks on top of it, so it enters those
# panels as a standalone reference marker instead of a point on the curve.
SCALED = [m for m in MODELS if m != "026base"]
DELAYS = sorted(evals.delay_frames.unique())

# Deployability on RTX 3060: p95 inference latency fits the delay's frame budget.
valid = latency.set_index(["model", "delay_frames"]).local_latency_valid_bucket.astype(bool)

# Cross-check against the precomputed frontier file.
frontier_file = pd.read_csv(RESULTS / "latency_capacity_frontier.csv")
for row in frontier_file.itertuples():
    eligible = set(row.eligible_models.split(","))
    benchmarked = {m for m in MODELS if valid.get((m, row.delay), False)}
    assert eligible == benchmarked, f"delay {row.delay}: frontier file {eligible} != benchmark {benchmarked}"

# Best over exposure per (model, delay) — capability at that delay.
best_over_d = evals.loc[evals.groupby(["model", "delay_frames"]).eval_net_stock_lcb.idxmax()].copy()
# Best over delay per (model, D_exp) — optimistic capability per checkpoint.
best_over_delay = evals.loc[evals.groupby(["model", "D_exp"]).eval_net_stock_lcb.idxmax()].copy()

# NLL curves use cooldown checkpoints only; prefix rows are pre-cooldown and not comparable.
# Incomplete cooldowns carry NaN NLL/FLOPs and drop out. A diverged cooldown (NLL worse than the
# same model's least-trained checkpoint) is excluded and reported.
cooldowns = runs[(runs.phase == "cooldown") & runs.validation_nll.notna()].copy()
nll_floor = cooldowns[cooldowns.D_exp == cooldowns.groupby("model").D_exp.transform("min")]
diverged = cooldowns.validation_nll > cooldowns.model.map(nll_floor.set_index("model").validation_nll)
DIVERGED = [(row.model, row.D_exp) for row in cooldowns[diverged].itertuples()]
for model, d_exp in DIVERGED:
    print(f"excluding diverged cooldown from NLL panels and fit: {model} @ 2^{d_exp:.0f}")
cooldowns = cooldowns[~diverged]

print(f"{len(evals)} evals, {len(cooldowns)} cooldown runs, models: {MODELS}")

# %% [markdown]
# ## Palette
#
# Model size is ordinal magnitude, so models take a one-hue blue ramp light→dark by size
# (dataviz reference palette, ordinal range steps 250–700). Exposure (D_exp) is a second
# sequential context and takes its own orange ramp. Direct labels carry identity; color
# carries order.

# %%
BLUE_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#2a78d6", "#256abf", "#184f95", "#0d366b"]  # steps 250..700
MODEL_COLOR = dict(zip(MODELS, BLUE_RAMP))
ORANGE_RAMP = ["#f6c8ab", "#f0a077", "#e57840", "#c04a15", "#7c2d0b"]  # one-hue ramp on the categorical orange
DEXP_COLOR = dict(zip([26.0, 27.0, 28.0, 29.0, 30.0], ORANGE_RAMP))
D_EXPS = [int(d) for d in DEXP_COLOR]
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "hal_blue", ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95", "#0d366b"]
)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e4df"

plt.rcParams.update(
    {
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "legend.frameon": False,
    }
)


def model_label(model: str) -> str:
    n = params_by_model[model]
    return f"{model} ({n / 1e6:.0f}M)"


def direct_label_lines(ax: plt.Axes, offset_frac: float = 0.01) -> None:
    """Label each line at its right end; push near-coincident labels apart vertically."""
    ends = []
    for line in ax.get_lines():
        label = line.get_label()
        if label.startswith("_"):
            continue
        x, y = np.asarray(line.get_xdata(), dtype=float), np.asarray(line.get_ydata(), dtype=float)
        ok = np.isfinite(y)
        if not ok.any():
            continue
        i = np.nonzero(ok)[0][-1]
        ends.append([x[i], y[i], label, line.get_color()])

    to_axes = ax.transData.transform
    pts = to_axes(np.array([[e[0], e[1]] for e in ends]))
    order = np.argsort(pts[:, 1])
    min_gap_px, near_px = 14.0, 120.0
    for prev, cur in zip(order[:-1], order[1:]):
        if abs(pts[cur, 0] - pts[prev, 0]) < near_px and pts[cur, 1] - pts[prev, 1] < min_gap_px:
            pts[cur, 1] = pts[prev, 1] + min_gap_px
    placed = ax.transData.inverted().transform(pts)

    x0, x1 = ax.get_xlim()
    pad = (x1 - x0) * offset_frac
    for (x, _, label, color), (_, y) in zip(ends, placed):
        ax.annotate(label, (x + pad, y), color=color, fontsize=9, va="center")


def mark_diverged(ax: plt.Axes, x_of: dict[tuple[str, float], float]) -> None:
    """Flag closed-loop points whose checkpoint diverged in training (kept in the data, not the fit)."""
    for model, d_exp in DIVERGED:
        point = best_over_delay[(best_over_delay.model == model) & (best_over_delay.D_exp == d_exp)]
        if point.empty or (model, d_exp) not in x_of:
            continue
        ax.annotate(
            "diverged ckpt",
            (x_of[(model, d_exp)], float(point.eval_net_stock_lcb.iloc[0])),
            textcoords="offset points",
            xytext=(6, 12),
            ha="left",
            fontsize=8,
            color=INK2,
        )


def param_axis_ticks(ax: plt.Axes) -> None:
    """Model-name ticks on a log param axis; the two 7M models share one tick."""
    seven_m = float(np.sqrt(params_by_model["L5"] * params_by_model["026base"]))
    ticks = [seven_m] + [float(params_by_model[m]) for m in MODELS[2:]]
    labels = ["L5 / 026base (7M)"] + [model_label(m) for m in MODELS[2:]]
    ax.set_xticks(ticks, labels, rotation=45, ha="right", fontsize=8)
    ax.set_xticks([], minor=True)


# %% [markdown]
# ## Panel A — iso-latency pareto
#
# For each imposed delay, each model's best net_stock_lcb over exposure. Filled marker =
# the (model, delay) pair fits the RTX 3060 frame budget; hollow = it does not. The dark
# step line is the deployable frontier: the best latency-valid model at each delay. The
# shaded band around the frontier spans LCB → point estimate for the frontier point.

# %%
fig, ax = plt.subplots(figsize=(10, 6.5))

for model in MODELS:
    sub = best_over_d[best_over_d.model == model].sort_values("delay_frames")
    color = MODEL_COLOR[model]
    ax.plot(sub.delay_frames, sub.eval_net_stock_lcb, color=color, lw=2, label=model_label(model), zorder=2)
    deploy = sub.delay_frames.map(lambda d, m=model: valid.get((m, d), False))
    ax.scatter(sub.delay_frames[deploy], sub.eval_net_stock_lcb[deploy], s=45, color=color, zorder=3)
    ax.scatter(
        sub.delay_frames[~deploy],
        sub.eval_net_stock_lcb[~deploy],
        s=45,
        facecolor="#fcfcfb",
        edgecolor=color,
        lw=1.6,
        zorder=3,
    )

frontier_rows = []
for delay in DELAYS:
    pool = best_over_d[
        (best_over_d.delay_frames == delay) & best_over_d.model.map(lambda m, d=delay: valid.get((m, d), False))
    ]
    if not pool.empty:
        frontier_rows.append(pool.loc[pool.eval_net_stock_lcb.idxmax()])
frontier = pd.DataFrame(frontier_rows)
ax.fill_between(
    frontier.delay_frames,
    frontier.eval_net_stock_lcb,
    frontier.eval_net_stock_per_min,
    color=INK,
    alpha=0.08,
    zorder=1,
)
ax.plot(
    frontier.delay_frames,
    frontier.eval_net_stock_lcb,
    color=INK,
    lw=3,
    drawstyle="steps-mid",
    label="deployable frontier",
    zorder=4,
)
for row in frontier.itertuples():
    ax.annotate(
        f"{row.model} @2^{row.D_exp:.0f}",
        (row.delay_frames, row.eval_net_stock_lcb),
        textcoords="offset points",
        xytext=(0, 9),
        ha="center",
        fontsize=8,
        color=INK,
    )

ax.set_xticks(DELAYS)
ax.set_xticklabels([f"{int(d)}\n{d * FRAME_MS:.0f} ms" for d in DELAYS])
ax.set_xlabel("imposed control delay (frames / inference budget)")
ax.set_ylabel("net_stock_lcb vs CPU (best over exposure)")
ax.set_title("Iso-latency pareto — capability per delay, RTX 3060 deployable frontier in black")
ax.legend(loc="lower right", ncols=2, fontsize=9)
fig.tight_layout()
fig.savefig(FIGURES / "A_iso_latency_pareto.png", dpi=150)

# %% [markdown]
# ## Panel B — iso-param: exposure scaling per model

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

ax = axes[0]
for model in MODELS:
    sub = best_over_delay[best_over_delay.model == model].sort_values("D_exp")
    ax.plot(
        sub.D_exp, sub.eval_net_stock_lcb, color=MODEL_COLOR[model], lw=2, marker="o", ms=6, label=model_label(model)
    )
ax.set_xticks(D_EXPS)
ax.set_xticklabels([f"$2^{{{e}}}$" for e in D_EXPS])
ax.set_xlabel("processed positions")
ax.set_ylabel("net_stock_lcb (best over delay)")
ax.set_title("Closed-loop")
ax.set_xlim(25.7, 30.9)
mark_diverged(ax, {(m, d): d for m, d in DIVERGED})
direct_label_lines(ax)

ax = axes[1]
for model in MODELS:
    sub = cooldowns[cooldowns.model == model].sort_values("D_exp")
    ax.plot(sub.D_exp, sub.validation_nll, color=MODEL_COLOR[model], lw=2, marker="o", ms=6, label=model_label(model))
ax.set_xticks(D_EXPS)
ax.set_xticklabels([f"$2^{{{e}}}$" for e in D_EXPS])
ax.set_xlabel("processed positions")
ax.set_ylabel("validation NLL")
ax.set_title("Validation NLL")
ax.set_xlim(25.7, 30.9)
direct_label_lines(ax)

fig.suptitle("Iso-param — exposure scaling (same 112k-replay corpus; exposure, not unique data)")
fig.tight_layout()
fig.savefig(FIGURES / "B_iso_param.png", dpi=150)

# %% [markdown]
# ## Panel C — iso-data: capacity scaling per exposure level

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

ax = axes[0]
for d_exp, color in DEXP_COLOR.items():
    sub = best_over_delay[(best_over_delay.D_exp == d_exp) & best_over_delay.model.isin(SCALED)].copy()
    sub["n"] = sub.model.map(params_by_model)
    sub = sub.sort_values("n")
    ax.semilogx(sub.n, sub.eval_net_stock_lcb, color=color, lw=2, marker="o", ms=6, label=f"$2^{{{d_exp:.0f}}}$ pos")
    ref = best_over_delay[(best_over_delay.D_exp == d_exp) & (best_over_delay.model == "026base")]
    ax.scatter(
        ref.model.map(params_by_model),
        ref.eval_net_stock_lcb,
        marker="D",
        s=40,
        facecolor="#fcfcfb",
        edgecolor=color,
        lw=1.6,
        zorder=3,
        label="026base (ref)" if d_exp == 26.0 else "_",
    )
ax.set_xlabel("total parameters")
ax.set_ylabel("net_stock_lcb (best over delay)")
ax.set_title("Closed-loop")
mark_diverged(ax, {(m, d): float(params_by_model[m]) for m, d in DIVERGED})
ax.legend(fontsize=9)

ax = axes[1]
for d_exp, color in DEXP_COLOR.items():
    sub = cooldowns[(cooldowns.D_exp == d_exp) & cooldowns.model.isin(SCALED)].sort_values("total_parameters")
    ax.semilogx(
        sub.total_parameters, sub.validation_nll, color=color, lw=2, marker="o", ms=6, label=f"$2^{{{d_exp:.0f}}}$ pos"
    )
    ref = cooldowns[(cooldowns.D_exp == d_exp) & (cooldowns.model == "026base")]
    ax.scatter(
        ref.total_parameters,
        ref.validation_nll,
        marker="D",
        s=40,
        facecolor="#fcfcfb",
        edgecolor=color,
        lw=1.6,
        zorder=3,
        label="026base (ref)" if d_exp == 26.0 else "_",
    )
ax.set_xlabel("total parameters")
ax.set_ylabel("validation NLL")
ax.set_title("Validation NLL")
ax.legend(fontsize=9)

for ax in axes:
    param_axis_ticks(ax)

fig.suptitle("Iso-data — capacity scaling at fixed exposure")
fig.tight_layout()
fig.savefig(FIGURES / "C_iso_data.png", dpi=150)

# %% [markdown]
# ## Panel D — compute frontier and iso-FLOP profiles
#
# Left: each model traces its exposure sweep in FLOPs; the envelope is the compute-pareto.
# Right: NLL interpolated (linear in log FLOPs, within each model's observed range) at fixed
# compute levels — the Chinchilla-style iso-FLOP profile over model size.

# %%
FLOP_LEVELS = [5e16, 1e17, 3e17, 1e18]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

ax = axes[0]
for model in MODELS:
    sub = cooldowns[cooldowns.model == model].sort_values("training_flops")
    ax.semilogx(
        sub.training_flops,
        sub.validation_nll,
        color=MODEL_COLOR[model],
        lw=2,
        marker="o",
        ms=6,
        label=model_label(model),
    )
for c in FLOP_LEVELS:
    ax.axvline(c, color=INK2, lw=0.8, ls=":", zorder=0)
ax.set_xlabel("training FLOPs")
ax.set_ylabel("validation NLL")
ax.set_title("Compute frontier (dotted = iso-FLOP slices)")
direct_label_lines(ax, offset_frac=0.004)

ax = axes[1]
level_colors = ["#86b6ef", "#5598e7", "#256abf", "#0d366b"]
for c, color in zip(FLOP_LEVELS, level_colors):
    xs, ys = [], []
    for model in SCALED:
        sub = cooldowns[cooldowns.model == model].sort_values("training_flops")
        if len(sub) < 2 or not (sub.training_flops.min() <= c <= sub.training_flops.max()):
            continue
        nll = float(np.interp(np.log10(c), np.log10(sub.training_flops), sub.validation_nll))
        xs.append(params_by_model[model])
        ys.append(nll)
    ax.semilogx(xs, ys, color=color, lw=2, marker="o", ms=6, label=f"C = {c:.0e}")
ax.set_xlabel("total parameters")
ax.set_ylabel("validation NLL (interpolated)")
ax.set_title("Iso-FLOP profiles")
ax.legend(fontsize=9)
param_axis_ticks(ax)

fig.suptitle("Compute scaling — cooldown checkpoints, FLOPs = per-run training_flops")
fig.tight_layout()
fig.savefig(FIGURES / "D_compute_frontier.png", dpi=150)

# %% [markdown]
# ## Panel E — model x delay heatmap
#
# Best net_stock_lcb over exposure. Hatched cell = not RTX 3060 latency-valid at that
# delay; blank = not evaluated.

# %%
grid = best_over_d.pivot_table(index="model", columns="delay_frames", values="eval_net_stock_lcb").reindex(MODELS)

fig, ax = plt.subplots(figsize=(10, 4.8))
mesh = ax.imshow(grid.to_numpy(), cmap=SEQ_CMAP, aspect="auto")
ax.set_xticks(range(len(DELAYS)), [f"{int(d)}" for d in DELAYS])
ax.set_yticks(range(len(MODELS)), [model_label(m) for m in MODELS])
ax.set_xlabel("imposed control delay (frames)")
ax.grid(False)

vmin, vmax = np.nanmin(grid.to_numpy()), np.nanmax(grid.to_numpy())
for i, model in enumerate(MODELS):
    for j, delay in enumerate(DELAYS):
        v = grid.iloc[i, j]
        if np.isnan(v):
            continue
        light = (v - vmin) / (vmax - vmin) < 0.55
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=INK if light else "#ffffff")
        if not valid.get((model, delay), False):
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="///", edgecolor=INK2, lw=0))

fig.colorbar(mesh, ax=ax, label="net_stock_lcb (best over exposure)", shrink=0.85)
ax.set_title("Capability grid — hatch = misses RTX 3060 frame budget, blank = not evaluated")
fig.tight_layout()
fig.savefig(FIGURES / "E_capability_grid.png", dpi=150)

# %% [markdown]
# ## Panel F — parametric fit and compute-optimal capacity
#
# The observed iso-FLOP profiles rise monotonically in N, so their minima sit at or below the
# smallest covered model. To locate the minima, fit the Chinchilla form
# L(N_eff, D) = E + A/N_eff^alpha + B/D^beta on the scaled-family cooldown runs, where
# N_eff = training_flops / (6 D) — the compute-relevant size from the FLOPs formula
# 6*D*(N_trunk + 36*N_decoder). The constraint C = 6 N_eff D then gives a closed-form optimum:
# N_opt^(alpha+beta) = (alpha A / (beta B)) (C/6)^beta.
#
# All fit runs are sub-epoch (D_over_U <= 0.22), so no data-repetition correction applies;
# predictions past ~2.4e9 positions (1 epoch) enter the repeated-data regime and are optimistic.

# %%
from scipy.optimize import minimize

fit_df = cooldowns[cooldowns.model.isin(SCALED)].copy()
fit_df["n_eff"] = fit_df.training_flops / (6 * fit_df.processed_positions)
# Probe models (L3/L4): prefix-only so far — their un-annealed NLL is not comparable to cooldown
# points, so they are shown as reference markers and kept out of the fit.
probes = runs[~runs.model.isin(MODELS) & runs.validation_nll.notna()].copy()
probes["n_eff"] = probes.training_flops / (6 * probes.processed_positions)
n_fit, d_fit, y_fit = fit_df.n_eff.to_numpy(), fit_df.processed_positions.to_numpy(), fit_df.validation_nll.to_numpy()


def huber_objective(p: np.ndarray) -> float:
    log_e, log_a, log_b, alpha, beta = p
    with np.errstate(over="ignore", invalid="ignore"):
        pred = np.exp(log_e) + np.exp(log_a) * n_fit**-alpha + np.exp(log_b) * d_fit**-beta
        r = pred - y_fit
        delta = 1e-3
        return float(np.sum(np.where(np.abs(r) < delta, 0.5 * r**2, delta * (np.abs(r) - 0.5 * delta))))


best = min(
    (
        minimize(
            huber_objective,
            [np.log(e0), la, lb, a0, b0],
            method="Nelder-Mead",
            options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-12},
        )
        for e0 in [1.2, 1.5, 1.7]
        for a0 in [0.1, 0.3, 0.5]
        for b0 in [0.1, 0.3, 0.5]
        for la in [0.0, 3.0, 6.0]
        for lb in [0.0, 3.0, 6.0]
    ),
    key=lambda r: r.fun,
)
E_FIT, A_FIT, B_FIT = np.exp(best.x[0]), np.exp(best.x[1]), np.exp(best.x[2])
ALPHA, BETA = best.x[3], best.x[4]
resid = E_FIT + A_FIT * n_fit**-ALPHA + B_FIT * d_fit**-BETA - y_fit
print(f"L(N_eff, D) = {E_FIT:.4f} + {A_FIT:.4g}/N^{ALPHA:.3f} + {B_FIT:.4g}/D^{BETA:.3f}")
print(f"rms residual {np.sqrt(np.mean(resid**2)):.4f} NLL, max {np.max(np.abs(resid)):.4f}, {len(fit_df)} points")


def loss_at(n_eff: np.ndarray, c: float) -> np.ndarray:
    return E_FIT + A_FIT * n_eff**-ALPHA + B_FIT * (c / (6 * n_eff)) ** -BETA


def n_eff_opt(c: float) -> float:
    return float((ALPHA * A_FIT / (BETA * B_FIT)) ** (1 / (ALPHA + BETA)) * (c / 6) ** (BETA / (ALPHA + BETA)))


n_eff_by_model = fit_df.groupby("model").n_eff.first()
C_PREDICT = 4.2e19

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

ax = axes[0]
for c, color in zip(FLOP_LEVELS, level_colors):
    grid_n = np.geomspace(n_eff_by_model.min() / 6, n_eff_by_model.max() * 1.5, 200)
    ax.plot(grid_n, loss_at(grid_n, c), color=color, lw=2, label=f"C = {c:.0e}")
    for model in SCALED:
        sub = cooldowns[cooldowns.model == model].sort_values("training_flops")
        if len(sub) < 2 or not (sub.training_flops.min() <= c <= sub.training_flops.max()):
            continue
        nll = float(np.interp(np.log10(c), np.log10(sub.training_flops), sub.validation_nll))
        ax.scatter(n_eff_by_model[model], nll, color=color, s=40, zorder=3)
    n_star = n_eff_opt(c)
    ax.scatter(
        n_star,
        loss_at(np.array([n_star]), c),
        marker="D",
        s=55,
        facecolor="#fcfcfb",
        edgecolor=color,
        lw=1.8,
        zorder=4,
    )
ax.set_xscale("log")
if not probes.empty:
    ax.scatter(
        probes.n_eff, probes.validation_nll, marker="s", s=45, facecolor="#fcfcfb", edgecolor=INK2, lw=1.6, zorder=3
    )
    ax.annotate(
        f"{', '.join(probes.model)} prefix\n(cooldown pending)",
        (float(probes.n_eff.mean()), float(probes.validation_nll.max())),
        textcoords="offset points",
        xytext=(0, 12),
        ha="center",
        fontsize=8,
        color=INK2,
    )
for model in SCALED:
    ax.axvline(n_eff_by_model[model], color=GRID, lw=0.8, zorder=0)
    ax.annotate(
        model,
        (n_eff_by_model[model], 0.99),
        xycoords=ax.get_xaxis_transform(),
        fontsize=8,
        color=INK2,
        ha="center",
        va="top",
    )
ax.set_xlabel("effective parameters (trunk + 36 x decoder)")
ax.set_ylabel("validation NLL")
ax.set_title("Fitted iso-FLOP parabolas (diamonds = predicted minima)")
ax.legend(fontsize=9, loc="upper right")

ax = axes[1]
c_grid = np.geomspace(1e16, 1e20, 100)
ax.loglog(
    c_grid,
    [n_eff_opt(c) for c in c_grid],
    color=MODEL_COLOR["L10"],
    lw=2,
    label=f"$N_{{opt}} \\propto C^{{{BETA / (ALPHA + BETA):.2f}}}$",
)
ax.axhspan(n_eff_by_model.min(), n_eff_by_model.max(), color=GRID, alpha=0.5, zorder=0, label="scaled family range")
for c in FLOP_LEVELS:
    ax.scatter(c, n_eff_opt(c), color=MODEL_COLOR["L10"], s=40, zorder=3)
n_star = n_eff_opt(C_PREDICT)
ax.scatter(C_PREDICT, n_star, marker="*", s=220, color="#eb6834", zorder=4, label=f"C = {C_PREDICT:.1e}")
d_star = C_PREDICT / (6 * n_star)
ax.annotate(
    f"N_eff = {n_star:.2e}\nD = {d_star:.2e} ({d_star / fit_df.unique_loss_positions.iloc[0]:.1f} epochs)\n"
    f"predicted NLL = {loss_at(np.array([n_star]), C_PREDICT)[0]:.3f}",
    (C_PREDICT, n_star),
    textcoords="offset points",
    xytext=(-10, -45),
    ha="right",
    fontsize=9,
)
ax.set_xlabel("training FLOPs")
ax.set_ylabel("compute-optimal effective parameters")
ax.set_title("Compute-optimal capacity, extrapolated")
ax.legend(fontsize=9, loc="upper left")

fig.suptitle("Iso-FLOP structure from the parametric fit (minima below the family range are extrapolations)")
fig.tight_layout()
fig.savefig(FIGURES / "F_isoflop_fit.png", dpi=150)

# Runs that bracket each predicted minimum: the nearest family sizes on each side of N_opt,
# each trained to D = C / (6 N_eff).
print("\nRuns needed to bracket each iso-FLOP minimum:")
for c in FLOP_LEVELS + [C_PREDICT]:
    n_star = n_eff_opt(c)
    lo = n_eff_by_model[n_eff_by_model <= n_star]
    hi = n_eff_by_model[n_eff_by_model > n_star]
    for side, series in (("below", lo.nlargest(2)), ("above", hi.nsmallest(2))):
        for model, n_eff in series.items():
            d_need = c / (6 * n_eff)
            covered = bool(((cooldowns.model == model) & (cooldowns.processed_positions >= d_need * 0.999)).any())
            note = "covered" if covered else f"NEEDED: cooldown @ 2^{np.log2(d_need):.1f} positions"
            print(f"  C={c:.1e}  {side:5s} minimum: {model:4s} D={d_need:.2e}  {note}")
    if lo.empty:
        print(f"  C={c:.1e}  below minimum: none — needs a model smaller than L5 (N_eff < {n_star:.2e})")

# %% [markdown]
# ## Frontier summary table

# %%
summary = frontier[["delay_frames", "model", "D_exp", "eval_net_stock_lcb", "eval_matches"]].copy()
summary["budget_ms"] = summary.delay_frames * FRAME_MS
summary["parameters"] = summary.model.map(params_by_model)
print(summary.to_string(index=False))
print(json.dumps({"figures": sorted(p.name for p in FIGURES.glob("*.png"))}, indent=2))
