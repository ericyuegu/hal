# %% [markdown]
# Backfill `eval/net_stock_lcb` and `eval/net_dmg_lcb` into past W&B runs.
#
# Both metrics are one-sided 95% lower confidence bounds on a net per-active-minute
# rate: net stocks = taken - lost, net damage = dealt - taken.
#
# Two tiers, by what each run logged:
# - CI tier: runs whose history has per-rate cluster-bootstrap CIs. The SE of the
#   net uses the conservative bound SE(a-b) <= SE(a) + SE(b) (the per-boot
#   covariance is not recoverable from the logged marginals).
# - sigma0 tier: older runs without CI keys. SE = sigma0 / sqrt(n_matches), with
#   sigma0 (per-match sd of the net rate) estimated from the CI-tier runs via
#   sigma0 = SE_net * sqrt(n_matches), pooled by median.
#
# Usage: `uv run notebooks/backfill_net_lcb.py` (dry run, prints a report), then
# `uv run notebooks/backfill_net_lcb.py --commit` to resume each run and log.

# %%
import math
import sys
from dataclasses import dataclass

import wandb

PROJECT = "hal"
Z = 1.645  # one-sided 95%
CI_Z = 1.96  # the logged CIs are two-sided 95%

# From /tmp/wandb_export_2026-08-09T13_46_48.424-07_00.csv.
RUN_IDS = [
    "shjnxxsu",
    "3sgvx6xv",
    "5b9x7mhq",
    "uw05bvm2",
    "vlim96s9",
    "m5e1kj7w",
    "rnbyn7n8",
    "1z6oyio7",
    "q3aojgfm",
    "obx3o3az",
    "4hoe86s7",
    "pu8jpmbf",
    "o3nihah5",
    "fxhtoxu8",
    "lxo3id9l",
    "6g1fud8e",
    "7tlk11io",
    "xgj1dwot",
]

RATE_KEYS = (
    "eval/stocks_taken_per_min",
    "eval/stocks_lost_per_min",
    "eval/damage_dealt_per_min",
    "eval/damage_taken_per_min",
)


# %%
@dataclass(frozen=True, slots=True)
class EvalPoint:
    step: float
    stocks_taken: float
    stocks_lost: float
    dmg_dealt: float
    dmg_taken: float
    matches: float | None
    # CI half-widths; None on sigma0-tier runs.
    hw_stocks_taken: float | None
    hw_stocks_lost: float | None
    hw_dmg_dealt: float | None
    hw_dmg_taken: float | None

    @property
    def has_ci(self) -> bool:
        return self.hw_stocks_taken is not None


def _half_width(row: dict, key: str) -> float | None:
    lo, hi = row.get(f"{key}_ci_lo"), row.get(f"{key}_ci_hi")
    if lo is None or hi is None or not (math.isfinite(lo) and math.isfinite(hi)):
        return None
    return (hi - lo) / 2


def eval_points(run: wandb.apis.public.Run) -> tuple[list[EvalPoint], bool]:
    """Distinct eval points from history, oldest first.

    Eval metrics are merged into the per-step log dict, so the same values repeat
    for thousands of rows; a point is kept where the rate tuple changes. Returns
    (points, has_global_step) — runs that predate the global-step-as-data
    convention fall back to `_step` for the x value.
    """
    points: list[EvalPoint] = []
    prev: tuple[float, ...] | None = None
    has_global_step = False
    for row in run.scan_history(page_size=2000):
        values = tuple(row.get(k) for k in RATE_KEYS)
        if any(v is None or not math.isfinite(v) for v in values):
            continue
        if values == prev:
            continue
        prev = values
        if row.get("global_step") is not None:
            has_global_step = True
        step = row.get("global_step", row["_step"])
        points.append(
            EvalPoint(
                step=float(step),
                stocks_taken=values[0],
                stocks_lost=values[1],
                dmg_dealt=values[2],
                dmg_taken=values[3],
                matches=row.get("eval/matches"),
                hw_stocks_taken=_half_width(row, "eval/stocks_taken_per_min"),
                hw_stocks_lost=_half_width(row, "eval/stocks_lost_per_min"),
                hw_dmg_dealt=_half_width(row, "eval/damage_dealt_per_min"),
                hw_dmg_taken=_half_width(row, "eval/damage_taken_per_min"),
            )
        )
    return points, has_global_step


# %%
def sigma0_estimates(points_by_run: dict[str, list[EvalPoint]]) -> tuple[float, float]:
    """Per-match sd of the two net rates, pooled by median over every CI-tier
    eval point that also logged a match count."""
    stock_sds: list[float] = []
    dmg_sds: list[float] = []
    for points in points_by_run.values():
        for p in points:
            if not p.has_ci or p.matches is None or p.matches <= 0:
                continue
            root_n = math.sqrt(p.matches)
            stock_sds.append((p.hw_stocks_taken + p.hw_stocks_lost) / CI_Z * root_n)
            dmg_sds.append((p.hw_dmg_dealt + p.hw_dmg_taken) / CI_Z * root_n)
    if not stock_sds:
        raise RuntimeError("no CI-tier eval points found; cannot estimate sigma0")
    stock_sds.sort()
    dmg_sds.sort()
    return stock_sds[len(stock_sds) // 2], dmg_sds[len(dmg_sds) // 2]


def lcbs(p: EvalPoint, *, sigma0_stock: float, sigma0_dmg: float, fallback_matches: float) -> tuple[float, float]:
    """(net_stock_lcb, net_dmg_lcb) for one eval point."""
    net_stock = p.stocks_taken - p.stocks_lost
    net_dmg = p.dmg_dealt - p.dmg_taken
    if p.has_ci:
        se_stock = (p.hw_stocks_taken + p.hw_stocks_lost) / CI_Z
        se_dmg = (p.hw_dmg_dealt + p.hw_dmg_taken) / CI_Z
    else:
        n = p.matches if p.matches is not None and p.matches > 0 else fallback_matches
        se_stock = sigma0_stock / math.sqrt(n)
        se_dmg = sigma0_dmg / math.sqrt(n)
    return net_stock - Z * se_stock, net_dmg - Z * se_dmg


def fallback_match_count(run: wandb.apis.public.Run) -> float:
    """Matches per eval wave for runs that logged neither CIs nor a match count.
    These runs predate instant restart, so one boot is one match and the config's
    matchup count is the wave's match count."""
    n = run.config.get("eval_n_matchups", run.config.get("eval_replicas"))
    if n is None:
        raise RuntimeError(f"run {run.id} has no eval/matches history and no eval_n_matchups/eval_replicas in config")
    return float(n)


# %%
def main(commit: bool) -> None:
    api = wandb.Api()
    runs = {run_id: api.run(f"{PROJECT}/{run_id}") for run_id in RUN_IDS}
    points_by_run: dict[str, list[EvalPoint]] = {}
    step_key_by_run: dict[str, bool] = {}
    for run_id, run in runs.items():
        points, has_global_step = eval_points(run)
        points_by_run[run_id] = points
        step_key_by_run[run_id] = has_global_step
        print(f"scanned {run_id} ({run.name[:50]}): {len(points)} eval points", flush=True)

    sigma0_stock, sigma0_dmg = sigma0_estimates(points_by_run)
    print(f"\nsigma0 (per-match sd, median-pooled): stocks {sigma0_stock:.3f}/min, damage {sigma0_dmg:.1f}/min\n")

    header = f"{'run':10} {'tier':7} {'points':>6} {'x-axis':11} {'final net_stock_lcb':>19} {'final net_dmg_lcb':>17}"
    print(header)
    for run_id, points in points_by_run.items():
        if not points:
            print(f"{run_id:10} {'-':7} {0:>6}  NO EVAL POINTS — skipped")
            continue
        run = runs[run_id]
        fallback = None if all(p.has_ci or p.matches for p in points) else fallback_match_count(run)
        tier = "ci" if points[-1].has_ci else "sigma0"
        x_axis = "global_step" if step_key_by_run[run_id] else "_step"
        stock_lcb, dmg_lcb = lcbs(
            points[-1], sigma0_stock=sigma0_stock, sigma0_dmg=sigma0_dmg, fallback_matches=fallback or 0.0
        )
        print(f"{run_id:10} {tier:7} {len(points):>6} {x_axis:11} {stock_lcb:>+19.3f} {dmg_lcb:>+17.1f}")

        if not commit:
            continue
        with wandb.init(project=PROJECT, id=run_id, resume="allow") as resumed:
            resumed.define_metric("eval/net_stock_lcb", step_metric="global_step")
            resumed.define_metric("eval/net_dmg_lcb", step_metric="global_step")
            for p in points:
                s, d = lcbs(p, sigma0_stock=sigma0_stock, sigma0_dmg=sigma0_dmg, fallback_matches=fallback or 0.0)
                resumed.log({"eval/net_stock_lcb": s, "eval/net_dmg_lcb": d, "global_step": p.step})
        print(f"{run_id:10} backfilled {len(points)} points")

    if not commit:
        print("\ndry run — re-run with --commit to write")


# %%
if __name__ == "__main__":
    main(commit="--commit" in sys.argv[1:])
