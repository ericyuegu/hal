"""Series behind the hand-written figures that evidence.json does not carry.

Writes figures_src/figure-data.json (inlined by build.py as ``window.HALFIG_DATA``):

* ``o50``        — validation NLL per 4,096 updates of the O50 Muon production run (W&B ``val/nll``)
                   next to the nine gameplay checkpoints from evidence.json.
* ``optimizer``  — final validation loss of both arms of the two optimizer tests (W&B).
* ``iso_flop``   — the 33 O39 capacity-sweep runs: N_eff, D, validation NLL, the published fit
                   L(N, D) and each point's residual. N_eff = trunk + 36 · decoder, the parameter count
                   the run's own FLOP formula 6 · D · N_eff uses (results/039_capacity_scaling).
                   Also ``checkpoints`` (every non-diverged scaled-family cooldown checkpoint, exact
                   iso-FLOP endpoints included), ``levels`` (measured NLL per model at four compute
                   budgets, read off exact hits or log-interpolated inside the model's range, as in
                   notebooks/039_scaling_viz.py), ``family`` (the N_eff range the fit saw),
                   ``u_total`` (distinct corpus positions) and ``production`` — the O50 run's budget in
                   the fit's units: D = 2³³ supervised positions (131,072 updates × 512 × 128), N_eff =
                   trunk + 20 · decoder with decoder = temporal decoder + group heads (the modules O39
                   counts as codec + temporal) and 20 = its sample_chunk_length, from the
                   parameter_count_contract in experiments/050_scaled_temporal_awr.py.
* ``tier``       — O54: S(D) = S₂₈ + A · (1 − (D/2²⁸)^−α) fitted per model through D = 2³⁰, and the
                   D = 2³¹ prediction next to the observation.

Run from the repo root:  uv run --no-project --with wandb,pandas,scipy docs/blog_workbench/figure_data.py [--results <039 csv>]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EVIDENCE = json.loads((HERE / "evidence.json").read_text())
STUDIES = {s["id"]: s for s in EVIDENCE["studies"]}
OUT = HERE / "figures_src" / "figure-data.json"
FIT = {"base": 1.6350, "aScale": 89.11, "aPower": 0.368, "dScale": 2009, "dPower": 0.498}
FLOP_LEVELS = [5e16, 1e17, 3e17, 1e18]
DIVERGENCE_NLL_MARGIN = 0.01
# experiments/050_scaled_temporal_awr.py Architecture.parameter_count_contract, run p1fyyp1z
O50 = {
    "trunk": 201_326_592,
    "temporal_decoder": 11_296_112,
    "group_heads": 1_594_211,
    "value_head": 1_049_089,
    "other": 1_230_790,
}
O50_TOTAL = 216_496_794
O50_POSITIONS = 131_072 * 512 * 128
O50_DECODER_PASSES = 20


def val_series(api: wandb.Api, run_id: str, key: str) -> list[tuple[int, float]]:
    run = api.run(f"ericyuegu/hal/{run_id}")
    rows = run.scan_history(keys=["global_step", key])
    out = [(int(r["global_step"]), float(r[key])) for r in rows if r.get(key) is not None]
    steps = [u for u, _ in out]
    assert steps == sorted(set(steps)), f"{run_id}: {key} has duplicate or unordered steps (a resumed run?)"
    return out


def val_at(api: wandb.Api, run_id: str, key: str, step: int) -> float:
    return dict(val_series(api, run_id, key))[step]


def o50(api: wandb.Api) -> dict:
    """Validation NLL from W&B; every archived 96-boot evaluation of the run from evidence.json."""
    s = STUDIES["training_checkpoint_series"]
    nll = val_series(api, s["run"], "val/nll")
    evals = []
    for eval_id, e in EVIDENCE["raw_evaluation_data"]["evaluations"].items():
        if s["run"] not in e.get("run_ids", []):
            continue
        step = re.search(r"step-0*(\d+)", e["sources"][0]["path"])
        assert step, f"{eval_id} has no checkpoint step in its source path"
        st, dmg = e["statistics"]["net_stocks_per_min"], e["statistics"]["net_damage_per_min"]
        evals.append(
            {
                "updates": int(step.group(1)),
                "evaluation_id": eval_id,
                "boots": e["statistics"]["n_boots"],
                "stock": st["pooled_mean"],
                "stock_ci95": st["ci95"],
                "damage": dmg["pooled_mean"],
                "damage_ci95": dmg["ci95"],
            }
        )
    evals.sort(key=lambda e: e["updates"])
    assert [e["updates"] for e in evals if e["updates"] in s["series"]["updates"]] == s["series"]["updates"]
    return {"run": s["run"], "val_updates": [u for u, _ in nll], "val_nll": [v for _, v in nll], "evals": evals}


def optimizer(api: wandb.Api) -> dict:
    out = {}
    for run_id, arm in [("o3nihah5", "AdamW"), ("vlim96s9", "Muon + AdamW")]:
        out[run_id] = {
            "arm": arm,
            "val_loss": float(api.run(f"ericyuegu/hal/{run_id}").summary["val/loss"]),
            "metric": "val/loss",
        }
    for run_id, arm in [
        ("p1fyyp1z", "Muon + AdamW"),
        ("tgpuo1be", "AdamW"),
    ]:  # both compared at their final 131,072nd update
        out[run_id] = {
            "arm": arm,
            "val_loss": val_at(api, run_id, "val/nll", 131072),
            "metric": "val/nll",
            "updates": 131072,
        }
    return out


def iso_flop(csv: Path) -> dict:
    if not csv.exists():
        raise SystemExit(
            f"{csv} is missing: results/ is untracked, pass --results <path to experiment_039_raw_outcomes.csv>"
        )
    df = pd.read_csv(csv, low_memory=False)
    runs = df[df.record_type == "run"].drop_duplicates("run_id").set_index("run_id")
    # the model × 2^k grid after cooldown; prefix checkpoints, crashed runs and the iso-FLOP
    # endpoints at non-power-of-two D are left out
    runs = runs[
        (runs.phase == "cooldown")
        & (runs.wandb_state == "finished")
        & runs.D_exp.notna()
        & runs.validation_nll.notna()
    ]
    pts = []
    for run_id, r in runs.sort_values(["total_parameters", "D_exp"]).iterrows():
        n_eff = int(r.trunk_parameters + 36 * r.decoder_parameters)
        d = 2.0 ** int(r.D_exp)
        pred = FIT["base"] + FIT["aScale"] / n_eff ** FIT["aPower"] + FIT["dScale"] / d ** FIT["dPower"]
        pts.append(
            {
                "run": run_id,
                "model": str(r.model),
                "layers": int(r.L),
                "d_model": int(r.d_model),
                "total_parameters": int(r.total_parameters),
                "n_eff": n_eff,
                "d_exp": int(r.D_exp),
                "training_flops": float(r.training_flops),
                "nll": float(r.validation_nll),
                "fit_nll": float(pred),
                "residual": float(r.validation_nll - pred),
            }
        )
    assert len(pts) == 33, f"expected the 33 cooled-down O39 grid runs, got {len(pts)}"

    u_total = float(runs.unique_loss_positions.iloc[0])
    assert (df[df.record_type == "run"].unique_loss_positions.dropna() == u_total).all(), (
        "runs disagree on corpus size"
    )

    # every scaled-family cooldown checkpoint the notebook's fit and iso-FLOP profiles use: exact
    # endpoints at non-power-of-two D included, checkpoints that regressed past the model's best
    # earlier NLL (the optimizer-failed L10 1e18 endpoint) excluded
    ck = df[df.record_type == "run"].drop_duplicates("run_id").set_index("run_id")
    ck = ck[
        (ck.model_family == "scaled")
        & (ck.phase == "cooldown")
        & (ck.wandb_state == "finished")
        & ck.validation_nll.notna()
        & ~ck.analysis_excluded.eq(True)
    ]
    ck = ck.drop_duplicates(subset=["model", "processed_positions"]).sort_values(["model", "processed_positions"])
    best_prior = ck.groupby("model").validation_nll.cummin().groupby(ck.model).shift(1)
    ck = ck[~(ck.validation_nll - best_prior > DIVERGENCE_NLL_MARGIN).fillna(False)]
    checkpoints = []
    for r in ck.sort_values(["total_parameters", "processed_positions"]).itertuples():
        n_eff = int(r.trunk_parameters + 36 * r.decoder_parameters)
        assert abs(n_eff - r.training_flops / (6 * r.processed_positions)) / n_eff < 1e-6, r.Index
        checkpoints.append(
            {
                "run": r.Index,
                "model": str(r.model),
                "n_eff": n_eff,
                "positions": float(r.processed_positions),
                "training_flops": float(r.training_flops),
                "nll": float(r.validation_nll),
                "exact": bool(r.endpoint_exact),
            }
        )
    assert len(checkpoints) == 35, f"expected 35 non-diverged scaled-family checkpoints, got {len(checkpoints)}"

    levels = []
    for c in FLOP_LEVELS:
        level = []
        for model, rows in ck.groupby("model", sort=False):
            rows = rows.sort_values("training_flops")
            exact = rows[np.isclose(rows.training_flops, c, rtol=0.01)]
            if not exact.empty:
                hit = exact.iloc[0]
                point = {"nll": float(hit.validation_nll), "exact": True, "runs": [hit.name]}
            elif len(rows) >= 2 and rows.training_flops.min() <= c <= rows.training_flops.max():
                above = int(np.searchsorted(rows.training_flops.to_numpy(), c))
                lo, hi = rows.iloc[above - 1], rows.iloc[above]
                nll = float(
                    np.interp(
                        np.log10(c),
                        np.log10([lo.training_flops, hi.training_flops]),
                        [lo.validation_nll, hi.validation_nll],
                    )
                )
                point = {
                    "nll": nll,
                    "exact": False,
                    "runs": [lo.name, hi.name],
                    "between_positions": [float(lo.processed_positions), float(hi.processed_positions)],
                }
            else:
                continue
            level.append(
                {
                    "model": str(model),
                    "n_eff": int(rows.trunk_parameters.iloc[0] + 36 * rows.decoder_parameters.iloc[0]),
                    **point,
                }
            )
        level.sort(key=lambda p: p["n_eff"])
        levels.append({"c": c, "points": level})
    assert [len(level["points"]) for level in levels] == [5, 6, 6, 3], [len(level["points"]) for level in levels]

    decoder = O50["temporal_decoder"] + O50["group_heads"]
    assert sum(O50.values()) == O50_TOTAL
    o50_n_eff = O50_TOTAL - decoder + O50_DECODER_PASSES * decoder
    return {
        "fit": FIT,
        "compute": "6 · D · N_eff",
        "n_eff": "trunk + 36 · decoder",
        "source_csv": str(csv),
        "points": pts,
        "checkpoints": checkpoints,
        "levels": levels,
        "family": {"n_min": min(c["n_eff"] for c in checkpoints), "n_max": max(c["n_eff"] for c in checkpoints)},
        "u_total": u_total,
        "production": {
            "run": "p1fyyp1z",
            "positions": O50_POSITIONS,
            "trunk": O50_TOTAL - decoder,
            "decoder": decoder,
            "passes": O50_DECODER_PASSES,
            "n_eff": o50_n_eff,
            "flops": 6.0 * O50_POSITIONS * o50_n_eff,
        },
    }


def tier() -> dict:
    s = STUDIES["capacity_and_latency"]
    by: dict[int, dict[int, dict]] = {}
    for a in s["arms"]:
        by.setdefault(int(a["name"].split("/")[0][1:]), {})[a["data_exponent"]] = a

    def curve(p, d_exp):
        s28, amp, alpha = p
        return s28 + amp * (1 - (2.0 ** (d_exp - 28)) ** (-alpha))

    models = []
    for width, rows in sorted(by.items()):
        d_exp = np.array([28.0, 29.0, 30.0])
        y = np.array([rows[int(e)]["stock"] for e in d_exp])
        # docs/experiments/054_bc_capacity_latency.md: A >= 0, 0.01 <= alpha <= 4; S28 free
        fit = least_squares(
            lambda p, e=d_exp, target=y: curve(p, e) - target,
            x0=[y[0], 0.5, 0.5],
            bounds=([-np.inf, 0, 0.01], [np.inf, np.inf, 4]),
        )
        first = rows[28]
        models.append(
            {
                "width": width,
                "delay_frames": first["delay_frames"],
                "bc_parameters": first["bc_parameters"],
                "run": first["run"],
                "fit": {"s28": float(fit.x[0]), "amplitude": float(fit.x[1]), "alpha": float(fit.x[2])},
                "points": [
                    {"d_exp": e, "stock": rows[e]["stock"], "ci95": rows[e]["stock_ci95"]} for e in (28, 29, 30, 31)
                ],
                "predicted_d31": float(curve(fit.x, np.array([31.0]))[0]),
                "observed_d31": rows[31]["stock"],
                "observed_d31_ci95": rows[31]["stock_ci95"],
            }
        )
    predicted = [m["width"] for m in sorted(models, key=lambda m: -m["predicted_d31"])]
    assert predicted == s["holdout"]["predicted_d31_order"], (predicted, s["holdout"]["predicted_d31_order"])
    return {"fit_through_d_exp": 30, "models": models, "holdout": s["holdout"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--results", type=Path, default=REPO / "results" / "039_capacity_scaling" / "experiment_039_raw_outcomes.csv"
    )
    args = ap.parse_args()
    api = wandb.Api()
    data = {"o50": o50(api), "optimizer": optimizer(api), "iso_flop": iso_flop(args.results), "tier": tier()}
    OUT.write_text(json.dumps(data, indent=1))
    print(OUT, f"{OUT.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
