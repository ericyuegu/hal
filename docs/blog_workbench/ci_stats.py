"""Treatment-minus-baseline differences with 95% intervals for every number the article quotes.

Reads evidence.json (its ``computed_differences`` and raw boot rows) plus
wandb_sweep_evals.json (W&B aggregates for runs whose match rows were not archived), and
writes ci.json. Three estimators:

* ``paired``  — a paired t-test over Dolphin boots. Both arms play the same seeded match-ups,
  so boot i of the treatment is paired with boot i of the baseline; the per-boot net rate
  differences d_i give Δ = mean(d), CI = Δ ± t_{0.975, n−1} · sd(d)/√n. Main effects average
  several such pairs boot by boot.
* ``welch``   — arms with boot rows on different boot sets: Welch's t-test on per-boot rates.
* ``independent`` — arms with only W&B aggregates: Δ ± 1.96·sqrt(SE_T² + SE_B²), SE
  backed out of each arm's one-sided 95% net LCB (cluster bootstrap where W&B logged it,
  otherwise the conservative LCB).

Run: uv run --no-project ci_stats.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).parent
DATA = json.loads((HERE / "evidence.json").read_text())
SWEEP = json.loads((HERE / "wandb_sweep_evals.json").read_text())
EVALS = DATA["raw_evaluation_data"]["evaluations"]
RESAMPLES = 10_000
Z = 1.96
LCB_Z = 1.645
METRICS = {"stock": ("stocks_taken", "stocks_lost"), "damage": ("damage_dealt", "damage_taken")}


def boots(eval_id: str) -> dict:
    """Per-boot net rates, keyed by boot index, with the match-up each boot played."""
    out = {}
    for r in EVALS[eval_id]["boots"]:
        minutes = r["active_frames"] / 3600.0
        if minutes <= 0:
            continue
        out[r["boot_index"]] = {
            "matchup": (r["ego_character"], r["opp_character"]),
            "stock": (r["stocks_taken"] - r["stocks_lost"]) / minutes,
            "damage": (r["damage_dealt"] - r["damage_taken"]) / minutes,
        }
    return out


def paired(treat_ids: list[str], control_ids: list[str], *, weights=None) -> dict:
    """Paired t-test on per-boot differences; ``weights`` lets a main effect average several (T − C) pairs."""
    arms = [boots(e) for e in treat_ids + control_ids]
    shared = sorted(set.intersection(*(set(a) for a in arms)))
    for i in shared:  # the schedule is seeded, so paired boots must have played the same match-up
        assert len({a[i]["matchup"] for a in arms}) == 1, f"boot {i} is not the same match-up in every arm"
    k = len(treat_ids)
    weights = weights or [1.0 / k] * k
    n = len(shared)
    out = {"method": "paired", "n_boots": n}
    for m in METRICS:
        d = np.array([sum(w * (arms[j][i][m] - arms[k + j][i][m]) for j, w in enumerate(weights)) for i in shared])
        se = d.std(ddof=1) / math.sqrt(n)
        half = stats.t.ppf(0.975, n - 1) * se
        out[m] = {"delta": float(d.mean()), "lo": float(d.mean() - half), "hi": float(d.mean() + half)}
    return out


def arm_se(run: str, metric: str) -> tuple[float, float]:
    """``run`` may carry a W&B eval namespace suffix, e.g. ``feic18gt@shared_k``."""
    run, _, ns = run.partition("@")
    s = {k.replace(f"eval/{ns}/" if ns else "eval/", "eval/"): v for k, v in SWEEP[run].items()}
    key = {"stock": "net_stock", "damage": "net_dmg"}[metric]
    mean = s.get(f"eval/{key}_per_min")
    if mean is None:  # June-era runs: only the four marginal rates were kept
        pos, neg = {"stock": ("stocks_taken", "stocks_lost"), "damage": ("damage_dealt", "damage_taken")}[metric]
        mean = s[f"eval/{pos}_per_min"] - s[f"eval/{neg}_per_min"]
    lcb = s.get(f"eval/{key}_cluster_bootstrap_lcb", s.get(f"eval/{key}_lcb"))
    return mean, (mean - lcb) / LCB_Z


def independent(treat: str, control: str) -> dict:
    out = {"method": "independent", "runs": [treat, control]}
    for m in METRICS:
        mt, st = arm_se(treat, m)
        mc, sc = arm_se(control, m)
        se = math.sqrt(st**2 + sc**2)
        out[m] = {"delta": mt - mc, "lo": mt - mc - Z * se, "hi": mt - mc + Z * se}
    return out


def welch(treat_id: str, control_id: str) -> dict:
    """Welch's t-test on per-boot rates for arms evaluated on different boot sets."""
    a, b = boots(treat_id), boots(control_id)
    out = {"method": "welch", "n_boots": {"treatment": len(a), "control": len(b)}}
    for m in METRICS:
        x = np.array([r[m] for r in a.values()])
        y = np.array([r[m] for r in b.values()])
        res = stats.ttest_ind(x, y, equal_var=False)
        lo, hi = res.confidence_interval(0.95)
        out[m] = {"delta": float(x.mean() - y.mean()), "lo": float(lo), "hi": float(hi)}
    return out


def study_pair(study_id: str, treatment: str | None = None) -> dict:
    """Paired t-test (shared match-ups) or Welch's t-test for a study whose evaluation ids the evidence file links."""
    study = next(s for s in DATA["studies"] if s["id"] == study_id)
    diffs = study["computed_differences"]
    d = diffs[0] if treatment is None else next(x for x in diffs if x["treatment"] == treatment)
    shared = "shared" in d["net_stocks_per_min"]["method"]
    est = (
        paired([d["treatment_evaluation_id"]], [d["control_evaluation_id"]])
        if shared
        else welch(d["treatment_evaluation_id"], d["control_evaluation_id"])
    )
    return {**est, "treatment": d["treatment"], "control": d["control"]}


def main() -> None:
    ci: dict[str, dict] = {}
    for sid, names in {
        "codec_legacy_cartesian": [None],
        "codec_cartesian_polar": [None],
        "projectile_inputs": [None],
        "past_action_encoding": [None],
        "onehot_button_pruning": [None],
        "lean_feature_bundle": [None],
        "rle_bpe_actions": [None],
        "o26_to_o42_package": [None],
        "old_head_capacity": [None],
        "future_state_expert": [None],
        "fixed_long_delay": [None],
        "rank_weighted_bc": [None],
        "awr_gamma": ["gamma 0.99855", "gamma 0.99920"],
        "endpoint_flow": [None],
        "attached_iql": [None],
        "exploration_modeling": ["XM K=1"],
        "data_source_mix": [None],
        "next_frame_loss_share": [None],
        "player_identity": ["Cody", "Master", "Mang0", "Platinum"],
        "optimizer_full_scale": [None],
    }.items():
        for name in names:
            ci[sid if name is None else f"{sid}:{name}"] = study_pair(sid, name)

    # O37 cells: H4 evaluations on one matchup schedule
    o37 = {
        "D0": "eval-8faedef527ee22bd",
        "D1": "eval-594758832356aa99",
        "D2": "eval-3c708684883d74e4",
        "D3": "eval-6de7c060d87a3563",
    }
    for cell in ("D1", "D2", "D3"):
        ci[f"o37:{cell}-D0"] = paired([o37[cell]], [o37["D0"]])
    ci["o37:across"] = paired([o37["D2"], o37["D3"]], [o37["D0"], o37["D1"]])
    ci["o37:within"] = paired([o37["D1"], o37["D3"]], [o37["D0"], o37["D2"]])

    # execution horizon: each O37 checkpoint at H1/H2/H4/H6, H1 is the baseline
    horizon = {
        "D0": {
            1: "eval-3af1bd0328372c4a",
            2: "eval-1323a35ba2ca7499",
            4: "eval-8faedef527ee22bd",
            6: "eval-4aee47edf0292576",
        },
        "D1": {
            1: "eval-6693c8d046f88f19",
            2: "eval-69e295c242a56d57",
            4: "eval-594758832356aa99",
            6: "eval-e2da17bfecfe3be9",
        },
        "D2": {
            1: "eval-529b88b2e3e0e1f5",
            2: "eval-74ba2519266056e0",
            4: "eval-3c708684883d74e4",
            6: "eval-538872d018746623",
        },
        "D3": {
            1: "eval-1dda80e14427aa4b",
            2: "eval-f0c055ff056da704",
            4: "eval-6de7c060d87a3563",
            6: "eval-60cff3532bd1c8bf",
        },
    }
    for cell, evs in horizon.items():
        for h in (2, 4, 6):
            ci[f"horizon:{cell}:H{h}-H1"] = paired([evs[h]], [evs[1]])

    # runs without archived match rows: independent SEs from W&B aggregates
    ci["awr_actor_weighting"] = independent("hwzv0k9a", "5wfk2esf")
    ci["exploration_modeling:XM K=8"] = independent("feic18gt@shared_k", "cqbbbg77")
    ci["windows_per_replay"] = independent("shjnxxsu", "uw05bvm2")
    batch = {128: "nt4gh4mk", 256: "phxqvexa", 512: "980k3yla", 1024: "7ok39dju", 2048: "rri7scjc"}
    for b, run in batch.items():
        if b != 2048:
            ci[f"batch:{b}-2048"] = independent(run, batch[2048])
    lr = {
        "0.014": "9luwu85n",
        "0.028": "wmwellfd",
        "0.040": "7hy8xjjh",
        "0.056": "2rbed6ze",
        "0.080": "5adh9mtu",
        "0.112": "zerhe8au",
    }
    for k, run in lr.items():
        if k != "0.014":
            ci[f"lr:{k}-0.014"] = independent(run, lr["0.014"])

    (HERE / "ci.json").write_text(json.dumps(ci, indent=1))
    for k, v in ci.items():
        s, d = v["stock"], v["damage"]
        print(
            f"{k:42s} {v['method']:11s} stock {s['delta']:+.3f} [{s['lo']:+.3f}, {s['hi']:+.3f}]   dmg {d['delta']:+.1f} [{d['lo']:+.1f}, {d['hi']:+.1f}]"
        )


if __name__ == "__main__":
    main()
