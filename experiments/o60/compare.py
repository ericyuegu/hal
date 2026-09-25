"""Compare the complete corrected O52 suite with the single final O60 suite."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import numpy.typing as npt


def load_boots(directory: Path) -> tuple[dict[str, object], npt.NDArray[np.float64]]:
    payload = json.loads((directory / "match_rows.json").read_text())
    metrics = json.loads((directory / "metrics.json").read_text())
    if payload["schema_version"] != 7:
        raise ValueError("only corrected schema-7 match rows are eligible")
    for field in ("boots", "completed_boots", "scheduled_boots"):
        if metrics[field] != 96:
            raise ValueError(f"incomplete evaluation: {field}={metrics[field]}")
    if metrics["crashed"] != 0:
        raise ValueError("evaluation contains crashes")
    protocol = payload["protocol"]
    reference = json.loads(Path(__file__).with_name("control-match-rows.json").read_text())["protocol"]
    reference["checkpoint_sha256"] = protocol["checkpoint_sha256"]
    if protocol != reference:
        raise ValueError("evaluation protocol differs from corrected control")
    totals = np.zeros((96, 3), dtype=np.float64)
    schedule: dict[int, tuple[int, int]] = {}
    ordinals: set[tuple[int, int]] = set()
    frames = np.zeros(96, dtype=np.int64)
    for row in payload["rows"]:
        boot = row["boot_index"]
        if not isinstance(boot, int) or not 0 <= boot < 96:
            raise ValueError("invalid boot index")
        key = (boot, row["match_ordinal"])
        if key in ordinals:
            raise ValueError("duplicate boot/match row")
        ordinals.add(key)
        matchup = (row["ego_character"], row["opp_character"])
        if row["stage"] != protocol["seed_stage"] or schedule.setdefault(boot, matchup) != matchup:
            raise ValueError("boot matchup changed")
        active = row["active_frames"]
        total = row["total_frames"]
        if not 0 <= active <= total:
            raise ValueError("invalid frame counts")
        frames[boot] += total
        if active:
            totals[boot] += (
                row["stocks_taken"] - row["stocks_lost"],
                row["damage_dealt"] - row["damage_taken"],
                active / 3600,
            )
    if set(schedule) != set(range(96)) or np.any(frames != 7200) or np.any(totals[:, 2] <= 0):
        raise ValueError("suite does not contain 96 complete active boots")
    encoded = json.dumps([schedule[index] for index in range(96)], separators=(",", ":")).encode()
    if hashlib.sha256(encoded).hexdigest() != protocol["matchup_schedule_sha256"]:
        raise ValueError("boot schedule hash differs")
    rates = totals[:, :2].sum(0) / totals[:, 2].sum()
    for value, key in zip(rates, ("net_stock_per_min", "net_dmg_per_min"), strict=True):
        if not math.isclose(value, metrics[key], rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError(f"rows disagree with persisted {key}")
    return protocol, totals


def paired_comparison(
    control: npt.NDArray[np.float64],
    treatment: npt.NDArray[np.float64],
    *,
    seed: int = 60,
    resamples: int = 2000,
) -> dict[str, object]:
    if control.shape != (96, 3) or treatment.shape != (96, 3):
        raise ValueError("paired comparison requires 96 boot totals")
    if not all(np.isfinite(values).all() and np.all(values[:, 2] > 0) for values in (control, treatment)):
        raise ValueError("invalid boot totals")
    index = np.random.default_rng(seed).integers(0, 96, size=(resamples, 96))
    points = []
    draws = []
    for values in (control, treatment):
        points.append(values[:, :2].sum(0) / values[:, 2].sum())
        sampled = values[index].sum(1)
        draws.append(sampled[:, :2] / sampled[:, 2, None])
    intervals = np.quantile(draws[0] - draws[1], [0.025, 0.975], axis=0)
    return {
        "seed": seed,
        "resamples": resamples,
        "direction": "AWR-minus-BC",
        "metrics": {
            name: {
                "awr": float(points[0][column]),
                "bc": float(points[1][column]),
                "difference": float(points[0][column] - points[1][column]),
                "ci95": intervals[:, column].tolist(),
            }
            for column, name in enumerate(("net_stocks_per_min", "net_damage_per_min"))
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("treatment", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    control_protocol, control = load_boots(args.control)
    treatment_protocol, treatment = load_boots(args.treatment)
    if control_protocol["checkpoint_sha256"] != "16c702fe3964a59c2f26d88207ef90137d93c07bf67a5d4213f4fde5d25b8631":
        raise ValueError("wrong AWR control checkpoint")
    result = paired_comparison(control, treatment)
    result["protocols"] = {"awr": control_protocol, "bc": treatment_protocol}
    result["evidence_sha256"] = {
        str(directory / filename): hashlib.sha256((directory / filename).read_bytes()).hexdigest()
        for directory in (args.control, args.treatment)
        for filename in ("match_rows.json", "metrics.json")
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
