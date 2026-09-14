"""Consolidate closed-loop blog evidence into one deduplicated JSON file."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import wandb

PROJECT = "ericyuegu/hal"
R2_ROOT = "r2:hal/runs"
RUN_ID_RE = re.compile(r"^[a-z0-9]{8}$")
ROW_FILENAMES = {"match_rows.json", "evaluation.json"}
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
FRAMES_PER_MINUTE = 3_600.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("docs/blog_workbench/evidence.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/blog_workbench/evidence.json"),
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--skip-r2", action="store_true")
    return parser.parse_args()


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    return str(value)


def collect_run_ids(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value} if RUN_ID_RE.fullmatch(value) else set()
    if isinstance(value, dict):
        found: set[str] = set()
        for item in value.values():
            found.update(collect_run_ids(item))
        return found
    if isinstance(value, list):
        found = set()
        for item in value:
            found.update(collect_run_ids(item))
        return found
    return set()


def useful_config(config: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "L_ctx",
        "batch_size",
        "d_model",
        "data_root",
        "eval_max_frames",
        "eval_n_matchups",
        "exec_horizon",
        "final_eval_n_matchups",
        "head_offsets",
        "max_steps",
        "n_heads",
        "n_layers",
        "replan_interval",
        "seed",
        "temporal_d_model",
        "temporal_layers",
        "transport_delay",
    )
    return {key: json_safe(config[key]) for key in keys if key in config}


def useful_summary(summary: dict[str, Any]) -> dict[str, Any]:
    wanted_fragments = (
        "net_stock_per_min",
        "net_dmg_per_min",
        "net_stock_lcb",
        "net_dmg_lcb",
        "completed_boots",
        "scheduled_boots",
        "/boots",
        "/matches",
        "/crashed",
        "/exec_horizon",
    )
    return {
        key: json_safe(value)
        for key, value in summary.items()
        if any(fragment in key for fragment in wanted_fragments)
    }


def fetch_run_metadata(run_id: str) -> tuple[str, dict[str, Any]]:
    run = wandb.Api(timeout=45).run(f"{PROJECT}/{run_id}")
    return run_id, {
        "run_id": run_id,
        "name": run.name,
        "url": run.url,
        "state": run.state,
        "config": useful_config(dict(run.config)),
        "aggregate_metrics": useful_summary(dict(run.summary)),
    }


def run_command(command: list[str], timeout: float = 120.0) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        timeout=timeout,
    )


def list_r2_sources(run_name: str) -> tuple[str, list[str], str | None]:
    root = f"{R2_ROOT}/{run_name}"
    command = [
        "rclone",
        "lsf",
        root,
        "--recursive",
        "--files-only",
        "--include",
        "**/match_rows.json",
        "--include",
        "**/evaluation.json",
    ]
    try:
        result = run_command(command)
    except subprocess.TimeoutExpired:
        return run_name, [], "R2 listing timed out"
    if result.returncode != 0:
        error = result.stderr.decode(errors="replace").strip()
        return run_name, [], error or f"rclone exited {result.returncode}"
    paths = [f"{root}/{line}" for line in result.stdout.decode().splitlines() if line]
    return run_name, paths, None


def read_r2_source(path: str) -> tuple[str, bytes | None, str | None]:
    try:
        result = run_command(["rclone", "cat", path])
    except subprocess.TimeoutExpired:
        return path, None, "R2 read timed out"
    if result.returncode != 0:
        error = result.stderr.decode(errors="replace").strip()
        return path, None, error or f"rclone exited {result.returncode}"
    return path, result.stdout, None


def local_sources(run_name: str) -> list[Path]:
    run_dir = Path("runs") / run_name
    if not run_dir.is_dir():
        return []
    return sorted(path for path in run_dir.rglob("*.json") if path.name in ROW_FILENAMES)


def normalize_row(row: dict[str, Any]) -> dict[str, int | float]:
    total_frames = int(row.get("total_frames", 0))
    return {
        "active_frames": int(row.get("active_frames", max(0, total_frames - 123))),
        "boot_index": int(row["boot_index"]),
        "damage_dealt": float(row["damage_dealt"]),
        "damage_taken": float(row["damage_taken"]),
        "ego_character": int(row["ego_character"]),
        "match_ordinal": int(row.get("match_ordinal", 0)),
        "opp_character": int(row["opp_character"]),
        "stage": int(row.get("stage", -1)),
        "stocks_lost": int(row["stocks_lost"]),
        "stocks_taken": int(row["stocks_taken"]),
        "total_frames": total_frames,
    }


def aggregate_boots(rows: list[dict[str, int | float]]) -> list[dict[str, int | float]]:
    grouped: dict[int, list[dict[str, int | float]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["boot_index"])].append(row)

    boots: list[dict[str, int | float]] = []
    for boot_index, boot_rows in sorted(grouped.items()):
        matchups = {(int(row["ego_character"]), int(row["opp_character"])) for row in boot_rows}
        if len(matchups) != 1:
            raise ValueError(f"boot {boot_index} contains multiple matchups: {sorted(matchups)}")
        ego_character, opp_character = next(iter(matchups))
        active = [row for row in boot_rows if int(row["active_frames"]) > 0]
        boots.append(
            {
                "active_frames": sum(int(row["active_frames"]) for row in active),
                "boot_index": boot_index,
                "damage_dealt": sum(float(row["damage_dealt"]) for row in active),
                "damage_taken": sum(float(row["damage_taken"]) for row in active),
                "ego_character": ego_character,
                "matches": len(active),
                "opp_character": opp_character,
                "stocks_lost": sum(int(row["stocks_lost"]) for row in active),
                "stocks_taken": sum(int(row["stocks_taken"]) for row in active),
                "total_frames": sum(int(row["total_frames"]) for row in boot_rows),
            }
        )
    return boots


def rate_statistics(
    numerator: np.ndarray,
    active_minutes: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> dict[str, float | int | list[float]]:
    valid = active_minutes > 0.0
    numerator = numerator[valid]
    active_minutes = active_minutes[valid]
    if not len(numerator):
        return {
            "boot_rate_mean": 0.0,
            "boot_rate_sd": 0.0,
            "boot_rate_se": 0.0,
            "ci95": [0.0, 0.0],
            "n_boots": 0,
            "pooled_mean": 0.0,
            "pooled_bootstrap_se": 0.0,
        }

    rates = numerator / active_minutes
    pooled = float(numerator.sum() / active_minutes.sum())
    sampled_num = numerator[bootstrap_indices].sum(axis=1)
    sampled_minutes = active_minutes[bootstrap_indices].sum(axis=1)
    sampled_rates = sampled_num / sampled_minutes
    lo, hi = np.percentile(sampled_rates, [2.5, 97.5])
    sample_sd = float(rates.std(ddof=1)) if len(rates) > 1 else 0.0
    return {
        "boot_rate_mean": float(rates.mean()),
        "boot_rate_sd": sample_sd,
        "boot_rate_se": sample_sd / math.sqrt(len(rates)),
        "ci95": [float(lo), float(hi)],
        "n_boots": len(rates),
        "pooled_mean": pooled,
        "pooled_bootstrap_se": float(sampled_rates.std(ddof=1)),
    }


def evaluation_statistics(boots: list[dict[str, int | float]]) -> dict[str, Any]:
    active = [boot for boot in boots if int(boot["active_frames"]) > 0]
    if not active:
        return {"n_boots": 0, "n_matches": 0}
    minutes = np.asarray(
        [int(boot["active_frames"]) / FRAMES_PER_MINUTE for boot in active],
        dtype=np.float64,
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0,
        len(active),
        size=(BOOTSTRAP_RESAMPLES, len(active)),
    )
    stock = np.asarray(
        [int(boot["stocks_taken"]) - int(boot["stocks_lost"]) for boot in active],
        dtype=np.float64,
    )
    damage = np.asarray(
        [float(boot["damage_dealt"]) - float(boot["damage_taken"]) for boot in active],
        dtype=np.float64,
    )
    return {
        "estimator": "sum(net outcome) / sum(active minutes)",
        "independent_unit": "Dolphin boot",
        "n_boots": len(active),
        "n_matches": sum(int(boot["matches"]) for boot in active),
        "net_damage_per_min": rate_statistics(damage, minutes, indices),
        "net_stocks_per_min": rate_statistics(stock, minutes, indices),
    }


def source_payload(
    source_type: str,
    path: str,
    content: bytes,
    run_id: str,
    run_name: str,
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        return None, f"invalid JSON: {error}"
    rows_value = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows_value, list) or not rows_value:
        return None, "no match rows"
    try:
        rows = [normalize_row(row) for row in rows_value]
        rows.sort(key=lambda row: (int(row["boot_index"]), int(row["match_ordinal"])))
        boots = aggregate_boots(rows)
    except (KeyError, TypeError, ValueError) as error:
        return None, f"invalid match rows: {error}"

    rows_hash = sha256(canonical_json(rows))
    protocol = payload.get("protocol", {})
    metrics = payload.get("metrics", {})
    return {
        "evaluation_id": f"eval-{rows_hash[:16]}",
        "raw_rows_sha256": rows_hash,
        "run_ids": [run_id],
        "run_names": [run_name],
        "sources": [
            {
                "content_sha256": sha256(content),
                "path": path,
                "type": source_type,
            }
        ],
        "source_schema_versions": [payload.get("schema_version")],
        "protocols": [json_safe(protocol)] if protocol else [],
        "saved_metrics": [json_safe(metrics)] if metrics else [],
        "boots": boots,
        "statistics": evaluation_statistics(boots),
    }, None


def merge_unique(values: list[Any], additions: list[Any]) -> list[Any]:
    known = {canonical_json(value) for value in values}
    for value in additions:
        encoded = canonical_json(value)
        if encoded not in known:
            values.append(value)
            known.add(encoded)
    return values


def merge_evaluation(target: dict[str, Any], addition: dict[str, Any]) -> None:
    target["run_ids"] = sorted(set(target["run_ids"]) | set(addition["run_ids"]))
    target["run_names"] = sorted(set(target["run_names"]) | set(addition["run_names"]))
    merge_unique(target["sources"], addition["sources"])
    merge_unique(target["source_schema_versions"], addition["source_schema_versions"])
    merge_unique(target["protocols"], addition["protocols"])
    merge_unique(target["saved_metrics"], addition["saved_metrics"])


def protocol_values(evaluation: dict[str, Any], key: str) -> list[Any]:
    values: list[Any] = []
    for protocol in evaluation["protocols"]:
        value = protocol.get(key)
        if value is None and isinstance(protocol.get("model"), dict):
            value = protocol["model"].get(key)
        if value is not None and value not in values:
            values.append(value)
    return values


def finalize_evaluation(evaluation: dict[str, Any]) -> None:
    expected_boots = [
        int(value) for value in protocol_values(evaluation, "n_matchups") if isinstance(value, int | float)
    ]
    n_boots = int(evaluation["statistics"].get("n_boots", 0))
    expected = max(expected_boots) if expected_boots else None
    descriptors: dict[str, Any] = {
        "checkpoint_sha256": protocol_values(evaluation, "checkpoint_sha256"),
        "execution_horizon": protocol_values(evaluation, "exec_horizon"),
        "suite": protocol_values(evaluation, "suite"),
    }
    descriptors = {key: value for key, value in descriptors.items() if value}
    evaluation["descriptor"] = descriptors
    evaluation["coverage"] = {
        "complete": None if expected is None else n_boots >= expected,
        "expected_boots": expected,
        "observed_boots": n_boots,
    }


def evaluation_target_error(
    evaluation: dict[str, Any],
    stock: float | None,
    damage: float | None,
) -> float | None:
    statistics = evaluation["statistics"]
    if not statistics.get("n_boots"):
        return None
    error = 0.0
    if stock is not None:
        actual_stock = statistics["net_stocks_per_min"]["pooled_mean"]
        stock_error = abs(actual_stock - stock)
        if stock_error > 0.02:
            return None
        error += stock_error / 0.02
    if damage is not None:
        actual_damage = statistics["net_damage_per_min"]["pooled_mean"]
        damage_error = abs(actual_damage - damage)
        if damage_error > 1.0:
            return None
        error += damage_error
    return error


def select_evaluation(
    evaluations: dict[str, dict[str, Any]],
    evaluation_ids: list[str],
    *,
    stock: float | None,
    damage: float | None,
) -> str | None:
    if stock is None and damage is None:
        return None
    matches: list[tuple[float, str]] = []
    for evaluation_id in evaluation_ids:
        error = evaluation_target_error(
            evaluations[evaluation_id],
            stock,
            damage,
        )
        if error is not None:
            matches.append((error, evaluation_id))
    if not matches:
        return None
    matches.sort()
    if len(matches) > 1 and math.isclose(matches[0][0], matches[1][0], abs_tol=1e-12):
        return None
    return matches[0][1]


def comparison_error(
    treatment: dict[str, Any],
    control: dict[str, Any],
    comparison: dict[str, Any],
) -> float | None:
    treatment_stats = treatment["statistics"]
    control_stats = control["statistics"]
    if not treatment_stats.get("n_boots") or not control_stats.get("n_boots"):
        return None
    score = 0.0
    stock_delta = comparison.get("stock_delta")
    if isinstance(stock_delta, int | float):
        observed = (
            treatment_stats["net_stocks_per_min"]["pooled_mean"] - control_stats["net_stocks_per_min"]["pooled_mean"]
        )
        error = abs(observed - stock_delta)
        if error > 0.03:
            return None
        score += error / 0.03
    damage_delta = comparison.get("damage_delta")
    if isinstance(damage_delta, int | float):
        observed = (
            treatment_stats["net_damage_per_min"]["pooled_mean"] - control_stats["net_damage_per_min"]["pooled_mean"]
        )
        error = abs(observed - damage_delta)
        if error > 1.5:
            return None
        score += error / 1.5
    treatment_horizon = treatment["descriptor"].get("execution_horizon", [])
    control_horizon = control["descriptor"].get("execution_horizon", [])
    if treatment_horizon and control_horizon and treatment_horizon != control_horizon:
        score += 10.0
    return score


def arm_for_comparison(
    arms: list[dict[str, Any]],
    requested_name: str,
) -> tuple[int, dict[str, Any]] | None:
    for index, arm in enumerate(arms):
        if arm.get("name") == requested_name:
            return index, arm
    requested_first_word = requested_name.lower().split(maxsplit=1)[0]
    candidates = [
        (index, arm)
        for index, arm in enumerate(arms)
        if str(arm.get("name", "")).lower().startswith(requested_first_word)
    ]
    return candidates[0] if len(candidates) == 1 else None


def comparison_list(study: dict[str, Any]) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    if isinstance(study.get("comparison"), dict):
        comparisons.append(study["comparison"])
    if isinstance(study.get("comparisons"), list):
        comparisons.extend(study["comparisons"])
    arms = study.get("arms", [])
    if len(arms) > 1 and not comparisons:
        control = arms[0]
        for arm in arms[1:]:
            if isinstance(arm.get("stock_delta"), int | float) or isinstance(arm.get("damage_delta"), int | float):
                comparisons.append(
                    {
                        "control": control.get("name"),
                        "damage_delta": arm.get("damage_delta"),
                        "stock_delta": arm.get("stock_delta"),
                        "treatment": arm.get("name"),
                    }
                )
    return comparisons


def select_comparison_evaluations(
    study: dict[str, Any],
    evaluations: dict[str, dict[str, Any]],
    evaluation_ids_by_run: dict[str, list[str]],
) -> None:
    arms = study.get("arms", [])
    if not arms:
        return
    links_by_index = {link["arm_index"]: link for link in study.get("evaluation_links", []) if "arm_index" in link}
    for comparison in comparison_list(study):
        treatment = arm_for_comparison(arms, str(comparison.get("treatment", "")))
        control = arm_for_comparison(arms, str(comparison.get("control", "")))
        if treatment is None or control is None:
            continue
        treatment_index, treatment_arm = treatment
        control_index, control_arm = control
        treatment_ids = evaluation_ids_by_run.get(str(treatment_arm.get("run")), [])
        control_ids = evaluation_ids_by_run.get(str(control_arm.get("run")), [])
        candidates: list[tuple[float, str, str]] = []
        for treatment_id in treatment_ids:
            for control_id in control_ids:
                error = comparison_error(
                    evaluations[treatment_id],
                    evaluations[control_id],
                    comparison,
                )
                if error is not None:
                    candidates.append((error, treatment_id, control_id))
        if not candidates:
            continue
        candidates.sort()
        best = candidates[0]
        if len(candidates) > 1 and math.isclose(best[0], candidates[1][0], abs_tol=1e-12):
            continue
        links_by_index[treatment_index]["selected_evaluation_id"] = best[1]
        links_by_index[treatment_index]["selection_method"] = "published difference match"
        links_by_index[control_index]["selected_evaluation_id"] = best[2]
        links_by_index[control_index]["selection_method"] = "published difference match"


def boot_arrays(evaluation: dict[str, Any], metric: str) -> dict[int, tuple[tuple[int, int], float, float]]:
    out: dict[int, tuple[tuple[int, int], float, float]] = {}
    for boot in evaluation["boots"]:
        active_frames = int(boot["active_frames"])
        if active_frames <= 0:
            continue
        if metric == "stock":
            numerator = int(boot["stocks_taken"]) - int(boot["stocks_lost"])
        else:
            numerator = float(boot["damage_dealt"]) - float(boot["damage_taken"])
        out[int(boot["boot_index"])] = (
            (int(boot["ego_character"]), int(boot["opp_character"])),
            float(numerator),
            active_frames / FRAMES_PER_MINUTE,
        )
    return out


def pooled_rate(values: list[tuple[tuple[int, int], float, float]]) -> float:
    return sum(value[1] for value in values) / sum(value[2] for value in values)


def difference_statistics(
    treatment: dict[str, Any],
    control: dict[str, Any],
    metric: str,
) -> dict[str, Any]:
    treatment_boots = boot_arrays(treatment, metric)
    control_boots = boot_arrays(control, metric)
    treatment_values = list(treatment_boots.values())
    control_values = list(control_boots.values())
    point = pooled_rate(treatment_values) - pooled_rate(control_values)
    shared = sorted(treatment_boots.keys() & control_boots.keys())
    matching = [index for index in shared if treatment_boots[index][0] == control_boots[index][0]]
    matching_fraction = (
        2 * len(matching) / (len(treatment_boots) + len(control_boots)) if treatment_boots or control_boots else 0.0
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    if matching_fraction >= 0.95:
        treatment_matched = [treatment_boots[index] for index in matching]
        control_matched = [control_boots[index] for index in matching]
        indices = rng.integers(
            0,
            len(matching),
            size=(BOOTSTRAP_RESAMPLES, len(matching)),
        )
        treatment_num = np.asarray([value[1] for value in treatment_matched])
        treatment_min = np.asarray([value[2] for value in treatment_matched])
        control_num = np.asarray([value[1] for value in control_matched])
        control_min = np.asarray([value[2] for value in control_matched])
        samples = treatment_num[indices].sum(axis=1) / treatment_min[indices].sum(axis=1) - control_num[indices].sum(
            axis=1
        ) / control_min[indices].sum(axis=1)
        boot_differences = treatment_num / treatment_min - control_num / control_min
        difference_sd = float(boot_differences.std(ddof=1)) if len(boot_differences) > 1 else 0.0
        method = "matchup-blocked bootstrap of shared boots"
        independent_units = len(matching)
    else:
        treatment_indices = rng.integers(
            0,
            len(treatment_values),
            size=(BOOTSTRAP_RESAMPLES, len(treatment_values)),
        )
        control_indices = rng.integers(
            0,
            len(control_values),
            size=(BOOTSTRAP_RESAMPLES, len(control_values)),
        )
        treatment_num = np.asarray([value[1] for value in treatment_values])
        treatment_min = np.asarray([value[2] for value in treatment_values])
        control_num = np.asarray([value[1] for value in control_values])
        control_min = np.asarray([value[2] for value in control_values])
        samples = treatment_num[treatment_indices].sum(axis=1) / treatment_min[treatment_indices].sum(
            axis=1
        ) - control_num[control_indices].sum(axis=1) / control_min[control_indices].sum(axis=1)
        difference_sd = None
        method = "independent bootstrap of boots"
        independent_units = {
            "control": len(control_values),
            "treatment": len(treatment_values),
        }
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return {
        "boot_difference_sd": difference_sd,
        "bootstrap_se": float(samples.std(ddof=1)),
        "ci95": [float(lo), float(hi)],
        "independent_units": independent_units,
        "matchup_pairing_fraction": matching_fraction,
        "method": method,
        "point_delta": point,
    }


def add_computed_differences(studies: list[dict[str, Any]], evaluations: dict[str, dict[str, Any]]) -> None:
    for study in studies:
        links = {
            link.get("arm_name"): link.get("selected_evaluation_id")
            for link in study.get("evaluation_links", [])
            if link.get("arm_name") and link.get("selected_evaluation_id")
        }
        computed: list[dict[str, Any]] = []
        for comparison in comparison_list(study):
            treatment_name = comparison.get("treatment")
            control_name = comparison.get("control")
            treatment_id = links.get(treatment_name)
            control_id = links.get(control_name)
            if treatment_id is None or control_id is None:
                treatment_arm = arm_for_comparison(study.get("arms", []), str(treatment_name or ""))
                control_arm = arm_for_comparison(study.get("arms", []), str(control_name or ""))
                if treatment_arm is not None:
                    treatment_id = links.get(treatment_arm[1].get("name"))
                if control_arm is not None:
                    control_id = links.get(control_arm[1].get("name"))
            if treatment_id is None or control_id is None:
                continue
            computed.append(
                {
                    "control": control_name,
                    "control_evaluation_id": control_id,
                    "net_damage_per_min": difference_statistics(
                        evaluations[treatment_id], evaluations[control_id], "damage"
                    ),
                    "net_stocks_per_min": difference_statistics(
                        evaluations[treatment_id], evaluations[control_id], "stock"
                    ),
                    "treatment": treatment_name,
                    "treatment_evaluation_id": treatment_id,
                }
            )
        if computed:
            study["computed_differences"] = computed


def add_study_links(
    studies: list[dict[str, Any]],
    evaluations: dict[str, dict[str, Any]],
    evaluation_ids_by_run: dict[str, list[str]],
) -> None:
    for study in studies:
        links: list[dict[str, Any]] = []
        for index, arm in enumerate(study.get("arms", [])):
            run_id = arm.get("run")
            if not isinstance(run_id, str):
                continue
            evaluation_ids = evaluation_ids_by_run.get(run_id, [])
            selected = select_evaluation(
                evaluations,
                evaluation_ids,
                stock=arm.get("stock"),
                damage=arm.get("damage"),
            )
            link: dict[str, Any] = {
                "arm_index": index,
                "arm_name": arm.get("name"),
                "run_id": run_id,
                "raw_evaluation_count": len(evaluation_ids),
            }
            if selected is not None:
                link["selected_evaluation_id"] = selected
            links.append(link)

        for series in study.get("series", []) if isinstance(study.get("series"), list) else []:
            run_id = series.get("run")
            horizons = series.get("horizons", [])
            stocks = series.get("stock", [])
            damages = series.get("damage", [])
            if not isinstance(run_id, str):
                continue
            for horizon, stock, damage in zip(horizons, stocks, damages, strict=True):
                selected = select_evaluation(
                    evaluations,
                    evaluation_ids_by_run.get(run_id, []),
                    stock=stock,
                    damage=damage,
                )
                link = {
                    "run_id": run_id,
                    "series_name": series.get("name"),
                    "setting": {"execution_horizon": horizon},
                }
                if selected is not None:
                    link["selected_evaluation_id"] = selected
                links.append(link)
        if links:
            study["evaluation_links"] = links
            select_comparison_evaluations(
                study,
                evaluations,
                evaluation_ids_by_run,
            )


def main() -> None:
    args = parse_args()
    document = json.loads(args.input.read_text())
    document.pop("raw_evaluation_data", None)
    for study in document.get("studies", []):
        study.pop("evaluation_links", None)
        study.pop("computed_differences", None)

    run_ids = sorted(collect_run_ids(document))
    print(f"Resolving {len(run_ids)} W&B runs", flush=True)
    runs: dict[str, dict[str, Any]] = {}
    metadata_errors: dict[str, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(fetch_run_metadata, run_id): run_id for run_id in run_ids}
        for future in concurrent.futures.as_completed(futures):
            run_id = futures[future]
            try:
                _, metadata = future.result()
            except Exception as error:  # W&B raises several transport-specific types.
                metadata_errors[run_id] = str(error)
                continue
            runs[run_id] = metadata

    r2_paths: dict[str, list[str]] = {}
    r2_errors: dict[str, str] = {}
    if not args.skip_r2:
        print(f"Listing R2 evidence for {len(runs)} runs", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(list_r2_sources, metadata["name"]): run_id for run_id, metadata in runs.items()}
            for future in concurrent.futures.as_completed(futures):
                run_id = futures[future]
                _, paths, error = future.result()
                r2_paths[run_id] = paths
                if error is not None:
                    r2_errors[run_id] = error

    source_items: list[tuple[str, str, bytes, str, str]] = []
    source_errors: list[dict[str, str]] = []
    remote_requests: list[tuple[str, str, str]] = []
    for run_id, metadata in runs.items():
        run_name = metadata["name"]
        local_relative_paths: set[str] = set()
        for path in local_sources(run_name):
            relative = path.relative_to(Path("runs") / run_name).as_posix()
            local_relative_paths.add(relative)
            source_items.append(("local", str(path), path.read_bytes(), run_id, run_name))
        for r2_path in r2_paths.get(run_id, []):
            relative = r2_path.removeprefix(f"{R2_ROOT}/{run_name}/")
            if relative in local_relative_paths:
                # Retain both source addresses without downloading the same bytes twice.
                local_match = next(
                    item
                    for item in source_items
                    if item[3] == run_id
                    and item[0] == "local"
                    and Path(item[1]).relative_to(Path("runs") / run_name).as_posix() == relative
                )
                source_items.append(("r2", r2_path, local_match[2], run_id, run_name))
            else:
                remote_requests.append((r2_path, run_id, run_name))

    if remote_requests:
        print(f"Reading {len(remote_requests)} R2 row files", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(read_r2_source, path): (path, run_id, run_name)
                for path, run_id, run_name in remote_requests
            }
            for future in concurrent.futures.as_completed(futures):
                path, run_id, run_name = futures[future]
                _, content, error = future.result()
                if error is not None or content is None:
                    source_errors.append({"path": path, "reason": error or "empty R2 response", "run_id": run_id})
                    continue
                source_items.append(("r2", path, content, run_id, run_name))

    evaluations: dict[str, dict[str, Any]] = {}
    ignored_sources: list[dict[str, str]] = []
    for source_type, path, content, run_id, run_name in source_items:
        evaluation, error = source_payload(source_type, path, content, run_id, run_name)
        if evaluation is None:
            ignored_sources.append({"path": path, "reason": error or "unknown", "run_id": run_id})
            continue
        evaluation_id = evaluation["evaluation_id"]
        if evaluation_id in evaluations:
            merge_evaluation(evaluations[evaluation_id], evaluation)
        else:
            evaluations[evaluation_id] = evaluation

    evaluations = dict(sorted(evaluations.items()))
    for evaluation in evaluations.values():
        finalize_evaluation(evaluation)
    evaluation_ids_by_run: dict[str, list[str]] = defaultdict(list)
    for evaluation_id, evaluation in evaluations.items():
        for run_id in evaluation["run_ids"]:
            evaluation_ids_by_run[run_id].append(evaluation_id)
    evaluation_ids_by_run = {
        run_id: sorted(evaluation_ids) for run_id, evaluation_ids in sorted(evaluation_ids_by_run.items())
    }
    for run_id, metadata in runs.items():
        metadata["evaluation_ids"] = evaluation_ids_by_run.get(run_id, [])

    add_study_links(document.get("studies", []), evaluations, evaluation_ids_by_run)
    add_computed_differences(document.get("studies", []), evaluations)
    usable_source_count = sum(len(evaluation["sources"]) for evaluation in evaluations.values())
    document["schema_version"] = 2
    document["purpose"] = (
        "Canonical, deduplicated evidence for the HAL experiment blog. "
        "It contains study descriptions, source locations, unique closed-loop evaluations, "
        "boot-level sufficient statistics, and derived uncertainty."
    )
    document["raw_evaluation_data"] = {
        "bootstrap": {
            "method": "percentile bootstrap of Dolphin boots",
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
        },
        "deduplication": {
            "duplicate_source_files_collapsed": usable_source_count - len(evaluations),
            "key": "SHA-256 of canonical per-match rows",
            "source_files_with_usable_rows": usable_source_count,
            "unique_evaluations": len(evaluations),
        },
        "evaluations": evaluations,
        "ignored_sources": sorted(ignored_sources, key=lambda item: (item["run_id"], item["path"])),
        "limitations": [
            "A run with no evaluation_ids has no retained match rows in the local or R2 archive.",
            "W&B aggregate metrics do not recover covariance or a paired difference interval.",
            "Boot-rate SD describes variation between boots. The headline mean is a pooled ratio and is stored separately.",
        ],
        "metadata_errors": metadata_errors,
        "r2_errors": r2_errors,
        "read_errors": sorted(source_errors, key=lambda item: (item["run_id"], item["path"])),
        "runs": dict(sorted(runs.items())),
        "unit": (
            "One record in evaluations[*].boots is one Dolphin process boot. "
            "Back-to-back instant-restart matches are summed inside that boot."
        ),
    }
    args.output.write_text(json.dumps(document, indent=2, sort_keys=False, allow_nan=False) + "\n")
    print(
        f"Wrote {args.output}: {len(evaluations)} unique evaluations from {usable_source_count} usable source files",
        flush=True,
    )


if __name__ == "__main__":
    main()
