"""Plan and launch stateful O51 sweep stages on Modal."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Annotated
from typing import Any
from typing import Literal
from typing import cast

import modal
import tyro

import wandb
from hal.training.o51_data import DATA_PROTOCOL
from hal.training.o51_sweep import GRID_EVAL_MATCHUPS
from hal.training.o51_sweep import ClosedLoopOutcome
from hal.training.o51_sweep import GridStage
from hal.training.o51_sweep import Stage
from hal.training.o51_sweep import SweepArm
from hal.training.o51_sweep import Treatment
from hal.training.o51_sweep import ValidationOutcome
from hal.training.o51_sweep import select_closed_loop_winner
from hal.training.o51_sweep import select_validation_winner
from hal.training.o51_sweep import stage_arms

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "launch_modal.py"
DEFAULT_STATE = ROOT / "runs" / "o51-sweep-v3-16l" / "launches.jsonl"
APP_PATTERN = re.compile(r"submitted Modal App (ap-[A-Za-z0-9]+), Function call (fc-[A-Za-z0-9]+)")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
STATE_SCHEMA = 2

LaunchEvent = Literal["launching", "failed", "uncertain", "launched"]


@dataclass(frozen=True, slots=True)
class PlanArgs:
    stage: Stage
    treatment: Path | None = None
    preflight_reports: Path | None = None
    """JSON object that maps production arm IDs, or ``*``, to report paths."""


@dataclass(frozen=True, slots=True)
class LaunchArgs:
    stage: Stage
    treatment: Path | None = None
    preflight_reports: Path | None = None
    """JSON object that maps production arm IDs, or ``*``, to report paths."""
    state: Path = DEFAULT_STATE
    max_arms: int | None = None
    gpu: str = "B200"
    disk_gib: int = 2048
    dry_run: bool = False


@dataclass(frozen=True, slots=True)
class SelectArgs:
    """Select a completed stage by its fixed validation rule."""

    stage: Stage
    runs: Path
    """JSON object that maps every arm ID to its full W&B run path."""
    output: Path
    """New treatment JSON for the next stage."""
    evidence: Path
    """New durable JSON record of the complete ranking."""
    treatment: Path | None = None


@dataclass(frozen=True, slots=True)
class RankArgs:
    """Screen a 3x3 grid to its two closed-loop finalists."""

    stage: GridStage
    runs: Path
    """JSON object that maps every arm ID to its full W&B training-run path."""
    evidence: Path
    """New durable JSON record of the validation ranking and finalists."""
    treatment: Path | None = None


@dataclass(frozen=True, slots=True)
class AdjudicateArgs:
    """Select a 3x3 grid winner from two complete closed-loop evaluations."""

    stage: GridStage
    runs: Path
    """JSON object that maps every arm ID to its full W&B training-run path."""
    evaluations: Path
    """JSON object that maps finalists to W&B runs containing closed-loop metrics."""
    output: Path
    """New treatment JSON for the next stage."""
    evidence: Path
    """New durable JSON record of validation screening and closed-loop ranking."""
    treatment: Path | None = None


type Command = (
    Annotated[PlanArgs, tyro.conf.subcommand(name="plan")]
    | Annotated[LaunchArgs, tyro.conf.subcommand(name="launch")]
    | Annotated[RankArgs, tyro.conf.subcommand(name="rank")]
    | Annotated[AdjudicateArgs, tyro.conf.subcommand(name="adjudicate")]
    | Annotated[SelectArgs, tyro.conf.subcommand(name="select")]
)


@dataclass(frozen=True, slots=True)
class PreparedArm:
    arm: SweepArm
    preflight_report: Path | None
    train_argv: tuple[str, ...]
    spec_sha256: str


def _treatment(path: Path | None) -> Treatment:
    return Treatment() if path is None else Treatment.load(path)


def _preflight_reports(path: Path | None) -> dict[str, Path]:
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read the preflight report map {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("the preflight report map must be a JSON object")
    reports: dict[str, Path] = {}
    for arm_id, report in payload.items():
        if not isinstance(arm_id, str) or not arm_id:
            raise ValueError("each preflight report key must be a non-empty arm ID")
        if not isinstance(report, str) or not report:
            raise ValueError(f"the preflight report path for {arm_id!r} must be a non-empty string")
        reports[arm_id] = Path(report)
    return reports


def _run_paths(path: Path, arms: tuple[SweepArm, ...]) -> dict[str, str]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read the W&B run map {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("the W&B run map must be a JSON object")
    expected = {arm.arm_id for arm in arms}
    actual = set(payload)
    if expected != actual:
        raise ValueError(
            f"W&B run map does not match the stage: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    invalid = sorted(
        arm_id for arm_id, run_path in payload.items() if not isinstance(run_path, str) or run_path.count("/") != 2
    )
    if invalid:
        raise ValueError(f"W&B run paths must be entity/project/run_id: {invalid}")
    return cast(dict[str, str], payload)


def _evaluation_paths(path: Path, finalists: tuple[str, str]) -> dict[str, str]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read the W&B evaluation map {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("the W&B evaluation map must be a JSON object")
    expected = set(finalists)
    actual = set(payload)
    if expected != actual:
        raise ValueError(
            f"W&B evaluation map does not match the finalists: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    invalid = sorted(
        arm_id for arm_id, run_path in payload.items() if not isinstance(run_path, str) or run_path.count("/") != 2
    )
    if invalid:
        raise ValueError(f"W&B evaluation paths must be entity/project/run_id: {invalid}")
    return cast(dict[str, str], payload)


def _run_config_mismatches(arm: SweepArm, run: Any) -> tuple[str, ...]:
    expected = {
        **asdict(arm.treatment),
        "target_positions": arm.target_positions,
        "tier_scale": arm.tier_scale,
        "seed": arm.seed,
        "data_protocol": DATA_PROTOCOL,
        "adam_eps": 1e-12,
        "reservoir_capacity": 17 * arm.treatment.batch_size,
        "replay_cooldown_batches": 16,
    }
    config = dict(run.config)
    mismatches = [
        name
        for name, value in expected.items()
        if json.dumps(config.get(name), sort_keys=True) != json.dumps(value, sort_keys=True)
    ]
    name = str(run.name)
    if not name.endswith(f"__{arm.arm_id}"):
        mismatches.append("run_name")
    if f"_o51-{arm.level}-" not in name:
        mismatches.append("model_level")
    return tuple(sorted(mismatches))


def _optional_float(value: object) -> float | None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    return float(value)


def _validation_outcome(arm: SweepArm, run_path: str, run: Any) -> ValidationOutcome:
    mismatches = _run_config_mismatches(arm, run)
    if mismatches:
        raise ValueError(f"W&B run {run_path} does not match {arm.arm_id}: {list(mismatches)}")
    summary = dict(run.summary)
    return ValidationOutcome(
        arm_id=arm.arm_id,
        run_path=run_path,
        state=str(run.state),
        processed_positions=_optional_float(summary.get("data/processed_loss_positions")),
        final_update=_optional_float(summary.get("global_step")),
        val_nll=_optional_float(summary.get("val/nll")),
        val_far_nll=_optional_float(summary.get("val/far_nll")),
        val_rollout_nll=_optional_float(summary.get("val/rollout_nll")),
    )


def _closed_loop_outcome(
    arm: SweepArm,
    training_run_path: str,
    evaluation_run_path: str,
    run: Any,
) -> ClosedLoopOutcome:
    config = dict(run.config)
    summary = dict(run.summary)
    mismatches: list[str] = []
    if evaluation_run_path == training_run_path:
        if summary.get("eval/backfilled") != 1:
            mismatches.append("eval/backfilled")
    else:
        expected_training_id = training_run_path.rsplit("/", maxsplit=1)[1]
        if config.get("training_wandb_id") != expected_training_id:
            mismatches.append("training_wandb_id")
        if config.get("eval_matchups") != GRID_EVAL_MATCHUPS:
            mismatches.append("eval_matchups")
    if mismatches:
        raise ValueError(f"W&B evaluation {evaluation_run_path} does not match {arm.arm_id}: {sorted(mismatches)}")
    return ClosedLoopOutcome(
        arm_id=arm.arm_id,
        run_path=evaluation_run_path,
        state=str(run.state),
        final_update=_optional_float(summary.get("global_step")),
        boots=_optional_float(summary.get("eval/boots")),
        crashed=_optional_float(summary.get("eval/crashed")),
        net_stock_per_min=_optional_float(summary.get("eval/net_stock_per_min")),
        net_stock_lcb=_optional_float(summary.get("eval/net_stock_lcb")),
        net_dmg_per_min=_optional_float(summary.get("eval/net_dmg_per_min")),
    )


def _validated_preflight_reports(reports: dict[str, Path]) -> dict[str, Path]:
    """Return report paths that the Modal source image will contain."""
    if not reports:
        return {}
    ignored = modal.FilePatternMatcher.from_file(ROOT / ".dockerignore")
    tracked = {Path(value) for value in _git("ls-files", "-z").split("\0") if value}
    validated: dict[str, Path] = {}
    root = ROOT.resolve()
    for arm_id, report in reports.items():
        if report.is_absolute():
            raise ValueError(f"preflight report for {arm_id!r} must be relative to the repository")
        try:
            candidate = (ROOT / report).resolve(strict=True)
            relative = candidate.relative_to(root)
        except (FileNotFoundError, ValueError) as error:
            raise ValueError(f"preflight report for {arm_id!r} is not a repository file: {report}") from error
        if not candidate.is_file():
            raise ValueError(f"preflight report for {arm_id!r} is not a regular file: {report}")
        if relative not in tracked:
            raise ValueError(f"preflight report for {arm_id!r} is not tracked by Git: {relative}")
        if ignored(relative):
            raise ValueError(f"preflight report for {arm_id!r} is excluded from the Modal image: {relative}")
        validated[arm_id] = relative
    return validated


def _report_for(arm: SweepArm, reports: dict[str, Path], *, required: bool) -> Path | None:
    if not arm.requires_preflight:
        return None
    report = reports.get(arm.arm_id, reports.get("*"))
    if report is None and required:
        raise ValueError(
            f"production arm {arm.arm_id} needs a preflight report; add its arm ID or '*' to --preflight-reports"
        )
    return report


def _spec_sha256(
    arm: SweepArm,
    train_argv: tuple[str, ...],
    *,
    git_sha: str,
    gpu: str,
    disk_gib: int,
) -> str:
    payload = {
        "arm_id": arm.arm_id,
        "stage": arm.stage,
        "git_sha": git_sha,
        "gpu": gpu,
        "disk_gib": disk_gib,
        "train_argv": train_argv,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _prepare_arms(
    arms: tuple[SweepArm, ...],
    reports: dict[str, Path],
    *,
    git_sha: str,
    gpu: str,
    disk_gib: int,
    require_preflight: bool,
) -> tuple[PreparedArm, ...]:
    prepared: list[PreparedArm] = []
    for arm in arms:
        report = _report_for(arm, reports, required=require_preflight)
        train_argv = arm.argv(preflight_report=report)
        prepared.append(
            PreparedArm(
                arm=arm,
                preflight_report=report,
                train_argv=train_argv,
                spec_sha256=_spec_sha256(
                    arm,
                    train_argv,
                    git_sha=git_sha,
                    gpu=gpu,
                    disk_gib=disk_gib,
                ),
            )
        )
    return tuple(prepared)


def _validate_record(record: object, path: Path, line_number: int) -> dict[str, object]:
    prefix = f"invalid launch state {path}:{line_number}"
    if not isinstance(record, dict):
        raise ValueError(f"{prefix}: each line must contain a JSON object")
    if record.get("schema_version") != STATE_SCHEMA:
        raise ValueError(f"{prefix}: expected schema version {STATE_SCHEMA}")
    required_strings = ("arm_id", "stage", "event", "attempt_id", "spec_sha256", "git_sha", "recorded_at")
    invalid_strings = [name for name in required_strings if not isinstance(record.get(name), str) or not record[name]]
    if invalid_strings:
        raise ValueError(f"{prefix}: invalid string fields {invalid_strings}")
    if record["event"] not in ("launching", "failed", "uncertain", "launched"):
        raise ValueError(f"{prefix}: unknown event {record['event']!r}")
    if not SHA256_PATTERN.fullmatch(cast(str, record["spec_sha256"])):
        raise ValueError(f"{prefix}: spec_sha256 is not a lowercase SHA-256 digest")
    argv = record.get("argv")
    if not isinstance(argv, list) or not all(isinstance(value, str) for value in argv):
        raise ValueError(f"{prefix}: argv must be a list of strings")
    if not isinstance(record.get("treatment"), dict):
        raise ValueError(f"{prefix}: treatment must be a JSON object")
    event = cast(LaunchEvent, record["event"])
    app_id = record.get("app_id")
    function_call_id = record.get("function_call_id")
    if event == "launched" and not (isinstance(app_id, str) and isinstance(function_call_id, str)):
        raise ValueError(f"{prefix}: only a launched event must contain both Modal IDs")
    if event != "launched" and (app_id is not None or function_call_id is not None):
        raise ValueError(f"{prefix}: only a launched event can contain Modal IDs")
    return record


def _records(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    latest: dict[str, dict[str, object]] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid launch state {path}:{line_number}: {error.msg}") from error
        record = _validate_record(raw, path, line_number)
        arm_id = cast(str, record["arm_id"])
        event = cast(LaunchEvent, record["event"])
        previous = latest.get(arm_id)
        if previous is None:
            if event != "launching":
                raise ValueError(f"invalid launch state {path}:{line_number}: first arm event must be launching")
        else:
            if record["spec_sha256"] != previous["spec_sha256"]:
                raise ValueError(f"invalid launch state {path}:{line_number}: arm ID changed its launch specification")
            previous_event = cast(LaunchEvent, previous["event"])
            same_attempt = record["attempt_id"] == previous["attempt_id"]
            if event == "launching":
                if previous_event != "failed" or same_attempt:
                    raise ValueError(
                        f"invalid launch state {path}:{line_number}: a new attempt can follow only a failed attempt"
                    )
            elif previous_event != "launching" or not same_attempt:
                raise ValueError(
                    f"invalid launch state {path}:{line_number}: terminal event does not match a launching attempt"
                )
        latest[arm_id] = record
    return latest


def _append(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def _state_lock(path: Path) -> Iterator[None]:
    lock_path = path.with_suffix(f"{path.suffix}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another O51 sweep launcher holds {lock_path}") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _pending_arms(
    arms: tuple[PreparedArm, ...],
    previous: dict[str, dict[str, object]],
) -> tuple[PreparedArm, ...]:
    pending: list[PreparedArm] = []
    for prepared in arms:
        record = previous.get(prepared.arm.arm_id)
        if record is None:
            pending.append(prepared)
            continue
        if record["spec_sha256"] != prepared.spec_sha256:
            raise RuntimeError(f"arm ID {prepared.arm.arm_id} already has a different launch specification")
        event = cast(LaunchEvent, record["event"])
        if event == "failed":
            pending.append(prepared)
        elif event != "launched":
            raise RuntimeError(
                f"arm {prepared.arm.arm_id} has an unresolved {event} attempt; reconcile Modal before retrying"
            )
    return tuple(pending)


def _event_payload(prepared: PreparedArm, git_sha: str, attempt_id: str, event: LaunchEvent) -> dict[str, object]:
    return {
        "schema_version": STATE_SCHEMA,
        "event": event,
        "attempt_id": attempt_id,
        "arm_id": prepared.arm.arm_id,
        "stage": prepared.arm.stage,
        "spec_sha256": prepared.spec_sha256,
        "git_sha": git_sha,
        "recorded_at": datetime.now(UTC).isoformat(),
        "argv": prepared.train_argv,
        "treatment": asdict(prepared.arm.treatment),
    }


def launch_command(prepared: PreparedArm, args: LaunchArgs, git_sha: str) -> tuple[str, ...]:
    app_name = f"hal-{prepared.arm.arm_id}-{git_sha[:7]}"
    return (
        sys.executable,
        str(LAUNCHER),
        "--gpu",
        args.gpu,
        "--disk-gib",
        str(args.disk_gib),
        "--app-name",
        app_name,
        "--",
        *prepared.train_argv,
    )


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _git_sha() -> str:
    if _git("status", "--porcelain"):
        raise RuntimeError("commit all working-tree changes before an O51 launch")
    return _git("rev-parse", "HEAD")


def _validate_launch_args(args: LaunchArgs) -> None:
    if args.max_arms is not None and args.max_arms < 1:
        raise ValueError("max_arms must be positive")
    if args.gpu != "B200":
        raise ValueError("O51 sweep launches require a B200")
    if args.disk_gib < 2048:
        raise ValueError("O51 sweep launches require at least 2048 GiB of ephemeral disk")


def _launch_pending(
    pending: tuple[PreparedArm, ...],
    args: LaunchArgs,
    git_sha: str,
) -> None:
    for prepared in pending:
        attempt_id = str(uuid.uuid4())
        launching = _event_payload(prepared, git_sha, attempt_id, "launching")
        _append(args.state, launching)
        command = launch_command(prepared, args, git_sha)
        try:
            result = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
        except BaseException:
            uncertain = _event_payload(prepared, git_sha, attempt_id, "uncertain")
            _append(args.state, uncertain)
            raise
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        if result.returncode:
            uncertain = _event_payload(prepared, git_sha, attempt_id, "uncertain")
            uncertain["returncode"] = result.returncode
            _append(args.state, uncertain)
            raise RuntimeError(f"launch failed for {prepared.arm.arm_id} with status {result.returncode}")
        match = APP_PATTERN.search(result.stdout + result.stderr)
        if match is None:
            uncertain = _event_payload(prepared, git_sha, attempt_id, "uncertain")
            _append(args.state, uncertain)
            raise RuntimeError(f"launch output for {prepared.arm.arm_id} did not contain Modal IDs")
        launched = _event_payload(prepared, git_sha, attempt_id, "launched")
        launched["app_id"] = match.group(1)
        launched["function_call_id"] = match.group(2)
        _append(args.state, launched)


def launch(args: LaunchArgs) -> None:
    _validate_launch_args(args)
    git_sha = _git_sha()
    arms = stage_arms(args.stage, _treatment(args.treatment))
    prepared = _prepare_arms(
        arms,
        _validated_preflight_reports(_preflight_reports(args.preflight_reports)),
        git_sha=git_sha,
        gpu=args.gpu,
        disk_gib=args.disk_gib,
        require_preflight=True,
    )
    if args.dry_run:
        pending = _pending_arms(prepared, _records(args.state))
        if args.max_arms is not None:
            pending = pending[: args.max_arms]
        for arm in pending:
            print(json.dumps({"arm_id": arm.arm.arm_id, "command": launch_command(arm, args, git_sha)}))
        return
    with _state_lock(args.state):
        pending = _pending_arms(prepared, _records(args.state))
        if args.max_arms is not None:
            pending = pending[: args.max_arms]
        _launch_pending(pending, args, git_sha)


def _plan(args: PlanArgs) -> None:
    reports = _validated_preflight_reports(_preflight_reports(args.preflight_reports))
    arms = stage_arms(args.stage, _treatment(args.treatment))
    payload = []
    for arm in arms:
        report = _report_for(arm, reports, required=False)
        payload.append(
            {
                "arm_id": arm.arm_id,
                "requires_preflight": arm.requires_preflight,
                "preflight_report": None if report is None else str(report),
                "argv": arm.argv(preflight_report=report),
            }
        )
    print(json.dumps(payload, indent=2))


def _json_text(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _check_output(path: Path, text: str) -> None:
    if path.exists() and path.read_text() != text:
        raise ValueError(f"refusing to replace different selection output: {path}")


def _write_output(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def _collect_validation(
    stage: Stage,
    center: Treatment,
    runs_path: Path,
    api: Any,
) -> tuple[tuple[SweepArm, ...], dict[str, ValidationOutcome], Any]:
    arms = stage_arms(stage, center)
    run_paths = _run_paths(runs_path, arms)
    outcomes = {
        arm.arm_id: _validation_outcome(arm, run_paths[arm.arm_id], api.run(run_paths[arm.arm_id])) for arm in arms
    }
    return arms, outcomes, select_validation_winner(arms, outcomes)


def _validation_rule() -> list[str]:
    return [
        "Every run must be terminal before screening.",
        "A candidate must finish its exact D/U endpoint and final optimizer update.",
        "Final validation NLL, far NLL, and rollout NLL must be finite.",
        "Rank eligible candidates by validation NLL, then far NLL, then rollout NLL, then arm ID.",
    ]


def rank(args: RankArgs) -> None:
    """Record the two validation finalists that require closed-loop evaluation."""
    center = _treatment(args.treatment)
    arms, outcomes, selection = _collect_validation(args.stage, center, args.runs, wandb.Api(timeout=30))
    del arms
    finalists = selection.ranking[:2]
    if len(finalists) != 2:
        raise ValueError("a 3x3 grid needs two eligible finalists")
    evidence = {
        "schema_version": 1,
        "stage": args.stage,
        "rule": _validation_rule(),
        "source_treatment": asdict(center),
        "final_eval_matchups": GRID_EVAL_MATCHUPS,
        "finalists": [{"arm_id": outcome.arm_id, "training_run": outcome.run_path} for outcome in finalists],
        "ranking": [asdict(outcome) for outcome in selection.ranking],
        "excluded": [{"arm_id": arm_id, "reasons": list(reasons)} for arm_id, reasons in selection.excluded],
    }
    text = _json_text(evidence)
    _check_output(args.evidence, text)
    _write_output(args.evidence, text)
    print(f"screened {args.stage}; evaluate {', '.join(outcome.arm_id for outcome in finalists)}")


def adjudicate(args: AdjudicateArgs) -> None:
    """Select a 3x3 grid only after evaluating its two validation finalists."""
    center = _treatment(args.treatment)
    api = wandb.Api(timeout=30)
    arms, validation_outcomes, validation = _collect_validation(args.stage, center, args.runs, api)
    finalists = validation.ranking[:2]
    if len(finalists) != 2:
        raise ValueError("a 3x3 grid needs two eligible finalists")
    finalist_ids = (finalists[0].arm_id, finalists[1].arm_id)
    training_paths = {outcome.arm_id: outcome.run_path for outcome in finalists}
    evaluation_paths = _evaluation_paths(args.evaluations, finalist_ids)
    arms_by_id = {arm.arm_id: arm for arm in arms}
    closed_loop_outcomes = {
        arm_id: _closed_loop_outcome(
            arms_by_id[arm_id],
            training_paths[arm_id],
            evaluation_paths[arm_id],
            api.run(evaluation_paths[arm_id]),
        )
        for arm_id in finalist_ids
    }
    selection = select_closed_loop_winner(arms, validation, closed_loop_outcomes)
    treatment_payload = asdict(selection.winner.treatment)
    evidence_payload = {
        "schema_version": 1,
        "stage": args.stage,
        "validation_rule": _validation_rule(),
        "closed_loop_rule": [
            f"Evaluate exactly the two best validation arms over {GRID_EVAL_MATCHUPS} fixed matchups.",
            "Both evaluations must finish the final checkpoint with all boots and no crashes.",
            "Rank by net-stock cluster-bootstrap lower bound, then mean net stock per minute.",
            "Break any remaining tie by net damage per minute, validation NLL, then arm ID.",
        ],
        "source_treatment": asdict(center),
        "winner_arm_id": selection.winner.arm_id,
        "winner_training_run": training_paths[selection.winner.arm_id],
        "winner_evaluation_run": closed_loop_outcomes[selection.winner.arm_id].run_path,
        "winner_treatment": treatment_payload,
        "validation_ranking": [asdict(outcome) for outcome in validation.ranking],
        "validation_excluded": [
            {"arm_id": arm_id, "reasons": list(reasons)} for arm_id, reasons in validation.excluded
        ],
        "closed_loop_ranking": [asdict(outcome) for outcome in selection.ranking],
    }
    treatment_text = _json_text(treatment_payload)
    evidence_text = _json_text(evidence_payload)
    _check_output(args.output, treatment_text)
    _check_output(args.evidence, evidence_text)
    _write_output(args.output, treatment_text)
    _write_output(args.evidence, evidence_text)
    print(f"selected {selection.winner.arm_id}; wrote {args.output} and {args.evidence}")


def select(args: SelectArgs) -> None:
    """Collect final W&B values and write one reproducible stage decision."""
    if args.stage in ("lr", "decay"):
        raise ValueError(f"{args.stage} is a 3x3 grid; use rank, evaluate both finalists, then adjudicate")
    center = _treatment(args.treatment)
    _arms, outcomes, selection = _collect_validation(args.stage, center, args.runs, wandb.Api(timeout=30))
    treatment_payload = asdict(selection.winner.treatment)
    evidence_payload = {
        "schema_version": 1,
        "stage": args.stage,
        "rule": _validation_rule(),
        "source_treatment": asdict(center),
        "winner_arm_id": selection.winner.arm_id,
        "winner_run": outcomes[selection.winner.arm_id].run_path,
        "winner_treatment": treatment_payload,
        "ranking": [asdict(outcome) for outcome in selection.ranking],
        "excluded": [{"arm_id": arm_id, "reasons": list(reasons)} for arm_id, reasons in selection.excluded],
    }
    treatment_text = _json_text(treatment_payload)
    evidence_text = _json_text(evidence_payload)
    _check_output(args.output, treatment_text)
    _check_output(args.evidence, evidence_text)
    _write_output(args.output, treatment_text)
    _write_output(args.evidence, evidence_text)
    print(f"selected {selection.winner.arm_id}; wrote {args.output} and {args.evidence}")


def main(args: Command) -> None:
    if isinstance(args, PlanArgs):
        _plan(args)
        return
    if isinstance(args, RankArgs):
        rank(args)
        return
    if isinstance(args, AdjudicateArgs):
        adjudicate(args)
        return
    if isinstance(args, SelectArgs):
        select(args)
        return
    launch(args)


if __name__ == "__main__":
    main(tyro.cli(cast(type[Command], Command)))
