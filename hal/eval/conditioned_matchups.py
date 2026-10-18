"""Evaluate conditioned actors against frozen anchors in MSL and Dolphin."""

import hashlib
import json
import math
import platform
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Literal

import melee
import numpy as np
import peppi_py
import torch
from peppi_py.game import EndMethod

from hal.data.behavior import EDGE_X
from hal.data.behavior import onsets
from hal.eval.h2h import stamp_replay_identity
from hal.eval.harness import SessionConfig
from hal.eval.harness import default_session_cfg
from hal.eval.harness import run_matches_vec
from hal.eval.match_summary import summarize_trajectory
from hal.eval.paired import mean_ci
from hal.eval.self_play_checkpoint import EvaluationCheckpoint
from hal.eval.self_play_checkpoint import load_evaluation_checkpoint
from hal.eval.self_play_checkpoint import sha256
from hal.eval.self_play_policy import PlayerCondition
from hal.eval.self_play_policy import SelfPlayPlayer
from hal.eval.self_play_policy import SelfPlayPolicy
from hal.policy import INCLUDED_STAGES
from hal.sim.inputs import native_controller_readback
from hal.sim.native import NativeFrameBatch
from hal.sim.native import NativeMatch
from hal.sim.native import NativePlayer
from hal.sim.native import NativeRolloutBatch
from hal.sim.native_bridge import library_sha256
from hal.sim.rollout import ObservationRow
from hal.sim.rollout import Slot
from hal.sim.rollout import VecMatch
from hal.sim.session import Matchup
from hal.sim.session import PlayerSetup
from hal.sim.trajectory import Trajectory
from hal.wire import POST_FIELD_SUFFIXES


@dataclass(frozen=True, slots=True)
class EvaluationStart:
    name: str
    stage: int
    characters: tuple[int, int]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.name) is None
            or type(self.stage) is not int
            or melee.Stage(self.stage) not in INCLUDED_STAGES
        ):
            raise ValueError("evaluation start needs a name and a supported stage")
        if len(self.characters) != 2:
            raise ValueError("evaluation start requires two characters")
        for character in self.characters:
            if type(character) is not int or character == melee.Character.NANA.value:
                raise ValueError("evaluation requires a selectable internal character kind")
            melee.Character(character)


@dataclass(frozen=True, slots=True)
class EvaluationSet:
    seeds: tuple[int, ...]
    starts: tuple[EvaluationStart, ...]

    def __post_init__(self) -> None:
        if (
            not self.seeds
            or len(set(self.seeds)) != len(self.seeds)
            or any(type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF for seed in self.seeds)
        ):
            raise ValueError("evaluation seeds must be distinct uint32 values")
        if (
            not self.starts
            or len({start.name for start in self.starts}) != len(self.starts)
            or len({(start.stage, tuple(sorted(start.characters))) for start in self.starts}) != len(self.starts)
        ):
            raise ValueError("evaluation starts must have distinct names and physical configurations")


@dataclass(frozen=True, slots=True)
class EvaluationProtocol:
    selection: EvaluationSet
    held_out: EvaluationSet
    max_frames: int = 28924

    def __post_init__(self) -> None:
        if type(self.max_frames) is not int or self.max_frames < 125:
            raise ValueError("evaluation must include active frames")
        if set(self.selection.seeds) & set(self.held_out.seeds):
            raise ValueError("selection and held-out seeds overlap")

        def keys(items: EvaluationSet) -> set[tuple[int, tuple[int, ...]]]:
            return {(start.stage, tuple(sorted(start.characters))) for start in items.starts}

        if keys(self.selection) & keys(self.held_out):
            raise ValueError("selection and held-out starts overlap, including port mirrors")


def pilot_protocol(*, max_frames: int = 28924) -> EvaluationProtocol:
    return EvaluationProtocol(
        EvaluationSet((17, 29), (EvaluationStart("fd-fox-falco", 25, (1, 22)),)),
        EvaluationSet((10017, 10029), (EvaluationStart("bf-fox-marth", 24, (1, 18)),)),
        max_frames,
    )


def read_protocol(path: Path) -> EvaluationProtocol:
    raw = json.loads(path.read_text())
    if (
        not isinstance(raw, dict)
        or set(raw) != {"format", "selection", "held_out", "max_frames"}
        or raw["format"] != "hal.o65.protocol.v1"
    ):
        raise ValueError("evaluation protocol schema changed")

    def dataset(name: str) -> EvaluationSet:
        value = raw[name]
        if not isinstance(value, dict) or set(value) != {"seeds", "starts"}:
            raise ValueError("evaluation set schema changed")
        starts = []
        for start in value["starts"]:
            if not isinstance(start, dict) or set(start) != {"name", "stage", "characters"}:
                raise ValueError("evaluation start schema changed")
            starts.append(EvaluationStart(start["name"], start["stage"], tuple(start["characters"])))
        return EvaluationSet(tuple(value["seeds"]), tuple(starts))

    return EvaluationProtocol(dataset("selection"), dataset("held_out"), raw["max_frames"])


def claim_held_out(output: Path, dataset: EvaluationSet, checkpoint_sha256: str) -> Path:
    """Consume a held-out set once within the shared results directory."""
    payload = json.dumps(asdict(dataset), sort_keys=True)
    identity = hashlib.sha256(payload.encode()).hexdigest()
    claim = output.parent / f".o65-held-out-{identity}.json"
    with claim.open("x") as target:
        json.dump(
            {"output": str(output.resolve()), "checkpoint_sha256": checkpoint_sha256, "set": asdict(dataset)},
            target,
            sort_keys=True,
        )
    return claim


@dataclass(frozen=True, slots=True)
class EvaluationMatch:
    simulator: str
    pair: str
    seed: int
    start: str
    focal_port: int
    active_frames: int
    net_stocks: int
    won: bool | None
    stop_reason: str
    events: dict[str, float]
    wall_seconds: float
    replay: str | None
    initial_rng: int | None


def score_match(
    trajectory: Trajectory,
    *,
    simulator: str,
    pair: str,
    seed: int,
    start: EvaluationStart,
    focal_port: int,
    stop_reason: str,
    wall_seconds: float,
    replay: Path | None = None,
) -> EvaluationMatch:
    initial_rng = int(trajectory.random_seed[0]) if len(trajectory) else None
    active = trajectory.frame_id >= 0
    if not active.any():
        raise ValueError("evaluation match contains no active frames")
    keep = np.flatnonzero(active)
    trajectory = Trajectory(
        trajectory.frame_id[keep],
        {port: {name: values[keep] for name, values in columns.items()} for port, columns in trajectory.post.items()},
        trajectory.random_seed[keep],
    )
    summary = summarize_trajectory(trajectory)
    stocks = {1: summary.p1_stocks_left, 2: summary.p2_stocks_left}
    if any(not 0 <= value <= 4 for value in stocks.values()):
        raise ValueError("evaluation expects fresh four-stock matches")
    minutes = len(trajectory) / 3600
    events: dict[str, float] = {}
    edge = EDGE_X[melee.Stage(start.stage)]
    tech_states = [
        melee.Action.NEUTRAL_TECH.value,
        melee.Action.FORWARD_TECH.value,
        melee.Action.BACKWARD_TECH.value,
        melee.Action.WALL_TECH.value,
        melee.Action.WALL_TECH_JUMP.value,
        melee.Action.CEILING_TECH.value,
    ]
    for role, port in (("ego", focal_port), ("opp", 3 - focal_port)):
        columns = trajectory.post[port]
        stock = columns["stock"]
        deaths = float(np.maximum(stock[:-1] - stock[1:], 0).sum())
        events[f"{role}_deaths_per_min"] = deaths / minutes
        events[f"{role}_ledge_grabs_per_min"] = (
            len(onsets(columns["action"] == melee.Action.EDGE_CATCHING.value)) / minutes
        )
        events[f"{role}_techs_per_min"] = len(onsets(np.isin(columns["action"], tech_states))) / minutes
        events[f"{role}_offstage_fraction"] = float((np.abs(columns["position_x"]) > edge).mean())
        damage = summary.p1_damage_taken if port == 1 else summary.p2_damage_taken
        events[f"{role}_damage_taken_per_min"] = damage / minutes
    events["timeout"] = float(stop_reason == "timeout")
    events["frame_budget"] = float(stop_reason == "frame_budget")
    events["match_minutes"] = minutes
    net = stocks[focal_port] - stocks[3 - focal_port]
    won = stocks[focal_port] > 0 if stop_reason == "terminal" else None
    if stop_reason == "terminal" and (stocks[1] == 0) == (stocks[2] == 0):
        raise ValueError("terminal match must have exactly one surviving player")
    return EvaluationMatch(
        simulator,
        pair,
        seed,
        start.name,
        focal_port,
        len(trajectory),
        net,
        won,
        stop_reason,
        events,
        wall_seconds,
        None if replay is None else str(replay),
        initial_rng if simulator == "dolphin" else seed,
    )


def _interval(values: Sequence[float]) -> dict[str, float | None]:
    mean, low, high = mean_ci(values)
    return {
        key: value if math.isfinite(value) else None for key, value in (("mean", mean), ("low", low), ("high", high))
    }


def summarize_matches(matches: Sequence[EvaluationMatch]) -> dict[str, object]:
    if not matches:
        raise ValueError("cannot summarize an empty evaluation")
    pairs: dict[str, list[EvaluationMatch]] = {}
    for match in matches:
        pairs.setdefault(match.pair, []).append(match)
    if any(len(pair) != 2 or {match.focal_port for match in pair} != {1, 2} for pair in pairs.values()):
        raise ValueError("paired evaluation needs both port orientations exactly once")
    rates = [
        sum(match.net_stocks for match in pair) * 3600 / sum(match.active_frames for match in pair)
        for pair in pairs.values()
    ]
    net = _interval(rates)
    low = net["low"]
    mean = net["mean"]
    lcb = None if low is None or mean is None else mean - (mean - low) * 1.645 / 1.96
    # Incomplete pairs cannot manufacture wins from a frame-budget stock lead.
    wins = [
        sum(float(match.won is True) for match in pair) / 2
        for pair in pairs.values()
        if all(match.won is not None for match in pair)
    ]
    win = _interval(wins)
    for key in ("low", "high"):
        bound = win[key]
        if bound is not None:
            win[key] = min(1.0, max(0.0, bound))
    return {
        "matches": len(matches),
        "paired_seeds": len(pairs),
        "completed_pairs": len(wins),
        "net_stocks_per_min": net,
        "net_stock_lcb": lcb,
        "win_rate": win,
        "events": {
            name: _interval([sum(match.events[name] for match in pair) / 2 for pair in pairs.values()])
            for name in matches[0].events
        },
        "active_frames_per_wall_second": sum(match.active_frames for match in matches)
        / sum(match.wall_seconds for match in matches),
    }


def compare_simulators(msl: Sequence[EvaluationMatch], dolphin: Sequence[EvaluationMatch]) -> dict[str, object]:
    def indexed(matches: Sequence[EvaluationMatch]) -> dict[tuple[str, int], EvaluationMatch]:
        result = {(match.pair, match.focal_port): match for match in matches}
        if len(result) != len(matches):
            raise ValueError("duplicate simulator match")
        summarize_matches(matches)
        return result

    native, live = indexed(msl), indexed(dolphin)
    if native.keys() != live.keys():
        raise ValueError("simulator comparisons require identical paired schedules")
    if any((native[key].seed, native[key].start) != (live[key].seed, live[key].start) for key in native):
        raise ValueError("simulator seeds or starts differ")
    names = tuple(next(iter(native.values())).events)
    pair_ids = sorted({pair for pair, _ in native})
    divergences = {
        name: _interval(
            [
                sum(native[pair, port].events[name] - live[pair, port].events[name] for port in (1, 2)) / 2
                for pair in pair_ids
            ]
        )
        for name in names
    }
    return {
        "direction": "MSL minus Dolphin",
        "unit": "paired seed and start",
        "events": divergences,
        "rng_note": "Policy sampling seeds match; stock Dolphin does not accept an injected game RNG seed.",
    }


def build_policy(
    checkpoints: tuple[EvaluationCheckpoint, EvaluationCheckpoint],
    conditions: tuple[PlayerCondition, PlayerCondition],
    *,
    seed: int,
    focal_port: int,
    history_mode: Literal["block", "sliding"] = "block",
) -> SelfPlayPolicy:
    players = {}
    for role, (checkpoint, condition) in enumerate(zip(checkpoints, conditions, strict=True)):
        port = focal_port if role == 0 else 3 - focal_port
        players[port] = SelfPlayPlayer(
            checkpoint.model,
            checkpoint.statistics,
            condition,
            port=port,
            seed=seed,
            logical_player=role,
            history_mode=history_mode,
        )
    return SelfPlayPolicy(players)


def native_observation_rows(frame: NativeFrameBatch) -> dict[Slot, list[ObservationRow]]:
    """Match Dolphin's processed readback while queues retain submitted factors."""
    if frame.frame_id.shape != (1,):
        raise ValueError("synchronous evaluation requires one native lane")
    flat = {name: value[0].item() for name, value in frame.columns.items()}
    observed = native_controller_readback(frame.applied_action)
    rows = {}
    for port in (1, 2):
        rows[Slot(0, port)] = [
            ObservationRow(int(frame.frame_id[0]), flat, observed[0, port - 1], bool(frame.reset[0]))
        ]
    return rows


def run_msl_match(
    simulator: NativeRolloutBatch,
    policy: SelfPlayPolicy,
    match: NativeMatch,
    *,
    max_frames: int,
) -> tuple[Trajectory, str, float]:
    started = time.perf_counter()
    frame = simulator.reset((match,), np.array([True]))
    columns: dict[str, list[float | int]] = {name: [] for name in frame.columns}
    frames: list[int] = []
    reason = "frame_budget"
    for at in range(max_frames):
        for name, value in frame.columns.items():
            columns[name].append(value[0].item())
        frames.append(int(frame.frame_id[0]))
        if bool(frame.terminated[0]) or bool(frame.truncated[0]):
            reason = "terminal" if bool(frame.terminated[0]) else "timeout"
            break
        if at == max_frames - 1:
            break
        rows = native_observation_rows(frame)
        plans = policy.plan_rows(rows)
        actions = np.stack([plans[Slot(0, port)][0] for port in (1, 2)]).astype(np.float32)
        frame = simulator.step(actions[None], np.array([True]))
    trajectory = Trajectory(
        np.asarray(frames, dtype=np.int32),
        {
            port: {
                suffix: np.asarray(columns[f"p{port}_{suffix}"])
                for suffix in POST_FIELD_SUFFIXES
                if f"p{port}_{suffix}" in columns
            }
            for port in (1, 2)
        },
        np.zeros(len(frames), dtype=np.uint32),
    )
    return trajectory, reason, time.perf_counter() - started


def dolphin_stop_reason(method: EndMethod | None, *, captured_frames: int, max_frames: int) -> str:
    """A short match must have a valid game-end record; it is not a budget stop."""
    if not 0 < captured_frames <= max_frames:
        raise ValueError("Dolphin capture exceeds its frame budget or is empty")
    if method == EndMethod.TIME:
        return "timeout"
    if method in (EndMethod.GAME, EndMethod.RESOLVED):
        return "terminal"
    if captured_frames == max_frames:
        return "frame_budget"
    raise RuntimeError(f"Dolphin match ended early without a completed game-end record: {method}")


def run_dolphin_match(
    policy: SelfPlayPolicy,
    start: EvaluationStart,
    characters: tuple[int, int],
    *,
    max_frames: int,
    replay_dir: Path,
    labels: dict[int, str],
    session: SessionConfig,
) -> tuple[Trajectory, str, float, Path]:
    matchup = Matchup(
        melee.Stage(start.stage), tuple(PlayerSetup(port, melee.Character(characters[port - 1])) for port in (1, 2))
    )
    started = time.perf_counter()
    boots = run_matches_vec(
        replace(session, replay_dir=replay_dir),
        [VecMatch(matchup, (1, 2))],
        lambda: policy,
        max_frames=max_frames,
        max_parallel=1,
        start_retries=0,
    )
    if len(boots) != 1 or len(boots[0]) != 1:
        raise RuntimeError("Dolphin evaluation failed; no match may be dropped")
    replay_files = list(replay_dir.rglob("*.slp"))
    if len(replay_files) != 1:
        raise RuntimeError("Dolphin evaluation must produce exactly one replay")
    replay = replay_files[0]
    stamp_replay_identity(replay, labels)
    game = peppi_py.read_slippi(str(replay), skip_frames=True)
    method = None if game.end is None else EndMethod(int(game.end.method))
    reason = dolphin_stop_reason(method, captured_frames=len(boots[0][0]), max_frames=max_frames)
    if start.stage == melee.Stage.POKEMON_STADIUM.value and game.start.is_frozen_ps is not True:
        raise ValueError("Dolphin evaluation requires frozen Stadium")
    return boots[0][0], reason, time.perf_counter() - started, replay


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def evaluate(
    checkpoint: Path,
    anchors: Sequence[Path],
    output: Path,
    *,
    protocol: EvaluationProtocol,
    split: Literal["selection", "held_out"],
    conditions: tuple[PlayerCondition, PlayerCondition],
    simulators: tuple[Literal["msl", "dolphin"], ...] = ("msl", "dolphin"),
    data_dir: Path | None = None,
    device: str = "cuda",
    history_mode: Literal["block", "sliding"] = "block",
    condition_grid: tuple[tuple[PlayerCondition, PlayerCondition], ...] | None = None,
    compare_events: bool = False,
    session: SessionConfig | None = None,
) -> dict[str, object]:
    if (
        not anchors
        or not simulators
        or len(set(simulators)) != len(simulators)
        or any(name not in ("msl", "dolphin") for name in simulators)
    ):
        raise ValueError("evaluation needs frozen anchors and distinct supported simulators")
    if split not in ("selection", "held_out"):
        raise ValueError("unknown evaluation split")
    if "msl" in simulators and data_dir is None:
        raise ValueError("MSL evaluation requires its data directory")
    if compare_events and set(simulators) != {"msl", "dolphin"}:
        raise ValueError("event validation requires both simulators")
    output.mkdir(parents=True, exist_ok=False)
    focal = load_evaluation_checkpoint(checkpoint, device=torch.device(device))
    frozen = [load_evaluation_checkpoint(anchor, device=torch.device(device)) for anchor in anchors]
    grid = (conditions,) if condition_grid is None else condition_grid
    if not grid or len(set(grid)) != len(grid):
        raise ValueError("evaluation condition grid must be nonempty and unique")
    if len({anchor.checkpoint_sha256 for anchor in frozen}) != len(frozen):
        raise ValueError("frozen anchors must have distinct checkpoint identities")
    cells: list[tuple[int, int, tuple[PlayerCondition, PlayerCondition], EvaluationCheckpoint]] = []
    for cell_index, cell_conditions in enumerate(grid):
        for anchor_index, anchor in enumerate(frozen):
            if compare_events and focal.policy_sha256 == anchor.policy_sha256:
                same_behavior = cell_conditions[0] == cell_conditions[1]
                if not bool(
                    torch.count_nonzero(focal.model.state_dict()["temporal.return_conditioner.embedding.weight"][:, 1])
                ):
                    same_behavior = (
                        replace(cell_conditions[0], lambda_value=cell_conditions[1].lambda_value) == cell_conditions[1]
                    )
                if same_behavior:
                    raise ValueError(
                        "simulator validation requires distinct policies; proxy lambda alone changes no behavior"
                    )
            build_policy((focal, anchor), cell_conditions, seed=0, focal_port=1, history_mode=history_mode)
            cells.append((cell_index, anchor_index, cell_conditions, anchor))
    selected = protocol.selection if split == "selection" else protocol.held_out
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    record: dict[str, object] = {
        "format": "hal.o65.evaluation-results.v1",
        "checkpoint_sha256": focal.checkpoint_sha256,
        "policy_sha256": focal.policy_sha256,
        "anchor_policy_sha256": [anchor.policy_sha256 for anchor in frozen],
        "anchor_sha256": [anchor.checkpoint_sha256 for anchor in frozen],
        "protocol": asdict(protocol),
        "split": split,
        "condition_grid": [[asdict(condition) for condition in pair] for pair in grid],
        "history_mode": history_mode,
        "u_star": 0.95,
        "treatment": "current snapshot at the declared identity, lambda, D, R and history mode",
        "control": "frozen anchor snapshots at their declared conditioning",
        "invariant_inputs": [
            "policy sampling seeds",
            "fresh four-stock zero-percent starts",
            "paired port orientations",
            "u_star=0.95",
            "feature statistics",
            "simulator builds",
        ],
        "timeout_frame": 28800,
        "inference_dtype": str(next(focal.model.parameters()).dtype),
        "git_sha": git_sha,
        "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if device.startswith("cuda") else None,
        },
        "checkpoint_provenance": focal.provenance,
        "anchor_provenance": [anchor.provenance for anchor in frozen],
        "metric_definitions": {
            "active_frames": "frame_id >= 0; each mirrored seed/start is one statistical unit",
            "net_stocks_per_min": "mean of paired net stocks divided by paired active minutes",
            "win_rate": "mean paired stockout wins; budget limits and timeouts remain unscored",
            "intervals": "two-sided 95% normal intervals over paired units; fewer than two units gives null bounds",
            "net_stock_lcb": "one-sided 95% normal lower bound over paired units",
            "offstage": "fraction of active frames past the walkable stage edge",
        },
        "rng_note": "Seeds control policy draws and MSL reset. Dolphin game RNG is recorded, not injected.",
        "groups": [],
    }
    if split == "held_out":
        claim_held_out(output, protocol.held_out, focal.checkpoint_sha256)
    write_json(output / "protocol.json", record)
    groups: list[dict[str, object]] = []
    for cell_index, anchor_index, conditions, anchor in cells:
        by_simulator: dict[str, list[EvaluationMatch]] = {}
        for simulator_name in simulators:
            matches: list[EvaluationMatch] = []
            session_cfg = (session or default_session_cfg()) if simulator_name == "dolphin" else None
            native = NativeRolloutBatch(data_dir, 1) if simulator_name == "msl" and data_dir is not None else None
            try:
                if native is not None:
                    record["msl"] = {
                        "library_sha256": native.library_sha256,
                        "manifest_sha256": native.manifest_sha256,
                        "adapter_sha256": library_sha256(),
                    }
                if session_cfg is not None:
                    record["dolphin"] = {
                        "session": {
                            key: str(value) if isinstance(value, Path) else value
                            for key, value in asdict(session_cfg).items()
                        },
                        "iso_sha256": sha256(Path(session_cfg.iso_path)),
                        "launcher_sha256": sha256(Path(session_cfg.dolphin_path)),
                        "binary_sha256": sha256(Path(session_cfg.dolphin_path).with_name("AppRun.wrapped"))
                        if Path(session_cfg.dolphin_path).with_name("AppRun.wrapped").is_file()
                        else sha256(Path(session_cfg.dolphin_path)),
                    }
                write_json(output / "protocol.json", record)
                for start in selected.starts:
                    for seed in selected.seeds:
                        for focal_port in (1, 2):
                            policy = build_policy(
                                (focal, anchor),
                                conditions,
                                seed=seed,
                                focal_port=focal_port,
                                history_mode=history_mode,
                            )
                            characters = (
                                start.characters if focal_port == 1 else (start.characters[1], start.characters[0])
                            )
                            pair = f"c{cell_index}/a{anchor_index}/{start.name}/{seed}"
                            replay = None
                            if native is not None:
                                match = NativeMatch(
                                    start.stage,
                                    (NativePlayer(1, characters[0]), NativePlayer(2, characters[1])),
                                    seed,
                                    max_frame=28800,
                                )
                                trajectory, reason, seconds = run_msl_match(
                                    native, policy, match, max_frames=protocol.max_frames
                                )
                            else:
                                if session_cfg is None:
                                    raise RuntimeError("Dolphin session is absent")
                                labels = {
                                    port: f"{'RL' if port == focal_port else 'REF'}-{(focal if port == focal_port else anchor).checkpoint_sha256[:8]}-"
                                    + hashlib.sha256(
                                        json.dumps(
                                            asdict(conditions[0 if port == focal_port else 1]), sort_keys=True
                                        ).encode()
                                    ).hexdigest()[:6]
                                    for port in (1, 2)
                                }
                                directory = (
                                    output
                                    / f"cell-{cell_index}-anchor-{anchor_index}"
                                    / start.name
                                    / f"{seed}-p{focal_port}"
                                )
                                trajectory, reason, seconds, replay = run_dolphin_match(
                                    policy,
                                    start,
                                    characters,
                                    max_frames=protocol.max_frames,
                                    replay_dir=directory,
                                    labels=labels,
                                    session=session_cfg,
                                )
                            matches.append(
                                score_match(
                                    trajectory,
                                    simulator=simulator_name,
                                    pair=pair,
                                    seed=seed,
                                    start=start,
                                    focal_port=focal_port,
                                    stop_reason=reason,
                                    wall_seconds=seconds,
                                    replay=replay,
                                )
                            )
                            write_json(
                                output / f"matches-{cell_index}-{anchor_index}-{simulator_name}.json",
                                [asdict(match) for match in matches],
                            )
            finally:
                if native is not None:
                    native.close()
            by_simulator[simulator_name] = matches
        group: dict[str, object] = {
            "anchor_sha256": anchor.checkpoint_sha256,
            "conditions": [asdict(condition) for condition in conditions],
            "metrics": {name: summarize_matches(matches) for name, matches in by_simulator.items()},
        }
        if compare_events:
            group["simulator_divergences"] = compare_simulators(by_simulator["msl"], by_simulator["dolphin"])
        groups.append(group)
    record["groups"] = groups
    write_json(output / "results.json", record)
    return record
