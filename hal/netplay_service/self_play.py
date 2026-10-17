"""Dedicated idle-demo worker. No queue reservations or replay uploads."""

import ctypes
import json
import os
import random
import signal
import tempfile
import time
from contextlib import ExitStack
from contextlib import closing
from dataclasses import asdict
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Protocol

import melee

from hal.eval import self_play
from hal.inference.api import PolicySpec
from hal.inference.api import PreparedInferenceProfile
from hal.inference.client import InferenceClient
from hal.netplay_service.domain import CHARACTERS
from hal.netplay_service.domain import IMITATIONS
from hal.netplay_service.domain import STAGES
from hal.netplay_service.stream import GameStreamState
from hal.netplay_service.stream import IdleStreamState
from hal.netplay_service.stream import write_stream_state
from hal.netplay_service.telemetry import TelemetryPublisher
from hal.sim.netplay import tested_dolphin_version
from hal.sim.session import Matchup
from hal.sim.session import PlayerSetup
from hal.sim.session import Session
from hal.sim.session import SessionCanceled


class CancelSignal(Protocol):
    def is_set(self) -> bool: ...
    def set(self) -> None: ...


@dataclass(frozen=True, slots=True)
class DemoConfig:
    iso_path: Path
    dolphin_path: Path
    display: str
    state_path: Path
    slippi_port: int
    seed: int
    game_number: int
    game_id: str
    stream_ids: tuple[int, int]
    max_frames: int
    pulse_environment: tuple[tuple[str, str], ...]
    git_sha: str
    policy_sha256: str
    checkpoint_sha256: str
    sampling_seeds: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DemoSelection:
    characters: tuple[str, str]
    imitations: tuple[str, str]
    stage: str


@dataclass(frozen=True, slots=True)
class DolphinStarted:
    pid: ctypes.c_int

    def __call__(self, pid: int) -> None:
        self.pid.value = pid


def selection(seed: int, game_number: int) -> DemoSelection:
    rng = random.Random(seed + game_number)
    return DemoSelection(
        (rng.choice(CHARACTERS).value, rng.choice(CHARACTERS).value),
        (rng.choice(IMITATIONS).value, rng.choice(IMITATIONS).value),
        rng.choice(STAGES).value,
    )


def worker(
    config: DemoConfig,
    spec: PolicySpec,
    context_frames: int,
    profiles: tuple[PreparedInferenceProfile, ...],
    connections: tuple[Connection, Connection],
    cancel: CancelSignal,
    engine_stop: CancelSignal,
    dolphin_pid: ctypes.c_int,
) -> None:
    # A separate process group contains the worker and its Dolphin descendants.
    os.setsid()
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from hal.netplay_service.runner import SlotHealthReporter

    os.environ["DISPLAY"] = config.display
    os.environ.update(dict(config.pulse_environment))
    chosen = selection(config.seed, config.game_number)
    record: dict[str, object] = {
        "schema_version": 1,
        "started_at": time.time(),
        **asdict(config),
        "iso_path": str(config.iso_path),
        "dolphin_path": str(config.dolphin_path),
        "state_path": str(config.state_path),
        "selection": asdict(chosen),
        "desired_return": 120,
        "temperature": 1,
        "physical_delay": 2,
    }
    # Pulse paths are local, and no account or stream key enters this record.
    record_path = config.state_path.with_name(f"self-play-{config.game_id}.json")
    record_path.write_text(json.dumps(record))
    try:
        if cancel.is_set():
            return
        with ExitStack() as resources:
            directory = resources.enter_context(tempfile.TemporaryDirectory(prefix="hal-self-play-"))
            clients = tuple(
                resources.enter_context(
                    closing(
                        InferenceClient(
                            spec,
                            context_frames,
                            connection,
                            engine_stop,
                            {p.fixed_prefix_frames: p for p in profiles},
                            timeout_seconds=0.2,
                        )
                    )
                )
                for connection in connections
            )
            health = resources.enter_context(SlotHealthReporter(0, config.state_path.with_suffix(".self-play.json")))
            health.configure_schedule(self_play.TIMING)
            health.connecting(2)
            telemetry = resources.enter_context(
                closing(
                    TelemetryPublisher(
                        config.state_path.with_name("telemetry.sock"),
                        config.state_path,
                        GameStreamState(
                            chosen.characters[0],
                            chosen.imitations[0],
                            120.0,
                            config.game_number,
                            config.game_id,
                            config.stream_ids[0],
                            1,
                            "self-play",
                            1,
                        ),
                    )
                )
            )
            with Session(
                config.iso_path,
                dolphin_path=config.dolphin_path,
                slippi_port=config.slippi_port,
                step_timeout_seconds=1,
                start_timeout_seconds=60,
                replay_dir=Path(directory),
                blocking_input=True,
                emulation_speed=1,
                polling_mode=True,
                canceled=cancel.is_set,
                on_dolphin_started=DolphinStarted(dolphin_pid),
                gfx_backend="OGL",
                dolphin_version=tested_dolphin_version(str(config.dolphin_path)),
                dolphin_batch=True,
                stream_output=True,
            ) as session:
                matchup = Matchup(
                    melee.Stage[chosen.stage],
                    tuple(
                        PlayerSetup(port, melee.Character[character])
                        for port, character in enumerate(chosen.characters, 1)
                    ),
                )
                metrics = self_play.run_match(
                    session,
                    matchup,
                    (clients[0], clients[1]),
                    config.stream_ids,
                    chosen.imitations,
                    telemetry,
                    health.observe_frame,
                    health.playing,
                    max_frames=config.max_frames,
                    observe_policy=health.observe_policy,
                )
                record["metrics"] = asdict(metrics)
    except SessionCanceled:
        record["canceled_at"] = time.time()
    finally:
        write_stream_state(config.state_path, IdleStreamState())
        config.state_path.with_suffix(".self-play.json").unlink(missing_ok=True)
        record["worker_finished_at"] = time.time()
        record_path.write_text(json.dumps(record))
