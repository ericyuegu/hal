"""Run one prepared Cody Fox policy through ranked sets and an OBS stream."""

import hashlib
import json
import math
import os
import signal
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Self

import melee
import torch
from loguru import logger
from websockets.exceptions import WebSocketException

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.eval.netplay import DolphinConnectionLost
from hal.eval.netplay import run_netplay_match
from hal.eval.ranked_replays import RankedReplayUploads
from hal.eval.replays import read_new_replay_end
from hal.eval.scheduling import FrameTiming
from hal.fixtures import ISO
from hal.fixtures import NETPLAY_EMULATOR
from hal.fixtures import ensure
from hal.inference.action_sequence_artifact import load_action_sequence_policy
from hal.inference.api import PreparedInferenceProfile
from hal.inference.api import RuntimeConfig
from hal.inference.engine import configure_inference_process
from hal.inference.engine import freeze_inference_runtime
from hal.inference.engine import start_inference_worker
from hal.netplay_service.obs import ObsStudio
from hal.netplay_service.stream import DisplayGroup
from hal.netplay_service.stream import PulseAudio
from hal.sim.netplay import ConnectAbandoned
from hal.sim.netplay import CountdownEnded
from hal.sim.netplay import NetplaySession
from hal.sim.netplay import NetplaySetup
from hal.sim.ranked import RankedMenu
from hal.sim.ranked import RankedScreen
from hal.sim.ranked import read_screen
from hal.sim.session import FrameTimeout


@dataclass(frozen=True, slots=True)
class RankedConfig:
    bundle: Path
    account: Path
    output: Path
    twitch_key: Path
    source_revision: str
    display: str = ":90"
    advantage: float = 120.0
    temperature: float = 1.0
    seed: int = 120647
    max_games: int = 0
    bandwidth_test: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.advantage) or not -20 <= self.advantage <= 140:
            raise ValueError("advantage must be finite and between -20 and 140")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        if self.max_games < 0:
            raise ValueError("max_games must be non-negative; zero runs until stopped")
        if len(self.source_revision) not in (8, 40) or any(c not in "0123456789abcdef" for c in self.source_revision):
            raise ValueError("source_revision must be the source Git commit")


def timing() -> FrameTiming:
    return FrameTiming(2, 1, 3, 4, 8)


def _digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _write(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def overlay(advantage: float) -> str:
    return f"HAL · Cody Fox · advantage {advantage:g} · Ranked"


class RankedStream:
    """Stream failure can restart OBS, but cannot stop or write to the player."""

    def __init__(self, config: RankedConfig, output: Path, environment: dict[str, str]) -> None:
        self.config = config
        self.output = output
        self.environment = environment
        self.playing = threading.Event()
        self.screen: RankedScreen | None = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="ranked-stream", daemon=True)

    def __enter__(self) -> Self:
        self._thread.start()
        if not self._ready.wait(20):
            self.__exit__(None, None, None)
            raise RuntimeError("OBS did not start; see stream-error.json")
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=20)
        if self._thread.is_alive():
            raise RuntimeError("ranked stream did not stop")

    def _run(self) -> None:
        screenshots = self.output / "screenshots"
        screenshots.mkdir()
        retained: deque[Path] = deque()
        backoff = 1.0
        last_saved = 0.0
        while not self._stop.is_set():
            studio = ObsStudio(self.config.display, self.output / "stream", self.environment)
            try:
                studio.start(self.config.twitch_key.read_text().strip(), bandwidth_test=self.config.bandwidth_test)
                self._ready.set()
                backoff = 1
                while not self._stop.wait(0.35):
                    studio.update(overlay(self.config.advantage))
                    _write(self.output / "obs-stats.json", {"at": time.time(), **studio.stats()})
                    captured = time.monotonic()
                    png = studio.screenshot()
                    if png is None:
                        self.screen = None
                        continue
                    temporary = self.output / "latest.partial.png"
                    temporary.write_bytes(png)
                    temporary.replace(self.output / "latest.png")
                    if captured - last_saved >= 2:
                        path = screenshots / f"{time.time_ns()}.png"
                        path.write_bytes(png)
                        retained.append(path)
                        if len(retained) > 300:
                            retained.popleft().unlink()
                        last_saved = captured
                    if self.playing.is_set():
                        self.screen = None
                        continue
                    try:
                        self.screen = read_screen(png, captured)
                        _write(self.output / "screen.json", asdict(self.screen))
                    except (ValueError, OSError, subprocess.SubprocessError) as error:
                        self.screen = None
                        _write(self.output / "vision-error.json", {"at": time.time(), "type": type(error).__name__})
            except (WebSocketException, OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
                self.screen = None
                _write(
                    self.output / "stream-error.json",
                    {"at": time.time(), "type": type(error).__name__, "message": str(error)},
                )
            finally:
                studio.close()
            if self._stop.wait(backoff):
                return
            backoff = min(30, backoff * 2)


def run(config: RankedConfig) -> Path:
    if torch.cuda.device_count() != 1 or "RTX PRO 6000 Blackwell" not in torch.cuda.get_device_name():
        raise RuntimeError("fixed ranked timing requires one validated RTX PRO 6000 Blackwell GPU")
    subprocess.run(["tesseract", "--version"], check=True, stdout=subprocess.DEVNULL)
    config.output.mkdir(parents=True, exist_ok=True)
    output = config.output / datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    output.mkdir()
    stop = threading.Event()

    def request_stop(_number: int, _frame: object) -> None:
        if stop.is_set():
            raise KeyboardInterrupt
        stop.set()

    previous_handlers = {number: signal.signal(number, request_stop) for number in (signal.SIGINT, signal.SIGTERM)}
    try:
        with RankedReplayUploads(config.output) as uploads:
            _run(config, output, stop, uploads.notify)
    except BaseException as error:
        _write(output / "status.json", {"state": "failed", "at": time.time(), "type": type(error).__name__})
        raise
    else:
        _write(output / "status.json", {"state": "stopped", "at": time.time()})
    finally:
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)
    return output


def _run(config: RankedConfig, output: Path, stop: threading.Event, notify_uploads: Callable[[], None]) -> None:
    iso, dolphin = ensure(ISO), ensure(NETPLAY_EMULATOR)
    account = json.loads(config.account.read_text())
    local_code = account.get("connectCode")
    if not isinstance(local_code, str) or not local_code:
        raise ValueError("Slippi account file has no connect code")
    configure_inference_process()
    runtime = RuntimeConfig(1, (2,), replan_interval_frames=4)
    policy = load_action_sequence_policy(
        config.bundle,
        device="cuda",
        seed=config.seed,
        compiled=True,
        history_mode="kv_cache",
        kv_update_frames=4,
    )
    # Preparation builds the compiled calls and CUDA graphs. The owner's fixed
    # timing replaces the qualification benchmark on this validated GPU.
    policy.prepare_prediction(runtime, 8, 3)
    profile = PreparedInferenceProfile(
        "ranked-delay-2",
        policy.checkpoint_sha256,
        "kv_cache",
        8,
        3,
        policy.prepared_update_shapes,
        1,
    )
    source_root = Path(__file__).parents[1]
    sources = [
        Path(__file__),
        source_root / "eval/ranked_replays.py",
        source_root / "sim/ranked.py",
        source_root / "sim/netplay.py",
        source_root / "sim/session.py",
        source_root / "netplay_service/obs.py",
        source_root / "netplay_service/stream.py",
    ]
    snapshot = output / "source"
    snapshot.mkdir()
    for source in sources:
        target = snapshot / source.relative_to(source_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    _write(
        output / "manifest.json",
        {
            "schema_version": 1,
            "source_revision": config.source_revision,
            "source_sha256": {str(p.relative_to(source_root)): _digest(p) for p in sources},
            "bundle_sha256": _digest(config.bundle),
            "checkpoint_sha256": policy.checkpoint_sha256,
            "account_sha256": _digest(config.account),
            "iso_sha256": _digest(iso),
            "dolphin_sha256": _digest(dolphin),
            "identity": "IBDW#0",
            "character": "FOX",
            "advantage": config.advantage,
            "temperature": config.temperature,
            "seed": policy.sampling_seed,
            "timing": asdict(timing()),
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "driver": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
            ).strip(),
            "qualification": "fixed timing authorized for the validated GPU; compilation only",
            "created_at": datetime.now(UTC).isoformat(),
        },
    )
    if stop.is_set():
        return
    freeze_inference_runtime()
    with (
        start_inference_worker(policy, profile, 0.0005) as client,
        DisplayGroup(1, 100, True, stream_display=config.display),
        PulseAudio(output / "pulse") as pulse,
    ):
        environment = dict(os.environ) | pulse.environment
        previous_environment = {key: os.environ.get(key) for key in pulse.environment}
        os.environ.update(pulse.environment)
        try:
            with RankedStream(config, output, environment) as stream, torch.compiler.set_stance("fail_on_recompile"):
                games = 0
                failures = 0
                replay_dir = output / "replays"
                setup = NetplaySetup(melee.Character.FOX, "", local_code=local_code)

                def record_action(action: str) -> None:
                    with (output / "actions.jsonl").open("a") as log:
                        log.write(
                            json.dumps(
                                {
                                    "at": time.time(),
                                    "action": action,
                                    "screen": None if stream.screen is None else asdict(stream.screen),
                                }
                            )
                            + "\n"
                        )

                def on_live() -> None:
                    stream.playing.set()
                    _write(output / "status.json", {"state": "playing", "game": games + 1, "at": time.time()})

                while not stop.is_set() and (not config.max_games or games < config.max_games):
                    menu = RankedMenu(lambda: stream.screen, on_action=record_action)
                    try:
                        with NetplaySession(
                            iso,
                            dolphin_path=dolphin,
                            user_json_path=config.account,
                            online_delay=2,
                            replay_dir=replay_dir,
                            slippi_port=51441,
                            realtime=True,
                            graphics_backend="OGL",
                            stream_output=True,
                            menu_driver=menu,
                            connect_timeout_seconds=7200,
                            connect_abandoned=stop.is_set,
                        ) as session:
                            rematch = False
                            while not stop.is_set() and (not config.max_games or games < config.max_games):
                                stream.playing.clear()
                                previous = set(replay_dir.rglob("*.slp"))
                                _write(output / "status.json", {"state": "menu", "game": games + 1, "at": time.time()})
                                try:
                                    result = run_netplay_match(
                                        session,
                                        setup,
                                        client,
                                        runtime,
                                        timing(),
                                        player_identity="IBDW#0",
                                        policy_settings=lambda: (config.advantage, config.temperature),
                                        max_frames=60 * 60 * 10,
                                        rematch=rematch,
                                        on_live=on_live,
                                    )
                                except CountdownEnded:
                                    # A new inference generation must start after a
                                    # countdown quit; the emulator can stay open.
                                    rematch = True
                                    session.submit(NEUTRAL_CONTROLLER_ACTION)
                                    _write(
                                        output / f"countdown-quit-{time.time_ns()}.json",
                                        {"at": time.time(), "game": games + 1},
                                    )
                                    continue
                                session.submit(NEUTRAL_CONTROLLER_ACTION)
                                stream.playing.clear()
                                rematch = True
                                deadline = time.monotonic() + 5
                                while True:
                                    try:
                                        replay = read_new_replay_end(replay_dir, previous)
                                        break
                                    except RuntimeError:
                                        if time.monotonic() >= deadline:
                                            raise
                                        time.sleep(0.05)
                                games += 1
                                failures = 0
                                _write(
                                    output / f"game-{games:04}.json",
                                    {
                                        "schema_version": 1,
                                        "game": games,
                                        "at": time.time(),
                                        "replay": str(replay.path.relative_to(output)),
                                        "replay_sha256": _digest(replay.path),
                                        "end_method": replay.method.name,
                                        "ego_port": result.ego_port,
                                        "opponent_port": result.opponent_port,
                                        "stage": result.stage,
                                        "frames": len(result.trajectory),
                                        "fps": result.game_fps,
                                        "frame_p95_ms": result.frame_interval_p95_ms,
                                        "inference_p95_ms": result.inference_p95_ms,
                                        "transport_corrections": result.transport_correction_frames,
                                        "wall_seconds": result.wall_seconds,
                                    },
                                )
                                notify_uploads()
                    except (DolphinConnectionLost, FrameTimeout, BrokenPipeError, EOFError) as error:
                        failures += 1
                        logger.warning("Ranked transport lost; restarting Dolphin ({}/3)", failures)
                        _write(
                            output / f"disconnect-{time.time_ns()}.json",
                            {
                                "at": time.time(),
                                "game": games + 1,
                                "type": type(error).__name__,
                                "replays": [str(p.relative_to(output)) for p in replay_dir.rglob("*.slp")],
                            },
                        )
                        if failures >= 3:
                            raise
                        stream.playing.clear()
                        stop.wait(2)
                    except ConnectAbandoned:
                        if not stop.is_set():
                            raise
        finally:
            for key, value in previous_environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
