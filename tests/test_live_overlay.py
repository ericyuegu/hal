import json
import socket
from contextlib import closing
from dataclasses import asdict
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.controller import ControllerAction
from hal.inference.api import ActionPlan
from hal.netplay_service.graphics import ASSETS
from hal.netplay_service.graphics import PanelRenderer
from hal.netplay_service.graphics import bar_endpoint
from hal.netplay_service.stream import GameStreamState
from hal.netplay_service.stream import IdleStreamState
from hal.netplay_service.stream import read_stream_state
from hal.netplay_service.stream import write_stream_state
from hal.netplay_service.telemetry import TelemetryPublisher
from hal.netplay_service.telemetry import TelemetryReceiver
from hal.netplay_service.telemetry import TelemetrySample
from hal.netplay_service.telemetry import decode_sample
from hal.wire import BUTTON_BITS

STATE = GameStreamState("FOX", "MASTER", 120, 1, "game-a", 0, 1)


def test_telemetry_expires_inputs_and_value_independently(tmp_path: Path, monkeypatch) -> None:
    now = [10.0]
    monkeypatch.setattr("hal.netplay_service.telemetry.time.monotonic", lambda: now[0])
    action = ControllerAction(1, -1, 0.5, 0, 0.6, 1, BUTTON_BITS["z"])
    with (
        closing(TelemetryReceiver(tmp_path / "socket")) as receiver,
        closing(TelemetryPublisher(tmp_path / "socket", tmp_path / "state", STATE)) as sender,
    ):
        sender.observe(10, action)
        sender.predicted(ActionPlan(0, 1, 1, 6, (), 40.0))
        assert receiver.poll(STATE, 10) == (action, 40)
        assert receiver.poll(STATE, 10.251) == (NEUTRAL_CONTROLLER_ACTION, 40)
        assert receiver.poll(STATE, 11.001) == (NEUTRAL_CONTROLLER_ACTION, None)
        assert receiver.poll(replace(STATE, game_id="game-b"), 10) == (NEUTRAL_CONTROLLER_ACTION, None)


def test_value_uses_six_frame_ema_and_resets_between_games(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("hal.netplay_service.telemetry.time.monotonic", lambda: 10.0)
    with (
        closing(TelemetryReceiver(tmp_path / "socket")) as receiver,
        closing(TelemetryPublisher(tmp_path / "socket", tmp_path / "state", STATE)) as sender,
    ):
        sender.predicted(ActionPlan(0, 1, 0, 0, (), 0.0))
        sender.predicted(ActionPlan(0, 1, 1, 6, (), 120.0))
        assert receiver.poll(STATE, 10)[1] == 60
        assert receiver.poll(replace(STATE, generation=2), 10)[1] is None
        sender.predicted(ActionPlan(0, 1, 2, 12, (), 120.0))
        assert receiver.poll(replace(STATE, generation=2), 10)[1] is None
        assert receiver.poll(IdleStreamState(), 10)[1] is None


def test_missing_overloaded_and_restarted_overlay_never_blocks_sender(tmp_path: Path) -> None:
    path = tmp_path / "socket"
    with closing(TelemetryPublisher(path, tmp_path / "state", STATE)) as sender:
        for _ in range(100):
            sender.observe(1, NEUTRAL_CONTROLLER_ACTION)
        with closing(TelemetryReceiver(path)):
            for frame in range(2000):
                sender.observe(frame, NEUTRAL_CONTROLLER_ACTION)
        sender.observe(2001, NEUTRAL_CONTROLLER_ACTION)
        with closing(TelemetryReceiver(path)) as restarted:
            sender.observe(2002, NEUTRAL_CONTROLLER_ACTION)
            import time

            restarted.poll(STATE, time.monotonic())
            assert restarted.sample.frame_id == 2002


def test_rejects_wrong_game_generation_rollback_and_malformed_samples(tmp_path: Path) -> None:
    sample = TelemetrySample("game-a", 0, 1, 20, 10, 10, NEUTRAL_CONTROLLER_ACTION, None, STATE)
    with (
        closing(TelemetryReceiver(tmp_path / "socket")) as receiver,
        socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sender,
    ):

        def send(value: TelemetrySample) -> None:
            sender.sendto(json.dumps(asdict(value)).encode(), str(tmp_path / "socket"))

        for bad in (
            replace(sample, game_id="other"),
            replace(sample, generation=2),
            replace(sample, sent_at=11),
            replace(sample, sent_at=0),
        ):
            send(bad)
        sender.sendto(b"{}", str(tmp_path / "socket"))
        assert receiver.poll(STATE, 10) == (NEUTRAL_CONTROLLER_ACTION, None)
        send(sample)
        receiver.poll(STATE, 10)
        send(replace(sample, frame_id=19, sent_at=10.1))
        receiver.poll(STATE, 10.1)
        assert receiver.sample == sample
    with pytest.raises(ValueError):
        decode_sample(json.dumps({**asdict(sample), "schema_version": 2}).encode())


def test_panel_uses_exact_website_sprites_and_all_assets_are_installed() -> None:
    from hal.netplay_service.domain import CHARACTERS

    root = Path(__file__).resolve().parents[1]
    for choice in CHARACTERS:
        assert (ASSETS / f"characters/{choice.value}.png").read_bytes() == (
            root / f"web/netplay/public/characters/{choice.value}.png"
        ).read_bytes()
    renderer = PanelRenderer()
    image = renderer.render(STATE, NEUTRAL_CONTROLLER_ACTION, 400.0, "")
    sprite = Image.open(ASSETS / "characters/FOX.png").convert("RGBA").resize((48, 48), Image.Resampling.NEAREST)
    for x, y in ((x, y) for x in range(48) for y in range(48)):
        if sprite.getpixel((x, y))[3] == 255:
            assert image.getpixel((x + 32, y + 85)) == sprite.getpixel((x, y))
    assert "Copyright (c) 2023 Frank Borden" in (ASSETS / "controller/LICENSE").read_text()
    assert (ASSETS / "controller/base.png").is_file()


def test_controller_uses_physical_button_readback_and_full_geometry() -> None:
    renderer = PanelRenderer()
    neutral = renderer.render(STATE, NEUTRAL_CONTROLLER_ACTION, None, "")
    z_only = renderer.render(STATE, replace(NEUTRAL_CONTROLLER_ACTION, buttons=BUTTON_BITS["z"]), None, "")
    # Z produces logical A in Melee. Physical A must stay released.
    assert neutral.getpixel((58 + 277, 342 + 96)) == z_only.getpixel((58 + 277, 342 + 96))
    pressed = renderer.render(
        STATE, replace(NEUTRAL_CONTROLLER_ACTION, buttons=BUTTON_BITS["a"] | BUTTON_BITS["l"]), None, ""
    )
    assert pressed.getpixel((58 + 277, 342 + 96)) != neutral.getpixel((58 + 277, 342 + 96))
    assert pressed.getpixel((58 + 43, 342 + 12)) != neutral.getpixel((58 + 43, 342 + 12))
    assert neutral.getpixel((58 + 45, 342 + 12))[3] == 255
    assert neutral.size == (456, 608)
    assert 342 - (308 + 18) >= 16
    assert 608 - (342 + 256) >= 0
    assert bar_endpoint(400) == 424
    assert bar_endpoint(-400) == 32
    assert bar_endpoint(0) == 228


def test_stream_state_schema_and_difficulty_reject_invalid_formats(tmp_path: Path) -> None:
    path = tmp_path / "state"
    write_stream_state(path, replace(STATE, mode="self-play"))
    assert read_stream_state(path).mode == "self-play"
    raw = json.loads(path.read_text())
    raw["schema_version"] = 1
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="schema"):
        read_stream_state(path)
    with pytest.raises(ValueError):
        replace(STATE, desired_return=float("nan"))


def test_difficulty_follows_submitted_conditioning_not_initial_selection(tmp_path: Path) -> None:
    from hal.inference.api import PolicyInput
    from hal.inference.api import PredictionRequest
    from hal.netplay_service.stream import difficulty_label

    with closing(TelemetryPublisher(tmp_path / "absent", tmp_path / "state", STATE)) as sender:
        for desired, label in ((20.0, "Easy"), (120.0, "Medium"), (None, "Unconditioned"), (19.9, "Custom")):
            item = PolicyInput(0, 0, 2, {}, NEUTRAL_CONTROLLER_ACTION, "IBDW#0", desired_return=desired)
            sender.requested(PredictionRequest(0, 1, 0, 0, (item,), ()))
            state = sender.state
            assert difficulty_label(state.desired_return) == label
            assert state.port == 2
            assert state.imitation == "IBDW#0"


def test_value_clips_only_bar_not_signed_numeric_display() -> None:
    renderer = PanelRenderer()
    clipped = renderer.render(STATE, NEUTRAL_CONTROLLER_ACTION, 400, "")
    limit = renderer.render(STATE, NEUTRAL_CONTROLLER_ACTION, 120, "")
    assert clipped.crop((32, 254, 425, 269)).tobytes() == limit.crop((32, 254, 425, 269)).tobytes()
    assert clipped.crop((300, 220, 425, 247)).tobytes() != limit.crop((300, 220, 425, 247)).tobytes()


def test_submitted_conditioning_reaches_renderer_without_gameplay_file_writes(tmp_path: Path, monkeypatch) -> None:
    import time

    from hal.inference.api import PolicyInput
    from hal.inference.api import PredictionRequest

    with (
        closing(TelemetryReceiver(tmp_path / "socket")) as receiver,
        closing(TelemetryPublisher(tmp_path / "socket", tmp_path / "state", STATE)) as sender,
    ):
        original = (tmp_path / "state").read_bytes()
        item = PolicyInput(0, 0, 2, {}, NEUTRAL_CONTROLLER_ACTION, "IBDW#0", desired_return=20.0)
        sender.requested(PredictionRequest(0, 1, 0, 0, (item,), ()))
        receiver.poll(STATE, time.monotonic())
        shown = receiver.state_for(STATE)
        assert shown.desired_return == 20 and shown.imitation == "IBDW#0" and shown.port == 2
        assert (tmp_path / "state").read_bytes() == original
