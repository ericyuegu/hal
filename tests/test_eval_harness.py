from collections.abc import Sequence
from pathlib import Path
from unittest.mock import Mock

import pytest

from hal.eval import harness
from hal.eval.harness import SessionConfig
from hal.eval.harness import _session_kwargs


def test_headless_eval_disables_dolphin_audio() -> None:
    cfg = SessionConfig(iso_path="iso", dolphin_path="dolphin")

    kwargs = _session_kwargs(cfg, slippi_port=51441, replay_dir=None)

    assert cfg.disable_audio is True
    assert kwargs["disable_audio"] is True


def test_rejected_replays_are_separate_from_accepted_boots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    replay_root = tmp_path / "replays" / "model-on-port1"
    cfg = SessionConfig(iso_path="iso", dolphin_path="dolphin", replay_dir=replay_root)
    attempts: list[tuple[int, ...]] = []

    def drive_wave(
        _cfg: object, indices: Sequence[int], _matches: object, _policy_factory: object, **_kwargs: object
    ) -> dict[int, list[str]]:
        attempt = len(attempts) + 1
        attempts.append(tuple(indices))
        for boot_index in indices:
            boot_dir = replay_root / f"boot_{boot_index:03d}"
            boot_dir.mkdir(parents=True, exist_ok=True)
            (boot_dir / "Game.slp").write_text(f"attempt {attempt}, boot {boot_index}")
        if attempt == 1:
            return {0: ["accepted boot 0"], 1: []}
        if attempt == 2:
            return {1: []}
        return {1: ["accepted boot 1"]}

    mocked_wave = Mock(side_effect=drive_wave)
    mocked_logger = Mock()
    monkeypatch.setattr(harness, "_drive_wave", mocked_wave)
    monkeypatch.setattr(harness, "logger", mocked_logger)

    boots = harness.run_matches_vec(cfg, [Mock(), Mock()], Mock(), max_frames=100, max_parallel=2)

    assert boots == [["accepted boot 0"], ["accepted boot 1"]]
    assert attempts == [(0, 1), (1,), (1,)]
    assert [call.kwargs["slippi_port_base"] for call in mocked_wave.call_args_list] == [51441, 51443, 51445]
    assert (replay_root / "boot_000/Game.slp").read_text() == "attempt 1, boot 0"
    assert (replay_root / "boot_001/Game.slp").read_text() == "attempt 3, boot 1"
    failed_root = replay_root.with_name("model-on-port1_failed_attempts")
    assert (failed_root / "boot_001/attempt_1/Game.slp").read_text() == "attempt 1, boot 1"
    assert (failed_root / "boot_001/attempt_2/Game.slp").read_text() == "attempt 2, boot 1"
    assert len(list(replay_root.rglob("*.slp"))) == 2
    assert mocked_logger.warning.call_count == 2
    assert all("produced no complete trajectory" in call.args[0] for call in mocked_logger.warning.call_args_list)


def test_clean_boot_leaves_no_failed_attempt_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    replay_root = tmp_path / "replays"
    cfg = SessionConfig(iso_path="iso", dolphin_path="dolphin", replay_dir=replay_root)

    def drive_wave(
        _cfg: object, _indices: object, _matches: object, _policy_factory: object, **_kwargs: object
    ) -> dict[int, list[str]]:
        boot_dir = replay_root / "boot_000"
        boot_dir.mkdir(parents=True)
        (boot_dir / "Game.slp").write_text("accepted")
        return {0: ["accepted"]}

    monkeypatch.setattr(harness, "_drive_wave", drive_wave)

    assert harness.run_matches_vec(cfg, [Mock()], Mock(), max_frames=100, max_parallel=1) == [["accepted"]]
    assert (replay_root / "boot_000/Game.slp").read_text() == "accepted"
    assert not replay_root.with_name("replays_failed_attempts").exists()


def test_final_failed_attempt_replay_is_preserved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    replay_root = tmp_path / "replays"
    cfg = SessionConfig(iso_path="iso", dolphin_path="dolphin", replay_dir=replay_root)

    def drive_wave(
        _cfg: object, _indices: object, _matches: object, _policy_factory: object, **_kwargs: object
    ) -> dict[int, list[str]]:
        boot_dir = replay_root / "boot_000"
        boot_dir.mkdir(parents=True)
        (boot_dir / "Game.slp").write_text("incomplete")
        return {0: []}

    monkeypatch.setattr(harness, "_drive_wave", drive_wave)

    assert harness.run_matches_vec(cfg, [Mock()], Mock(), max_frames=100, max_parallel=1, start_retries=0) == [[]]
    assert not list(replay_root.rglob("*.slp"))
    assert (replay_root.with_name("replays_failed_attempts") / "boot_000/attempt_1/Game.slp").read_text() == (
        "incomplete"
    )


def test_existing_failed_attempt_replay_is_not_overwritten(tmp_path: Path) -> None:
    replay_root = tmp_path / "replays"
    boot_dir = replay_root / "boot_000"
    boot_dir.mkdir(parents=True)
    replay = boot_dir / "Game.slp"
    replay.write_text("new evidence")
    destination = replay_root.with_name("replays_failed_attempts") / "boot_000/attempt_1/Game.slp"
    destination.parent.mkdir(parents=True)
    destination.write_text("prior evidence")

    with pytest.raises(FileExistsError, match="failed-attempt replay already exists"):
        harness._preserve_failed_replays(replay_root, (0,), 1)

    assert replay.read_text() == "new evidence"
    assert destination.read_text() == "prior evidence"
