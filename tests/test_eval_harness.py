from hal.eval.harness import SessionConfig
from hal.eval.harness import _session_kwargs


def test_headless_eval_disables_dolphin_audio() -> None:
    cfg = SessionConfig(iso_path="iso", dolphin_path="dolphin")

    kwargs = _session_kwargs(cfg, slippi_port=51441, replay_dir=None)

    assert cfg.disable_audio is True
    assert kwargs["disable_audio"] is True
