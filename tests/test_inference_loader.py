from types import SimpleNamespace

import pytest

from hal.inference import loader
from hal.inference import o50
from hal.inference import o59


def test_o59_defaults_to_kv_cache_and_allows_window_control(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = SimpleNamespace(backend=o59.O59_BACKEND, backend_version=o59.O59_BACKEND_VERSION)
    monkeypatch.setattr(loader, "read_policy_manifest", lambda _path: manifest)
    modes: list[str] = []

    def load(_path: object, **options: object) -> object:
        modes.append(str(options["history_mode"]))
        return object()

    monkeypatch.setattr(o59, "load_o59_policy", load)
    loader.load_policy("policy.hal")
    loader.load_policy("policy.hal", history_mode="window")
    assert modes == ["kv_cache", "window"]


def test_o50_keeps_window_default_and_rejects_kv_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = SimpleNamespace(backend=o50.O50_BACKEND, backend_version=o50.O50_BACKEND_VERSION)
    monkeypatch.setattr(loader, "read_policy_manifest", lambda _path: manifest)
    loaded: list[object] = []
    monkeypatch.setattr(o50, "load_o50_policy", lambda *_args, **_options: loaded.append(object()))
    loader.load_policy("policy.hal")
    assert len(loaded) == 1
    with pytest.raises(ValueError, match="unsupported by backend"):
        loader.load_policy("policy.hal", history_mode="kv_cache")


def test_history_mode_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unsupported policy backend"):
        loader.resolve_history_mode("unknown", "auto")


def test_history_mode_rejects_invalid_value() -> None:
    with pytest.raises(ValueError, match="unsupported history mode"):
        loader.resolve_history_mode(o50.O50_BACKEND, "invalid")  # type: ignore[arg-type]
