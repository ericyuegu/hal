import importlib.util
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).parents[1] / "experiments" / "052_policy_world_v8_loader.py"
_SPEC = importlib.util.spec_from_file_location("test_exp052", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_git_sha_uses_modal_source_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    sha = "a" * 40
    monkeypatch.setenv("HAL_GIT_SHA", sha)
    monkeypatch.setattr(
        _MODULE.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("git must not run when HAL_GIT_SHA is set"),
    )

    assert _MODULE._git_sha() == sha


def test_git_sha_rejects_invalid_modal_source_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HAL_GIT_SHA", "not-a-sha")

    with pytest.raises(ValueError, match="40-character lowercase hexadecimal"):
        _MODULE._git_sha()
