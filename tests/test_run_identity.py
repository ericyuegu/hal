import subprocess
from pathlib import Path

import pytest

from hal.training.runs import source_git_sha


def test_pinned_source_identity_works_outside_a_git_checkout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HAL_GIT_SHA", "a" * 40)
    assert source_git_sha(tmp_path) == "a" * 40


@pytest.mark.parametrize("value", ("", "main", "a" * 39, "A" * 40, "a" * 40 + "\n"))
def test_invalid_pinned_identity_is_rejected(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("HAL_GIT_SHA", value)
    with pytest.raises(ValueError, match="full lowercase commit SHA"):
        source_git_sha()


def test_local_source_identity_uses_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HAL_GIT_SHA", raising=False)
    root = Path(__file__).resolve().parents[1]
    expected = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    assert source_git_sha(root) == expected


def test_missing_git_checkout_is_not_reported_as_a_valid_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("HAL_GIT_SHA", raising=False)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    with pytest.raises(subprocess.CalledProcessError):
        source_git_sha(tmp_path)
