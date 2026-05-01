import os
from pathlib import Path
from typing import Final

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SSBM_HOME = Path(os.getenv("HAL_SSBM_HOME", str(Path.home() / "data" / "ssbm")))


def _env_path(var: str, default: Path) -> str:
    return os.getenv(var, str(default))


REPO_DIR: Final[str] = _env_path("HAL_REPO_DIR", _REPO_ROOT)
ISO_PATH: Final[str] = _env_path("HAL_ISO_PATH", _SSBM_HOME / "ssbm.ciso")
EMULATOR_PATH: Final[str] = _env_path("HAL_EMULATOR_PATH", _SSBM_HOME / "squashfs-root" / "AppRun")
EVAL_REPLAY_DIR: Final[str] = _env_path("HAL_REPLAY_DIR", _SSBM_HOME / "replays")
