import os
from pathlib import Path
from typing import Final

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_HOME = Path(os.getenv("HAL_DATA_HOME", str(Path.home() / "data")))


def _env_path(var: str, default: Path) -> str:
    return os.getenv(var, str(default))


REPO_DIR: Final[str] = _env_path("HAL_REPO_DIR", _REPO_ROOT)
ISO_PATH: Final[str] = _env_path("HAL_ISO_PATH", _DATA_HOME / "dolphin" / "ssbm.ciso")
EMULATOR_PATH: Final[str] = _env_path(
    "HAL_EMULATOR_PATH", _DATA_HOME / "dolphin" / "exiai" / "squashfs-root" / "AppRun"
)
EVAL_REPLAY_DIR: Final[str] = _env_path("HAL_REPLAY_DIR", _DATA_HOME / "scratch" / "replays")
# Small .7z replay archive used as the integration-test fixture. Skip cleanly
# when absent so the test suite runs on a fresh checkout.
DEV_ARCHIVE_PATH: Final[str] = _env_path("HAL_DEV_ARCHIVE", _DATA_HOME / "raw" / "dev.7z")
# MDS shards produced from the dev archive; used by the round-trip integration test.
DEV_MDS_DIR: Final[str] = _env_path("HAL_DEV_MDS_DIR", _DATA_HOME / "processed" / "dev" / "mds")
