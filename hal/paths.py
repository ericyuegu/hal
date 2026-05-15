import os
from pathlib import Path
from typing import Final

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _REPO_ROOT / "data"


def _env_path(var: str, default: Path) -> str:
    return os.getenv(var, str(default))


REPO_DIR: Final[str] = str(_REPO_ROOT)
ISO_PATH: Final[str] = _env_path("HAL_ISO_PATH", _DATA_DIR / "emulator" / "ssbm.ciso")
EMULATOR_PATH: Final[str] = _env_path(
    "HAL_EMULATOR_PATH", _DATA_DIR / "emulator" / "exiai" / "squashfs-root" / "AppRun"
)
DEV_ARCHIVE_PATH: Final[str] = _env_path("HAL_DEV_ARCHIVE", _DATA_DIR / "raw" / "dev.7z")
DEV_MDS_DIR: Final[str] = _env_path("HAL_DEV_MDS_DIR", _DATA_DIR / "processed" / "dev" / "mds")


def repo_relative(p: Path | str) -> Path:
    """Return ``p`` relative to ``REPO_DIR`` when it lives under the repo;
    otherwise return its absolute (normalized) path.

    Normalization is done via ``os.path.abspath`` (resolves ``..`` / ``.``
    and CWD-relative inputs) but does NOT follow symlinks. This keeps
    symlinked-in-place fixtures — e.g. ``data/raw/ranked-1.7z`` pointing at
    ``~/data/raw/ranked-1.7z`` — serialized as their in-repo path so shared
    MDS bundles stay portable across collaborator machines.
    """
    abs_p = Path(os.path.abspath(p))
    try:
        return abs_p.relative_to(_REPO_ROOT)
    except ValueError:
        return abs_p
