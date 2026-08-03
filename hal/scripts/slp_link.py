"""CLI: emit slippilab viewer URLs for `.slp` files (e.g. closed-loop replays under `runs/`).

slippilab's vite dev server serves files from its `public/` dir. This stages each
`.slp` into a served directory, mirroring its absolute path so replays that share a
basename (`Game_<timestamp>.slp` across many match dirs) stay distinct, and prints
`<url>/?replayUrl=...`.

Setup once: `cd ~/src/slippilab && npm run dev` (vite, port 5173), and SSH-forward
`-L 5173:localhost:5173`. The served dir is symlinked into slippilab's `public/`.

Usage:
    python -m hal.scripts.slp_link runs/<run>/replays/match_000/Game_*.slp   # globs/files
    python -m hal.scripts.slp_link runs/<run>                                # all .slp under a dir
"""

import urllib.parse
from pathlib import Path
from typing import Annotated

import tyro
from loguru import logger

from hal.data.slp_finalize import finalize_bytes
from hal.data.slp_finalize import is_finalized
from hal.paths import REPO_DIR

# Served dir + how slippilab reaches it. The mount is a symlink under slippilab's
# `public/`; vite then serves staged slps at `<URL>/<MOUNT>/<mirrored path>`. Repo-local
# scratch (gitignored), owned by this CLI — not borrowed from any notebook.
SERVE_DIR = Path(REPO_DIR) / "data" / "scratch" / "slippilab"
SLIPPILAB_PUBLIC = Path("~/src/slippilab/public").expanduser()
SLIPPILAB_URL = "http://localhost:5173"
SERVE_MOUNT = "hal-runs"


def _ensure_mount() -> None:
    SERVE_DIR.mkdir(parents=True, exist_ok=True)
    mount = SLIPPILAB_PUBLIC / SERVE_MOUNT
    if mount.is_symlink():
        if mount.resolve() == SERVE_DIR.resolve():
            return
        mount.unlink()  # points at a stale served dir (repo moved / SERVE_DIR changed)
    elif mount.exists():
        raise SystemExit(f"{mount} exists and is not a symlink to {SERVE_DIR}")
    if not SLIPPILAB_PUBLIC.exists():
        raise SystemExit(f"slippilab public/ not found at {SLIPPILAB_PUBLIC}")
    mount.symlink_to(SERVE_DIR)
    logger.info(f"symlinked {mount} -> {SERVE_DIR}")


def _staged_path(slp: Path) -> Path:
    """Where `slp` is served from: its absolute path, mirrored under the served dir.

    Basenames repeat across match dirs (`Game_<ts>.slp`, `boot_*/<policy>.slp`), so a
    served name must encode the whole source path — and two sources must never map to
    one name, which would silently serve the wrong replay. Mirroring the tree inherits
    uniqueness from the filesystem; flattening to one name cannot (any separator can
    also occur inside a directory name).
    """
    resolved = slp.resolve()
    return SERVE_DIR / resolved.relative_to(resolved.anchor)


def _link(slp: Path) -> str:
    """Stage one `.slp` under the served dir; return its viewer URL."""
    staged = _staged_path(slp)
    if staged.is_symlink() and not staged.exists():
        staged.unlink()  # dangling: the source it was staged from is gone
    elif staged.is_file() and not staged.is_symlink() and is_finalized(slp):
        staged.unlink()  # a repaired copy of what was then a mid-game .slp; the source has since closed
    if not staged.exists():
        staged.parent.mkdir(parents=True, exist_ok=True)
        # A match killed mid-game leaves an unfinalized .slp (rawLength == 0)
        # that slippilab can't parse; stage a finalized copy instead of a
        # symlink so the viewer always works. Finalized files just get symlinked.
        if is_finalized(slp):
            staged.symlink_to(slp.resolve())
        else:
            finalized = finalize_bytes(slp.read_bytes())
            if finalized is None:
                raise SystemExit(f"not a Slippi .slp file: {slp}")
            staged.write_bytes(finalized)
    served = staged.relative_to(SERVE_DIR).as_posix()
    replay_url = f"{SLIPPILAB_URL}/{SERVE_MOUNT}/{urllib.parse.quote(served)}"
    return f"{SLIPPILAB_URL}/?replayUrl={urllib.parse.quote(replay_url, safe=':/')}"


def _collect(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        p = p if p.is_absolute() else Path(REPO_DIR) / p
        if p.is_dir():
            out.extend(sorted(p.rglob("*.slp")))
        elif p.is_file():
            out.append(p)
        else:
            raise SystemExit(f"no such file or directory: {p}")
    return list(dict.fromkeys(out))  # a dir and a file inside it may both be named


def slp_link(paths: Annotated[list[Path], tyro.conf.Positional]) -> None:
    """Print a slippilab URL for each `.slp` (files or dirs to walk)."""
    slps = _collect(paths)
    if not slps:
        raise SystemExit("no .slp files found")
    _ensure_mount()
    for slp in slps:
        print(f"{slp.relative_to(Path(REPO_DIR)) if slp.is_relative_to(Path(REPO_DIR)) else slp}")
        print(f"  {_link(slp)}")


def main() -> None:
    tyro.cli(slp_link)


if __name__ == "__main__":
    main()
