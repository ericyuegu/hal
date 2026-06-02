"""CLI: emit slippilab viewer URLs for `.slp` files (e.g. closed-loop replays under `runs/`).

slippilab's vite dev server serves files from its `public/` dir. This stages each
`.slp` into a served directory under a collision-free name (runs replays are all
`Game_<timestamp>.slp` across many match dirs) and prints `<url>/?replayUrl=...`.

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
# `public/`; vite then serves staged slps at `<URL>/<MOUNT>/<name>`. Repo-local
# scratch (gitignored), owned by this CLI — not borrowed from any notebook.
SERVE_DIR = Path(REPO_DIR) / "data" / "scratch" / "slippilab"
SLIPPILAB_PUBLIC = Path("~/src/slippilab/public").expanduser()
SLIPPILAB_URL = "http://localhost:5173"
SERVE_MOUNT = "hal-runs"


def _ensure_mount() -> None:
    SERVE_DIR.mkdir(parents=True, exist_ok=True)
    mount = SLIPPILAB_PUBLIC / SERVE_MOUNT
    if mount.exists():
        return
    if not SLIPPILAB_PUBLIC.exists():
        raise SystemExit(f"slippilab public/ not found at {SLIPPILAB_PUBLIC}")
    mount.symlink_to(SERVE_DIR)
    logger.info(f"symlinked {mount} -> {SERVE_DIR}")


def _link(slp: Path) -> str:
    """Stage one `.slp` into the served dir under a collision-free name; return its URL."""
    try:
        rel = slp.resolve().relative_to(Path(REPO_DIR).resolve())
        name = "__".join(rel.parts)
    except ValueError:
        name = slp.name
    staged = SERVE_DIR / name
    if not staged.exists():
        # A match killed mid-game leaves an unfinalized .slp (rawLength == 0)
        # that slippilab can't parse; stage a finalized copy instead of a
        # symlink so the viewer always works. Finalized files just get symlinked.
        if is_finalized(slp):
            staged.symlink_to(slp.resolve())
        else:
            staged.write_bytes(finalize_bytes(slp.read_bytes()))
    replay_url = f"{SLIPPILAB_URL}/{SERVE_MOUNT}/{staged.name}"
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
    return out


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
