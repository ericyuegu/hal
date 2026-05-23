"""Surface freshly-recorded .slps as slippilab URLs.

Dolphin writes one .slp per match into ``Session(replay_dir=...)``. The
first one has a guaranteed GameEnd footer (Instant Match auto-restart
truncates subsequent .slps when the Session shuts down); peppi refuses
to parse the truncated ones, so eval flows always pick the first.

These helpers don't move files; the experiment can rename + symlink
into a slippilab ``public/`` mount as it sees fit.
"""

import urllib.parse
from pathlib import Path


def first_new_slp(replay_dir: Path, before: set[str], match_start_mtime: float) -> Path | None:
    """The first .slp Dolphin wrote during this Session — the only one
    guaranteed to have a GameEnd footer. ``before`` is the set of .slp
    filenames present before the match started; ``match_start_mtime`` is
    a wall-clock guard for racy filesystems."""
    new = [p for p in replay_dir.glob("*.slp") if p.name not in before and p.stat().st_mtime >= match_start_mtime]
    if not new:
        return None
    new.sort(key=lambda p: p.stat().st_mtime)
    return new[0]


def slippilab_url(slippilab_base: str, serve_mount: str, slp_path: Path) -> str:
    """Build a slippilab viewer URL for an .slp served at
    ``{slippilab_base}/{serve_mount}/{slp_path.name}``."""
    replay_url = f"{slippilab_base}/{serve_mount}/{slp_path.name}"
    return f"{slippilab_base}/?replayUrl={urllib.parse.quote(replay_url, safe=':/')}"
