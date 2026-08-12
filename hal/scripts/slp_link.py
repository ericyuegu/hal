"""CLI: emit slippilab viewer URLs for local or R2 `.slp` files.

slippilab's vite dev server serves files from its `public/` dir. This stages each
`.slp` into a served directory, mirroring its absolute path so replays that share a
basename (`Game_<timestamp>.slp` across many match dirs) stay distinct, and prints
`<url>/?replayUrl=...`.

R2 inputs use short-lived presigned URLs, so the replay is downloaded directly by
the browser only when its link is opened. The R2 bucket must allow browser GETs
from the slippilab origin via CORS.

Setup once: `cd ~/src/slippilab && npm run dev` (vite, port 5173), and SSH-forward
`-L 5173:localhost:5173`. The served dir is symlinked into slippilab's `public/`.

Usage:
    python -m hal.scripts.slp_link runs/<run>/replays/match_000/Game_*.slp   # globs/files
    python -m hal.scripts.slp_link runs/<run>                                # all .slp under a dir
    python -m hal.scripts.slp_link r2:hal/runs/<run>/                        # all R2 .slps under prefix
"""

import fnmatch
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import tyro
from botocore.exceptions import BotoCoreError
from botocore.exceptions import ClientError
from loguru import logger

from hal import r2
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
R2_SCHEME = "r2:"
DEFAULT_EXPIRES_IN = 3_600
MIN_EXPIRES_IN = 1
MAX_EXPIRES_IN = 604_800


@dataclass(frozen=True, slots=True)
class R2Object:
    bucket: str
    key: str

    def __str__(self) -> str:
        return f"{R2_SCHEME}{self.bucket}/{self.key}"


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
    return _viewer_link(replay_url)


def _viewer_link(replay_url: str) -> str:
    return f"{SLIPPILAB_URL}/?replayUrl={urllib.parse.quote(replay_url, safe=':/')}"


def _parse_r2(value: str) -> R2Object:
    """Parse ``r2:bucket/key-or-prefix`` without normalizing the object key."""
    remote = value.removeprefix(R2_SCHEME)
    bucket, separator, key = remote.partition("/")
    if not separator or not bucket:
        raise SystemExit(f"invalid R2 locator {value!r}; expected r2:<bucket>/<object-or-prefix>")
    return R2Object(bucket=bucket, key=key)


def _collect_r2(locator: R2Object, client) -> list[R2Object]:  # type: ignore[no-untyped-def]
    """Resolve one exact `.slp` key or recursively list a prefix."""
    try:
        if locator.key.endswith(".slp"):
            client.head_object(Bucket=locator.bucket, Key=locator.key)
            return [locator]

        objects: list[R2Object] = []
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=locator.bucket, Prefix=locator.key):
            objects.extend(
                R2Object(locator.bucket, item["Key"])
                for item in page.get("Contents", [])
                if item["Key"].endswith(".slp")
            )
        return sorted(objects, key=lambda obj: obj.key)
    except (BotoCoreError, ClientError) as exc:
        raise SystemExit(f"failed to inspect {locator}: {exc}") from exc


def _cors_origin_matches(pattern: str, origin: str) -> bool:
    return pattern == "*" or fnmatch.fnmatchcase(origin, pattern)


def _validate_r2_cors(bucket: str, client) -> None:  # type: ignore[no-untyped-def]
    """Require a browser-readable GET rule for slippilab's localhost origin."""
    try:
        rules = client.get_bucket_cors(Bucket=bucket).get("CORSRules", [])
    except (BotoCoreError, ClientError) as exc:
        raise SystemExit(
            f"cannot verify CORS for R2 bucket {bucket!r}: {exc}. "
            f"Allow GET from {SLIPPILAB_URL} before using browser links."
        ) from exc

    allowed = any(
        "GET" in {method.upper() for method in rule.get("AllowedMethods", [])}
        and any(_cors_origin_matches(pattern, SLIPPILAB_URL) for pattern in rule.get("AllowedOrigins", []))
        for rule in rules
    )
    if not allowed:
        raise SystemExit(
            f"R2 bucket {bucket!r} does not allow browser GETs from {SLIPPILAB_URL}. "
            "Add a bucket CORS rule with "
            f"AllowedOrigins=[{SLIPPILAB_URL!r}] and AllowedMethods=['GET']."
        )


def _r2_link(obj: R2Object, client, expires_in: int) -> str:  # type: ignore[no-untyped-def]
    replay_url = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": obj.bucket, "Key": obj.key},
        ExpiresIn=expires_in,
    )
    return _viewer_link(replay_url)


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


def slp_link(
    paths: Annotated[list[str], tyro.conf.Positional],
    expires_in: int = DEFAULT_EXPIRES_IN,
) -> None:
    """Print slippilab URLs for local paths/directories or ``r2:bucket/prefix`` inputs."""
    if not MIN_EXPIRES_IN <= expires_in <= MAX_EXPIRES_IN:
        raise SystemExit(f"--expires-in must be between {MIN_EXPIRES_IN} and {MAX_EXPIRES_IN} seconds")

    local_paths = [Path(value) for value in paths if not value.startswith(R2_SCHEME)]
    remote_locators = [_parse_r2(value) for value in paths if value.startswith(R2_SCHEME)]
    slps = _collect(local_paths)

    client = None
    remote_objects: list[R2Object] = []
    if remote_locators:
        try:
            client = r2.client()
        except r2.R2Error as exc:
            raise SystemExit(str(exc)) from exc
        for locator in remote_locators:
            remote_objects.extend(_collect_r2(locator, client))
        remote_objects = list(dict.fromkeys(remote_objects))

    if not slps and not remote_objects:
        raise SystemExit("no .slp files found")

    if slps:
        _ensure_mount()
    for slp in slps:
        print(f"{slp.relative_to(Path(REPO_DIR)) if slp.is_relative_to(Path(REPO_DIR)) else slp}")
        print(f"  {_link(slp)}")

    if remote_objects:
        assert client is not None
        for bucket in dict.fromkeys(obj.bucket for obj in remote_objects):
            _validate_r2_cors(bucket, client)
        for obj in remote_objects:
            print(obj)
            print(f"  {_r2_link(obj, client, expires_in)}")


def main() -> None:
    tyro.cli(slp_link)


if __name__ == "__main__":
    main()
