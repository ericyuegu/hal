"""CLI: emit Slippilab viewer URLs for local or R2 `.slp` files.

Slippilab's Vite dev server serves files from its `public/` dir. Replay-only
links mirror the source path so repeated basenames stay distinct. Replay and
advantage pairs use a short bundle ID and fixed `match.slp` and
`advantage.json` names.

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
import hashlib
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
BUNDLE_MOUNT = "b"
BUNDLE_ID_LENGTH = 16
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


def _bundle_id(slp: Path, sidecar: Path) -> str:
    identity = f"{slp.resolve()}\0{sidecar.resolve()}".encode()
    return hashlib.sha256(identity).hexdigest()[:BUNDLE_ID_LENGTH]


def _stage_bundle(slp: Path, sidecar: Path) -> str:
    if sidecar.suffix.lower() != ".json" or not sidecar.is_file():
        raise SystemExit(f"advantage sidecar does not exist or is not JSON: {sidecar}")
    bundle_id = _bundle_id(slp, sidecar)
    bundle = SERVE_DIR / BUNDLE_MOUNT / bundle_id
    staged_replay = bundle / "match.slp"
    staged_sidecar = bundle / "advantage.json"

    expected = ((staged_replay, slp.resolve()), (staged_sidecar, sidecar.resolve()))
    for staged, source in expected:
        if staged.is_symlink() and (not staged.exists() or staged.resolve() != source):
            staged.unlink()
        elif staged.exists() and (not staged.is_symlink() or staged.resolve() != source):
            raise SystemExit(f"bundle ID collision at {staged}")
        if not staged.exists():
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.symlink_to(source)
    return bundle_id


def _link(slp: Path, *, advantage: Path | None = None, slippilab_url: str | None = None) -> str:
    """Stage one `.slp` under the served dir; return its viewer URL."""
    base_url = SLIPPILAB_URL if slippilab_url is None else slippilab_url.rstrip("/")
    if advantage is not None:
        if not is_finalized(slp):
            raise SystemExit("advantage bundles require a finalized Slippi replay")
        bundle_id = _stage_bundle(slp, advantage)
        return f"{base_url}/?{urllib.parse.urlencode({'bundle': bundle_id})}"

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
    replay_url = f"{base_url}/{SERVE_MOUNT}/{urllib.parse.quote(served)}"
    return _viewer_link(replay_url, slippilab_url=base_url)


def _viewer_link(
    replay_url: str,
    *,
    advantage_url: str | None = None,
    slippilab_url: str | None = None,
) -> str:
    base_url = SLIPPILAB_URL if slippilab_url is None else slippilab_url.rstrip("/")
    query = {"replayUrl": replay_url}
    if advantage_url is not None:
        query["advantageUrl"] = advantage_url
    return f"{base_url}/?{urllib.parse.urlencode(query, safe=':/')}"


def link_replay(slp: Path, *, advantage: Path | None = None, slippilab_url: str | None = None) -> str:
    """Stage one local replay and optional value sidecar, then return a viewer URL."""
    _ensure_mount()
    return _link(slp, advantage=advantage, slippilab_url=slippilab_url)


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
    advantage: str | None = None,
    slippilab_url: str = SLIPPILAB_URL,
) -> None:
    """Print slippilab URLs for local paths/directories or ``r2:bucket/prefix`` inputs."""
    if not MIN_EXPIRES_IN <= expires_in <= MAX_EXPIRES_IN:
        raise SystemExit(f"--expires-in must be between {MIN_EXPIRES_IN} and {MAX_EXPIRES_IN} seconds")

    local_paths = [Path(value) for value in paths if not value.startswith(R2_SCHEME)]
    remote_locators = [_parse_r2(value) for value in paths if value.startswith(R2_SCHEME)]
    if advantage is not None and remote_locators:
        raise SystemExit("--advantage requires exactly one local replay")
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
    if advantage is not None and len(slps) != 1:
        raise SystemExit("--advantage requires exactly one local replay")

    if slps:
        _ensure_mount()
    for slp in slps:
        print(f"{slp.relative_to(Path(REPO_DIR)) if slp.is_relative_to(Path(REPO_DIR)) else slp}")
        sidecar = None if advantage is None else Path(advantage)
        print(f"  {_link(slp, advantage=sidecar, slippilab_url=slippilab_url)}")

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
