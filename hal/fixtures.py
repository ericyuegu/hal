"""Cloud-backed integration fixtures.

Each fixture has a sha256 of the downloaded blob and a local destination
under `<repo>/data/`. `ensure(fix)` downloads on first use and verifies the
sha256; subsequent calls no-op when the local file (or extracted tree)
already matches.

Two download backends, picked per-fixture:
- Private artifacts (slp data, MDS, ISO) live in a Cloudflare R2 bucket.
  Credentials come from env vars `AWS_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`,
  `AWS_SECRET_ACCESS_KEY`, `AWS_BUCKET` — see `.env.example`. The bucket is
  private; the ISO is Nintendo-copyrighted and distribution is legally
  fraught even among collaborators. Hand out creds out-of-band only.
- Public upstream artifacts (Dolphin AppImage) are fetched straight from
  GitHub releases — pinned by tag, verified by sha256. Don't re-host
  upstream binaries; if a release is ever yanked, drop a one-time mirror
  into R2 then.
"""

import hashlib
import os
import shutil
import subprocess
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import IO
from typing import Final
from typing import Literal

import boto3
from loguru import logger
from tqdm import tqdm

from hal.paths import REPO_DIR

DATA_DIR: Final[Path] = Path(REPO_DIR) / "data"
_CACHE_DIR: Final[Path] = DATA_DIR / ".cache"
_SENTINEL: Final[str] = ".sha256"
_TODO_SHA: Final[str] = "TODO_FILL_IN_SHA256"
_CHUNK: Final[int] = 1 << 20  # 1 MiB

Extract = Literal["tar_zst", "appimage"]


@dataclass(frozen=True, slots=True)
class Fixture:
    name: str
    sha256: str
    size_bytes: int
    dest: Path
    r2_key: str | None = None
    url: str | None = None
    extract: Extract | None = None

    def __post_init__(self) -> None:
        if (self.r2_key is None) == (self.url is None):
            raise ValueError(f"{self.name}: exactly one of r2_key or url must be set")


DEV_ARCHIVE: Final[Fixture] = Fixture(
    name="dev.7z",
    r2_key="fixtures/dev.7z",
    sha256="6c3a1815bc78f44786b91a366195b0b506e81105f998c15b47c48946488622eb",
    size_bytes=36_818_126,
    dest=Path("data/raw/dev.7z"),
)
DEV_MDS: Final[Fixture] = Fixture(
    name="dev-mds",
    r2_key="fixtures/dev-mds.tar.zst",
    sha256="8a5de5bcf23bfc55bb9b86cab01fd180897b8c249d4fd291941430f7573f5810",
    size_bytes=11_532_643,
    dest=Path("data/processed/dev/mds"),
    extract="tar_zst",
)
ISO: Final[Fixture] = Fixture(
    name="ssbm.ciso",
    r2_key="fixtures/ssbm.ciso",
    sha256="b7de482eb955c8a96b6746dfa043b69ae7bf6c7c2a09ac382b9da126faa7055c",
    size_bytes=1_449_165_376,
    dest=Path("data/emulator/ssbm.ciso"),
)
DOLPHIN_EXIAI: Final[Fixture] = Fixture(
    name="dolphin-exiai",
    url="https://github.com/vladfi1/slippi-Ishiiruka/releases/download/exi-ai-0.2.0/Slippi_Online-x86_64-ExiAI.AppImage",
    sha256="87e9ef6d80ed03354a1647d0616016dbc91399aa9e86a69ae5a398edd0a0c2bd",
    size_bytes=0,
    dest=Path("data/emulator/exiai"),
    extract="appimage",
)

ALL: Final[tuple[Fixture, ...]] = (DEV_ARCHIVE, DEV_MDS, ISO, DOLPHIN_EXIAI)
BY_NAME: Final[dict[str, Fixture]] = {f.name: f for f in ALL}


class FixtureError(RuntimeError):
    pass


def _r2_client():  # type: ignore[no-untyped-def]
    """Lazy boto3 S3 client against R2's S3-compatible endpoint."""
    missing = [v for v in ("AWS_ENDPOINT_URL", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY") if not os.environ.get(v)]
    if missing:
        raise FixtureError(f"missing env vars for R2 fetch: {missing}. See .env.example.")
    return boto3.client(
        "s3",
        endpoint_url=os.environ["AWS_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


def _stream(fout: IO[bytes], reader, total: int, label: str) -> str:  # type: ignore[no-untyped-def]
    """Pipe `reader.read()` chunks into `fout`, return hex sha256.

    `total` is used only for the tqdm bar — pass 0 when unknown.
    """
    h = hashlib.sha256()
    bar = tqdm(total=total or None, unit="B", unit_scale=True, unit_divisor=1024, desc=label)
    try:
        while True:
            chunk = reader.read(_CHUNK)
            if not chunk:
                break
            fout.write(chunk)
            h.update(chunk)
            bar.update(len(chunk))
    finally:
        bar.close()
    return h.hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(fix: Fixture, cache_path: Path) -> str:
    """Download `fix` into `cache_path`, return the observed sha256."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".partial")
    if fix.r2_key is not None:
        client = _r2_client()
        bucket = os.environ.get("AWS_BUCKET")
        if not bucket:
            raise FixtureError("AWS_BUCKET env var not set")
        obj = client.get_object(Bucket=bucket, Key=fix.r2_key)
        with tmp.open("wb") as f:
            digest = _stream(f, obj["Body"], obj.get("ContentLength", fix.size_bytes), fix.name)
    else:
        assert fix.url is not None
        with urllib.request.urlopen(fix.url) as resp, tmp.open("wb") as f:
            total = int(resp.headers.get("Content-Length") or fix.size_bytes or 0)
            digest = _stream(f, resp, total, fix.name)
    tmp.rename(cache_path)
    return digest


def _extract_tar_zst(tarball: Path, dest: Path) -> None:
    """Extract a .tar.zst into `dest`. dest itself is the deepest dir we
    write into; the tarball is expected to contain the contents directly
    (not a leading wrapper dir). We extract a tree and move it into place
    atomically via a sibling staging dir.
    """
    staging = dest.parent / (dest.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    with tarfile.open(tarball, "r:zst") as tar:
        tar.extractall(staging, filter="data")
    if dest.exists():
        shutil.rmtree(dest)
    staging.rename(dest)


def _extract_appimage(appimage: Path, dest: Path) -> None:
    """`<appimage> --appimage-extract` produces `squashfs-root/` in the cwd.
    Final layout: `<dest>/squashfs-root/AppRun`. Survives FUSE-less envs.
    """
    appimage.chmod(0o755)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    subprocess.run([str(appimage.resolve()), "--appimage-extract"], cwd=dest, check=True)


def _written_sha(target: Path, is_dir: bool) -> str | None:
    """Return the sha256 we recorded for `target`, or None if absent."""
    if is_dir:
        sentinel = target / _SENTINEL
        if not sentinel.is_file():
            return None
        return sentinel.read_text().strip()
    if not target.is_file():
        return None
    return _file_sha256(target)


def _record_sha(target: Path, is_dir: bool, digest: str) -> None:
    if is_dir:
        (target / _SENTINEL).write_text(digest + "\n")


def ensure(fix: Fixture) -> Path:
    """Idempotent fetch. Returns the absolute local path of `fix.dest`."""
    if fix.sha256 == _TODO_SHA:
        raise FixtureError(
            f"fixture {fix.name!r} has placeholder sha256 — "
            "fill in `hal/fixtures.py` after uploading to R2 (see plan)."
        )

    target = Path(REPO_DIR) / fix.dest
    is_dir = fix.extract is not None
    existing = _written_sha(target, is_dir=is_dir)
    if existing == fix.sha256:
        logger.info(f"skip {fix.name} (sha match)")
        return target

    if fix.extract is None:
        # `name` already carries its own extension (dev.7z, ssbm.ciso).
        cache_path = _CACHE_DIR / fix.name
    else:
        ext = {"tar_zst": ".tar.zst", "appimage": ".AppImage"}[fix.extract]
        cache_path = _CACHE_DIR / f"{fix.name}{ext}"

    cached_sha = _file_sha256(cache_path) if cache_path.is_file() else None
    if cached_sha != fix.sha256:
        logger.info(f"fetch {fix.name} -> {cache_path}")
        observed = _download(fix, cache_path)
        if observed != fix.sha256:
            cache_path.unlink(missing_ok=True)
            raise FixtureError(f"sha256 mismatch for {fix.name}: expected {fix.sha256}, got {observed}")

    target.parent.mkdir(parents=True, exist_ok=True)
    if fix.extract == "tar_zst":
        _extract_tar_zst(cache_path, target)
        _record_sha(target, is_dir=True, digest=fix.sha256)
    elif fix.extract == "appimage":
        _extract_appimage(cache_path, target)
        _record_sha(target, is_dir=True, digest=fix.sha256)
    else:
        if target.exists():
            target.unlink()
        shutil.copy2(cache_path, target)

    logger.info(f"ready {fix.name} -> {target}")
    return target


def ensure_all() -> None:
    for fix in ALL:
        ensure(fix)
