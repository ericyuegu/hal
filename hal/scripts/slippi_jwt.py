"""CLI: mint a fresh Slippi JWT from the Slippi Launcher's stored credentials.

slippi.gg's GraphQL API (`https://internal.slippi.gg/graphql`) authenticates with a
Firebase ID token as `Authorization: Bearer <jwt>`. The Slippi Launcher is an Electron
app, so its Firebase session lives in the renderer's IndexedDB rather than in any
config file, under the key `firebase:authUser:<apiKey>`.

The `accessToken` cached there is an ID token with a one-hour lifetime, so a stored
copy is almost always expired. The durable credential is the sibling `refreshToken`,
which this CLI exchanges for a fresh ID token at Google's secure-token endpoint --
the same exchange the Launcher performs.

The JWT is the only thing written to stdout, so the command pipes cleanly. Its claims
go to stderr.

Log out of the Launcher (or change the account password) and the refresh token is
revoked; re-run after logging back in.

Usage:
    python -m hal.scripts.slippi_jwt                    # print a fresh JWT
    curl -H "Authorization: Bearer $(python -m hal.scripts.slippi_jwt)" ...
"""

import base64
import datetime
import json
import os
import platform
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import tyro
from loguru import logger

# Electron's `app.getPath("userData")` for the Launcher, per platform.
LAUNCHER_DIR_BY_SYSTEM = {
    "Darwin": Path("~/Library/Application Support/Slippi Launcher"),
    "Linux": Path("~/.config/Slippi Launcher"),
    "Windows": Path("~/AppData/Roaming/Slippi Launcher"),
}
LEVELDB_SUBDIR = Path("IndexedDB/file__0.indexeddb.leveldb")
TOKEN_ENDPOINT = "https://securetoken.googleapis.com/v1/token"
EXPECTED_AUDIENCE = "slippi"

# IndexedDB values are V8 structured-clone blobs, not JSON, so the record is scraped
# by key name. `stsTokenManager` serializes `refreshToken` before `accessToken`, so
# the first long token-shaped run after the key is the refresh token. The window is
# wider than any observed refresh token (~204 chars) and narrower than the gap to
# the next record.
AUTH_USER_KEY = re.compile(rb"firebase:authUser:([A-Za-z0-9_\-]+)")
REFRESH_TOKEN_KEY = re.compile(rb"refreshToken")
TOKEN_RUN = re.compile(rb"[A-Za-z0-9_\-]{80,}")
SCAN_WINDOW = 400


@dataclass(frozen=True, slots=True)
class Credentials:
    """The Launcher's stored Firebase session: the project it belongs to, and the
    long-lived token that mints ID tokens for it."""

    api_key: str
    refresh_token: str


def leveldb_dir() -> Path:
    """Locate the Launcher's IndexedDB store. `HAL_SLIPPI_LAUNCHER_DIR` overrides."""
    override = os.getenv("HAL_SLIPPI_LAUNCHER_DIR")
    if override:
        launcher = Path(override).expanduser()
    else:
        system = platform.system()
        if system not in LAUNCHER_DIR_BY_SYSTEM:
            raise SystemExit(
                f"no known Slippi Launcher location for platform {system!r}; "
                "set HAL_SLIPPI_LAUNCHER_DIR to the Launcher's userData directory"
            )
        launcher = LAUNCHER_DIR_BY_SYSTEM[system].expanduser()

    store = launcher / LEVELDB_SUBDIR
    if not store.is_dir():
        raise SystemExit(f"Slippi Launcher IndexedDB not found at {store}; is the Launcher installed?")
    return store


def read_credentials(store: Path) -> Credentials:
    """Scrape the newest stored session out of the leveldb files.

    Chrome journals recent writes to `*.log` and compacts older ones into `*.ldb`, and
    a re-login appends a new record rather than replacing the old one. Reading files
    oldest-first and taking the last match in each therefore lands on the newest
    session; an older, revoked token would be rejected by the endpoint.
    """
    files = sorted(store.glob("*.log"), key=lambda p: p.stat().st_mtime)
    files += sorted(store.glob("*.ldb"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise SystemExit(f"no leveldb files under {store}")

    api_key: str | None = None
    refresh_token: str | None = None
    for path in files:
        data = path.read_bytes()
        for match in AUTH_USER_KEY.finditer(data):
            api_key = match.group(1).decode()
        for match in REFRESH_TOKEN_KEY.finditer(data):
            run = TOKEN_RUN.search(data, match.end(), match.end() + SCAN_WINDOW)
            if run is not None:
                refresh_token = run.group().decode()

    if api_key is None or refresh_token is None:
        raise SystemExit(
            f"no Slippi credentials in {store}; log in with the Slippi Launcher first. "
            "Close the Launcher before re-running so it cannot rotate the token mid-read."
        )
    return Credentials(api_key=api_key, refresh_token=refresh_token)


def mint(credentials: Credentials) -> str:
    """Exchange the refresh token for a fresh Firebase ID token."""
    body = urllib.parse.urlencode({"grant_type": "refresh_token", "refresh_token": credentials.refresh_token}).encode()
    url = f"{TOKEN_ENDPOINT}?{urllib.parse.urlencode({'key': credentials.api_key})}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=body)) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise SystemExit(
            f"token refresh rejected ({exc.code}): {detail}. "
            "The stored refresh token is revoked on logout or password change; "
            "log in again with the Slippi Launcher."
        ) from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"cannot reach {TOKEN_ENDPOINT}: {exc.reason}") from exc
    return payload["id_token"]


def claims(jwt: str) -> dict:
    """Decode the JWT payload. Unverified -- for reporting, never for authorization."""
    encoded = jwt.split(".")[1]
    return json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))


def slippi_jwt() -> None:
    """Print a fresh Slippi JWT to stdout; log its claims to stderr."""
    jwt = mint(read_credentials(leveldb_dir()))
    decoded = claims(jwt)
    if decoded.get("aud") != EXPECTED_AUDIENCE:
        raise SystemExit(f"minted token has audience {decoded.get('aud')!r}, expected {EXPECTED_AUDIENCE!r}")

    expires = datetime.datetime.fromtimestamp(decoded["exp"], datetime.UTC)
    user = decoded.get("https://slippi.gg/jwt/claims", {}).get("USER")
    logger.info(f"minted JWT for {decoded.get('email')} (slippi user {user}), expires {expires.isoformat()}")
    print(jwt)


def main() -> None:
    tyro.cli(slippi_jwt)


if __name__ == "__main__":
    main()
