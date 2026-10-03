"""Host-bound operator credentials, compatible with the September 2026 store."""

import hmac
import os
import re
import stat
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


def _validate_name(name: str) -> None:
    if re.fullmatch(r"[a-z][a-z0-9-]{0,63}", name) is None:
        raise ValueError("invalid credential name")


def _transform(name: str, operation: Literal["encrypt", "decrypt"], data: bytes) -> bytes:
    _validate_name(name)
    option = "--with-key=host" if operation == "encrypt" else "--refuse-null"
    result = subprocess.run(
        ["systemd-creds", "--user", f"--name={name}", option, operation, "-", "-"],
        input=data,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        # Command output can contain credential material. Report only the exit code.
        raise RuntimeError(f"systemd-creds {operation} failed with exit {result.returncode}")
    return result.stdout


@dataclass(frozen=True, slots=True)
class CredentialStore:
    directory: Path

    def _check_directory(self) -> None:
        info = self.directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise RuntimeError("credential directory owner or permissions are unsafe")

    def load(self, name: str) -> bytes:
        _validate_name(name)
        self._check_directory()
        fd = os.open(self.directory / f"{name}.cred", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise RuntimeError("credential file owner or permissions are unsafe")
            return _transform(name, "decrypt", source.read())

    def save(self, name: str, data: bytes) -> None:
        _validate_name(name)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._check_directory()
        encrypted = _transform(name, "encrypt", data)
        if not hmac.compare_digest(_transform(name, "decrypt", encrypted), data):
            raise RuntimeError("credential encryption round-trip failed")
        with tempfile.NamedTemporaryFile(dir=self.directory, prefix=f".{name}-", delete=False) as output:
            temporary = Path(output.name)
            try:
                output.write(encrypted)
                output.flush()
                os.fsync(output.fileno())
                temporary.replace(self.directory / f"{name}.cred")
            finally:
                temporary.unlink(missing_ok=True)


def _environment(data: bytes) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in data.decode().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        name, separator, value = line.partition("=")
        if not separator or re.fullmatch(r"[A-Z_][A-Z_0-9]*", name) is None or name in values:
            raise ValueError("credential environment must contain unique NAME=value lines")
        values[name] = value
    return values


def admin_environment(store: CredentialStore, base: Mapping[str, str]) -> dict[str, str]:
    """Keep decrypted values in the child environment, never arguments or files."""
    values = dict(base) | _environment(store.load("runner-env"))
    access = _environment(store.load("cloudflare-access"))
    token = store.load("admin-token").decode().strip()
    if not token or not access["CF_ACCESS_CLIENT_ID"] or not access["CF_ACCESS_CLIENT_SECRET"]:
        raise ValueError("admin credentials must not be empty")
    if values.get("HAL_NETPLAY_API_URL", "").rstrip("/") != "https://20xx.xyz":
        raise ValueError("stored operator credentials must target https://20xx.xyz")
    values.update(
        HAL_NETPLAY_ADMIN_TOKEN=token,
        HAL_NETPLAY_ADMIN_ACCESS_CLIENT_ID=access["CF_ACCESS_CLIENT_ID"],
        HAL_NETPLAY_ADMIN_ACCESS_CLIENT_SECRET=access["CF_ACCESS_CLIENT_SECRET"],
    )
    return values
