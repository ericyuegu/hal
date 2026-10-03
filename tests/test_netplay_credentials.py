import os
import sys
from pathlib import Path

import pytest

from hal.netplay_service.credentials import CredentialStore
from hal.netplay_service.credentials import admin_environment


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CredentialStore:
    executable = tmp_path / "systemd-creds"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import base64, sys\n"
        "name = sys.argv[2].removeprefix('--name=')\n"
        "mode = sys.argv[4]\n"
        "data = sys.stdin.buffer.read()\n"
        "assert sys.argv[1] == '--user' and sys.argv[5:] == ['-', '-']\n"
        "prefix = b'ciphertext:' + name.encode() + b':'\n"
        "if mode == 'encrypt':\n"
        "    assert sys.argv[3] == '--with-key=host'\n"
        "    sys.stdout.buffer.write(prefix + base64.b64encode(data))\n"
        "else:\n"
        "    assert mode == 'decrypt' and sys.argv[3] == '--refuse-null'\n"
        "    if not data.startswith(prefix):\n"
        "        sys.stderr.write('private credential material')\n"
        "        sys.exit(23)\n"
        "    sys.stdout.buffer.write(base64.b64decode(data[len(prefix):]))\n"
    )
    executable.chmod(0o700)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    return CredentialStore(tmp_path / "credentials")


def test_roundtrip_uses_private_files_and_atomic_replacement(store: CredentialStore) -> None:
    store.save("admin-token", b"first secret")
    assert store.load("admin-token") == b"first secret"
    store.save("admin-token", b"replacement")
    assert store.load("admin-token") == b"replacement"
    assert store.directory.stat().st_mode & 0o777 == 0o700
    path = store.directory / "admin-token.cred"
    assert path.stat().st_mode & 0o777 == 0o600
    assert b"replacement" not in path.read_bytes()
    assert list(store.directory.iterdir()) == [path]


def test_rejects_unsafe_files_directories_and_names(store: CredentialStore, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid credential name"):
        store.save("../escape", b"secret")
    store.save("admin-token", b"secret")
    path = store.directory / "admin-token.cred"
    path.chmod(0o644)
    with pytest.raises(RuntimeError, match="file owner or permissions"):
        store.load("admin-token")
    path.chmod(0o600)
    store.directory.chmod(0o755)
    with pytest.raises(RuntimeError, match="directory owner or permissions"):
        store.load("admin-token")
    store.directory.chmod(0o700)
    link = tmp_path / "symlink"
    link.symlink_to(store.directory, target_is_directory=True)
    with pytest.raises(RuntimeError, match="directory owner or permissions"):
        CredentialStore(link).load("admin-token")
    path.unlink()
    path.symlink_to(tmp_path / "systemd-creds")
    with pytest.raises(OSError):
        store.load("admin-token")


def test_decrypt_failure_does_not_expose_command_output(store: CredentialStore) -> None:
    store.save("admin-token", b"secret")
    (store.directory / "admin-token.cred").write_bytes(b"invalid")
    with pytest.raises(RuntimeError) as error:
        store.load("admin-token")
    assert str(error.value) == "systemd-creds decrypt failed with exit 23"


def test_admin_environment_preserves_existing_store_contract(store: CredentialStore) -> None:
    store.save("runner-env", b"HAL_NETPLAY_API_URL=https://20xx.xyz\nAWS_BUCKET=hal\n")
    store.save("cloudflare-access", b"CF_ACCESS_CLIENT_ID=test-id\nCF_ACCESS_CLIENT_SECRET=test-secret\n")
    store.save("admin-token", b"test-token\n")
    base = {"PATH": "/usr/bin"}
    values = admin_environment(store, base)
    assert values["HAL_NETPLAY_ADMIN_TOKEN"] == "test-token"
    assert values["HAL_NETPLAY_ADMIN_ACCESS_CLIENT_ID"] == "test-id"
    assert values["HAL_NETPLAY_ADMIN_ACCESS_CLIENT_SECRET"] == "test-secret"
    assert values["AWS_BUCKET"] == "hal"
    assert base == {"PATH": "/usr/bin"}
    store.save("runner-env", b"HAL_NETPLAY_API_URL=https://wrong.example\n")
    with pytest.raises(ValueError, match="must target"):
        admin_environment(store, base)


def test_duplicate_credential_environment_keys_fail(store: CredentialStore) -> None:
    store.save("runner-env", b"AWS_BUCKET=first\nAWS_BUCKET=second\n")
    with pytest.raises(ValueError, match="unique NAME=value"):
        admin_environment(store, {})
