"""Keep this host's operator credentials encrypted with systemd-creds."""

import hmac
import os
import re
import subprocess
import tempfile
from pathlib import Path

DIRECTORY = Path('/home/ericgu/src/hal/runs/netplay/credentials')


def transform(name: str, operation: str, data: bytes) -> bytes:
    if re.fullmatch(r'[a-z][a-z0-9-]{0,63}', name) is None:
        raise ValueError('Invalid credential name')
    option = '--with-key=host' if operation == 'encrypt' else '--refuse-null'
    result = subprocess.run(['systemd-creds', '--user', '--name=' + name, option, operation, '-', '-'],
        input=data, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(f'systemd-creds {operation} failed with exit {result.returncode}')
    return result.stdout


def save(name: str, data: bytes) -> None:
    encrypted = transform(name, 'encrypt', data)
    if not hmac.compare_digest(transform(name, 'decrypt', encrypted), data):
        raise RuntimeError('Credential encryption round-trip failed')
    DIRECTORY.mkdir(parents=True, exist_ok=True, mode=0o700)
    if DIRECTORY.is_symlink() or DIRECTORY.stat().st_uid != os.getuid():
        raise RuntimeError('Credential directory has the wrong owner')
    DIRECTORY.chmod(0o700)
    with tempfile.NamedTemporaryFile(dir=DIRECTORY, prefix='.' + name + '-', delete=False) as output:
        temporary = Path(output.name)
        try:
            output.write(encrypted)
            output.flush()
            os.fsync(output.fileno())
            temporary.replace(DIRECTORY / (name + '.cred'))
        finally:
            temporary.unlink(missing_ok=True)


def load(name: str) -> bytes:
    if re.fullmatch(r'[a-z][a-z0-9-]{0,63}', name) is None:
        raise ValueError('Invalid credential name')
    path = DIRECTORY / (name + '.cred')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as source:
        stat = os.fstat(source.fileno())
        if stat.st_uid != os.getuid() or stat.st_mode & 0o077:
            raise RuntimeError('Credential file owner or permissions are unsafe')
        return transform(name, 'decrypt', source.read())
