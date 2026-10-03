"""Run the maintained admin CLI with the existing encrypted operator store."""

import argparse
import os
import subprocess
from pathlib import Path

from hal.netplay_service.credentials import CredentialStore
from hal.netplay_service.credentials import admin_environment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credentials", required=True, type=Path, help="existing systemd-creds directory")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="hal-netplay-admin command and arguments")
    args = parser.parse_args()
    if not args.command:
        parser.error("an admin command is required")
    environment = admin_environment(CredentialStore(args.credentials), os.environ)
    result = subprocess.run(["hal-netplay-admin", *args.command], env=environment, check=False)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
