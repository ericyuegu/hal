"""Administrative commands for the netplay service."""

import argparse
import os
from pathlib import Path

from hal.netplay_service.queue import QueueStore
from hal.netplay_service.replays import ensure_replay_lifecycle


def main() -> None:
    parser = argparse.ArgumentParser(prog="hal-netplay-admin")
    commands = parser.add_subparsers(dest="command", required=True)
    invite = commands.add_parser("invite", help="create a player-bound invite code")
    invite.add_argument("label")
    invite.add_argument(
        "--database",
        type=Path,
        default=Path(os.environ.get("HAL_NETPLAY_DATABASE", "runs/netplay/queue.sqlite3")),
    )
    commands.add_parser("install-replay-lifecycle", help="install the private replay 30-day lifecycle rule")
    args = parser.parse_args()
    if args.command == "invite":
        print(QueueStore(args.database).create_invite(args.label))
    elif args.command == "install-replay-lifecycle":
        ensure_replay_lifecycle()
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
