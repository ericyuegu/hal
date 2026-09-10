"""Administrative commands for the netplay service."""

import argparse

from hal.netplay_service.replays import ensure_replay_lifecycle


def main() -> None:
    parser = argparse.ArgumentParser(prog="hal-netplay-admin")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("install-replay-lifecycle", help="install the private replay 30-day lifecycle rule")
    args = parser.parse_args()
    if args.command == "install-replay-lifecycle":
        ensure_replay_lifecycle()
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
