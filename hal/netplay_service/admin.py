"""Administrative commands for the netplay service."""

import argparse
import json
import os
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from typing import Final

from hal import r2
from hal.data.schema import Rank
from hal.inference.action_sequence_artifact import read_action_sequence_artifact
from hal.netplay_service.assets import account_key
from hal.netplay_service.assets import ensure_uploaded
from hal.netplay_service.assets import policy_bundle_key
from hal.netplay_service.assets import sha256_file
from hal.netplay_service.domain import CHARACTERS
from hal.netplay_service.domain import IMITATIONS
from hal.netplay_service.domain import STAGES
from hal.netplay_service.domain import Choice
from hal.netplay_service.domain import PolicyConfig
from hal.netplay_service.domain import account_connect_code
from hal.netplay_service.queue_client import Account
from hal.netplay_service.queue_client import AdminClient
from hal.netplay_service.queue_client import admin_endpoint
from hal.netplay_service.replays import ensure_replay_lifecycle
from hal.representation.player_identity import RANK_PLAYER_IDS

_SINCE: Final[re.Pattern[str]] = re.compile(r"([0-9]+)([smhd])")
_UNIT_SECONDS: Final[dict[str, int]] = {"s": 1, "m": 60, "h": 3600, "d": 86400}
# The policy resolves these imitations without the bundle's connect-code vocabulary.
_VOCABULARY_FREE_IMITATIONS: Final[frozenset[str]] = frozenset(
    {"MASKED", *(Rank(player_id).name for player_id in RANK_PLAYER_IDS)}
)


def check_imitations(imitations: tuple[Choice, ...], vocabulary_codes: tuple[str, ...]) -> None:
    """Reject imitations that the policy would fail to resolve on the runner."""
    known = frozenset(vocabulary_codes)
    missing = sorted(
        choice.value
        for choice in imitations
        if choice.value not in _VOCABULARY_FREE_IMITATIONS and choice.value not in known
    )
    if missing:
        raise ValueError(f"the bundle vocabulary lacks imitation codes: {', '.join(missing)}")


def policy_config(
    bundle_sha256: str, vocabulary_sha256: str, capability_version: int, supported_delays: tuple[int, ...]
) -> PolicyConfig:
    delays = tuple(delay for delay in (2, 3) if delay in supported_delays)
    if capability_version < 2 or delays != (2, 3):
        raise ValueError("netplay needs a capability-v2 bundle that supports delays 2 and 3")
    # The roster stays the current imitation list until the page redesign adds its own.
    return PolicyConfig(
        bundle_sha256=bundle_sha256,
        bundle_r2_key=policy_bundle_key(bundle_sha256),
        vocabulary_sha256=vocabulary_sha256,
        characters=CHARACTERS,
        imitations=IMITATIONS,
        stages=STAGES,
        online_delays=delays,
        desired_return_range=(0.0, 40.0),
        default_desired_return=20.0,
        temperature_range=(0.8, 1.1),
        default_temperature=1.0,
        masked_identity=False,
    )


def policy_config_for(bundle: Path) -> PolicyConfig:
    """Validate every bundle member and derive the published config from it."""
    artifact = read_action_sequence_artifact(bundle)
    config = policy_config(
        sha256_file(bundle),
        artifact.vocabulary.sha256,
        artifact.capability_version,
        artifact.spec.supported_transport_delays,
    )
    check_imitations(config.imitations, artifact.vocabulary.codes)
    return config


def publish_policy(bundle: Path, config: PolicyConfig, admin: AdminClient, remote: Any, bucket: str) -> bool:
    uploaded = ensure_uploaded(remote, bucket, bundle, config.bundle_r2_key, config.bundle_sha256)
    admin.put_policy(config)
    return uploaded


def upload_accounts(paths: Sequence[Path], admin: AdminClient, remote: Any, bucket: str) -> tuple[Account, ...]:
    if not paths:
        raise ValueError("accounts upload needs at least one account")
    codes = [account_connect_code(path) for path in paths]
    for code in codes:
        if codes.count(code) > 1:
            raise ValueError(f"connect code {code} appears twice")
    accounts: list[Account] = []
    for path, code in zip(paths, codes, strict=True):
        digest = sha256_file(path)
        key = account_key(digest)
        ensure_uploaded(remote, bucket, path, key, digest)
        accounts.append(Account(code, key, digest))
    uploaded = tuple(accounts)
    admin.put_accounts(uploaded)
    return uploaded


def parse_since(value: str, now: float) -> float:
    match = _SINCE.fullmatch(value)
    if match is None:
        raise ValueError("--since must look like 30m, 1h, or 2d")
    return now - int(match[1]) * _UNIT_SECONDS[match[2]]


def _admin_client() -> AdminClient:
    return AdminClient(admin_endpoint(os.environ))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="hal-netplay-admin")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("install-replay-lifecycle", help="install the private replay 30-day lifecycle rule")
    publish = commands.add_parser("publish-policy", help="upload a bundle and make it the active policy")
    publish.add_argument("bundle", type=Path)
    accounts = commands.add_parser("accounts", help="manage bot Slippi accounts")
    account_commands = accounts.add_subparsers(dest="account_command", required=True)
    upload = account_commands.add_parser("upload", help="upload user.json files and replace the account list")
    upload.add_argument("paths", type=Path, nargs="+")
    commands.add_parser("status", help="print sessions, accounts, capacity, and the active policy")
    events = commands.add_parser("events", help="print the event timeline as JSON lines")
    scope = events.add_mutually_exclusive_group()
    scope.add_argument("--job")
    scope.add_argument("--session")
    events.add_argument("--since", help="for example 30m, 1h, or 2d")
    commands.add_parser("pause", help="stop accepting new reservations")
    commands.add_parser("resume", help="accept new reservations again")
    args = parser.parse_args(argv)

    if args.command == "install-replay-lifecycle":
        ensure_replay_lifecycle()
        return
    admin = _admin_client()
    try:
        if args.command == "publish-policy":
            config = policy_config_for(args.bundle)
            remote = r2.client()
            try:
                publish_policy(args.bundle, config, admin, remote, r2.bucket())
            finally:
                remote.close()
            print(json.dumps(config.to_payload(), indent=2, sort_keys=True))
        elif args.command == "accounts":
            remote = r2.client()
            try:
                uploaded = upload_accounts(args.paths, admin, remote, r2.bucket())
            finally:
                remote.close()
            for account in uploaded:
                print(f"{account.connect_code} {account.r2_key}")
        elif args.command == "status":
            print(json.dumps(admin.status(), indent=2, sort_keys=True))
        elif args.command == "events":
            since = None if args.since is None else parse_since(args.since, time.time())
            for event in admin.events(job=args.job, session=args.session, since=since):
                print(json.dumps(event, sort_keys=True))
        elif args.command in ("pause", "resume"):
            admin.set_paused(args.command == "pause")
        else:
            raise AssertionError(args.command)
    finally:
        admin.close()


if __name__ == "__main__":
    main()
