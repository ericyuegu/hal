"""Use the existing encrypted credentials for this reviewed deployment."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "x-pilot-master120/control"))
import hal_credentials


def environment():
    values = os.environ.copy()
    for line in hal_credentials.load("runner-env").decode().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    access = dict(line.split("=", 1) for line in hal_credentials.load("cloudflare-access").decode().splitlines() if "=" in line)
    values["HAL_NETPLAY_ADMIN_TOKEN"] = hal_credentials.load("admin-token").decode().strip()
    values["HAL_NETPLAY_ADMIN_ACCESS_CLIENT_ID"] = access["CF_ACCESS_CLIENT_ID"]
    values["HAL_NETPLAY_ADMIN_ACCESS_CLIENT_SECRET"] = access["CF_ACCESS_CLIENT_SECRET"]
    assert values["HAL_NETPLAY_API_URL"].rstrip("/") == "https://20xx.xyz"
    return values


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("status", "publish-policy", "resume"))
    args = parser.parse_args()
    values = environment()
    if args.operation == "publish-policy":
        from hal.netplay_service.queue_client import AdminClient, admin_endpoint
        client = AdminClient(admin_endpoint(values))
        try:
            current = client.status()["policy"]
        finally:
            client.close()
        candidate = json.loads(Path(__file__).with_name("policy.json").read_text())
        changes = {key for key in set(current) | set(candidate) if current.get(key) != candidate.get(key)}
        if changes - {"imitations"}:
            raise RuntimeError("Publication would change more than the player list: " + str(sorted(changes)))
        print("Policy changes:", sorted(changes), flush=True)
    command = ["uv", "run", "hal-netplay-admin", args.operation]
    if args.operation == "publish-policy":
        command.append("/tmp/hal-o59-production.halpolicy")
    subprocess.run(command, env=values, check=True)
