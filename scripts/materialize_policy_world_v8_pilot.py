"""Materialize and audit the bounded policy-world-v8 pilot corpus."""

import json

import tyro

from hal.scripts.scaleup_policy_world_v8 import PolicyWorldV8PilotConfig
from hal.scripts.scaleup_policy_world_v8 import build_policy_world_v8_pilot

if __name__ == "__main__":
    result = build_policy_world_v8_pilot(tyro.cli(PolicyWorldV8PilotConfig))
    print(json.dumps(result, indent=2, sort_keys=True))
