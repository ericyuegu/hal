"""Upload the SHA-256-pinned ranked-1 supplemental index for Modal workers."""

import json
from dataclasses import dataclass
from pathlib import Path

import tyro

from hal.scripts.scaleup_policy_world_v8 import stage_rank_one_metadata


@dataclass(frozen=True, slots=True)
class Args:
    path: Path = Path("data/processed/ranked-anonymized-1/index.jsonl")
    staging_root: str = "r2:hal/processed/_staging/policy-world-v8"


if __name__ == "__main__":
    args = tyro.cli(Args)
    print(json.dumps(stage_rank_one_metadata(args.path, args.staging_root), indent=2, sort_keys=True))
