"""Digest the ordered 44-source 059 sufficient-stat mixture.

Run with ``--control`` and the control checkout on PYTHONPATH to use the old
owner; omit it for the refactored owner. Local published stats are unchanged.
"""

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

from hal import streams


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="checkout containing the local published data cache")
    parser.add_argument("--control", action="store_true")
    args = parser.parse_args()
    if args.control:
        from hal.training.ego_stats import load_consolidated_mixture_stats
    else:
        from hal.data.feature_stats import load_consolidated_mixture_stats

    sources = streams.POLICY_WORLD_V8_SOURCES
    paths = [args.repo / source.local / "stats.json" for source in sources]
    weights = [float(streams.POLICY_WORLD_V8_TRAIN_REPLAYS[source.name]) for source in sources]
    stats = load_consolidated_mixture_stats(paths, weights, expected_mds_schema_version=7)
    payload = json.dumps(
        {name: dataclasses.asdict(value) for name, value in sorted(stats.items())},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    print(json.dumps({"count": len(stats), "sha256": hashlib.sha256(payload).hexdigest()}))


if __name__ == "__main__":
    main()
