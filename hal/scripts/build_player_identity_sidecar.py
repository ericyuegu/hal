"""Build the O49 professional connect-code sidecar from local manifests."""

from __future__ import annotations

import json
from pathlib import Path

import tyro

from hal import streams
from hal.training.player_identity import ManifestInput
from hal.training.player_identity import build_player_identity_sidecar


def build(
    professional_root: Path = Path("data/processed/professional"),
    output: Path = Path("data/processed/player-identity-v1/professional-code-v1.jsonl.gz"),
) -> None:
    """Build from ``<root>/<slug>/mds-policy-world-v7/manifest.jsonl``."""
    inputs = tuple(
        ManifestInput(
            name=streams.PROFESSIONAL_POLICY_WORLD_V7[slug].name,
            path=professional_root / slug / "mds-policy-world-v7" / "manifest.jsonl",
        )
        for slug in streams.PROFESSIONAL_PLAYER_SLUGS
    )
    missing = [str(item.path) for item in inputs if not item.path.is_file()]
    if missing:
        raise FileNotFoundError(f"professional manifests are missing: {missing[:5]}")
    result = build_player_identity_sidecar(inputs, output)
    summary = {name: value for name, value in result.items() if name not in {"codes", "display_names"}}
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    tyro.cli(build)
