"""Count per-character player identities over a checkpoint's own training rows."""

import dataclasses
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import TypedDict
from typing import cast

import torch
import tyro

from hal import streams
from hal.data.index import ReplayIndexEntry
from hal.data.player_identity import PlayerIdentitySidecar
from hal.data.player_identity import load_player_identity_sidecar
from hal.data.policy_schema import policy_replay_identity
from hal.data.schema import Rank
from hal.data.schema import rank_from_player_name
from hal.eval.matchups import MATCHUP_PRIOR
from hal.policy import INCLUDED_STAGES


class IdentityRow(TypedDict):
    identity: int
    character: int
    count: int


@dataclasses.dataclass(frozen=True, slots=True)
class CountResult:
    identities: tuple[IdentityRow, ...]
    train_rows: int
    sha256: str


def count_identities(path: Path, sidecar: PlayerIdentitySidecar, *, professional: bool) -> CountResult:
    counts: Counter[tuple[int, int]] = Counter()
    digest = hashlib.sha256()
    rows: set[int] = set()
    replays: set[str] = set()
    with path.open("rb") as source:
        for line in source:
            digest.update(line)
            if not line.strip():
                continue
            entry = ReplayIndexEntry.from_dict(json.loads(line))
            annotation = entry.annotation
            if annotation is None or annotation.split != "train":
                continue
            if entry.schema_version != 7 or annotation.schema_version != 7:
                raise ValueError(f"{path}: expected policy-world-v8 replay metadata schema 7")
            replay = policy_replay_identity(entry.path)
            if annotation.mds_row_idx in rows or replay in replays:
                raise ValueError(f"{path}: duplicate training row or replay")
            rows.add(annotation.mds_row_idx)
            replays.add(replay)
            players = sorted(entry.players, key=lambda player: player.port)
            if len(players) != 2:
                raise ValueError(f"{path}: expected two occupied player ports")
            if professional:
                identities = sidecar.by_replay[replay]
            else:
                identities = tuple(int(rank_from_player_name(player.name)) for player in players)
                if int(Rank.UNKNOWN) in identities:
                    raise ValueError(f"{path}: ranked training row has an unknown identity")
            for player, identity in zip(players, identities, strict=True):
                counts[identity, player.character] += 1
    if not rows or rows != set(range(len(rows))):
        raise ValueError(f"{path}: training rows must be nonempty and contiguous")
    return CountResult(
        tuple(IdentityRow(identity=i, character=c, count=n) for (i, c), n in sorted(counts.items())),
        len(rows),
        digest.hexdigest(),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class Args:
    checkpoint: Path
    manifests_root: Path
    sidecar: Path = Path("data/processed/player-identity-v1/professional-code-v1.jsonl.gz")
    output: Path = Path("experiments/o65/pilot-sampler.json")
    provenance: Path = Path("experiments/o65/pilot-sampler-source.json")


def main(args: Args) -> None:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = cast(dict[str, object], checkpoint["cfg"])
    names = cast(tuple[str, ...], config["source_names"])
    if not names or len(names) != len(set(names)):
        raise ValueError("checkpoint source names must be nonempty and unique")
    sidecar = load_player_identity_sidecar(args.sidecar, expected_sha256=cast(str, config["player_sidecar_sha256"]))
    counts: Counter[tuple[int, int]] = Counter()
    sources: dict[str, object] = {}
    for name in names:
        path = args.manifests_root / name / "manifest.jsonl"
        result = count_identities(path, sidecar, professional=name.startswith("professional-"))
        if result.train_rows != streams.POLICY_WORLD_V8_TRAIN_REPLAYS[name]:
            raise ValueError(f"{name}: replay manifest does not cover the checkpoint training rows")
        for row in result.identities:
            counts[row["identity"], row["character"]] += row["count"]
        sources[name] = {
            "manifest": str(path),
            "sha256": result.sha256,
            "train_rows": result.train_rows,
            "checkpoint_train_index_sha256": checkpoint["provenance"]["source_manifest_sha256"][name],
        }
    sampler = {
        "format": "hal.o65.sampler.v2",
        "identities": [IdentityRow(identity=i, character=c, count=n) for (i, c), n in sorted(counts.items())],
        "matchups": [{"characters": [a.value, b.value], "weight": n} for a, b, n in MATCHUP_PRIOR],
        "stages": [stage.value for stage in INCLUDED_STAGES],
        "stocks": [4],
        "percents": [0.0],
        "data_floor": 1,
        "tempering": 1.0,
        "masked_rate": float(cast(float, config["identity_dropout"])),
        "deployment_rate": 0.0,
    }
    provenance = {
        "format": "hal.o65.sampler-source.v2",
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "sources": sources,
        "identity_sidecar_sha256": sidecar.sha256,
        "count_unit": "player appearances per character in the checkpoint training rows; no validation/test rows",
        "matchup_prior": "hal.eval.matchups.MATCHUP_PRIOR (259 ranked train pairs, minimum 10 replays)",
    }
    args.output.write_text(json.dumps(sampler, indent=2) + "\n")
    args.provenance.write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main(tyro.cli(Args))
