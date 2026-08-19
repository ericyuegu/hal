"""Build replay-ID-to-player-rank metadata for compact policy training."""

import gzip
import hashlib
import io
import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory

import tyro

from hal.data.index import ReplayIndexEntry
from hal.data.index import read_jsonl
from hal.data.mds import open_shard
from hal.data.mds import read_shard_index
from hal.data.policy_schema import policy_replay_identity
from hal.data.schema import Rank
from hal.data.schema import rank_from_player_name

RANK_SIDECAR_SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rank_pair(entry: ReplayIndexEntry) -> tuple[Rank, Rank]:
    by_port = {player.port: rank_from_player_name(player.name) for player in entry.players}
    if set(by_port) != {1, 2}:
        raise ValueError(f"{entry.path}: expected human ports 1 and 2, got {sorted(by_port)}")
    ranks = (by_port[1], by_port[2])
    allowed = {Rank.PLATINUM, Rank.DIAMOND, Rank.MASTER}
    if set(ranks) - allowed:
        names = {player.port: player.name for player in entry.players}
        raise ValueError(f"{entry.path}: ranked replay has unsupported player tier(s): {names}")
    return ranks


def _rank_rows(manifest: Path) -> Iterator[tuple[str, int, int]]:
    seen: set[str] = set()
    for entry in read_jsonl(manifest, verify_schema_version=False):
        if entry.annotation is None:
            continue
        replay_id = policy_replay_identity(entry.path)
        if replay_id in seen:
            raise ValueError(f"duplicate replay ID {replay_id} in {manifest}")
        seen.add(replay_id)
        p1_rank, p2_rank = _rank_pair(entry)
        yield replay_id, int(p1_rank), int(p2_rank)


def audit_policy_coverage(
    policy_root: Path,
    expected_ids: set[str],
) -> dict[str, dict[str, int | str]]:
    """Scan compact policy rows and prove exact sidecar ID coverage."""
    observed_ids: set[str] = set()
    report: dict[str, dict[str, int | str]] = {}
    with TemporaryDirectory(prefix="hal-rank-audit-") as scratch:
        for split in ("train", "val", "test"):
            shards = read_shard_index(policy_root, split)
            expected_rows = sum(int(info["samples"]) for info in shards)
            ordered_digest = hashlib.sha256()
            row_number = 0
            for info in shards:
                with open_shard(policy_root, split, info, Path(scratch)) as reader:
                    for sample in reader:
                        replay_id = str(sample["replay_id"])
                        if replay_id in observed_ids:
                            raise ValueError(f"compact policy dataset repeats replay ID {replay_id}")
                        observed_ids.add(replay_id)
                        ordered_digest.update(replay_id.encode("ascii") + b"\n")
                        row_number += 1
                        if row_number % 10_000 == 0:
                            print(f"[audit] {split}: {row_number}/{expected_rows} replay IDs", flush=True)
            if row_number != expected_rows:
                raise ValueError(f"{split}: read {row_number} rows, expected {expected_rows}")
            report[split] = {
                "rows": row_number,
                "ordered_replay_id_sha256": ordered_digest.hexdigest(),
            }

    missing = expected_ids - observed_ids
    unexpected = observed_ids - expected_ids
    if missing or unexpected:
        raise ValueError(
            "compact policy/sidecar replay-ID mismatch: "
            f"missing={len(missing)}, unexpected={len(unexpected)}, "
            f"first_missing={sorted(missing)[:3]}, first_unexpected={sorted(unexpected)[:3]}"
        )
    report["all"] = {
        "rows": len(observed_ids),
        "set_replay_id_sha256": hashlib.sha256(
            b"".join(replay_id.encode("ascii") + b"\n" for replay_id in sorted(observed_ids))
        ).hexdigest(),
    }
    return report


def build_rank_sidecar(
    manifest: Path = Path("data/processed/ranked-anonymized-1/mds-v7/manifest.jsonl"),
    output: Path = Path("data/processed/ranked-anonymized-1/mds-policy-v7/ranks-v1.jsonl.gz"),
    policy_root: Path | None = None,
) -> None:
    """Create a deterministic compressed rank lookup from a canonical manifest."""
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    rows = sorted(_rank_rows(manifest))
    if not rows:
        raise ValueError(f"{manifest} contains no materialized replay rows")
    counts = Counter(rank for _, p1_rank, p2_rank in rows for rank in (p1_rank, p2_rank))
    header = {
        "rank_sidecar_schema_version": RANK_SIDECAR_SCHEMA_VERSION,
        "source_manifest_sha256": _sha256(manifest),
        "rows": len(rows),
        "player_rank_counts": {str(rank): count for rank, count in sorted(counts.items())},
    }

    policy_audit = None
    if policy_root is not None:
        policy_audit = audit_policy_coverage(policy_root, {replay_id for replay_id, _, _ in rows})

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with (
        temporary.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        io.TextIOWrapper(compressed, encoding="utf-8") as text,
    ):
        text.write(json.dumps(header, sort_keys=True, separators=(",", ":")) + "\n")
        for row in rows:
            text.write(json.dumps(row, separators=(",", ":")) + "\n")
    temporary.replace(output)
    print(
        json.dumps(
            {
                **header,
                "output": str(output),
                "policy_audit": policy_audit,
                "sha256": _sha256(output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    tyro.cli(build_rank_sidecar)
