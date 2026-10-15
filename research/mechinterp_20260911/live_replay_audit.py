"""Audit fixed gameplay SLPs before reconstructing offline O50 contexts.

This is an audit-only reconstruction of the one-off CPU trace's replay path. It
does not load a checkpoint or run a model forward pass. The original trace
source was ``/tmp/live_replay_attention_probe.py`` (SHA-256 prefix
``36abbd73071cac44``); this module preserves its fixed input rule and makes
the replay and port checks reproducible in the worktree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np
import peppi_py

from hal.data.extract import extract_replay
from hal.data.slp_finalize import is_finalized
from hal.training import dataloader
from hal.training import features
from hal.wire import peppi_port_to_libmelee

CONTEXT_LENGTH = 256
WINDOW_START = 700
BOOT_INDICES = (0, 1, 2)
CONDITIONS = ("early_24576", "slump_57344")
CONTROLLED_PORT = 1
MASKED_PLAYER_ID = 0


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of one immutable input file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_frame_ids(frame: np.ndarray, *, minimum_rows: int = CONTEXT_LENGTH) -> None:
    """Reject short or restart-crossing replay windows."""
    if frame.ndim != 1:
        raise ValueError(f"frame ids must be one-dimensional, got {frame.shape}")
    if len(frame) < WINDOW_START + minimum_rows:
        raise ValueError(
            f"replay has {len(frame)} rows; need at least {WINDOW_START + minimum_rows} for the fixed window"
        )
    if np.any(np.diff(frame) < 0):
        raise ValueError("replay contains an instant-restart boundary")


def replay_path(replay_root: Path, condition: str, boot: int) -> Path:
    """Resolve exactly one replay for one fixed condition and boot."""
    if condition not in CONDITIONS:
        raise ValueError(f"unsupported condition {condition!r}")
    if boot not in BOOT_INDICES:
        raise ValueError(f"unsupported boot {boot!r}")
    paths = sorted((replay_root / condition / f"boot_{boot:03d}").glob("*.slp"))
    if len(paths) != 1:
        raise ValueError(f"expected one SLP for {condition} boot {boot}, got {paths}")
    return paths[0]


def extract_context(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Extract one fixed p1 context and return columns plus provenance."""
    if not is_finalized(path):
        raise ValueError(f"replay is not finalized: {path}")
    source = extract_replay(str(path))
    if source is None:
        raise ValueError(f"replay extraction failed: {path}")
    frame = np.asarray(source["frame"])
    validate_frame_ids(frame)
    required = ("stage", "p1_character", "p2_character", "p1_action", "p2_action")
    missing = sorted(name for name in required if name not in source)
    if missing:
        raise ValueError(f"replay is missing required columns {missing}: {path}")
    for name in required:
        values = np.asarray(source[name])[WINDOW_START : WINDOW_START + CONTEXT_LENGTH]
        if not np.isfinite(values).all():
            raise ValueError(f"required column {name!r} is not finite in {path}")
    source["p1_player_id"] = np.full(len(frame), MASKED_PLAYER_ID, dtype=np.int32)
    source["p2_player_id"] = np.full(len(frame), MASKED_PLAYER_ID, dtype=np.int32)
    window = dataloader.make_window(
        source,
        ego_prefix="p1",
        start=WINDOW_START,
        pad=0,
        length=CONTEXT_LENGTH,
        projection=features.ITEM_PLAYER_PROJECTION,
    )
    window["ctx_pad"] = np.int64(0)
    columns = dataloader.collate_windows([window])
    return columns, {
        "path": str(path),
        "sha256": sha256_file(path),
        "frame_start": int(frame[WINDOW_START]),
        "frame_stop": int(frame[WINDOW_START + CONTEXT_LENGTH - 1]),
        "row_count": len(frame),
        "raw_column_count": len(source),
        "projected_column_count": len(features.ITEM_PLAYER_PROJECTION.columns),
        "context_length": CONTEXT_LENGTH,
        "context_start_row": WINDOW_START,
        "ego_prefix": "p1",
        "masked_player_id": MASKED_PLAYER_ID,
        "frame_resets": int(np.count_nonzero(np.diff(frame) < 0)),
    }


def start_port_record(path: Path) -> list[dict[str, object]]:
    """Return the SLP game-start ports used by the p1/p2 extractor mapping."""
    game = peppi_py.read_slippi(str(path), skip_frames=True)
    return [
        {
            "peppi_port": int(player.port),
            "libmelee_port": int(peppi_port_to_libmelee(player.port)),
            "character_external": int(player.character),
            "player_type": int(player.type),
        }
        for player in game.start.players
    ]


def verify_controlled_port(path: Path, *, expected_ego: str, expected_opponent: str) -> dict[str, object]:
    """Verify p1 is port 1 and its matchup agrees with pilot metadata."""
    source = extract_replay(str(path))
    if source is None:
        raise ValueError(f"replay extraction failed: {path}")
    import melee

    p1 = melee.Character(int(np.asarray(source["p1_character"])[0])).name
    p2 = melee.Character(int(np.asarray(source["p2_character"])[0])).name
    if (p1, p2) != (expected_ego, expected_opponent):
        raise ValueError(f"replay matchup {(p1, p2)} differs from {(expected_ego, expected_opponent)}")
    starts = start_port_record(path)
    libmelee_ports = sorted(int(cast(int, row["libmelee_port"])) for row in starts)
    if libmelee_ports != [1, 2]:
        raise ValueError(f"expected libmelee start ports [1, 2], got {libmelee_ports}")
    return {
        "controlled_port": CONTROLLED_PORT,
        "p1_maps_to_controlled_port": libmelee_ports[0] == CONTROLLED_PORT,
        "p1_character": p1,
        "p2_character": p2,
        "start_players": starts,
    }


def load_protocol(result_path: Path) -> Mapping[str, object]:
    """Load the pilot protocol and require its masked identity contract."""
    raw = json.loads(result_path.read_bytes())
    protocol = raw.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("pilot result has no protocol mapping")
    if protocol.get("effective_player_id") != MASKED_PLAYER_ID:
        raise ValueError("pilot effective player identity differs from the fixed masked identity")
    return cast(Mapping[str, object], protocol)


def audit(replay_root: Path, result_path: Path) -> dict[str, object]:
    """Audit the six preselected contexts without model execution."""
    protocol = load_protocol(result_path)
    schedule = protocol.get("schedule")
    if not isinstance(schedule, list) or len(schedule) < max(BOOT_INDICES) + 1:
        raise ValueError("pilot protocol has no complete fixed matchup schedule")
    contexts = []
    for condition in CONDITIONS:
        for boot in BOOT_INDICES:
            expected_raw = schedule[boot]
            if not isinstance(expected_raw, Mapping):
                raise ValueError(f"schedule row {boot} is not an object")
            expected = cast(Mapping[str, object], expected_raw)
            path = replay_path(replay_root, condition, boot)
            columns, record = extract_context(path)
            control = verify_controlled_port(
                path,
                expected_ego=str(expected["ego"]),
                expected_opponent=str(expected["opponent"]),
            )
            record.update({"condition": condition, "boot": boot, "columns": len(columns), **control})
            contexts.append(record)
    return {
        "schema_version": 1,
        "audit_only": True,
        "protocol": {
            "replay_root": str(replay_root),
            "pilot_result": str(result_path),
            "conditions": CONDITIONS,
            "boots": BOOT_INDICES,
            "context_start_row": WINDOW_START,
            "context_length": CONTEXT_LENGTH,
            "controlled_port": CONTROLLED_PORT,
            "masked_player_id": MASKED_PLAYER_ID,
            "model_execution": "none",
        },
        "contexts": contexts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--pilot-result", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = audit(args.replay_root, args.pilot_result)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
