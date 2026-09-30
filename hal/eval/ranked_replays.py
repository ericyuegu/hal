"""Upload completed ranked records without blocking controller input."""

import hashlib
import json
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Self

import melee
import melee.enums
import peppi_py
from botocore.exceptions import BotoCoreError
from botocore.exceptions import ClientError
from loguru import logger

from hal import r2
from hal.netplay_service.replays import ReplayMetadata
from hal.netplay_service.replays import UploadedReplay
from hal.netplay_service.replays import upload_replay


def _digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _write(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def upload_game(
    record: Path,
    *,
    bucket: str,
    upload: Callable[[Path, ReplayMetadata], UploadedReplay] = upload_replay,
) -> bool:
    """A game record commits a complete replay; an upload receipt commits R2."""
    raw = record.read_bytes()
    record_sha256 = hashlib.sha256(raw).hexdigest()
    row = json.loads(raw)
    if not isinstance(row, dict) or type(row.get("schema_version")) is not int or row["schema_version"] != 1:
        raise ValueError(f"unsupported ranked game record: {record}")
    receipt = record.parent / "uploads" / record.name
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if (
            not isinstance(saved, dict)
            or saved.get("schema_version") != 1
            or saved.get("bucket") != bucket
            or saved.get("record_sha256") != record_sha256
            or saved.get("sha256") != row.get("replay_sha256")
        ):
            raise ValueError(f"ranked upload receipt differs from the game record: {receipt}")
        return False

    replay = (record.parent / row["replay"]).resolve()
    if not replay.is_relative_to((record.parent / "replays").resolve()):
        raise ValueError("ranked replay must be inside the run replay directory")
    digest = _digest(replay)
    if digest != row["replay_sha256"]:
        raise ValueError(f"ranked replay hash differs from the game record: {replay}")
    manifest = json.loads((record.parent / "manifest.json").read_text())
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("unsupported ranked run manifest")
    game = peppi_py.read_slippi(str(replay), skip_frames=True)
    match = game.start.match
    if game.end is None or match is None or not match.id.startswith("mode.ranked-"):
        raise ValueError("ranked upload requires a completed ranked replay")
    if melee.enums.to_internal_stage(game.start.stage).value != row["stage"]:
        raise ValueError("replay stage differs from the game record")
    if game.end.method.name != row["end_method"]:
        raise ValueError("replay end differs from the game record")
    ego_port = row["ego_port"]
    if type(ego_port) is not int or ego_port not in (1, 2):
        raise ValueError("ranked game has an invalid local port")
    players = [player for player in game.start.players if int(player.port) + 1 == ego_port]
    if len(players) != 1 or players[0].netplay is None:
        raise ValueError("ranked replay has no local netplay identity")
    player_code = players[0].netplay.code.replace("＃", "#")
    placements = {} if game.end.players is None else {int(p.port) + 1: p.placement for p in game.end.players}
    placement = placements.get(ego_port)
    result = "win" if placement == 0 else "loss" if placement == 1 else "unresolved"
    metadata = ReplayMetadata(
        reservation_id="ranked-" + hashlib.sha256(f"{match.id}/{match.tiebreaker}".encode()).hexdigest(),
        player_code=player_code,
        game_number=match.game,
        actual_stage=melee.Stage(row["stage"]).name,
        result=result,
        policy_sha256=manifest["bundle_sha256"],
        git_sha=manifest["source_revision"],
        started_at=datetime.fromisoformat(game.metadata["startAt"]),
        ended_at=datetime.fromtimestamp(row["at"], UTC),
    )
    uploaded = upload(replay, metadata)
    if uploaded.sha256 != digest or uploaded.size != replay.stat().st_size:
        raise ValueError("uploaded replay identity differs from the local replay")
    receipt.parent.mkdir(exist_ok=True)
    _write(
        receipt,
        {"schema_version": 1, "bucket": bucket, "record_sha256": record_sha256, **asdict(uploaded)},
    )
    logger.info("Ranked replay uploaded: {} ({} bytes)", uploaded.key, uploaded.size)
    return True


def upload_pending(root: Path, *, stop: threading.Event | None = None) -> int:
    """Retry records from previous runs; never select an unfinished .slp."""
    uploaded = 0
    bucket = r2.bucket()
    remote = r2.client(timeout_seconds=5)
    try:
        send = partial(upload_replay, client=remote)
        for record in sorted(root.glob("*/game-*.json")):
            if stop is not None and stop.is_set():
                break
            if not record.stem.removeprefix("game-").isdigit():
                continue
            try:
                uploaded += upload_game(record, bucket=bucket, upload=send)
            except (BotoCoreError, ClientError, OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
                _write(
                    record.parent / "upload-error.json",
                    {"at": time.time(), "record": record.name, "type": type(error).__name__, "message": str(error)},
                )
                logger.warning("Ranked replay upload deferred for {}: {}", record, error)
    finally:
        remote.close()
    return uploaded


class RankedReplayUploads:
    """Keep retry state on disk and network work outside the game loop."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread = threading.Thread(target=self._run, name="ranked-replay-upload", daemon=True)

    def __enter__(self) -> Self:
        if r2.missing_credentials():
            raise ValueError("R2 credentials are required for ranked replay uploads")
        r2.bucket()
        self._thread.start()
        return self

    def notify(self) -> None:
        self._wake.set()

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=45)
        if self._thread.is_alive():
            logger.warning("Ranked upload is still finishing; pending records remain on disk")

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.clear()
            try:
                upload_pending(self.root, stop=self._stop)
            except (BotoCoreError, ClientError, OSError, RuntimeError, ValueError) as error:
                logger.warning("Ranked replay upload pass failed: {}", error)
            self._wake.wait(15)
