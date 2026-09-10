"""Durable FIFO queue for netplay reservations."""

import hashlib
import secrets
import sqlite3
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Final

from hal.netplay_service.domain import TERMINAL_STATUSES
from hal.netplay_service.domain import Job
from hal.netplay_service.domain import JobCredentials
from hal.netplay_service.domain import JobStatus
from hal.netplay_service.domain import MatchChoices
from hal.netplay_service.domain import validate_character
from hal.netplay_service.domain import validate_imitation
from hal.netplay_service.domain import validate_player_code
from hal.netplay_service.domain import validate_stage

_SCHEMA_VERSION: Final[int] = 1
_ACTIVE_SQL: Final[str] = "'queued','leased','connecting','playing','rematch_wait','rematch_ready'"


class QueueError(RuntimeError):
    pass


class AuthenticationError(QueueError):
    pass


class InvalidTransitionError(QueueError):
    pass


class InviteError(QueueError):
    pass


class ActiveJobError(QueueError):
    pass


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode()).digest()


class QueueStore:
    """SQLite queue with short transactions and expiring runner leases."""

    def __init__(self, path: str | Path, *, now: Callable[[], float] = time.time) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._now = now
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    @contextmanager
    def _transaction(self):  # type: ignore[no-untyped-def]
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version not in (0, _SCHEMA_VERSION):
                raise QueueError(f"unsupported netplay queue schema {version}")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                f"""
                CREATE TABLE IF NOT EXISTS invites (
                    digest BLOB PRIMARY KEY,
                    label TEXT NOT NULL,
                    player_code TEXT,
                    active INTEGER NOT NULL CHECK (active IN (0, 1)),
                    created_at REAL NOT NULL,
                    last_used_at REAL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    token_digest BLOB NOT NULL,
                    player_code TEXT NOT NULL,
                    character TEXT NOT NULL,
                    imitation TEXT NOT NULL,
                    online_delay INTEGER NOT NULL CHECK (online_delay IN (2, 3)),
                    requested_stage TEXT,
                    status TEXT NOT NULL,
                    queue_seq INTEGER NOT NULL,
                    retry_front INTEGER NOT NULL DEFAULT 0 CHECK (retry_front IN (0, 1)),
                    attempt INTEGER NOT NULL DEFAULT 0,
                    game_count INTEGER NOT NULL DEFAULT 0,
                    connect_code TEXT,
                    actual_stage TEXT,
                    last_result TEXT,
                    error_code TEXT,
                    connect_deadline REAL,
                    rematch_deadline REAL,
                    cancel_after_game INTEGER NOT NULL DEFAULT 0 CHECK (cancel_after_game IN (0, 1)),
                    lease_owner TEXT,
                    lease_expires_at REAL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_one_active_player
                ON jobs(player_code) WHERE status IN ({_ACTIVE_SQL});
                CREATE INDEX IF NOT EXISTS idx_jobs_queue
                ON jobs(retry_front DESC, queue_seq) WHERE status = 'queued';
                CREATE INDEX IF NOT EXISTS idx_jobs_lease
                ON jobs(lease_expires_at) WHERE lease_owner IS NOT NULL;
                CREATE TABLE IF NOT EXISTS games (
                    job_id TEXT NOT NULL REFERENCES jobs(id),
                    game_number INTEGER NOT NULL,
                    actual_stage TEXT NOT NULL,
                    result TEXT NOT NULL,
                    replay_key TEXT,
                    replay_sha256 TEXT,
                    replay_size INTEGER,
                    replay_etag TEXT,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (job_id, game_number)
                );
                """
            )
            connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            connection.execute("PRAGMA optimize")

    def create_invite(self, label: str) -> str:
        if not label.strip():
            raise ValueError("invite label must be non-empty")
        code = secrets.token_urlsafe(24)
        with self._transaction() as connection:
            connection.execute(
                "INSERT INTO invites(digest, label, active, created_at) VALUES (?, ?, 1, ?)",
                (_digest(code), label.strip(), self._now()),
            )
        return code

    def add_invite(self, code: str, label: str) -> None:
        if len(code) < 20:
            raise ValueError("invite codes must contain at least 20 characters")
        if not label.strip():
            raise ValueError("invite label must be non-empty")
        with self._transaction() as connection:
            connection.execute(
                "INSERT INTO invites(digest, label, active, created_at) VALUES (?, ?, 1, ?)",
                (_digest(code), label.strip(), self._now()),
            )

    def create_job(self, invite_code: str, player_code: str, choices: MatchChoices) -> JobCredentials:
        validate_player_code(player_code)
        if not invite_code:
            raise InviteError("invite code is required")
        job_id = secrets.token_urlsafe(18)
        token = secrets.token_urlsafe(32)
        timestamp = self._now()
        try:
            with self._transaction() as connection:
                invite = connection.execute(
                    "SELECT player_code, active FROM invites WHERE digest = ?",
                    (_digest(invite_code),),
                ).fetchone()
                if invite is None or not invite["active"]:
                    raise InviteError("invite code is invalid")
                if invite["player_code"] is not None and invite["player_code"] != player_code:
                    raise InviteError("invite code belongs to a different player code")
                connection.execute(
                    "UPDATE invites SET player_code = ?, last_used_at = ? WHERE digest = ?",
                    (player_code, timestamp, _digest(invite_code)),
                )
                queue_seq = int(connection.execute("SELECT COALESCE(MAX(queue_seq), 0) + 1 FROM jobs").fetchone()[0])
                connection.execute(
                    """
                    INSERT INTO jobs(
                        id, token_digest, player_code, character, imitation, online_delay,
                        requested_stage, status, queue_seq, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, NULL, 'queued', ?, ?, ?)
                    """,
                    (
                        job_id,
                        _digest(token),
                        player_code,
                        choices.character,
                        choices.imitation,
                        choices.online_delay,
                        queue_seq,
                        timestamp,
                        timestamp,
                    ),
                )
        except sqlite3.IntegrityError as error:
            if "jobs.player_code" in str(error):
                raise ActiveJobError(f"player {player_code} already has an active reservation") from error
            raise
        return JobCredentials(self.get_job(job_id, token), token)

    def get_job(self, job_id: str, token: str) -> Job:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None or not secrets.compare_digest(bytes(row["token_digest"]), _digest(token)):
                raise AuthenticationError("job credentials are invalid")
            return self._job(connection, row)

    def get_worker_job(self, job_id: str, worker_id: str) -> Job:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None or row["lease_owner"] != worker_id:
                raise InvalidTransitionError("worker does not own this job")
            return self._job(connection, row)

    def _job(self, connection: sqlite3.Connection, row: sqlite3.Row) -> Job:
        status = JobStatus(row["status"])
        position = None
        if status is JobStatus.QUEUED:
            position = 1 + int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM jobs
                    WHERE status = 'queued' AND (
                        retry_front > ? OR (retry_front = ? AND queue_seq < ?)
                    )
                    """,
                    (row["retry_front"], row["retry_front"], row["queue_seq"]),
                ).fetchone()[0]
            )
        return Job(
            id=row["id"],
            player_code=row["player_code"],
            choices=MatchChoices(
                character=row["character"],
                imitation=row["imitation"],
                online_delay=row["online_delay"],
                requested_stage=row["requested_stage"],
            ),
            status=status,
            queue_position=position,
            attempt=row["attempt"],
            game_count=row["game_count"],
            connect_code=row["connect_code"],
            actual_stage=row["actual_stage"],
            last_result=row["last_result"],
            error_code=row["error_code"],
            connect_deadline=row["connect_deadline"],
            rematch_deadline=row["rematch_deadline"],
            cancel_after_game=bool(row["cancel_after_game"]),
            lease_owner=row["lease_owner"],
            lease_expires_at=row["lease_expires_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def queue_depth(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM jobs WHERE status = 'queued'").fetchone()[0])

    def active_count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute(f"SELECT COUNT(*) FROM jobs WHERE status IN ({_ACTIVE_SQL})").fetchone()[0])

    def claim_next(self, worker_id: str, *, lease_seconds: float = 15.0) -> Job | None:
        if not worker_id:
            raise ValueError("worker_id must be non-empty")
        timestamp = self._now()
        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT * FROM jobs WHERE status = 'queued'
                ORDER BY retry_front DESC, queue_seq ASC LIMIT 1
                """
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """
                UPDATE jobs SET status = 'leased', retry_front = 0, attempt = attempt + 1,
                    lease_owner = ?, lease_expires_at = ?, updated_at = ? WHERE id = ?
                """,
                (worker_id, timestamp + lease_seconds, timestamp, row["id"]),
            )
            claimed = connection.execute("SELECT * FROM jobs WHERE id = ?", (row["id"],)).fetchone()
            assert claimed is not None
            return self._job(connection, claimed)

    def heartbeat(self, job_id: str, worker_id: str, *, lease_seconds: float = 15.0) -> None:
        timestamp = self._now()
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE jobs SET lease_expires_at = ?, updated_at = ?
                WHERE id = ? AND lease_owner = ? AND status NOT IN ('complete','failed','canceled','no_show')
                """,
                (timestamp + lease_seconds, timestamp, job_id, worker_id),
            )
            if cursor.rowcount != 1:
                raise InvalidTransitionError("worker does not own an active job lease")

    def mark_connecting(self, job_id: str, worker_id: str, connect_code: str, *, timeout_seconds: float = 60) -> None:
        self._worker_transition(
            job_id,
            worker_id,
            (JobStatus.LEASED,),
            JobStatus.CONNECTING,
            connect_code=validate_player_code(connect_code),
            connect_deadline=self._now() + timeout_seconds,
        )

    def mark_playing(self, job_id: str, worker_id: str) -> None:
        self._worker_transition(
            job_id,
            worker_id,
            (JobStatus.CONNECTING, JobStatus.REMATCH_READY),
            JobStatus.PLAYING,
            connect_deadline=None,
            rematch_deadline=None,
        )

    def mark_no_show(self, job_id: str, worker_id: str) -> None:
        timestamp = self._now()
        with self._transaction() as connection:
            self._owned_job(connection, job_id, worker_id, (JobStatus.CONNECTING,))
            connection.execute(
                """
                UPDATE jobs SET status = 'no_show', connect_deadline = NULL,
                    lease_owner = NULL, lease_expires_at = NULL, updated_at = ? WHERE id = ?
                """,
                (timestamp, job_id),
            )

    def finish_game(self, job_id: str, worker_id: str, *, actual_stage: str, result: str) -> JobStatus:
        validate_stage(actual_stage)
        if not result:
            raise ValueError("game result must be non-empty")
        timestamp = self._now()
        with self._transaction() as connection:
            row = self._owned_job(connection, job_id, worker_id, (JobStatus.PLAYING,))
            game_number = int(row["game_count"]) + 1
            terminal = game_number >= 5 or bool(row["cancel_after_game"])
            next_status = JobStatus.COMPLETE if terminal else JobStatus.REMATCH_WAIT
            rematch_deadline = None if terminal else timestamp + 60
            connection.execute(
                """
                UPDATE jobs SET status = ?, game_count = ?, actual_stage = ?, last_result = ?,
                    rematch_deadline = ?, lease_owner = CASE WHEN ? THEN NULL ELSE lease_owner END,
                    lease_expires_at = CASE WHEN ? THEN NULL ELSE lease_expires_at END, updated_at = ?
                WHERE id = ?
                """,
                (
                    next_status.value,
                    game_number,
                    actual_stage,
                    result,
                    rematch_deadline,
                    terminal,
                    terminal,
                    timestamp,
                    job_id,
                ),
            )
            connection.execute(
                """
                INSERT INTO games(job_id, game_number, actual_stage, result, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (job_id, game_number, actual_stage, result, timestamp),
            )
            return next_status

    def request_rematch(self, job_id: str, token: str, *, character: str, imitation: str, stage: str) -> Job:
        validate_character(character)
        validate_imitation(imitation)
        validate_stage(stage)
        timestamp = self._now()
        with self._transaction() as connection:
            row = self._authenticated_job(connection, job_id, token)
            if JobStatus(row["status"]) is not JobStatus.REMATCH_WAIT:
                raise InvalidTransitionError("job is not waiting for a rematch")
            deadline = row["rematch_deadline"]
            if deadline is None or deadline <= timestamp:
                self._finish_expired_rematch(connection, job_id, timestamp)
                raise InvalidTransitionError("rematch window expired")
            connection.execute(
                """
                UPDATE jobs SET status = 'rematch_ready', character = ?, imitation = ?,
                    requested_stage = ?, rematch_deadline = NULL, updated_at = ? WHERE id = ?
                """,
                (character, imitation, stage, timestamp, job_id),
            )
            updated = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            assert updated is not None
            return self._job(connection, updated)

    def cancel(self, job_id: str, token: str) -> Job:
        timestamp = self._now()
        with self._transaction() as connection:
            row = self._authenticated_job(connection, job_id, token)
            status = JobStatus(row["status"])
            if status in TERMINAL_STATUSES:
                return self._job(connection, row)
            if status is JobStatus.PLAYING:
                connection.execute(
                    "UPDATE jobs SET cancel_after_game = 1, updated_at = ? WHERE id = ?",
                    (timestamp, job_id),
                )
            else:
                connection.execute(
                    """
                    UPDATE jobs SET status = 'canceled', lease_owner = NULL, lease_expires_at = NULL,
                        connect_deadline = NULL, rematch_deadline = NULL, updated_at = ? WHERE id = ?
                    """,
                    (timestamp, job_id),
                )
            updated = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            assert updated is not None
            return self._job(connection, updated)

    def fail(self, job_id: str, worker_id: str, error_code: str, *, retryable: bool) -> JobStatus:
        if not error_code:
            raise ValueError("error_code must be non-empty")
        timestamp = self._now()
        with self._transaction() as connection:
            row = self._owned_job(
                connection,
                job_id,
                worker_id,
                (
                    JobStatus.LEASED,
                    JobStatus.CONNECTING,
                    JobStatus.PLAYING,
                    JobStatus.REMATCH_WAIT,
                    JobStatus.REMATCH_READY,
                ),
            )
            retry = retryable and int(row["attempt"]) < 2
            status = JobStatus.QUEUED if retry else JobStatus.FAILED
            connection.execute(
                """
                UPDATE jobs SET status = ?, retry_front = ?, error_code = ?, lease_owner = NULL,
                    lease_expires_at = NULL, connect_deadline = NULL, rematch_deadline = NULL,
                    updated_at = ? WHERE id = ?
                """,
                (status.value, retry, error_code, timestamp, job_id),
            )
            return status

    def reap_expired(self) -> int:
        timestamp = self._now()
        changed = 0
        with self._transaction() as connection:
            changed += connection.execute(
                """
                UPDATE jobs SET status = 'no_show', lease_owner = NULL, lease_expires_at = NULL,
                    connect_deadline = NULL, updated_at = ?
                WHERE status = 'connecting' AND connect_deadline <= ?
                """,
                (timestamp, timestamp),
            ).rowcount
            expired_rematches = connection.execute(
                "SELECT id FROM jobs WHERE status = 'rematch_wait' AND rematch_deadline <= ?",
                (timestamp,),
            ).fetchall()
            for row in expired_rematches:
                self._finish_expired_rematch(connection, row["id"], timestamp)
            changed += len(expired_rematches)
            expired_leases = connection.execute(
                """
                SELECT id, attempt FROM jobs WHERE lease_owner IS NOT NULL AND lease_expires_at <= ?
                    AND status NOT IN ('complete','failed','canceled','no_show')
                """,
                (timestamp,),
            ).fetchall()
            for row in expired_leases:
                retry = int(row["attempt"]) < 2
                connection.execute(
                    """
                    UPDATE jobs SET status = ?, retry_front = ?, error_code = 'lease_expired',
                        lease_owner = NULL, lease_expires_at = NULL, connect_deadline = NULL,
                        rematch_deadline = NULL, updated_at = ? WHERE id = ?
                    """,
                    ("queued" if retry else "failed", retry, timestamp, row["id"]),
                )
            changed += len(expired_leases)
        return changed

    def record_replay(
        self,
        job_id: str,
        game_number: int,
        *,
        key: str,
        sha256: str,
        size: int,
        etag: str,
    ) -> None:
        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT replay_key, replay_sha256, replay_size, replay_etag FROM games
                WHERE job_id = ? AND game_number = ?
                """,
                (job_id, game_number),
            ).fetchone()
            if row is None:
                raise InvalidTransitionError("game is absent")
            existing = (row["replay_key"], row["replay_sha256"], row["replay_size"], row["replay_etag"])
            requested = (key, sha256, size, etag)
            if existing == requested:
                return
            if row["replay_key"] is not None:
                raise InvalidTransitionError("game already has a different replay")
            cursor = connection.execute(
                """
                UPDATE games SET replay_key = ?, replay_sha256 = ?, replay_size = ?, replay_etag = ?
                WHERE job_id = ? AND game_number = ? AND replay_key IS NULL
                """,
                (key, sha256, size, etag, job_id, game_number),
            )
            if cursor.rowcount != 1:
                raise InvalidTransitionError("game replay changed during recording")

    def _worker_transition(
        self,
        job_id: str,
        worker_id: str,
        expected: tuple[JobStatus, ...],
        target: JobStatus,
        **values: object,
    ) -> None:
        timestamp = self._now()
        with self._transaction() as connection:
            self._owned_job(connection, job_id, worker_id, expected)
            assignments = ["status = ?", "updated_at = ?"]
            arguments: list[object] = [target.value, timestamp]
            for name, value in values.items():
                if name not in {"connect_code", "connect_deadline", "rematch_deadline"}:
                    raise AssertionError(f"unsupported transition field {name}")
                assignments.append(f"{name} = ?")
                arguments.append(value)
            arguments.append(job_id)
            connection.execute(f"UPDATE jobs SET {', '.join(assignments)} WHERE id = ?", arguments)

    def _authenticated_job(self, connection: sqlite3.Connection, job_id: str, token: str) -> sqlite3.Row:
        row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None or not secrets.compare_digest(bytes(row["token_digest"]), _digest(token)):
            raise AuthenticationError("job credentials are invalid")
        return row

    def _owned_job(
        self,
        connection: sqlite3.Connection,
        job_id: str,
        worker_id: str,
        expected: tuple[JobStatus, ...],
    ) -> sqlite3.Row:
        row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None or row["lease_owner"] != worker_id:
            raise InvalidTransitionError("worker does not own this job")
        if JobStatus(row["status"]) not in expected:
            raise InvalidTransitionError(f"job status {row['status']!r} is not valid for this operation")
        return row

    @staticmethod
    def _finish_expired_rematch(connection: sqlite3.Connection, job_id: str, timestamp: float) -> None:
        connection.execute(
            """
            UPDATE jobs SET status = 'complete', lease_owner = NULL, lease_expires_at = NULL,
                rematch_deadline = NULL, updated_at = ? WHERE id = ?
            """,
            (timestamp, job_id),
        )
