"""Per-replay metadata: ``ReplayIndexEntry`` shared by all three pipeline stages.

Stage 1 emits one entry per slp into ``index.jsonl`` (start/end/metadata
blocks only, no frame iteration). Stage 3 writes the subset that landed in
MDS into ``manifest.jsonl`` with ``Stage3Annotation`` populated.

Integer ids are slp-native (peppi-py vocabulary) — see CLAUDE.md
(Architecture → Conventions / Footguns) for the stage/character/port translation rules.
LRAS = "L + R + A + Start" controller-combo forfeit; valid only when the slp
ended via ``NO_CONTEST`` — ``GameOutcome`` enforces this at construction.

``extract_index_entry`` returns ``None`` on peppi parse failure (~0.5% of the
mang0 corpus — usually an out-of-spec ``end.method`` byte).
"""

import dataclasses
import hashlib
import json
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any
from typing import Literal
from typing import cast
from typing import get_args

import peppi_py

from hal.data.schema import SCHEMA_VERSION
from hal.wire import peppi_port_to_libmelee as _peppi_port_to_libmelee


class EndMethod(IntEnum):
    """slp end.method values per the Slippi spec.

    Mirrors `peppi_py.game.EndMethod` exactly — adding new values here without
    a peppi update would break parsing.
    """

    UNRESOLVED = 0
    TIME = 1  # match timer expired
    GAME = 2  # one player ran out of stocks
    RESOLVED = 3  # sudden-death resolution
    NO_CONTEST = 7  # someone LRAS'd or disconnected


# slp Player.type values. EMPTY (unused port slot) is filtered out before
# PlayerEntry is constructed, so it is not reachable here. Slip kept loose
# because future slp revisions may introduce new values we haven't seen.
PlayerType = Literal["HUMAN", "CPU", "DEMO"]

# `metadata.playedOn` values seen in the wild. None = older slps that omit
# the field entirely. Kept loose for the same reason as PlayerType.
PlayedOn = Literal["dolphin", "console", "network"]

Split = Literal["train", "val", "test"]

_VALID_PORTS: tuple[int, ...] = (1, 2, 3, 4)
_RANK_KEYWORDS: tuple[str, ...] = ("platinum", "diamond", "master")


@dataclass(frozen=True)
class GameOutcome:
    """How a slp ended.

    Couples `end_method` with `lras_initiator` so the invariant
    "an lras_initiator only exists for NO_CONTEST games" is enforced at
    construction time. Constructing a `GameOutcome` with a contradiction
    raises `ValueError`.
    """

    end_method: EndMethod
    lras_initiator: int | None  # libmelee port (1..4) iff end_method == NO_CONTEST

    def __post_init__(self) -> None:
        if self.lras_initiator is None:
            return
        if self.end_method != EndMethod.NO_CONTEST:
            raise ValueError(
                f"lras_initiator={self.lras_initiator} is only valid when "
                f"end_method=NO_CONTEST; got {self.end_method.name}"
            )
        if self.lras_initiator not in _VALID_PORTS:
            raise ValueError(f"lras_initiator must be in {_VALID_PORTS} or None; got {self.lras_initiator}")

    @property
    def completed(self) -> bool:
        """True iff the game ran to a definitive result (stocks, timer, or
        sudden-death resolution). False for UNRESOLVED and NO_CONTEST."""
        return self.end_method in (EndMethod.TIME, EndMethod.GAME, EndMethod.RESOLVED)

    def to_dict(self) -> dict[str, Any]:
        return {"end_method": int(self.end_method), "lras_initiator": self.lras_initiator}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GameOutcome:
        return cls(end_method=EndMethod(data["end_method"]), lras_initiator=data["lras_initiator"])


@dataclass(frozen=True)
class Stage3Annotation:
    """Atomic annotation written by process_replays when an entry lands in MDS.

    Either all five fields are populated together or the entire annotation is
    None — the per-field nullability of the previous design is no longer
    representable.

    ``schema_version`` pins the column set the shards were written with. A
    consumer (training, round-trip) refuses to load a manifest whose version
    doesn't match its expected ``SCHEMA_VERSION``; this prevents silent
    drift between schema-incompatible builds.
    """

    replay_uuid: int
    split: Split
    mds_row_idx: int
    frame_count_actual: int
    schema_version: int

    def __post_init__(self) -> None:
        if self.split not in ("train", "val", "test"):
            raise ValueError(f"split must be one of train/val/test; got {self.split!r}")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Stage3Annotation:
        if "schema_version" not in data:
            raise ValueError(
                "manifest entry has no schema_version; this manifest was built "
                "before SCHEMA_VERSION was introduced. Rerun process_replays."
            )
        return cls(**data)


@dataclass(frozen=True)
class PlayerEntry:
    port: int  # 1..4 (libmelee convention; slp/peppi use 0..3)
    character: int  # slp-native (in-game) character id
    costume: int
    player_type: PlayerType
    code: str | None  # netplay connect code, e.g. "ZAIN#0"
    name: str | None  # netplay display name

    def __post_init__(self) -> None:
        if self.port not in _VALID_PORTS:
            raise ValueError(f"port must be in {_VALID_PORTS}; got {self.port}")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlayerEntry:
        return cls(**data)


@dataclass(frozen=True)
class ReplayIndexEntry:
    path: str  # absolute path
    slp_version: tuple[int, int, int]
    stage: int  # slp-native stage id
    players: list[PlayerEntry]
    frame_count: int  # metadata.lastFrame; 0 if missing
    timestamp: str | None  # ISO 8601 from metadata.startAt
    played_on: PlayedOn | None
    outcome: GameOutcome | None  # None iff slp has no end block (truncated / in-progress)
    rank_filename: str | None  # heuristic rank label inferred from filename
    sha1: str | None  # sha1 hex digest of the whole file (None if compute_sha1=False)

    annotation: Stage3Annotation | None = None

    def __post_init__(self) -> None:
        if len(self.slp_version) != 3:
            raise ValueError(f"slp_version must be a 3-tuple; got {self.slp_version!r}")

    def player_for_port(self, port: int) -> PlayerEntry | None:
        for p in self.players:
            if p.port == port:
                return p
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "slp_version": list(self.slp_version),
            "stage": self.stage,
            "players": [p.to_dict() for p in self.players],
            "frame_count": self.frame_count,
            "timestamp": self.timestamp,
            "played_on": self.played_on,
            "outcome": self.outcome.to_dict() if self.outcome is not None else None,
            "rank_filename": self.rank_filename,
            "sha1": self.sha1,
            "annotation": self.annotation.to_dict() if self.annotation is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReplayIndexEntry:
        outcome = data.get("outcome")
        annotation = data.get("annotation")
        return cls(
            path=data["path"],
            slp_version=tuple(data["slp_version"]),
            stage=data["stage"],
            players=[PlayerEntry.from_dict(p) for p in data["players"]],
            frame_count=data["frame_count"],
            timestamp=data.get("timestamp"),
            played_on=data.get("played_on"),
            outcome=GameOutcome.from_dict(outcome) if outcome is not None else None,
            rank_filename=data.get("rank_filename"),
            sha1=data.get("sha1"),
            annotation=Stage3Annotation.from_dict(annotation) if annotation is not None else None,
        )


def replay_uuid_from_path(path: str | Path) -> int:
    """Stable int32 hash of an absolute replay path."""
    digest = hashlib.md5(str(path).encode()).digest()
    return struct.unpack("i", digest[:4])[0]


def _sha1(path: Path, chunk_bytes: int = 1 << 20) -> str:
    """Streaming sha1 over the whole file.

    Full-file hash costs ~5 ms / MB; index builds are once-off so the extra I/O
    is acceptable. For archive-streaming, the file is already in tmpfs so the
    second read is essentially free.
    """
    h = hashlib.sha1()
    with path.open("rb") as f:
        while chunk := f.read(chunk_bytes):
            h.update(chunk)
    return h.hexdigest()


def _rank_from_filename(path: Path) -> str | None:
    name = path.name.lower()
    for rank in _RANK_KEYWORDS:
        if rank in name:
            return rank
    return None


def _raw_player_type_name(t: Any) -> str:
    name = getattr(t, "name", None)
    return str(name) if name is not None else str(t)


def _narrow_player_type(name: str) -> PlayerType:
    """Narrow peppi's player-type name to the PlayerType Literal.

    EMPTY is a peppi-internal placeholder for unused port slots and is
    filtered out by callers before they reach this function. Any other
    value outside the Literal is rejected — silently coercing would let an
    out-of-spec ``end.method``-style failure mode through.
    """
    if name not in get_args(PlayerType):
        raise ValueError(f"unknown player type: {name!r}")
    return cast(PlayerType, name)


def extract_index_entry(replay_path: Path, *, compute_sha1: bool = True) -> ReplayIndexEntry | None:
    """Parse a .slp file's start/end/metadata blocks (no frame iteration) and
    return a `ReplayIndexEntry`. Returns None on parse failure (caller logs)."""
    # Indexing walks 100k+ files; one malformed slp shouldn't kill the job, so
    # we surface failures as None rather than propagating exceptions.
    try:
        g = peppi_py.read_slippi(str(replay_path), skip_frames=True)
    except Exception:
        return None

    md = g.metadata or {}
    md_players: dict[str, Any] = md.get("players") or {}

    players: list[PlayerEntry] = []
    for sp in g.start.players:
        type_name = _raw_player_type_name(sp.type).upper()
        if type_name == "EMPTY":
            continue
        port = _peppi_port_to_libmelee(sp.port)
        # metadata.players is keyed by 0-indexed port as a string
        md_entry = md_players.get(str(port - 1)) or {}
        names = md_entry.get("names") or {}
        players.append(
            PlayerEntry(
                port=port,
                character=int(sp.character),
                costume=int(sp.costume),
                player_type=_narrow_player_type(type_name),
                code=names.get("code") or None,
                name=names.get("netplay") or None,
            )
        )
    players.sort(key=lambda p: p.port)

    outcome: GameOutcome | None = None
    end = g.end
    if end is not None:
        end_method = EndMethod(int(end.method))
        lras = end.lras_initiator
        lras_port = _peppi_port_to_libmelee(lras) if lras is not None else None
        try:
            outcome = GameOutcome(end_method=end_method, lras_initiator=lras_port)
        except ValueError:
            # peppi reported an inconsistent end block (lras_initiator on a non-NO_CONTEST
            # game). Drop the row rather than fabricate a fix.
            return None

    last_frame = md.get("lastFrame")
    frame_count = int(last_frame) if last_frame is not None else 0

    return ReplayIndexEntry(
        path=str(replay_path.resolve()),
        slp_version=tuple(g.start.slippi.version),
        stage=int(g.start.stage),
        players=players,
        frame_count=frame_count,
        timestamp=md.get("startAt"),
        played_on=md.get("playedOn"),
        outcome=outcome,
        rank_filename=_rank_from_filename(replay_path),
        sha1=_sha1(replay_path) if compute_sha1 else None,
    )


def write_jsonl(path: Path, entries: list[ReplayIndexEntry], *, append: bool = False) -> None:
    mode = "a" if append else "w"
    with path.open(mode) as f:
        for entry in entries:
            f.write(json.dumps(entry.to_dict()) + "\n")


def read_jsonl(path: Path, *, verify_schema_version: bool = True) -> Iterator[ReplayIndexEntry]:
    """Read a jsonl of ``ReplayIndexEntry`` rows.

    When ``verify_schema_version`` is True (default), each row that carries a
    ``Stage3Annotation`` must have ``schema_version == hal.data.schema.SCHEMA_VERSION``.
    Mismatch raises — a manifest from a different schema build is not safe to
    co-mingle with shards built against the current schema. Unannotated rows
    (Stage 1 index) skip the check.
    """
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = ReplayIndexEntry.from_dict(json.loads(line))
            if (
                verify_schema_version
                and entry.annotation is not None
                and entry.annotation.schema_version != SCHEMA_VERSION
            ):
                raise ValueError(
                    f"manifest {path} row schema_version={entry.annotation.schema_version} "
                    f"!= SCHEMA_VERSION={SCHEMA_VERSION}. Rerun process_replays."
                )
            yield entry
