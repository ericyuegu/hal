"""Per-replay metadata schema used by build_index → filter_replays → process_replays.

A `ReplayIndexEntry` describes everything we know about a single .slp file
*without* iterating its frames: slp version, stage, players (port + character +
costume + netplay code/name), match length, end outcome, timestamps. It is the
load-bearing artifact for filtering datasets by rank/character/version/etc., and
for closed-loop compatibility checks at training/eval time (slp_version drift
matters for bit-exact reproduction).

Three-stage pipeline
--------------------

The data pipeline separates *what slps exist*, *which slps to use for this
dataset*, and *how to extract frames from them*:

1. **`build_index.py`** — walks a replay tree once and emits `index.jsonl`,
   one `ReplayIndexEntry` per slp. Reads only the start/end/metadata blocks
   (no frame iteration), so it scales to 100k+ replays in minutes. Reusable
   across arbitrarily many filtered datasets. `annotation` is None on every
   row at this stage.
2. **`filter_replays.py`** — pure function on `index.jsonl`. Composable
   predicates (rank, characters, slp version, frame count, completion)
   produce a `paths.txt` for Stage 3. No slp opens; cuts are seconds.
3. **`process_replays.py`** — reads `paths.txt`, parses each slp's frame data
   in parallel (peppi-py), writes MDS shards plus a `manifest.jsonl` sidecar.
   Manifest rows are the index rows that actually landed in MDS, each with
   `annotation` populated (split, mds_row_idx, frame_count_actual,
   replay_uuid).

So the same dataclass serves two artifacts:

- `index.jsonl` — built once by Stage 1; `annotation` is None on each row.
- `manifest.jsonl` — written by Stage 3 alongside MDS shards; rows are a
  subset of the index, with `annotation` populated.

LRAS = "L + R + A + Start", the GameCube controller combo a player holds to
forfeit a Melee match. It is only meaningful when the slp ended via
NO_CONTEST; that invariant is enforced by `GameOutcome`.

Conventions
-----------

We default to **slp-native** integer ids (what peppi-py exposes, matching the
bytes on disk) for: `stage`, `character`, `costume`, `slp_version`,
`end_method`, `frame_count`. The one libmelee accommodation: **ports are
stored 1..4** (libmelee convention) rather than peppi's 0..3.

Footgun: `stage` is the slp-native id (e.g. Fountain of Dreams = 2). libmelee's
`melee.Stage.FOUNTAIN_OF_DREAMS.value` is 8 — a different remapping. Comparing
`entry.stage == melee.Stage.FOUNTAIN_OF_DREAMS.value` will silently misbehave.
Use the slp-native id directly, or convert via a stage-id table at the
consumption site. Character ids are also slp-native but happen to coincide with
libmelee's enum values (verified against 56k+ frames in
`notebooks/peppi_vs_libmelee.py`).

Some slps are unparseable by peppi-rs but readable by libmelee — the most
common cause is an out-of-spec `end.method` byte (e.g. value 9 in the wild,
not in peppi's exhaustive enum {0,1,2,3,7}). `extract_index_entry` returns
None for these; observed rate is ~0.5% on the mang0 corpus.
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

import peppi_py


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

    Either all four fields are populated together or the entire annotation is
    None — the per-field nullability of the previous design is no longer
    representable.
    """

    replay_uuid: int
    split: Split
    mds_row_idx: int
    frame_count_actual: int

    def __post_init__(self) -> None:
        if self.split not in ("train", "val", "test"):
            raise ValueError(f"split must be one of train/val/test; got {self.split!r}")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Stage3Annotation:
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
    sha1_partial: str | None  # first 4KB sha1 hex digest (for dedupe)

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
            "sha1_partial": self.sha1_partial,
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
            sha1_partial=data.get("sha1_partial"),
            annotation=Stage3Annotation.from_dict(annotation) if annotation is not None else None,
        )


def replay_uuid_from_path(path: str | Path) -> int:
    """Stable int32 hash of an absolute replay path."""
    digest = hashlib.md5(str(path).encode()).digest()
    return struct.unpack("i", digest[:4])[0]


def _sha1_partial(path: Path, n_bytes: int = 4096) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        h.update(f.read(n_bytes))
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
    if name not in ("HUMAN", "CPU", "DEMO"):
        raise ValueError(f"unknown player type: {name!r}")
    return name  # type: ignore[return-value]


def _peppi_port_to_libmelee(p: Any) -> int:
    """peppi Port enum is 0-indexed (P1=0); libmelee uses 1..4."""
    return int(getattr(p, "value", p)) + 1


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
        sha1_partial=_sha1_partial(replay_path) if compute_sha1 else None,
    )


def write_jsonl(path: Path, entries: list[ReplayIndexEntry], *, append: bool = False) -> None:
    mode = "a" if append else "w"
    with path.open(mode) as f:
        for entry in entries:
            f.write(json.dumps(entry.to_dict()) + "\n")


def read_jsonl(path: Path) -> Iterator[ReplayIndexEntry]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield ReplayIndexEntry.from_dict(json.loads(line))
