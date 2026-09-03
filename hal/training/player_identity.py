"""Portable ego-player identities for compact policy training.

Professional identity comes from the exact Slippi connect code stored in each
canonical manifest. Ranked-anonymous rows use their three public rank labels.
Nicknames are retained only for reports: they are not identity keys.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Final

import numpy as np

from hal.data.policy_schema import policy_replay_identity
from hal.data.schema import Rank

PLAYER_IDENTITY_SCHEMA_VERSION: Final[int] = 1
MASKED_PLAYER_ID: Final[int] = 0
FIRST_CONNECT_CODE_ID: Final[int] = int(Rank.MASTER) + 1
RANK_PLAYER_IDS: Final[frozenset[int]] = frozenset({int(Rank.PLATINUM), int(Rank.DIAMOND), int(Rank.MASTER)})


def _trimmed(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def encode_player_codes(codes: tuple[str, ...]) -> bytes:
    """Encode the ordered connect-code vocabulary for checkpoint storage."""
    return json.dumps(codes, ensure_ascii=False, separators=(",", ":")).encode()


def decode_player_codes(payload: bytes) -> tuple[str, ...]:
    """Decode and validate an ordered connect-code vocabulary."""
    values = json.loads(payload)
    if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values):
        raise ValueError("player-code vocabulary must be a list of non-empty strings")
    codes = tuple(values)
    if codes != tuple(sorted(set(codes))):
        raise ValueError("player-code vocabulary must be sorted and unique")
    return codes


def player_code_sha256(codes: tuple[str, ...]) -> str:
    return hashlib.sha256(encode_player_codes(codes)).hexdigest()


@dataclass(frozen=True, slots=True)
class PlayerVocabulary:
    """The train-only identity vocabulary embedded in every O49 checkpoint."""

    codes: tuple[str, ...]
    display_names: tuple[str | None, ...] = ()

    def __post_init__(self) -> None:
        if self.codes != tuple(sorted(set(self.codes))):
            raise ValueError("connect codes must be sorted and unique")
        if self.display_names and len(self.display_names) != len(self.codes):
            raise ValueError("one display name is required per connect code")

    @property
    def size(self) -> int:
        return FIRST_CONNECT_CODE_ID + len(self.codes)

    @property
    def sha256(self) -> str:
        return player_code_sha256(self.codes)

    def id_for_code(self, connect_code: str) -> int:
        """Resolve one exact connect code, raising rather than masking an OOV request."""
        code = _trimmed(connect_code)
        if code is None:
            raise ValueError("connect code must be non-empty")
        try:
            return FIRST_CONNECT_CODE_ID + self.codes.index(code)
        except ValueError as error:
            raise KeyError(f"connect code {code!r} is absent from the training vocabulary") from error

    def id_for_rank(self, rank: Rank) -> int:
        if int(rank) not in RANK_PLAYER_IDS:
            raise ValueError(f"rank conditioning requires Platinum, Diamond, or Master; got {rank!r}")
        return int(rank)


@dataclass(frozen=True, slots=True)
class PlayerIdentitySidecar:
    """Professional replay identities plus their portable train vocabulary."""

    vocabulary: PlayerVocabulary
    by_replay: dict[str, tuple[int, int]]
    header: Mapping[str, Any]
    sha256: str


@dataclass(frozen=True, slots=True)
class ReplayPlayerLookup:
    """Attach p1/p2 IDs before the existing dataloader samples an ego side."""

    professional: Mapping[str, tuple[int, int]]

    def ids(self, compact: Mapping[str, object]) -> tuple[int, int]:
        """Return both IDs without allocating per-frame columns."""
        replay_id = str(compact["replay_id"])
        if replay_id in self.professional:
            return self.professional[replay_id]
        ranks = (
            int(np.asarray(compact["p1_rank"]).item()),
            int(np.asarray(compact["p2_rank"]).item()),
        )
        if set(ranks) - RANK_PLAYER_IDS:
            raise KeyError(
                f"replay {replay_id} is absent from the professional sidecar and has unsupported ranks {ranks}"
            )
        return ranks

    def __call__(self, compact: Mapping[str, object]) -> dict[str, np.ndarray]:
        frames = int(np.asarray(compact["num_frames"]).item())
        p1_id, p2_id = self.ids(compact)
        return {
            "p1_player_id": np.full(frames, p1_id, dtype=np.int32),
            "p2_player_id": np.full(frames, p2_id, dtype=np.int32),
        }


@dataclass(frozen=True, slots=True)
class ManifestInput:
    name: str
    path: Path


@dataclass(frozen=True, slots=True)
class _ManifestRow:
    replay_id: str
    split: str
    codes: tuple[str | None, str | None]
    names: tuple[str | None, str | None]


def _manifest_rows(source: ManifestInput) -> tuple[list[_ManifestRow], str, dict[str, Counter[str]]]:
    rows: list[_ManifestRow] = []
    digest = hashlib.sha256()
    names_by_code: dict[str, Counter[str]] = {}
    with source.path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            digest.update(line)
            if not line.strip():
                continue
            raw = json.loads(line)
            annotation = raw.get("annotation")
            if annotation is None:
                continue
            split = str(annotation["split"])
            players = sorted(raw.get("players", ()), key=lambda player: int(player["port"]))
            if len(players) != 2 or len({int(player["port"]) for player in players}) != 2:
                raise ValueError(f"{source.path}:{line_number}: expected two distinct occupied ports")
            codes = (_trimmed(players[0].get("code")), _trimmed(players[1].get("code")))
            names = (_trimmed(players[0].get("name")), _trimmed(players[1].get("name")))
            if split == "train":
                for name, code in zip(names, codes, strict=True):
                    if code is not None and name is not None:
                        names_by_code.setdefault(code, Counter())[name] += 1
            rows.append(
                _ManifestRow(
                    replay_id=policy_replay_identity(str(raw["path"])),
                    split=split,
                    codes=codes,
                    names=names,
                )
            )
    return rows, digest.hexdigest(), names_by_code


def _write_gzip_lines(path: Path, lines: Iterable[bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
        for line in lines:
            compressed.write(line)
            compressed.write(b"\n")
    temporary.replace(path)


def build_player_identity_sidecar(inputs: Iterable[ManifestInput], output: Path) -> dict[str, Any]:
    """Build a deterministic professional replay-to-connect-code sidecar."""
    sources = tuple(sorted(inputs, key=lambda item: item.name))
    if not sources or len({source.name for source in sources}) != len(sources):
        raise ValueError("manifest inputs must be non-empty and have unique names")

    all_rows: list[_ManifestRow] = []
    manifest_hashes: dict[str, str] = {}
    names_by_code: dict[str, Counter[str]] = {}
    for source in sources:
        rows, digest, source_names = _manifest_rows(source)
        all_rows.extend(rows)
        manifest_hashes[source.name] = digest
        for code, counts in source_names.items():
            names_by_code.setdefault(code, Counter()).update(counts)

    replay_ids = [row.replay_id for row in all_rows]
    if len(set(replay_ids)) != len(replay_ids):
        duplicates = [replay_id for replay_id, count in Counter(replay_ids).items() if count > 1]
        raise ValueError(f"professional manifests repeat replay IDs: {duplicates[:3]}")

    codes = tuple(sorted({code for row in all_rows if row.split == "train" for code in row.codes if code is not None}))
    code_to_id = {code: FIRST_CONNECT_CODE_ID + index for index, code in enumerate(codes)}
    display_names = tuple(
        names_by_code[code].most_common(1)[0][0] if names_by_code.get(code) else None for code in codes
    )
    split_rows = Counter(row.split for row in all_rows)
    split_sides = Counter({split: count * 2 for split, count in split_rows.items()})
    split_present_codes = Counter(row.split for row in all_rows for code in row.codes if code is not None)
    split_present_names = Counter(row.split for row in all_rows for name in row.names if name is not None)
    split_in_vocabulary = Counter(
        row.split for row in all_rows for code in row.codes if code is not None and code in code_to_id
    )
    exact_names = {name for row in all_rows for name in row.names if name is not None}
    casefolded_names = {name.casefold() for name in exact_names}
    casefolded_codes = Counter(code.casefold() for code in codes)
    vocabulary = PlayerVocabulary(codes, display_names)
    header: dict[str, Any] = {
        "schema_version": PLAYER_IDENTITY_SCHEMA_VERSION,
        "identity_key": "exact_connect_code_after_outer_whitespace_trim",
        "nicknames_are_identity_keys": False,
        "missing_or_oov_player_id": MASKED_PLAYER_ID,
        "source_names": [source.name for source in sources],
        "manifest_sha256": manifest_hashes,
        "rows": dict(sorted(split_rows.items())),
        "player_sides": dict(sorted(split_sides.items())),
        "present_connect_code_sides": dict(sorted(split_present_codes.items())),
        "present_nickname_sides": dict(sorted(split_present_names.items())),
        "in_training_vocabulary_sides": dict(sorted(split_in_vocabulary.items())),
        "unique_connect_codes_all_splits": len({code for row in all_rows for code in row.codes if code is not None}),
        "unique_connect_codes_train": len(codes),
        "casefold_connect_code_collision_groups_train": sum(count > 1 for count in casefolded_codes.values()),
        "unique_nicknames_exact_all_splits": len(exact_names),
        "unique_nicknames_casefolded_for_audit_only": len(casefolded_names),
        "codes": list(codes),
        "display_names": list(display_names),
        "vocabulary_size": vocabulary.size,
        "vocabulary_sha256": vocabulary.sha256,
    }

    def lines() -> Iterable[bytes]:
        yield json.dumps(header, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        for row in sorted(all_rows, key=lambda item: item.replay_id):
            ids = [MASKED_PLAYER_ID if code is None else code_to_id.get(code, MASKED_PLAYER_ID) for code in row.codes]
            yield json.dumps([row.replay_id, *ids], separators=(",", ":")).encode()

    _write_gzip_lines(output, lines())
    return {**header, "output": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


def load_player_identity_sidecar(path: Path, *, expected_sha256: str | None = None) -> PlayerIdentitySidecar:
    """Load and validate a sidecar without changing any identity semantics."""
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"player sidecar SHA-256 {digest} != expected {expected_sha256}")
    try:
        lines = gzip.decompress(payload).splitlines()
    except (EOFError, OSError) as error:
        raise ValueError(f"{path} is not a valid gzip sidecar") from error
    if not lines:
        raise ValueError(f"{path} is empty")
    header = json.loads(lines[0])
    if header.get("schema_version") != PLAYER_IDENTITY_SCHEMA_VERSION:
        raise ValueError(f"unsupported player sidecar header: {header}")
    codes = tuple(header.get("codes", ()))
    display_names = tuple(header.get("display_names", ()))
    vocabulary = PlayerVocabulary(codes, display_names)
    if header.get("vocabulary_size") != vocabulary.size or header.get("vocabulary_sha256") != vocabulary.sha256:
        raise ValueError("player sidecar vocabulary metadata is inconsistent")
    expected_rows = sum(int(value) for value in header.get("rows", {}).values())
    if len(lines) - 1 != expected_rows:
        raise ValueError(f"player sidecar has {len(lines) - 1} rows, expected {expected_rows}")

    by_replay: dict[str, tuple[int, int]] = {}
    for line_number, line in enumerate(lines[1:], start=2):
        row = json.loads(line)
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError(f"player sidecar line {line_number} must be [replay_id, p1_id, p2_id]")
        replay_id, p1_id, p2_id = row
        if not isinstance(replay_id, str) or len(replay_id) != 32:
            raise ValueError(f"player sidecar line {line_number} has an invalid replay ID")
        ids = (int(p1_id), int(p2_id))
        if any(value < 0 or value >= vocabulary.size for value in ids):
            raise ValueError(f"player sidecar line {line_number} has invalid player IDs {ids}")
        if replay_id in by_replay:
            raise ValueError(f"player sidecar repeats replay ID {replay_id}")
        by_replay[replay_id] = ids
    return PlayerIdentitySidecar(vocabulary, by_replay, header, digest)


def vocabulary_from_checkpoint_buffer(value: np.ndarray | bytes) -> PlayerVocabulary:
    payload = value if isinstance(value, bytes) else np.asarray(value, dtype=np.uint8).tobytes()
    return PlayerVocabulary(decode_player_codes(payload))


def vocabulary_buffer(vocabulary: PlayerVocabulary) -> np.ndarray:
    return np.frombuffer(encode_player_codes(vocabulary.codes), dtype=np.uint8).copy()
