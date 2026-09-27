"""Player identity encoding shared by training, artifacts, and inference."""

import hashlib
import json
from dataclasses import dataclass
from typing import Final

import numpy as np

from hal.data.schema import Rank

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
    """The ordered identity vocabulary embedded in policy checkpoints."""

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


def vocabulary_from_checkpoint_buffer(value: np.ndarray | bytes) -> PlayerVocabulary:
    payload = value if isinstance(value, bytes) else np.asarray(value, dtype=np.uint8).tobytes()
    return PlayerVocabulary(decode_player_codes(payload))


def vocabulary_buffer(vocabulary: PlayerVocabulary) -> np.ndarray:
    return np.frombuffer(encode_player_codes(vocabulary.codes), dtype=np.uint8).copy()
