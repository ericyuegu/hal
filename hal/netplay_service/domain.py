"""Validated values shared by the netplay API and runner."""

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Final


@dataclass(frozen=True, slots=True)
class Choice:
    value: str
    label: str


CHARACTERS: Final[tuple[Choice, ...]] = (
    Choice("FOX", "Fox"),
    Choice("FALCO", "Falco"),
    Choice("MARTH", "Marth"),
    Choice("SHEIK", "Sheik"),
    Choice("JIGGLYPUFF", "Jigglypuff"),
    Choice("PEACH", "Peach"),
    Choice("CPTFALCON", "Captain Falcon"),
    Choice("POPO", "Ice Climbers"),
    Choice("PIKACHU", "Pikachu"),
    Choice("SAMUS", "Samus"),
    Choice("YOSHI", "Yoshi"),
    Choice("LUIGI", "Luigi"),
    Choice("GANONDORF", "Ganondorf"),
    Choice("MARIO", "Mario"),
    Choice("DOC", "Dr. Mario"),
    Choice("LINK", "Link"),
    Choice("YOUNG_LINK", "Young Link"),
    Choice("ZELDA", "Zelda"),
    Choice("MEWTWO", "Mewtwo"),
    Choice("NESS", "Ness"),
    Choice("ROY", "Roy"),
    Choice("GAMEANDWATCH", "Mr. Game & Watch"),
    Choice("PICHU", "Pichu"),
    Choice("DK", "Donkey Kong"),
    Choice("BOWSER", "Bowser"),
    Choice("KIRBY", "Kirby"),
)

IMITATIONS: Final[tuple[Choice, ...]] = (
    Choice("IBDW#0", "iBDW"),
    Choice("ZAIN#0", "Zain"),
    Choice("MANG#0", "Mang0"),
    Choice("LEFFEN#0", "Leffen"),
    Choice("PIPLUP#0", "Pipsqueak"),
    Choice("AMSA#0", "aMSa"),
    Choice("HBOX#1", "Hungrybox"),
    Choice("PLATINUM", "Platinum rank"),
    Choice("DIAMOND", "Diamond rank"),
    Choice("MASTER", "Master rank"),
)

STAGES: Final[tuple[Choice, ...]] = (
    Choice("BATTLEFIELD", "Battlefield"),
    Choice("FINAL_DESTINATION", "Final Destination"),
    Choice("DREAMLAND", "Dream Land"),
    Choice("POKEMON_STADIUM", "Pokémon Stadium"),
    Choice("YOSHIS_STORY", "Yoshi's Story"),
    Choice("FOUNTAIN_OF_DREAMS", "Fountain of Dreams"),
)

CHARACTER_VALUES: Final[frozenset[str]] = frozenset(choice.value for choice in CHARACTERS)
IMITATION_VALUES: Final[frozenset[str]] = frozenset(choice.value for choice in IMITATIONS)
STAGE_VALUES: Final[frozenset[str]] = frozenset(choice.value for choice in STAGES)
_PLAYER_CODE = re.compile(r"[A-Z0-9]{1,8}#[0-9]{1,4}")


class JobStatus(StrEnum):
    QUEUED = "queued"
    LEASED = "leased"
    CONNECTING = "connecting"
    PLAYING = "playing"
    REMATCH_WAIT = "rematch_wait"
    REMATCH_READY = "rematch_ready"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELED = "canceled"
    NO_SHOW = "no_show"


TERMINAL_STATUSES: Final[frozenset[JobStatus]] = frozenset(
    (JobStatus.COMPLETE, JobStatus.FAILED, JobStatus.CANCELED, JobStatus.NO_SHOW)
)


def validate_player_code(value: str) -> str:
    if _PLAYER_CODE.fullmatch(value) is None:
        raise ValueError("player_code must be an exact uppercase Slippi connect code such as CRYO#610")
    return value


def validate_character(value: str) -> str:
    if value not in CHARACTER_VALUES:
        raise ValueError(f"unsupported character {value!r}")
    return value


def validate_imitation(value: str) -> str:
    if value not in IMITATION_VALUES:
        raise ValueError(f"unsupported imitation {value!r}")
    return value


def validate_stage(value: str) -> str:
    if value not in STAGE_VALUES:
        raise ValueError(f"unsupported stage {value!r}")
    return value


def validate_delay(value: int) -> int:
    if value not in (2, 3) or isinstance(value, bool):
        raise ValueError("online_delay must be 2 or 3")
    return value


@dataclass(frozen=True, slots=True)
class MatchChoices:
    character: str
    imitation: str
    online_delay: int
    requested_stage: str | None = None

    def __post_init__(self) -> None:
        validate_character(self.character)
        validate_imitation(self.imitation)
        validate_delay(self.online_delay)
        if self.requested_stage is not None:
            validate_stage(self.requested_stage)


@dataclass(frozen=True, slots=True)
class Job:
    id: str
    player_code: str
    choices: MatchChoices
    status: JobStatus
    queue_position: int | None
    attempt: int
    game_count: int
    connect_code: str | None
    actual_stage: str | None
    last_result: str | None
    error_code: str | None
    connect_deadline: float | None
    rematch_deadline: float | None
    cancel_after_game: bool
    lease_owner: str | None
    lease_expires_at: float | None
    created_at: float
    updated_at: float


@dataclass(frozen=True, slots=True)
class JobCredentials:
    job: Job
    token: str
