"""Validated values shared by the netplay API and runner."""

import json
import math
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final
from typing import cast

from hal.inference.api import DESIRED_RETURN_RANGE
from hal.netplay_service.roster import PROFESSIONAL_ROSTER


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
    Choice("MASKED", "No player identity"),
    *(Choice(code, label) for code, label, _ in PROFESSIONAL_ROSTER),
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
# One window bounds both connecting and choosing a rematch; a player may idle this long before the slot is released.
CONNECT_TIMEOUT_SECONDS: Final[int] = 60
IDLE_TIMEOUT_SECONDS: Final[int] = 600
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


def validate_desired_return(value: float | None) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        or not DESIRED_RETURN_RANGE[0] <= value <= DESIRED_RETURN_RANGE[1]
    ):
        raise ValueError("desired_return must be in [-20, 140] or null")
    return float(value)


def validate_temperature(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        or not 0.8 <= value <= 1.1
    ):
        raise ValueError("temperature must be in [0.8, 1.1]")
    return float(value)


@dataclass(frozen=True, slots=True)
class MatchChoices:
    character: str
    imitation: str
    online_delay: int
    requested_stage: str | None = None
    desired_return: float | None = 20.0
    temperature: float = 1.0

    def __post_init__(self) -> None:
        validate_character(self.character)
        validate_imitation(self.imitation)
        validate_delay(self.online_delay)
        if self.requested_stage is not None:
            validate_stage(self.requested_stage)
        validate_desired_return(self.desired_return)
        validate_temperature(self.temperature)


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
    policy_revision: int = 0


@dataclass(frozen=True, slots=True)
class JobCredentials:
    job: Job
    token: str


POLICY_CONFIG_VERSION: Final[int] = 1
_SHA256: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_POLICY_FIELDS: Final[frozenset[str]] = frozenset(
    (
        "schema_version",
        "bundle_sha256",
        "bundle_r2_key",
        "vocabulary_sha256",
        "characters",
        "imitations",
        "stages",
        "online_delays",
        "desired_return_range",
        "default_desired_return",
        "temperature_range",
        "default_temperature",
        "masked_identity",
    )
)


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    """The active policy as published to the queue Worker, which checks the same rules."""

    bundle_sha256: str
    bundle_r2_key: str
    vocabulary_sha256: str
    characters: tuple[Choice, ...]
    imitations: tuple[Choice, ...]
    stages: tuple[Choice, ...]
    online_delays: tuple[int, ...]
    desired_return_range: tuple[float, float]
    default_desired_return: float
    temperature_range: tuple[float, float]
    default_temperature: float
    masked_identity: bool

    def __post_init__(self) -> None:
        for name, value in (("bundle_sha256", self.bundle_sha256), ("vocabulary_sha256", self.vocabulary_sha256)):
            if _SHA256.fullmatch(value) is None:
                raise ValueError(f"policy config {name} must be lowercase SHA-256 hex")
        if not self.bundle_r2_key:
            raise ValueError("policy config bundle_r2_key must be non-empty")
        # The runner parses every leased job against the static domain, so a published choice or range
        # outside it would let the Worker lease a job that the runner then rejects.
        for name, choices, supported in (
            ("characters", self.characters, CHARACTER_VALUES),
            ("imitations", self.imitations, IMITATION_VALUES),
            ("stages", self.stages, STAGE_VALUES),
        ):
            values = [choice.value for choice in choices]
            if not values or len(set(values)) != len(values) or not all(c.value and c.label for c in choices):
                raise ValueError(f"policy config {name} must be non-empty, labeled, and unique")
            if not set(values) <= supported:
                raise ValueError(f"policy config {name} has unsupported values {sorted(set(values) - supported)}")
        if not self.online_delays or any(delay not in (2, 3) for delay in self.online_delays):
            raise ValueError("policy config online_delays must be a non-empty subset of (2, 3)")
        for name, (low, high), default in (
            ("desired_return", self.desired_return_range, self.default_desired_return),
            ("temperature", self.temperature_range, self.default_temperature),
        ):
            if not (math.isfinite(low) and math.isfinite(high) and low < high and low <= default <= high):
                raise ValueError(f"policy config {name} range and default are invalid")
        validate_desired_return(self.desired_return_range[0])
        validate_desired_return(self.desired_return_range[1])
        validate_temperature(self.temperature_range[0])
        validate_temperature(self.temperature_range[1])

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLICY_CONFIG_VERSION,
            "bundle_sha256": self.bundle_sha256,
            "bundle_r2_key": self.bundle_r2_key,
            "vocabulary_sha256": self.vocabulary_sha256,
            "characters": [{"value": choice.value, "label": choice.label} for choice in self.characters],
            "imitations": [{"value": choice.value, "label": choice.label} for choice in self.imitations],
            "stages": [{"value": choice.value, "label": choice.label} for choice in self.stages],
            "online_delays": list(self.online_delays),
            "desired_return_range": list(self.desired_return_range),
            "default_desired_return": self.default_desired_return,
            "temperature_range": list(self.temperature_range),
            "default_temperature": self.default_temperature,
            "masked_identity": self.masked_identity,
        }

    @classmethod
    def from_payload(cls, payload: object) -> PolicyConfig:
        if not isinstance(payload, dict) or set(payload) != _POLICY_FIELDS:
            raise ValueError("policy config fields changed")
        fields = cast(dict[str, object], payload)
        if _integer(fields["schema_version"], "schema_version") != POLICY_CONFIG_VERSION:
            raise ValueError(f"policy config schema_version must be {POLICY_CONFIG_VERSION}")
        return cls(
            bundle_sha256=_text(fields["bundle_sha256"], "bundle_sha256"),
            bundle_r2_key=_text(fields["bundle_r2_key"], "bundle_r2_key"),
            vocabulary_sha256=_text(fields["vocabulary_sha256"], "vocabulary_sha256"),
            characters=_choices(fields["characters"], "characters"),
            imitations=_choices(fields["imitations"], "imitations"),
            stages=_choices(fields["stages"], "stages"),
            online_delays=tuple(
                _integer(delay, "online_delays") for delay in _list(fields["online_delays"], "online_delays")
            ),
            desired_return_range=_range(fields["desired_return_range"], "desired_return_range"),
            default_desired_return=_number(fields["default_desired_return"], "default_desired_return"),
            temperature_range=_range(fields["temperature_range"], "temperature_range"),
            default_temperature=_number(fields["default_temperature"], "default_temperature"),
            masked_identity=_boolean(fields["masked_identity"], "masked_identity"),
        )


def _text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _number(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    return float(value)


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return cast(list[object], value)


def _range(value: object, name: str) -> tuple[float, float]:
    items = _list(value, name)
    if len(items) != 2:
        raise ValueError(f"{name} must have two numbers")
    return _number(items[0], name), _number(items[1], name)


def _choices(value: object, name: str) -> tuple[Choice, ...]:
    choices: list[Choice] = []
    for item in _list(value, name):
        if not isinstance(item, dict) or set(item) != {"value", "label"}:
            raise ValueError(f"{name} entries need exactly value and label")
        entry = cast(dict[str, object], item)
        choices.append(Choice(_text(entry["value"], name), _text(entry["label"], name)))
    return tuple(choices)


def account_connect_code(path: Path) -> str:
    """Read the connect code from a Slippi user.json without falling back to defaults."""
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read Slippi account JSON {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Slippi account JSON {path} must contain an object")
    value = payload.get("connectCode")
    if not isinstance(value, str):
        raise ValueError(f"Slippi account JSON {path} has no connectCode")
    return validate_player_code(value)
