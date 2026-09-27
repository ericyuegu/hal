"""Mirrored model matches, replay identity, and paired outcome records."""

import json
import struct
import time
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any
from typing import Final
from typing import Literal

import melee
import numpy as np
import torch
from loguru import logger

from hal.data.extract import extract_replay
from hal.data.slp_finalize import trim_to_last_frame
from hal.eval.cross_stage import STARTING_STOCKS
from hal.eval.harness import SessionConfig
from hal.eval.harness import default_session_cfg
from hal.eval.harness import run_matches_vec
from hal.eval.harness import usable_cpus
from hal.eval.match_summary import summarize_trajectory
from hal.eval.matchups import MatchConfig
from hal.eval.matchups import mirrored_configs
from hal.eval.policy import PolicyBatchAdapter
from hal.eval.policy import PolicySettings
from hal.eval.scheduling import FrameTiming
from hal.inference.action_sequence_artifact import read_action_sequence_artifact
from hal.inference.action_sequence_policy import ActionSequencePolicy
from hal.inference.api import RuntimeConfig
from hal.inference.checkpoints import resolve_checkpoint
from hal.inference.engine import ModelRegistry
from hal.policy import INCLUDED_STAGES
from hal.sim.rollout import ChunkPolicy
from hal.sim.rollout import ObservationRow
from hal.sim.rollout import PolicyRuntimeSpec
from hal.sim.rollout import Slot
from hal.sim.rollout import VecMatch
from hal.sim.session import Matchup
from hal.sim.session import PlayerSetup
from hal.sim.trajectory import Trajectory
from hal.wire import BUTTON_BITS

PolicyBuilder = Callable[[Mapping[int, str], int], ChunkPolicy]


@dataclass(frozen=True, slots=True)
class H2HModel:
    name: str
    artifact: str
    player_identity: str | None = "IBDW#0"
    desired_return: float | None = None
    temperature: float = 1.0


class H2HPolicyFactory:
    def __init__(
        self,
        policies_by_name: Mapping[str, ActionSequencePolicy],
        settings_by_name: Mapping[str, PolicySettings],
        runtime_by_checkpoint: Mapping[str, RuntimeConfig],
        timing: FrameTiming,
        *,
        observed_actions: bool,
    ) -> None:
        self.policies_by_name = dict(policies_by_name)
        self.settings_by_name = dict(settings_by_name)
        self.runtime_by_checkpoint = dict(runtime_by_checkpoint)
        self.timing = timing
        self.observed_actions = observed_actions

    def __call__(self, model_by_port: Mapping[int, str], seed: int) -> H2HPolicy:
        settings_by_checkpoint: dict[str, dict[int, PolicySettings]] = {}
        policies_by_checkpoint: dict[str, ActionSequencePolicy] = {}
        for port, name in model_by_port.items():
            policy = self.policies_by_name[name]
            checkpoint = policy.checkpoint_sha256
            policies_by_checkpoint[checkpoint] = policy
            settings_by_checkpoint.setdefault(checkpoint, {})[port] = self.settings_by_name[name]
        adapters_by_checkpoint = {}
        for index, (checkpoint, policy) in enumerate(policies_by_checkpoint.items()):
            policy.reset_prediction(seed=seed + index)
            adapters_by_checkpoint[checkpoint] = PolicyBatchAdapter(
                policy,
                self.runtime_by_checkpoint[checkpoint],
                self.timing,
                settings_by_port=settings_by_checkpoint[checkpoint],
                observed_actions=self.observed_actions,
            )
        return H2HPolicy(
            {
                port: adapters_by_checkpoint[self.policies_by_name[name].checkpoint_sha256]
                for port, name in model_by_port.items()
            }
        )


def prepare_h2h_policies(
    models: Sequence[H2HModel],
    *,
    max_parallel: int,
    profile: Literal["local", "official-059"],
    device: str,
    compiled: bool,
) -> H2HPolicyFactory:
    if len(models) != 2 or models[0].name == models[1].name:
        raise ValueError("head-to-head needs two distinct model names")
    if max_parallel < 1:
        raise ValueError("max_parallel must be positive")
    is_official = profile == "official-059"
    timing = FrameTiming(0, 0, 2 if is_official else 0, 2, 4)
    artifacts_by_name = {
        model.name: read_action_sequence_artifact(resolve_checkpoint(model.artifact)) for model in models
    }
    registry = ModelRegistry()
    policies_by_checkpoint: dict[str, ActionSequencePolicy] = {}
    policies_by_name: dict[str, ActionSequencePolicy] = {}
    settings_by_name: dict[str, PolicySettings] = {}
    runtime_by_checkpoint: dict[str, RuntimeConfig] = {}
    target = torch.device(device)
    for model in models:
        artifact = artifacts_by_name[model.name]
        if 0 not in artifact.spec.supported_transport_delays:
            raise ValueError("local H2H requires an artifact that declares the zero-delay profile")
        checkpoint = artifact.checkpoint_sha256
        if checkpoint not in policies_by_checkpoint:
            weights = registry.register_artifact(
                artifact,
                device=target,
                inference_dtype=torch.bfloat16 if target.type == "cuda" else torch.float32,
            )
            policy = ActionSequencePolicy(
                weights,
                artifact.statistics,
                artifact.vocabulary.codes,
                spec=artifact.spec,
                checkpoint_sha256=checkpoint,
                return_p90=artifact.return_p90,
                capability_version=artifact.capability_version,
                device=target,
                seed=0,
                compiled=compiled,
                history_mode="window" if is_official else "kv_cache",
                kv_update_frames=2,
            )
            port_count = sum(other.checkpoint_sha256 == checkpoint for other in artifacts_by_name.values())
            runtime = RuntimeConfig(max_parallel * port_count, (0,), replan_interval_frames=2)
            policy.prepare_prediction(runtime, 4, timing.fixed_prefix_frames)
            policies_by_checkpoint[checkpoint] = policy
            runtime_by_checkpoint[checkpoint] = runtime
        policies_by_name[model.name] = policies_by_checkpoint[checkpoint]
        settings_by_name[model.name] = PolicySettings(
            model.player_identity,
            artifact.return_p90 if model.desired_return is None else model.desired_return,
            model.temperature,
        )
    return H2HPolicyFactory(
        policies_by_name, settings_by_name, runtime_by_checkpoint, timing, observed_actions=not is_official
    )


# Per match. 7200 frames = 2 minutes, the training-time closed-loop eval budget.
DEFAULT_MAX_FRAMES: Final[int] = 7200
# libmelee's stage-select cursor navigation is flaky under concurrent fast-forward load.
DEFAULT_START_RETRIES: Final[int] = 3
# Version 2 names stock leaders directly. ``MatchRecord.from_dict`` still reads version 1 fields.
MATCH_RECORD_SCHEMA_VERSION: Final[int] = 3
# Stick magnitude that counts as "the policy moved this stick". Melee's dead-band gate is
# 0.2875 per axis, so 0.3 is just outside it and reads as deliberate motion.
_STICK_ACTIVE_MAGNITUDE: Final[float] = 0.3
# Buttons the action space can press. START is excluded by policy (it pauses the match).
_ACTION_BUTTONS: Final[tuple[str, ...]] = tuple(name for name in BUTTON_BITS if name != "start")


@dataclass(frozen=True, slots=True)
class MatchSpec:
    """One head-to-head match: a config plus which model sits on which port."""

    config: MatchConfig
    orientation: int  # 0 = model A on port 1; 1 = the mirror
    model_port_1: str
    model_port_2: str
    character_port_1: melee.Character
    character_port_2: melee.Character

    @property
    def match_id(self) -> str:
        """Globally unique, self-describing match name (also the replay basename)."""
        return f"config{self.config.config_id:04d}-{path_safe(self.model_port_1)}-on-port1"

    def model_on_port(self, port: int) -> str:
        if port == 1:
            return self.model_port_1
        if port == 2:
            return self.model_port_2
        raise ValueError(f"head-to-head uses ports 1 and 2, got {port}")

    def vec_match(self) -> VecMatch:
        return VecMatch(
            matchup=Matchup(
                stage=self.config.stage,
                players=(
                    PlayerSetup(port=1, character=self.character_port_1, cpu_level=0),
                    PlayerSetup(port=2, character=self.character_port_2, cpu_level=0),
                ),
            ),
            model_ports=(1, 2),
        )


def match_specs(
    configs: Sequence[MatchConfig],
    *,
    name_a: str,
    name_b: str,
    fixed_characters: Mapping[str, melee.Character] | None = None,
) -> list[MatchSpec]:
    """Both orientations of every config, with optional model-bound characters."""
    if name_a == name_b:
        raise ValueError(f"the two models need distinct names, got {name_a!r} twice")
    character_overrides = dict(fixed_characters or {})
    unexpected = character_overrides.keys() - {name_a, name_b}
    if unexpected:
        raise ValueError(f"fixed characters name unknown models: {sorted(unexpected)}")
    specs: list[MatchSpec] = []
    for config in configs:
        for orientation in (0, 1):
            port_1, port_2 = (name_a, name_b) if orientation == 0 else (name_b, name_a)
            specs.append(
                MatchSpec(
                    config=config,
                    orientation=orientation,
                    model_port_1=port_1,
                    model_port_2=port_2,
                    character_port_1=character_overrides.get(port_1, config.character_port_1),
                    character_port_2=character_overrides.get(port_2, config.character_port_2),
                )
            )
    return specs


def path_safe(name: str) -> str:
    """Model name reduced to a filesystem- and URL-safe token, keeping it readable."""
    return "".join(c if c.isalnum() or c in "-_." else "-" for c in name)


# ---------------------------------------------------------------------------
# The two-model batched policy
# ---------------------------------------------------------------------------


class H2HPolicy:
    """Route complete observation chunks, collecting ports that share a policy."""

    def __init__(self, by_port: Mapping[int, ChunkPolicy]) -> None:
        if not by_port:
            raise ValueError("H2HPolicy needs at least one port -> policy entry")
        self.by_port = dict(by_port)
        self._groups: list[tuple[ChunkPolicy, set[int]]] = []
        for port, policy in self.by_port.items():
            for candidate, ports in self._groups:
                if candidate is policy:
                    ports.add(port)
                    break
            else:
                self._groups.append((policy, {port}))
        self._runtime_spec = self._groups[0][0].runtime_spec
        if any(policy.runtime_spec != self._runtime_spec for policy, _ in self._groups):
            raise ValueError("head-to-head policies must use the same rollout timing profile")

    @property
    def runtime_spec(self) -> PolicyRuntimeSpec:
        return self._runtime_spec

    def plan_rows(self, rows: Mapping[Slot, Sequence[ObservationRow]]) -> Mapping[Slot, np.ndarray]:
        actions_by_slot: dict[Slot, np.ndarray] = {}
        for policy, ports in self._groups:
            selected = {slot: values for slot, values in rows.items() if slot.port in ports}
            if not selected:
                continue
            actions = policy.plan_rows(selected)
            if actions.keys() != selected.keys():
                raise RuntimeError("H2HPolicy returned actions for the wrong slots")
            actions_by_slot.update(actions)
        if actions_by_slot.keys() != rows.keys():
            raise RuntimeError("H2HPolicy has no policy for an observed port")
        return actions_by_slot


@dataclass(slots=True)
class _OrientationPolicyFactory:
    builder: PolicyBuilder
    model_by_port: Mapping[int, str]
    seed: int
    wave: int = 0

    def __call__(self) -> ChunkPolicy:
        policy = self.builder(self.model_by_port, self.seed + 2 * self.wave)
        self.wave += 1
        return policy


# ---------------------------------------------------------------------------
# Replay identity stamping
# ---------------------------------------------------------------------------

# UBJSON / Slippi raw-stream layout, the same envelope ``hal.data.slp_finalize`` repairs.
_SLP_HEADER: Final[bytes] = b"{U\x03raw[$U#l"
_RAW_LENGTH_OFFSET: Final[int] = len(_SLP_HEADER)
_RAW_START: Final[int] = _RAW_LENGTH_OFFSET + 4
_EVENT_PAYLOADS: Final[int] = 0x35
_GAME_START: Final[int] = 0x36
# Game Start block offsets from the Slippi replay spec. They count from the event's
# COMMAND BYTE, not from its payload, and each block is indexed by 0-based player index.
# libmelee and peppi read the same two blocks (see melee.console: 0x1A5 + 0x1F * i).
_NAMETAG_OFFSET: Final[int] = 0x161
_NAMETAG_SIZE: Final[int] = 16
_DISPLAY_NAME_OFFSET: Final[int] = 0x1A5
_DISPLAY_NAME_SIZE: Final[int] = 31
# Payload bytes a Game Start event needs to carry the display-name block (3.9.0 and up).
_DISPLAY_NAME_MIN_PAYLOAD: Final[int] = _DISPLAY_NAME_OFFSET - 1 + 4 * _DISPLAY_NAME_SIZE
# Melee shows at most 8 characters of a nametag.
_NAMETAG_CHARACTERS: Final[int] = 8
_METADATA_OPEN: Final[bytes] = b"U\x08metadata{"


def _ub_key(name: str) -> bytes:
    """UBJSON object key: a uint8-length string with no type marker."""
    raw = name.encode("utf-8")
    if len(raw) > 0xFF:
        raise ValueError(f"UBJSON key too long: {name!r}")
    return b"U" + bytes([len(raw)]) + raw


def _ub_string(value: str) -> bytes:
    """UBJSON string value: the ``S`` marker plus a uint8-length string."""
    return b"S" + _ub_key(value)


def _names_object(label: str) -> bytes:
    """The ``names`` member the Slippi launcher reads for its replay listing."""
    return _ub_key("names") + b"{" + _ub_key("netplay") + _ub_string(label) + b"}"


def _event_sizes(data: bytes) -> dict[int, int]:
    """Payload size of every command, from the stream's own EVENT_PAYLOADS event."""
    if data[_RAW_START] != _EVENT_PAYLOADS:
        raise ValueError(f"expected EVENT_PAYLOADS ({_EVENT_PAYLOADS:#x}) at {_RAW_START}, got {data[_RAW_START]:#x}")
    declared = data[_RAW_START + 1]
    sizes = {_EVENT_PAYLOADS: declared}
    cursor = _RAW_START + 2
    for _ in range((declared - 1) // 3):
        sizes[data[cursor]] = struct.unpack_from(">H", data, cursor + 1)[0]
        cursor += 3
    return sizes


def _game_start_offset(data: bytes) -> int:
    """Offset of the Game Start event's command byte, which the block offsets count from."""
    sizes = _event_sizes(data)
    cursor = _RAW_START + 1 + sizes[_EVENT_PAYLOADS]
    while cursor < len(data) and data[cursor] in sizes:
        if data[cursor] == _GAME_START:
            if sizes[_GAME_START] < _DISPLAY_NAME_MIN_PAYLOAD:
                raise ValueError(f"Game Start payload is {sizes[_GAME_START]} bytes; too old to carry display names")
            return cursor
        cursor += 1 + sizes[data[cursor]]
    raise ValueError("no Game Start event in this .slp")


def _write_fixed_field(buffer: bytearray, offset: int, size: int, text: str) -> None:
    """Write ``text`` shift-jis encoded, NUL terminated, into a fixed-width field.

    Truncation happens on a character boundary so a multi-byte character never splits.
    """
    encoded = b""
    for character in text:
        candidate = encoded + character.encode("shift_jis", errors="replace")
        if len(candidate) > size - 1:
            break
        encoded = candidate
    buffer[offset : offset + size] = encoded.ljust(size, b"\x00")


def _nametag(label: str) -> str:
    """In-game nametag form of a label: uppercase alphanumerics, 8 characters."""
    return "".join(c for c in label.upper() if c.isalnum())[:_NAMETAG_CHARACTERS]


def _metadata_with_names(region: bytes, labels: Mapping[int, str]) -> bytes:
    """Insert per-player ``names`` members into the trailing metadata object.

    UBJSON objects here carry no count or type optimization, so a member can be spliced
    in directly after an object's opening brace. The metadata block sits outside the
    ``raw`` length prefix, so the insert needs no length fixup. A block that already
    carries ``names`` is left alone: a second member of the same name would be a
    duplicate key.
    """
    if not region.startswith(_METADATA_OPEN):
        raise ValueError("no metadata object at the end of the raw stream")
    if _ub_key("names") in region:
        return region
    players_key = _ub_key("players") + b"{"
    players_at = region.find(players_key)
    if players_at < 0:
        entries = b"".join(
            _ub_key(str(port - 1)) + b"{" + _names_object(label) + b"}" for port, label in sorted(labels.items())
        )
        return region[: len(_METADATA_OPEN)] + players_key + entries + b"}" + region[len(_METADATA_OPEN) :]
    out = region
    for port, label in sorted(labels.items()):
        player_key = _ub_key(str(port - 1)) + b"{"
        at = out.find(player_key, players_at)
        if at < 0:  # Dolphin recorded no entry for this port; add the whole player object.
            at = players_at + len(players_key)
            out = out[:at] + player_key + _names_object(label) + b"}" + out[at:]
        else:
            at += len(player_key)
            out = out[:at] + _names_object(label) + out[at:]
    return out


def stamp_replay_identity(replay: str | Path, labels: Mapping[int, str]) -> None:
    """Write the policy labels into the .slp player-name fields, in place.

    ``labels`` maps a libmelee port (1..4) to the name of the policy that drove it. The
    label lands in three places a viewer reads: the in-game nametag and the game-start
    display name (both inside the Game Start event) and the metadata ``netplay`` name
    (the launcher's replay listing). Every replay is then self-describing.

    Stamping happens after the match, not at generation time: libmelee can set neither a
    nametag nor a display name for a local versus match, so there is no generation-time
    path to take. The written bytes are fixed width in the Game Start event, so repeated
    stamping is idempotent.

    Raises ``ValueError`` when the file is not a finalized Slippi raw stream.
    """
    path = Path(replay)
    data = bytearray(path.read_bytes())
    if bytes(data[:_RAW_LENGTH_OFFSET]) != _SLP_HEADER:
        raise ValueError(f"{path}: not a Slippi raw stream")
    raw_length = struct.unpack_from(">i", data, _RAW_LENGTH_OFFSET)[0]
    if raw_length <= 0:
        raise ValueError(f"{path}: rawLength is 0; the file must be finalized first")
    start = _game_start_offset(bytes(data))
    for port, label in labels.items():
        index = port - 1
        if not 0 <= index < 4:
            raise ValueError(f"{path}: port {port} is outside the 1..4 range")
        _write_fixed_field(data, start + _NAMETAG_OFFSET + index * _NAMETAG_SIZE, _NAMETAG_SIZE, _nametag(label))
        _write_fixed_field(data, start + _DISPLAY_NAME_OFFSET + index * _DISPLAY_NAME_SIZE, _DISPLAY_NAME_SIZE, label)
    metadata_at = _RAW_START + raw_length
    path.write_bytes(bytes(data[:metadata_at]) + _metadata_with_names(bytes(data[metadata_at:]), labels))


def replay_display_names(replay: str | Path) -> dict[int, str]:
    """Game-start display name per libmelee port, for ports that carry one."""
    data = Path(replay).read_bytes()
    start = _game_start_offset(data)
    out: dict[int, str] = {}
    for index in range(4):
        at = start + _DISPLAY_NAME_OFFSET + index * _DISPLAY_NAME_SIZE
        name = data[at : at + _DISPLAY_NAME_SIZE].split(b"\x00", 1)[0].decode("shift_jis", errors="replace")
        if name:
            out[index + 1] = name
    return out


# ---------------------------------------------------------------------------
# Dead-policy tripwire
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PortInputStats:
    """Controller activity of one port, read back from the recorded .slp.

    A dead or degenerate policy shows near-zero stick motion, near-zero button frames, or
    a single distinct action for the whole match. ``button_start_frac`` must be exactly
    0.0: START is excluded from the action space because it pauses the match.
    """

    main_stick_active_frac: float
    c_stick_active_frac: float
    any_button_frac: float
    button_start_frac: float
    distinct_actions: int


def _peppi_panicked(error: BaseException) -> bool:
    """True for peppi's arrow2 panic.

    pyo3 raises ``PanicException``, a ``BaseException`` whose module (``pyo3_runtime``)
    is synthetic and cannot be imported, so the class is identified by name. It escapes
    the ``except Exception`` inside ``extract_replay``.
    """
    return type(error).__qualname__ == "PanicException"


def replay_input_stats(replay: str | Path) -> dict[int, PortInputStats] | None:
    """Per-port controller activity of a recorded match, or None if it cannot be read.

    A match cut off at the frame budget can end between the two ports' post-frame events.
    ``slp_finalize`` repairs the envelope to the last complete event, which still leaves
    peppi with port columns of unequal length, and peppi panics. Such a replay reports
    None here; the frame-granular repair belongs to the replay-analysis layer.
    """
    try:
        columns = extract_replay(str(replay))
    except BaseException as error:  # peppi panics are BaseExceptions; see _peppi_panicked
        if not _peppi_panicked(error):
            raise
        logger.warning(f"replay_input_stats: peppi panicked on {replay}")
        return None
    if columns is None:
        return None
    out: dict[int, PortInputStats] = {}
    # The START tripwire only counts in-game frames: the pipe->slp latency of +1 frame can
    # land a menu-era press inside the countdown, which is not a policy defect.
    in_game = columns["frame"] >= 0
    for port, prefix in ((1, "p1"), (2, "p2")):
        main_x = columns[f"{prefix}_main_stick_x"]
        main_y = columns[f"{prefix}_main_stick_y"]
        c_x = columns[f"{prefix}_c_stick_x"]
        c_y = columns[f"{prefix}_c_stick_y"]
        buttons = np.stack([columns[f"{prefix}_button_{name}"] for name in _ACTION_BUTTONS], axis=1)
        actions = np.concatenate([np.stack([main_x, main_y, c_x, c_y], axis=1), buttons.astype(np.float32)], axis=1)
        out[port] = PortInputStats(
            main_stick_active_frac=float(np.mean(np.hypot(main_x, main_y) > _STICK_ACTIVE_MAGNITUDE)),
            c_stick_active_frac=float(np.mean(np.hypot(c_x, c_y) > _STICK_ACTIVE_MAGNITUDE)),
            any_button_frac=float(np.mean(buttons.any(axis=1))),
            button_start_frac=float(np.mean(columns[f"{prefix}_button_start"][in_game] > 0)) if in_game.any() else 0.0,
            distinct_actions=int(len(np.unique(actions, axis=0))),
        )
    return out


# ---------------------------------------------------------------------------
# Match records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MatchOutcome:
    """What one completed match produced. Absent when the match never ran."""

    total_frames: int
    active_frames: int  # frames with canonical id >= 0 (post-GO); the frozen protocol
    stocks_left_port_1: int
    stocks_left_port_2: int
    stocks_lost_port_1: int
    stocks_lost_port_2: int
    damage_taken_port_1: float
    damage_taken_port_2: float
    damage_dealt_port_1: float
    damage_dealt_port_2: float
    decided: bool  # a port reached zero stocks, so the match ended on a knockout
    hit_frame_budget: bool
    stock_leader_port: int | None  # None on a stock tie at the budget
    stock_leader_model: str | None

    def stocks_lost(self, port: int) -> int:
        return self.stocks_lost_port_1 if port == 1 else self.stocks_lost_port_2

    def damage_dealt(self, port: int) -> float:
        return self.damage_dealt_port_1 if port == 1 else self.damage_dealt_port_2

    def damage_taken(self, port: int) -> float:
        return self.damage_taken_port_1 if port == 1 else self.damage_taken_port_2


@dataclass(frozen=True, slots=True)
class MatchRecord:
    """One row of ``matches.jsonl``: who played what, where, and with which result.

    A boot that never started still produces a record (``outcome is None``), so the
    analysis sees the dropout instead of a silently shorter sample.
    """

    match_id: str
    config_id: int
    orientation: int
    boot_index: int  # equal to config_id: one match per boot, one boot per config
    stage: str
    stage_id: int
    model_port_1: str
    model_port_2: str
    character_port_1: str
    character_port_2: str
    character_id_port_1: int  # libmelee-internal character ids, the space the model uses
    character_id_port_2: int
    replay_path: str | None
    replay_status: str  # "ok" | "unreadable" | "missing"
    identity_stamped: bool
    outcome: MatchOutcome | None
    input_stats_port_1: PortInputStats | None
    input_stats_port_2: PortInputStats | None
    replay_trimmed: bool = False

    def model_on_port(self, port: int) -> str:
        return self.model_port_1 if port == 1 else self.model_port_2

    def port_of_model(self, model: str) -> int:
        if model == self.model_port_1:
            return 1
        if model == self.model_port_2:
            return 2
        raise ValueError(f"{model!r} did not play match {self.match_id}")

    def character_of_port(self, port: int) -> str:
        return self.character_port_1 if port == 1 else self.character_port_2

    def input_stats_of_port(self, port: int) -> PortInputStats | None:
        return self.input_stats_port_1 if port == 1 else self.input_stats_port_2

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MatchRecord:
        """Rebuild from a persisted row (round-trips ``as_dict``)."""
        values = dict(data)
        values.setdefault("replay_trimmed", False)
        outcome = values["outcome"]
        if outcome is not None and "winner_port" in outcome:
            outcome = dict(outcome)
            outcome["stock_leader_port"] = outcome.pop("winner_port")
            outcome["stock_leader_model"] = outcome.pop("winner_model")
        stats = {
            f"input_stats_port_{port}": (
                None
                if values[f"input_stats_port_{port}"] is None
                else PortInputStats(**values[f"input_stats_port_{port}"])
            )
            for port in (1, 2)
        }
        return cls(
            **{k: v for k, v in values.items() if k not in {"outcome", "input_stats_port_1", "input_stats_port_2"}},
            outcome=None if outcome is None else MatchOutcome(**outcome),
            **stats,
        )


def check_input_stats(records: Sequence[MatchRecord]) -> None:
    """Fail loud on a dead or forbidden policy, from the recorded controller traces.

    Call this only after the records are written: the evidence must survive the raise.
    Two conditions are errors, not observations. START pressed at all means a policy
    escaped the action space and can pause the match. A single distinct action over a
    whole match means the policy emitted a constant input and the head-to-head measures
    nothing.
    """
    problems: list[str] = []
    for record in records:
        for port in (1, 2):
            stats = record.input_stats_of_port(port)
            if stats is None:
                if record.outcome is not None:
                    problems.append(
                        f"{record.match_id} port {port} ({record.model_on_port(port)}): "
                        f"input statistics are missing; replay status is {record.replay_status}"
                    )
                continue
            model = record.model_on_port(port)
            if stats.button_start_frac != 0.0:
                problems.append(
                    f"{record.match_id} port {port} ({model}): START pressed on "
                    f"{stats.button_start_frac:.4f} of frames; START is not in the action space"
                )
            if stats.distinct_actions <= 1:
                problems.append(
                    f"{record.match_id} port {port} ({model}): {stats.distinct_actions} distinct action(s); "
                    "the policy is dead"
                )
    if problems:
        raise ValueError("head-to-head input tripwire tripped:\n  " + "\n  ".join(problems))


def _modification_time(path: Path) -> float:
    return path.stat().st_mtime


def _claim_replay(boot_dir: Path, match_id: str) -> tuple[Path | None, str]:
    """Rename the match's .slp to ``<match_id>.slp``, and report ``"ok"`` or ``"missing"``.

    Dolphin names replays by wall-clock second, so concurrent boots collide on the
    basename. The match id is globally unique and self-describing, so downstream analysis
    can key on the basename alone. The match's own file is the newest in the boot
    directory: a start retry can leave an earlier stub behind.
    """
    target = boot_dir / f"{match_id}.slp"
    if not target.is_file():
        if not boot_dir.is_dir():
            return None, "missing"
        candidates = sorted(boot_dir.glob("*.slp"))
        if not candidates:
            return None, "missing"
        max(candidates, key=_modification_time).rename(target)
    return target, "ok"


def match_record(
    spec: MatchSpec,
    traj: Trajectory | None,
    boot_dir: Path,
    *,
    max_frames: int,
    stamp_identity: bool = True,
    verify_inputs: bool = True,
) -> MatchRecord:
    """Build one record from a driven match and its recorded replay."""
    replay, replay_status = _claim_replay(boot_dir, spec.match_id)
    stamped = False
    if replay is not None and stamp_identity:
        labels = {1: spec.model_port_1, 2: spec.model_port_2}
        try:
            stamp_replay_identity(replay, labels)
            stamped = True
        except (ValueError, IndexError, struct.error) as error:
            # A truncated file can die inside the offset math before any explicit check;
            # a bad replay must cost its own record only, never the rest of the sweep.
            logger.warning(f"match_record: cannot stamp identities into {replay}: {error}")
    stats: dict[int, PortInputStats] | None = None
    replay_trimmed = False
    if replay is not None and verify_inputs:
        stats = replay_input_stats(replay)
        if stats is None and trim_to_last_frame(replay):
            replay_trimmed = True
            logger.info(f"match_record: trimmed a torn final frame from {replay}")
            stats = replay_input_stats(replay)
        if stats is None:
            replay_status = "unreadable"

    outcome: MatchOutcome | None = None
    if traj is not None:
        summary = summarize_trajectory(traj)
        stocks_left = {1: summary.p1_stocks_left, 2: summary.p2_stocks_left}
        damage_taken = {1: summary.p1_damage_taken, 2: summary.p2_damage_taken}
        stock_leader_port: int | None = None
        if stocks_left[1] != stocks_left[2]:
            stock_leader_port = 1 if stocks_left[1] > stocks_left[2] else 2
        outcome = MatchOutcome(
            total_frames=len(traj),
            active_frames=int((traj.frame_id >= 0).sum()),
            stocks_left_port_1=stocks_left[1],
            stocks_left_port_2=stocks_left[2],
            stocks_lost_port_1=STARTING_STOCKS - stocks_left[1],
            stocks_lost_port_2=STARTING_STOCKS - stocks_left[2],
            damage_taken_port_1=damage_taken[1],
            damage_taken_port_2=damage_taken[2],
            damage_dealt_port_1=damage_taken[2],
            damage_dealt_port_2=damage_taken[1],
            decided=min(stocks_left.values()) <= 0,
            hit_frame_budget=len(traj) >= max_frames - 1,
            stock_leader_port=stock_leader_port,
            stock_leader_model=None if stock_leader_port is None else spec.model_on_port(stock_leader_port),
        )
    return MatchRecord(
        match_id=spec.match_id,
        config_id=spec.config.config_id,
        orientation=spec.orientation,
        boot_index=spec.config.config_id,
        stage=spec.config.stage.name,
        stage_id=int(spec.config.stage.value),
        model_port_1=spec.model_port_1,
        model_port_2=spec.model_port_2,
        character_port_1=spec.character_port_1.name,
        character_port_2=spec.character_port_2.name,
        character_id_port_1=int(spec.character_port_1.value),
        character_id_port_2=int(spec.character_port_2.value),
        replay_path=None if replay is None else str(replay),
        replay_status=replay_status,
        replay_trimmed=replay_trimmed,
        identity_stamped=stamped,
        outcome=outcome,
        input_stats_port_1=None if stats is None else stats[1],
        input_stats_port_2=None if stats is None else stats[2],
    )


# ---------------------------------------------------------------------------
# Driving
# ---------------------------------------------------------------------------


def run_orientation(
    specs: Sequence[MatchSpec],
    builder: PolicyBuilder,
    *,
    session_cfg: SessionConfig,
    max_frames: int,
    max_parallel: int,
    seed: int,
    start_retries: int,
) -> list[Trajectory | None]:
    """Drive one orientation's matches. Every spec here shares one port -> model map.

    Returns one trajectory per spec, or None where the boot produced no match. Each wave
    gets fresh policies with distinct decode seeds, so no rolling state and no sampling
    stream is reused across waves.
    """
    if not specs:
        return []
    by_port = {1: specs[0].model_port_1, 2: specs[0].model_port_2}
    if any((s.model_port_1, s.model_port_2) != (by_port[1], by_port[2]) for s in specs):
        raise ValueError("run_orientation needs a constant port -> model map across its specs")
    boots = run_matches_vec(
        session_cfg,
        [s.vec_match() for s in specs],
        _OrientationPolicyFactory(builder, by_port, seed),
        max_frames=max_frames,
        max_parallel=max_parallel,
        start_retries=start_retries,
    )
    # No instant restart: each boot is exactly one match, or empty if it never started.
    return [boot[0] if boot else None for boot in boots]


def orientation_replay_dir(out_dir: Path, name_a: str, orientation: int) -> Path:
    """Replay root of one orientation, named for the port model A sits on."""
    return out_dir / "replays" / f"{path_safe(name_a)}-on-port{1 if orientation == 0 else 2}"


def run_h2h(
    build_policy: PolicyBuilder,
    *,
    name_a: str,
    name_b: str,
    n_configs: int,
    out_dir: str | Path,
    stages: Sequence[melee.Stage] = INCLUDED_STAGES,
    max_frames: int = DEFAULT_MAX_FRAMES,
    max_parallel: int = 0,
    start_retries: int = DEFAULT_START_RETRIES,
    seed: int = 0,
    session_cfg: SessionConfig | None = None,
    stamp_identity: bool = True,
    verify_inputs: bool = True,
    fixed_characters: Mapping[str, melee.Character] | None = None,
    meta: Mapping[str, Any] | None = None,
    on_orientation_done: Callable[[int], None] | None = None,
) -> list[MatchRecord]:
    """Run the full mirrored sweep and return one record per match.

    ``n_configs`` prior-drawn configs each play twice, once per orientation, for
    ``2 * n_configs`` matches. Below ``out_dir`` this writes ``meta.json``,
    ``matches.jsonl`` (appended after each orientation, so a crash in the second sweep
    cannot lose the first) and the replays. Nothing is uploaded: the caller owns transfer.

    ``fixed_characters`` binds selected characters to models as the models swap ports. Without
    it, the original characters stay pinned to ports. ``max_parallel`` 0 means one concurrent
    Dolphin boot per USABLE CPU. ``session_cfg`` defaults
    to the standard headless eval session; a supplied one has its ``replay_dir`` replaced
    per orientation. ``meta`` is merged into ``meta.json``, which is where a caller
    records the checkpoints and decode settings behind its policy builders.

    The input tripwire (``check_input_stats``) runs last, after every record is on disk.
    It raises, so a caller that must move the artifacts off a disposable box has to upload
    from a ``finally`` block. ``on_orientation_done(orientation)`` fires after each
    orientation's records are flushed, so such a caller can also ship partial evidence
    before the second sweep — a kill mid-sweep then costs one orientation, not both.
    """
    if name_a == name_b:
        raise ValueError(f"the two models need distinct names, got {name_a!r} twice")
    if n_configs < 1:
        raise ValueError(f"n_configs must be >= 1, got {n_configs}")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    specs = match_specs(
        mirrored_configs(n_configs, stages=stages),
        name_a=name_a,
        name_b=name_b,
        fixed_characters=fixed_characters,
    )
    parallel = max_parallel or usable_cpus()

    run_meta: dict[str, Any] = {
        "record_schema_version": MATCH_RECORD_SCHEMA_VERSION,
        "model_a": name_a,
        "model_b": name_b,
        "n_configs": n_configs,
        "n_matches": len(specs),
        "stages": [s.name for s in stages],
        "max_frames": max_frames,
        "max_parallel": parallel,
        "start_retries": start_retries,
        "seed": seed,
        "instant_match_restart": False,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **(dict(meta) if meta else {}),
    }
    meta_path = out_path / "meta.json"
    meta_path.write_text(json.dumps(run_meta, indent=2, sort_keys=True))

    records_path = out_path / "matches.jsonl"
    records_path.unlink(missing_ok=True)
    records: list[MatchRecord] = []
    for orientation in (0, 1):
        subset = [s for s in specs if s.orientation == orientation]
        replay_root = orientation_replay_dir(out_path, name_a, orientation)
        replay_root.mkdir(parents=True, exist_ok=True)
        cfg = (
            default_session_cfg(replay_root, instant_match_restart=False)
            if session_cfg is None
            else replace(session_cfg, replay_dir=replay_root, instant_match_restart=False)
        )
        logger.info(
            f"h2h orientation {orientation}: {len(subset)} matches, "
            f"port 1 = {subset[0].model_port_1}, port 2 = {subset[0].model_port_2}, max_parallel = {parallel}"
        )
        started = time.monotonic()
        trajectories = run_orientation(
            subset,
            build_policy,
            session_cfg=cfg,
            max_frames=max_frames,
            max_parallel=parallel,
            # The orientations are separate sweeps. Offset their seeds so the two never
            # reuse the same decode stream.
            seed=seed + 100_000 * orientation,
            start_retries=start_retries,
        )
        run_meta[f"orientation_{orientation}_seconds"] = time.monotonic() - started
        with records_path.open("a") as fh:
            for spec, traj in zip(subset, trajectories, strict=True):
                record = match_record(
                    spec,
                    traj,
                    replay_root / f"boot_{spec.config.config_id:03d}",
                    max_frames=max_frames,
                    stamp_identity=stamp_identity,
                    verify_inputs=verify_inputs,
                )
                records.append(record)
                fh.write(json.dumps(record.as_dict(), sort_keys=True) + "\n")
        if on_orientation_done is not None:
            on_orientation_done(orientation)

    completed = [r for r in records if r.outcome is not None]
    run_meta.update(
        {
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "matches_completed": len(completed),
            "matches_failed": len(records) - len(completed),
        }
    )
    meta_path.write_text(json.dumps(run_meta, indent=2, sort_keys=True))
    logger.info(f"h2h: {len(completed)}/{len(records)} matches completed -> {records_path}")
    if verify_inputs:
        check_input_stats(records)
    return records


def load_records(path: str | Path) -> list[MatchRecord]:
    """Read a ``matches.jsonl`` back into records."""
    with Path(path).open() as fh:
        return [MatchRecord.from_dict(json.loads(line)) for line in fh if line.strip()]
