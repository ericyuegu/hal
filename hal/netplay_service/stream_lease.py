"""Private, expiring handoff from queue control to the overlay service."""

import json
import math
import os
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

from hal.netplay_service.queue_client import StreamGrant
from hal.netplay_service.queue_client import parse_session_state

MAX_AGE_SECONDS = 30.0


@dataclass(frozen=True, slots=True)
class StreamLease:
    updated_at: float
    queue_depth: int
    grant: StreamGrant | None = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.updated_at) not in (int, float) or not math.isfinite(self.updated_at) or self.updated_at < 0:
            raise ValueError("invalid stream lease timestamp")
        if type(self.queue_depth) is not int or self.queue_depth < 0:
            raise ValueError("invalid stream lease queue depth")
        if self.grant is not None and (self.grant.slot != 0 or not self.grant.key):
            raise ValueError("invalid stream grant")

    def current_grant(self, now: float) -> StreamGrant | None:
        return self.grant if 0 <= now - self.updated_at < MAX_AGE_SECONDS else None


def write_stream_lease(path: Path, lease: StreamLease) -> None:
    """Replace the lease atomically; the Twitch key must remain owner-only."""
    payload = {
        "schema_version": 1,
        "updated_at": lease.updated_at,
        "queue_depth": lease.queue_depth,
        "stream": None if lease.grant is None else {"slot": lease.grant.slot, "key": lease.grant.key},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as output:
        os.fchmod(output.fileno(), 0o600)
        json.dump(payload, output, allow_nan=False)
    temporary.replace(path)


def read_stream_lease(path: Path) -> StreamLease:
    raw = json.loads(path.read_text())
    if (
        not isinstance(raw, dict)
        or set(raw) != {"schema_version", "updated_at", "queue_depth", "stream"}
        or type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
    ):
        raise ValueError("unsupported stream lease schema")
    state = parse_session_state({"draining": False, "stream": raw["stream"]})
    return StreamLease(raw["updated_at"], raw["queue_depth"], state.stream)
