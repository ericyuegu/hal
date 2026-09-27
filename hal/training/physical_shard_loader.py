"""Pickle records used by 059 replay-ring checkpoints.

Keep these definitions at their original import path so current checkpoints
load without a serialization migration.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PhysicalRow:
    """A stable row locator independent of Mosaic's global sample map."""

    source: str
    shard: int
    row: int


@dataclass(frozen=True, slots=True)
class RingSlotDescriptor:
    slot: int
    locator: PhysicalRow
    epoch: int
    replay_checksum: int
