"""Picklable replay-rank metadata joins for training workers."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ReplayRankLookup:
    """Attach a replay port's rank after deterministic window sampling."""

    by_replay: dict[str, tuple[int, int]]

    def __call__(self, replay_id: str, ego_prefix: str, window: dict[str, np.ndarray]) -> None:
        try:
            ranks = self.by_replay[replay_id]
        except KeyError as error:
            raise KeyError(f"rank sidecar has no entry for replay {replay_id}") from error
        if ego_prefix == "p1":
            rank = ranks[0]
        elif ego_prefix == "p2":
            rank = ranks[1]
        else:
            raise ValueError(f"unexpected ego prefix {ego_prefix!r}")
        window["ego_rank"] = np.asarray(rank, dtype=np.uint8)
