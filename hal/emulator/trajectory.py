"""Columnar per-frame trajectory + constructors from .slp / MDS / live capture.

A ``Trajectory`` is the comparison currency: ``diff`` consumes two of them
without caring which side came from where. We keep only post-frame data here
— pre-frame controller features live in ``ControllerInputs`` /
``MdsControllerSource`` and are not duplicated.

Layout: ``post[libmelee_port][field]`` is a 1D ndarray of length N. Field
names are the MDS column suffixes from ``hal.wire.POST_FIELD_SUFFIXES``, which
match peppi-py's (renamed) ``Post`` and libmelee's canonical ``Post`` 1:1, so
no translation layer is needed between the three input paths.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import peppi_py

from hal.wire import POST_FIELD_SUFFIXES

# Post-frame field names used by diff. Same vocabulary on .slp (peppi), MDS,
# and live-capture sides.
POST_FIELDS: tuple[str, ...] = POST_FIELD_SUFFIXES


@dataclass(frozen=True, slots=True)
class Trajectory:
    """Columnar per-frame data covering N frames.

    ``post[port][field]`` is a 1D ndarray of length N. ``port`` keys are
    libmelee port ints (1..4). ``frame_id`` is the slp frame index (peppi
    convention, starting at -123). ``random_seed`` is the per-frame Slippi RNG
    state — used as a tripwire in ``diff``.
    """

    frame_id: np.ndarray
    post: dict[int, dict[str, np.ndarray]]
    random_seed: np.ndarray

    def __len__(self) -> int:
        return int(self.frame_id.shape[0])

    def take(self, n: int) -> Trajectory:
        return Trajectory(
            frame_id=self.frame_id[:n],
            post={p: {k: v[:n] for k, v in cols.items()} for p, cols in self.post.items()},
            random_seed=self.random_seed[:n],
        )

    @classmethod
    def from_slp(cls, path: str | Path) -> Trajectory:
        """Read a .slp directly via peppi-py — aliases peppi's SoA arrays.

        peppi compactly stores only occupied ports in ``frames.ports``; the
        libmelee port number for ``frames.ports[i]`` comes from
        ``start.players[i].port``.
        """
        from hal.wire import peppi_port_to_libmelee

        game = peppi_py.read_slippi(str(path), skip_frames=False)
        frames = game.frames
        n = len(frames.id)
        post: dict[int, dict[str, np.ndarray]] = {}
        for sp, port_data in zip(game.start.players, frames.ports, strict=True):
            libmelee_port = peppi_port_to_libmelee(sp.port)
            leader_post = port_data.leader.post
            cols = {field: _peppi_post_field(leader_post, field, n) for field in POST_FIELDS}
            post[libmelee_port] = cols
        return cls(
            frame_id=np.asarray(frames.id),
            post=post,
            random_seed=np.asarray(frames.start.random_seed),
        )

    @classmethod
    def from_mds_rows(cls, columns: dict[str, np.ndarray], port_to_mds_prefix: dict[int, str]) -> Trajectory:
        """Project MDS columns into a Trajectory.

        ``port_to_mds_prefix`` maps libmelee port (1..4) -> ``"p1"|"p2"``.
        Derived from ``Matchup.port_to_mds_prefix`` at the call site so this
        function stays decoupled from manifest details.

        ``random_seed`` is filled with zeros — the MDS schema does not store
        per-frame seed today. Diff treats a flat-zero seed array as "unknown,
        skip" rather than asserting against it.
        """
        post: dict[int, dict[str, np.ndarray]] = {}
        for port, prefix in port_to_mds_prefix.items():
            post[port] = {suffix: columns[f"{prefix}_{suffix}"] for suffix in POST_FIELD_SUFFIXES}
        n = len(columns["frame"])
        return cls(
            frame_id=columns["frame"],
            post=post,
            random_seed=np.zeros(n, dtype=np.uint32),
        )

    @classmethod
    def from_capture(cls, frames: Sequence[dict], ports: Sequence[int]) -> Trajectory:
        """Transpose row-by-row CanonicalFrame dicts (from ``Session.step``)
        into columnar form.

        ``ports`` lists which libmelee ports are active in this match — we
        index ``frame['ports'][port]`` for each one.
        """
        n = len(frames)
        frame_id = np.empty(n, dtype=np.int32)
        seed = np.empty(n, dtype=np.uint32)
        post: dict[int, dict[str, np.ndarray]] = {
            p: {f: np.empty(n, dtype=np.float64) for f in POST_FIELDS} for p in ports
        }

        for i, frame in enumerate(frames):
            frame_id[i] = frame["id"]
            start = frame.get("start")
            seed[i] = start["random_seed"] if start else 0
            for p in ports:
                pd = frame["ports"].get(p)
                if pd is None:
                    continue
                pf = pd["leader"]["post"]
                cols = post[p]
                pos = pf["position"]
                cols["position_x"][i] = pos["x"]
                cols["position_y"][i] = pos["y"]
                cols["percent"][i] = pf["percent"]
                cols["shield"][i] = pf["shield"]
                cols["stock"][i] = pf["stock"]
                cols["direction"][i] = pf["direction"]
                cols["action"][i] = pf["action"]
                cols["jumps_used"][i] = pf.get("jumps_used") or 0
                cols["airborne"][i] = pf.get("airborne") or 0
                cols["hurtbox_state"][i] = pf.get("hurtbox_state") or 0
                cols["hitlag_left"][i] = pf.get("hitlag_left") or 0.0
        return cls(frame_id=frame_id, post=post, random_seed=seed)


def _peppi_post_field(post: object, field: str, n: int) -> np.ndarray:
    """Pull one named post-field out of peppi's nested SoA.

    Position lives under ``post.position.{x,y}`` rather than as flat fields,
    so we special-case it. Optional fields that are entirely absent on this
    slp version are filled with NaN, matching MDS's mask convention for
    float columns (hal/data/extract._mask_value). ``diff`` then compares
    with ``equal_nan=True`` so masked-on-both-sides reads as equal.
    """
    if field == "position_x":
        return np.asarray(post.position.x)
    if field == "position_y":
        return np.asarray(post.position.y)
    raw = getattr(post, field, None)
    if raw is None:
        return np.full(n, np.nan, dtype=np.float32)
    return np.asarray(raw)
