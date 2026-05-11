"""Columnar per-frame trajectory + constructors from .slp / MDS / live capture.

A ``Trajectory`` is the comparison currency. ``diff`` consumes two of them
without caring which side came from where.

Layout: ``post`` is a per-libmelee-port dict of column-name -> 1D ndarray.
Field names mirror peppi's post-frame block (``position_x``, ``position_y``,
``state``, ``percent``, ``shield``, ``stocks``, ``direction``, ``jumps``,
``airborne``, ``hurtbox_state``, ``hitlag``). Per-frame ``random_seed`` is
top-level for the seed tripwire.

We keep only post-frame data here. Pre-frame controller features are owned
by ``ControllerInputs`` / ``MdsControllerSource``; we don't duplicate.
"""

from collections.abc import Sequence
from pathlib import Path

import attrs
import numpy as np
import peppi_py

# Post-frame fields compared by the round-trip diff. Names mirror MDS columns
# in hal/data/schema._gamestate_columns AND peppi's post block, modulo two
# renames ("stock" in MDS vs "stocks" in peppi; "action" in MDS vs "state" in
# peppi). We canonicalize on peppi names here.
POST_FIELDS: tuple[str, ...] = (
    "position_x",
    "position_y",
    "percent",
    "shield",
    "stocks",
    "direction",
    "state",
    "jumps",
    "airborne",
    "hurtbox_state",
    "hitlag",
)

# Map MDS column suffix (under hal/data/schema._gamestate_columns) to the peppi
# post-field name. The two disagree on "stock" (MDS) vs "stocks" (peppi),
# "action" (MDS) vs "state" (peppi), and a couple of suffixes.
_MDS_TO_PEPPI_POST: dict[str, str] = {
    "position_x": "position_x",
    "position_y": "position_y",
    "percent": "percent",
    "shield": "shield",
    "stock": "stocks",
    "direction": "direction",
    "action": "state",
    "jumps_used": "jumps",
    "airborne": "airborne",
    "hurtbox_state": "hurtbox_state",
    "hitlag_left": "hitlag",
}


@attrs.frozen(slots=True)
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
        ``start.players[i].port`` (peppi's 0..3 + 1).
        """
        game = peppi_py.read_slippi(str(path), skip_frames=False)
        frames = game.frames
        n = len(frames.id)
        post: dict[int, dict[str, np.ndarray]] = {}
        for sp, port_data in zip(game.start.players, frames.ports, strict=True):
            libmelee_port = int(getattr(sp.port, "value", sp.port)) + 1
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
            cols = {
                peppi_name: columns[f"{prefix}_{mds_suffix}"] for mds_suffix, peppi_name in _MDS_TO_PEPPI_POST.items()
            }
            post[port] = cols
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
                cols["stocks"][i] = pf["stocks"]
                cols["direction"][i] = pf["direction"]
                cols["state"][i] = pf["state"]
                cols["jumps"][i] = pf.get("jumps") or 0
                cols["airborne"][i] = pf.get("airborne") or 0
                cols["hurtbox_state"][i] = pf.get("hurtbox_state") or 0
                cols["hitlag"][i] = pf.get("hitlag") or 0.0
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
