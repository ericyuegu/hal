"""Compare two trajectories for bit-exact agreement on post-frame fields.

The diff is symmetric in source: round-trip uses ``diff(live, mds_truth)`` but
the same function compares slp-vs-MDS losslessness (no Dolphin involved) or
two parallel rollouts. ``random_seed`` is reported as a tripwire — divergence
is a strong signal that RNG-driven physics has entered the picture, but it
does not fail the diff because we cannot inject the seed into stock Dolphin
yet.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from hal.emulator.trajectory import POST_FIELDS
from hal.emulator.trajectory import Trajectory


@dataclass(frozen=True, slots=True)
class FieldDivergence:
    port: int
    field: str
    first_diff_frame: int  # index into the compared range
    a_value: float
    b_value: float


@dataclass(frozen=True, slots=True)
class DiffReport:
    passed: bool
    n_frames: int
    divergences: tuple[FieldDivergence, ...]
    seed_diverged_at: int | None  # None means "agreement (or seed unknown)"

    def summary(self) -> str:
        if self.passed:
            base = f"PASS ({self.n_frames} frames bit-exact across {len(POST_FIELDS)} post-fields × ports)"
        else:
            d0 = self.divergences[0]
            base = (
                f"FAIL: first divergence at frame {d0.first_diff_frame}, "
                f"port {d0.port}.{d0.field} (a={d0.a_value!r} b={d0.b_value!r}); "
                f"{len(self.divergences)} field(s) diverged"
            )
        if self.seed_diverged_at is not None:
            base += f" | seed tripwire: divergence at frame {self.seed_diverged_at}"
        else:
            base += " | seed tripwire: agree (or unknown)"
        return base


def diff(
    a: Trajectory,
    b: Trajectory,
    *,
    fields: Iterable[str] = POST_FIELDS,
    max_frames: int | None = None,
) -> DiffReport:
    """Compare two trajectories field-by-field, port-by-port.

    The two trajectories must agree on which libmelee ports are populated. The
    range compared is ``min(len(a), len(b), max_frames)``. ``random_seed`` is
    reported separately if both sides have non-zero seeds (zeros mean "seed
    not captured" — e.g. ``Trajectory.from_mds_rows``).
    """
    n = min(len(a), len(b))
    if max_frames is not None:
        n = min(n, max_frames)

    if set(a.post) != set(b.post):
        raise ValueError(f"port mismatch: a={set(a.post)} b={set(b.post)}")

    divergences: list[FieldDivergence] = []
    fields_tuple = tuple(fields)
    for port in sorted(a.post):
        a_cols = a.post[port]
        b_cols = b.post[port]
        for field_name in fields_tuple:
            if field_name not in a_cols or field_name not in b_cols:
                continue
            a_arr = a_cols[field_name][:n]
            b_arr = b_cols[field_name][:n]
            a_floating = np.issubdtype(a_arr.dtype, np.floating)
            b_floating = np.issubdtype(b_arr.dtype, np.floating)
            if np.array_equal(a_arr, b_arr, equal_nan=a_floating and b_floating):
                continue
            # Mask convention: NaN on EITHER side means "this source didn't
            # record the field at this position" (slp version gap, no live
            # capture, etc.). Skip those positions — equality there is
            # unknowable, and the alternative (false-positive divergence) is
            # noise we always have to filter out by hand.
            either_nan = np.zeros(n, dtype=bool)
            if a_floating:
                either_nan |= np.isnan(a_arr)
            if b_floating:
                either_nan |= np.isnan(b_arr)
            mismatches = np.flatnonzero((a_arr != b_arr) & ~either_nan)
            if mismatches.size == 0:
                continue
            i = int(mismatches[0])
            divergences.append(
                FieldDivergence(
                    port=port,
                    field=field_name,
                    first_diff_frame=i,
                    a_value=float(a_arr[i]),
                    b_value=float(b_arr[i]),
                )
            )

    seed_diverged_at: int | None = None
    a_seed = a.random_seed[:n]
    b_seed = b.random_seed[:n]
    if np.any(a_seed != 0) and np.any(b_seed != 0):
        mismatches = np.flatnonzero(a_seed != b_seed)
        if mismatches.size > 0:
            seed_diverged_at = int(mismatches[0])

    divergences.sort(key=lambda d: (d.first_diff_frame, d.port, d.field))
    return DiffReport(
        passed=not divergences,
        n_frames=n,
        divergences=tuple(divergences),
        seed_diverged_at=seed_diverged_at,
    )
