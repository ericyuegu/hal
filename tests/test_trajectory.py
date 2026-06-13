"""Trajectory.from_capture mask-convention tests (no Dolphin / .slp needed)."""

import numpy as np

from hal.sim.trajectory import Trajectory


def _post(**overrides: object) -> dict:
    post = {
        "position": {"x": 1.0, "y": 2.0},
        "percent": 0.0,
        "shield": 60.0,
        "stock": 4,
        "direction": 1.0,
        "action": 14,
        "jumps_used": 0,
        "airborne": 0,
        "hurtbox_state": 0,
        "hitlag_left": 0.0,
    }
    post.update(overrides)
    return post


def _frame(i: int, ports_post: dict[int, dict]) -> dict:
    return {
        "id": i,
        "start": {"random_seed": 0},
        "ports": {p: {"leader": {"post": post}} for p, post in ports_post.items()},
    }


def test_from_capture_absent_optional_field_is_nan() -> None:
    """An optional post-field libmelee reports as None must land as NaN (the
    masked-on-this-build convention from_slp uses), not silently 0."""
    post = _post()
    del post["jumps_used"]  # libmelee returned None for this slp/build
    traj = Trajectory.from_capture([_frame(-123, {1: post})], ports=(1,))
    assert np.isnan(traj.post[1]["jumps_used"][0])


def test_from_capture_present_zero_optional_field_stays_zero() -> None:
    """A genuine 0 (e.g. jumps_used=0 on the ground) must survive, not be
    collapsed to NaN — the old ``or 0`` and a naive NaN fix both get this wrong."""
    traj = Trajectory.from_capture([_frame(-123, {1: _post(jumps_used=0)})], ports=(1,))
    assert traj.post[1]["jumps_used"][0] == 0.0


def test_from_capture_absent_port_is_nan() -> None:
    """A port with no data this frame must be NaN, not uninitialized garbage."""
    traj = Trajectory.from_capture([_frame(-123, {1: _post()})], ports=(1, 2))
    assert np.all(np.isnan(traj.post[2]["percent"]))


def test_flatten_leader_matches_capture_post_fields() -> None:
    """flatten_canonical_frame's LEADER block covers exactly POST_FIELD_SUFFIXES, identical to
    from_capture — the drift guard for the shared accessor. flatten additionally emits a nana
    follower block (gamestate-only) that from_capture does not carry; it mirrors the same suffixes
    and is masked (NaN) when there's no follower (non-Ice-Climbers)."""
    from hal.training.canonical import flatten_canonical_frame
    from hal.wire import POST_FIELD_SUFFIXES

    frame = _frame(-123, {1: _post(), 2: _post()})
    capture_fields = set(Trajectory.from_capture([frame], ports=(1, 2)).post[1])
    flat = flatten_canonical_frame(frame)
    leader = {k[3:] for k in flat if k.startswith("p1_") and not k.startswith("p1_nana_")}
    nana = {k.removeprefix("p1_nana_") for k in flat if k.startswith("p1_nana_")}
    assert capture_fields == set(POST_FIELD_SUFFIXES) == leader
    assert nana == set(POST_FIELD_SUFFIXES)
    assert all(np.isnan(flat[f"p1_nana_{s}"]) for s in POST_FIELD_SUFFIXES)


def test_flatten_emits_real_nana_when_follower_present() -> None:
    """Ice Climbers: a follower in the canonical frame yields real (unmasked) nana gamestate."""
    from hal.training.canonical import flatten_canonical_frame

    frame = _frame(-123, {1: _post()})
    frame["ports"][1]["follower"] = {"post": _post(percent=88.0)}
    flat = flatten_canonical_frame(frame)
    assert flat["p1_nana_percent"] == 88.0
    assert not np.isnan(flat["p1_nana_action"])
