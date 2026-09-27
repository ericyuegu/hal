"""Unit tests for the model-vs-model head-to-head runner. No Dolphin, no GPU."""

import json
import struct
from dataclasses import replace
from pathlib import Path

import melee
import numpy as np
import pytest

from hal.eval.h2h import H2HPolicy
from hal.eval.h2h import MatchRecord
from hal.eval.h2h import PortInputStats
from hal.eval.h2h import check_input_stats
from hal.eval.h2h import load_records
from hal.eval.h2h import match_record
from hal.eval.h2h import match_specs
from hal.eval.h2h import replay_display_names
from hal.eval.h2h import run_h2h
from hal.eval.h2h import stamp_replay_identity
from hal.eval.harness import SessionConfig
from hal.eval.matchups import mirrored_configs
from hal.eval.matchups import startable_matchups
from hal.policy import INCLUDED_STAGES
from hal.sim.rollout import PolicyRuntimeSpec
from hal.sim.rollout import Slot
from hal.sim.trajectory import Trajectory

# ---------------------------------------------------------------------------
# Config schedule
# ---------------------------------------------------------------------------


def test_mirrored_configs_are_deterministic():
    assert mirrored_configs(16) == mirrored_configs(16)


def test_mirrored_configs_are_prefix_stable():
    assert mirrored_configs(8) == mirrored_configs(32)[:8]


def test_mirrored_configs_cycle_the_stages():
    configs = mirrored_configs(len(INCLUDED_STAGES) * 3)
    assert [c.stage for c in configs] == list(INCLUDED_STAGES) * 3
    assert [c.config_id for c in configs] == list(range(len(configs)))


def test_mirrored_configs_rejects_no_stages():
    with pytest.raises(ValueError, match="stages"):
        mirrored_configs(4, stages=())


def test_startable_matchups_avoids_unstartable_sheik():
    matchups = startable_matchups(400)
    assert len(matchups) == 400
    assert all(a is not melee.Character.SHEIK for a, _ in matchups)
    assert any(b is melee.Character.SHEIK for _, b in matchups)


def test_startable_matchups_is_prefix_stable():
    assert startable_matchups(16) == startable_matchups(64)[:16]


def test_match_specs_pair_the_orientations():
    configs = mirrored_configs(5)
    specs = match_specs(configs, name_a="alpha", name_b="beta")
    assert len(specs) == 2 * len(configs)
    for config in configs:
        pair = [s for s in specs if s.config.config_id == config.config_id]
        assert [s.orientation for s in pair] == [0, 1]
        # The models swap ports; the characters and stage stay pinned to the ports.
        assert (pair[0].model_port_1, pair[0].model_port_2) == ("alpha", "beta")
        assert (pair[1].model_port_1, pair[1].model_port_2) == ("beta", "alpha")
        assert {s.config for s in pair} == {config}
    assert len({s.match_id for s in specs}) == len(specs)


def test_match_specs_rejects_one_model_twice():
    with pytest.raises(ValueError, match="distinct names"):
        match_specs(mirrored_configs(2), name_a="alpha", name_b="alpha")


def test_vec_match_pins_characters_to_ports():
    spec = match_specs(mirrored_configs(1), name_a="alpha", name_b="beta")[1]
    vec = spec.vec_match()
    assert vec.model_ports == (1, 2)
    assert vec.matchup.stage is spec.config.stage
    assert {p.port: p.character for p in vec.matchup.players} == {
        1: spec.character_port_1,
        2: spec.character_port_2,
    }


def test_match_specs_fixed_character_follows_model_across_orientations():
    specs = match_specs(
        mirrored_configs(1),
        name_a="IBDW#0",
        name_b="MASTER",
        fixed_characters={"IBDW#0": melee.Character.FOX},
    )

    assert specs[0].model_port_1 == "IBDW#0"
    assert specs[0].character_port_1 is melee.Character.FOX
    assert specs[1].model_port_2 == "IBDW#0"
    assert specs[1].character_port_2 is melee.Character.FOX


# ---------------------------------------------------------------------------
# The two-model batched policy
# ---------------------------------------------------------------------------


class _RecordingPolicy:
    def __init__(self, label: str) -> None:
        self.label = label
        self.calls = []
        self.runtime_spec = PolicyRuntimeSpec(8, 4, 2, 0, 14)

    def plan_rows(self, rows):
        self.calls.append(tuple(rows))
        return {slot: self.label for slot in rows}


def test_h2h_policy_batches_each_model_once():
    first, second = _RecordingPolicy("alpha"), _RecordingPolicy("beta")
    policy = H2HPolicy({1: first, 2: second})
    rows = {Slot(match, port): [] for match in (0, 1) for port in (1, 2)}
    actions = policy.plan_rows(rows)
    assert set(actions) == set(rows)
    assert all(actions[slot] == "alpha" for slot in rows if slot.port == 1)
    assert all(actions[slot] == "beta" for slot in rows if slot.port == 2)
    assert first.calls == [(Slot(0, 1), Slot(1, 1))]
    assert second.calls == [(Slot(0, 2), Slot(1, 2))]


def test_h2h_policy_collects_both_ports_for_shared_checkpoint():
    shared = _RecordingPolicy("shared")
    policy = H2HPolicy({1: shared, 2: shared})
    rows = {Slot(match, port): [] for match in (0, 1) for port in (1, 2)}
    assert set(policy.plan_rows(rows)) == set(rows)
    assert shared.calls == [tuple(rows)]


def test_h2h_policy_skips_idle_models():
    first, second = _RecordingPolicy("alpha"), _RecordingPolicy("beta")
    H2HPolicy({1: first, 2: second}).plan_rows({Slot(0, 1): []})
    assert len(first.calls) == 1
    assert second.calls == []


def test_h2h_policy_rejects_wrong_response_slots():
    class Dropping(_RecordingPolicy):
        def plan_rows(self, rows):
            return {}

    with pytest.raises(RuntimeError, match="wrong slots"):
        H2HPolicy({1: Dropping("alpha")}).plan_rows({Slot(0, 1): []})


def test_h2h_policy_rejects_empty_or_incompatible_routing():
    with pytest.raises(ValueError, match="port"):
        H2HPolicy({})
    first, second = _RecordingPolicy("alpha"), _RecordingPolicy("beta")
    second.runtime_spec = PolicyRuntimeSpec(8, 4, 1, 0, 14)
    with pytest.raises(ValueError, match="timing"):
        H2HPolicy({1: first, 2: second})


# ---------------------------------------------------------------------------
# Match records
# ---------------------------------------------------------------------------


def _trajectory(*, frames: int, damage_port_1: float, damage_port_2: float, stocks_left) -> Trajectory:
    """Synthetic match: each port takes its damage in one hit, then loses stocks at the end.

    ``cumulative_damage`` sums positive percent deltas on frames where the stock is
    unchanged, so a single step in percent while the stock is flat reads back exactly.
    """
    frame_id = np.arange(-123, -123 + frames, dtype=np.int32)
    post = {}
    for port, damage in ((1, damage_port_1), (2, damage_port_2)):
        percent = np.zeros(frames, dtype=np.float64)
        percent[frames // 2 :] = damage
        stock = np.full(frames, 4.0, dtype=np.float64)
        stock[-1] = stocks_left[port]
        post[port] = {"percent": percent, "stock": stock}
    return Trajectory(frame_id=frame_id, post=post, random_seed=np.zeros(frames, dtype=np.uint32))


def _spec(orientation: int = 0):
    return match_specs(mirrored_configs(1), name_a="alpha", name_b="beta")[orientation]


def test_match_record_outcome_invariants(tmp_path):
    spec = _spec()
    traj = _trajectory(frames=400, damage_port_1=61.0, damage_port_2=25.0, stocks_left={1: 2, 2: 0})

    record = match_record(spec, traj, tmp_path / "boot_000", max_frames=7200, verify_inputs=False)

    outcome = record.outcome
    assert outcome is not None
    assert outcome.total_frames == 400
    assert outcome.active_frames == 400 - 123
    # stocks_lost is the complement of stocks_left against the Melee default of 4.
    assert (outcome.stocks_lost_port_1, outcome.stocks_lost_port_2) == (2, 4)
    # Damage one port took is damage the other dealt.
    assert outcome.damage_taken_port_1 == pytest.approx(61.0)
    assert outcome.damage_dealt_port_2 == pytest.approx(outcome.damage_taken_port_1)
    assert outcome.damage_dealt_port_1 == pytest.approx(outcome.damage_taken_port_2)
    # The stock leader maps back to the model on that port.
    assert outcome.stock_leader_port == 1
    assert outcome.stock_leader_model == spec.model_port_1 == "alpha"
    assert outcome.decided is True
    assert outcome.hit_frame_budget is False
    assert record.boot_index == record.config_id == spec.config.config_id


def test_match_record_mirror_gives_the_stock_lead_to_the_other_model(tmp_path):
    traj = _trajectory(frames=200, damage_port_1=10.0, damage_port_2=20.0, stocks_left={1: 3, 2: 0})

    forward = match_record(_spec(0), traj, tmp_path / "a", max_frames=7200, verify_inputs=False)
    mirror = match_record(_spec(1), traj, tmp_path / "b", max_frames=7200, verify_inputs=False)

    assert forward.outcome is not None and mirror.outcome is not None
    assert forward.outcome.stock_leader_model == "alpha"
    assert mirror.outcome.stock_leader_model == "beta"


def test_match_record_stock_tie_has_no_leader(tmp_path):
    traj = _trajectory(frames=7200, damage_port_1=50.0, damage_port_2=50.0, stocks_left={1: 2, 2: 2})

    record = match_record(_spec(), traj, tmp_path / "boot_000", max_frames=7200, verify_inputs=False)

    assert record.outcome is not None
    assert record.outcome.stock_leader_port is None
    assert record.outcome.stock_leader_model is None
    assert record.outcome.decided is False
    assert record.outcome.hit_frame_budget is True


def test_match_record_for_a_boot_that_never_started(tmp_path):
    record = match_record(_spec(), None, tmp_path / "boot_000", max_frames=7200, verify_inputs=False)

    assert record.outcome is None
    assert record.replay_path is None
    assert record.replay_status == "missing"
    assert record.replay_trimmed is False
    assert record.identity_stamped is False


def test_match_record_round_trips_through_json(tmp_path):
    traj = _trajectory(frames=300, damage_port_1=5.0, damage_port_2=8.0, stocks_left={1: 0, 2: 1})
    record = replace(
        match_record(_spec(), traj, tmp_path / "boot_000", max_frames=7200, verify_inputs=False),
        input_stats_port_1=PortInputStats(0.4, 0.1, 0.3, 0.0, 512),
        input_stats_port_2=None,
    )
    path = tmp_path / "matches.jsonl"
    path.write_text(json.dumps(record.as_dict()) + "\n")

    assert load_records(path) == [record]


def test_match_record_loads_old_winner_field_names(tmp_path):
    traj = _trajectory(frames=300, damage_port_1=5.0, damage_port_2=8.0, stocks_left={1: 0, 2: 1})
    record = match_record(_spec(), traj, tmp_path / "boot_000", max_frames=7200, verify_inputs=False)
    old = record.as_dict()
    outcome = old["outcome"]
    assert outcome is not None
    outcome["winner_port"] = outcome.pop("stock_leader_port")
    outcome["winner_model"] = outcome.pop("stock_leader_model")
    path = tmp_path / "matches.jsonl"
    path.write_text(json.dumps(old) + "\n")

    assert load_records(path) == [record]


def test_match_record_loads_old_rows_without_replay_trimmed(tmp_path):
    record = match_record(_spec(), None, tmp_path / "boot_000", max_frames=7200, verify_inputs=False)
    old = record.as_dict()
    old.pop("replay_trimmed")
    path = tmp_path / "matches.jsonl"
    path.write_text(json.dumps(old) + "\n")

    assert load_records(path) == [record]


# ---------------------------------------------------------------------------
# Dead-policy tripwire
# ---------------------------------------------------------------------------


def _record_with_stats(port_1: PortInputStats, port_2: PortInputStats) -> MatchRecord:
    record = match_record(_spec(), None, Path("/nonexistent-h2h-boot-dir"), max_frames=7200, verify_inputs=False)
    return replace(record, input_stats_port_1=port_1, input_stats_port_2=port_2)


def test_check_input_stats_accepts_live_policies():
    healthy = PortInputStats(
        main_stick_active_frac=0.55,
        c_stick_active_frac=0.04,
        any_button_frac=0.31,
        button_start_frac=0.0,
        distinct_actions=873,
    )
    check_input_stats([_record_with_stats(healthy, healthy)])


def test_check_input_stats_rejects_a_start_press():
    healthy = PortInputStats(0.5, 0.05, 0.3, 0.0, 500)
    pressed_start = PortInputStats(0.5, 0.05, 0.3, 0.002, 500)
    with pytest.raises(ValueError, match="START"):
        check_input_stats([_record_with_stats(healthy, pressed_start)])


def test_check_input_stats_rejects_a_dead_policy():
    healthy = PortInputStats(0.5, 0.05, 0.3, 0.0, 500)
    dead = PortInputStats(0.0, 0.0, 0.0, 0.0, 1)
    with pytest.raises(ValueError, match="dead"):
        check_input_stats([_record_with_stats(dead, healthy)])


def test_check_input_stats_rejects_missing_stats_from_a_completed_match(tmp_path):
    record = match_record(
        _spec(),
        _trajectory(frames=300, damage_port_1=5.0, damage_port_2=8.0, stocks_left={1: 3, 2: 2}),
        tmp_path / "boot_000",
        max_frames=7200,
        verify_inputs=False,
    )

    with pytest.raises(ValueError, match="input statistics are missing"):
        check_input_stats([record])


def test_match_record_repairs_a_torn_final_frame(tmp_path, monkeypatch):
    boot = tmp_path / "boot_000"
    boot.mkdir()
    (boot / "Game.slp").write_bytes(b"test replay")
    healthy = {
        1: PortInputStats(0.5, 0.05, 0.3, 0.0, 500),
        2: PortInputStats(0.5, 0.05, 0.3, 0.0, 500),
    }
    calls = 0

    def fake_stats(_path):
        nonlocal calls
        calls += 1
        return None if calls == 1 else healthy

    monkeypatch.setattr("hal.eval.h2h.replay_input_stats", fake_stats)
    monkeypatch.setattr("hal.eval.h2h.trim_to_last_frame", lambda _path: True)

    record = match_record(
        _spec(),
        _trajectory(frames=300, damage_port_1=5.0, damage_port_2=8.0, stocks_left={1: 3, 2: 2}),
        boot,
        max_frames=7200,
        stamp_identity=False,
    )

    assert calls == 2
    assert record.replay_status == "ok"
    assert record.replay_trimmed is True
    assert record.input_stats_port_1 == healthy[1]
    assert record.input_stats_port_2 == healthy[2]


# ---------------------------------------------------------------------------
# Replay identity stamping
# ---------------------------------------------------------------------------

_SLP_HEADER = b"{U\x03raw[$U#l"
_GAME_START_SIZE = 760


def _synthetic_slp(metadata: bytes) -> bytes:
    """Smallest finalized .slp envelope that carries a modern Game Start event.

    EVENT_PAYLOADS declares one command (Game Start, 760 bytes, the 3.19 layout), then the
    Game Start event itself follows with an all-zero payload.
    """
    payloads = bytes([0x35, 4, 0x36]) + struct.pack(">H", _GAME_START_SIZE)
    game_start = bytes([0x36]) + bytes(_GAME_START_SIZE)
    raw = payloads + game_start
    return _SLP_HEADER + struct.pack(">i", len(raw)) + raw + metadata


def test_stamp_writes_display_names_and_nametags(tmp_path):
    path = tmp_path / "match.slp"
    path.write_bytes(_synthetic_slp(b"U\x08metadata{}}"))

    stamp_replay_identity(path, {1: "019-factored", 2: "016-base"})

    assert replay_display_names(path) == {1: "019-factored", 2: "016-base"}
    # Pin the wire offsets with the literal spec values, independent of the module's
    # constants: peppi and libmelee count both blocks from the Game Start COMMAND byte.
    data = path.read_bytes()
    game_start = len(_SLP_HEADER) + 4 + 5  # header + rawLength + the EVENT_PAYLOADS event
    assert data[game_start] == 0x36
    assert data[game_start + 0x1A5 : game_start + 0x1A5 + 31] == b"019-factored".ljust(31, b"\x00")
    assert data[game_start + 0x1A5 + 31 : game_start + 0x1A5 + 62] == b"016-base".ljust(31, b"\x00")
    # Nametag block: uppercase alphanumerics, at most 8 characters.
    assert data[game_start + 0x161 : game_start + 0x161 + 16] == b"019FACTO".ljust(16, b"\x00")
    assert data[game_start + 0x161 + 16 : game_start + 0x161 + 32] == b"016BASE".ljust(16, b"\x00")
    assert data.endswith(b"}}")


def test_stamp_writes_metadata_names(tmp_path):
    path = tmp_path / "match.slp"
    path.write_bytes(_synthetic_slp(b"U\x08metadata{}}"))

    stamp_replay_identity(path, {1: "alpha", 2: "beta"})

    metadata = path.read_bytes().split(b"U\x08metadata", 1)[1]
    assert b"U\x07players{" in metadata
    assert b"U\x05names{U\x07netplaySU\x05alpha" in metadata
    assert b"U\x05names{U\x07netplaySU\x04beta" in metadata
    assert metadata.endswith(b"}}")


def test_stamp_injects_into_an_existing_players_object(tmp_path):
    # Dolphin's own metadata: player entries keyed by 0-based port, no names member.
    metadata = (
        b"U\x08metadata{U\x07players{U\x010{U\ncharacters{U\x017l\x00\x00\x1b\x1c}}"
        b"U\x011{U\ncharacters{U\x0218l\x00\x00\x1b\x1c}}}U\x08playedOnSU\x07dolphin}}"
    )
    path = tmp_path / "match.slp"
    path.write_bytes(_synthetic_slp(metadata))

    stamp_replay_identity(path, {1: "alpha", 2: "beta"})

    out = path.read_bytes().split(b"U\x08metadata", 1)[1]
    assert b"U\x010{U\x05names{U\x07netplaySU\x05alpha}U\ncharacters{" in out
    assert b"U\x011{U\x05names{U\x07netplaySU\x04beta}U\ncharacters{" in out
    assert b"U\x08playedOnSU\x07dolphin" in out


def test_stamp_is_idempotent(tmp_path):
    path = tmp_path / "match.slp"
    path.write_bytes(_synthetic_slp(b"U\x08metadata{}}"))

    stamp_replay_identity(path, {1: "alpha", 2: "beta"})
    once = path.read_bytes()
    stamp_replay_identity(path, {1: "alpha", 2: "beta"})

    assert path.read_bytes() == once


def test_stamp_rejects_an_unfinalized_replay(tmp_path):
    path = tmp_path / "match.slp"
    data = bytearray(_synthetic_slp(b"U\x08metadata{}}"))
    struct.pack_into(">i", data, len(_SLP_HEADER), 0)
    path.write_bytes(bytes(data))

    with pytest.raises(ValueError, match="rawLength"):
        stamp_replay_identity(path, {1: "alpha"})


def test_stamp_rejects_a_non_slippi_file(tmp_path):
    path = tmp_path / "match.slp"
    path.write_bytes(b"not a replay at all")

    with pytest.raises(ValueError, match="not a Slippi raw stream"):
        stamp_replay_identity(path, {1: "alpha"})


# ---------------------------------------------------------------------------
# The whole sweep, with the emulator replaced
# ---------------------------------------------------------------------------


def test_run_h2h_writes_records_replays_and_meta(tmp_path, monkeypatch):
    """The in-process final-eval path, with ``run_matches_vec`` standing in for Dolphin."""
    seeds: list[list[int]] = []

    def fake_run_matches_vec(session_cfg, matches, policy_factory, *, max_frames, max_parallel, start_retries):
        # Exercise the builders exactly as a wave would, and record their decode seeds.
        router = policy_factory()
        seeds.append([router.by_port[port].seed for port in sorted(router.by_port)])
        boots = []
        for index in range(len(matches)):
            boot_dir = Path(session_cfg.replay_dir) / f"boot_{index:03d}"
            boot_dir.mkdir(parents=True, exist_ok=True)
            (boot_dir / "Game_20260804T000000.slp").write_bytes(_synthetic_slp(b"U\x08metadata{}}"))
            boots.append([_trajectory(frames=300, damage_port_1=40.0, damage_port_2=10.0, stocks_left={1: 3, 2: 0})])
        return boots

    class _SeededPolicy:
        def __init__(self, seed: int) -> None:
            self.seed = seed

        runtime_spec = PolicyRuntimeSpec(8, 4, 2, 0, 14)

        def plan_rows(self, rows):
            return dict.fromkeys(rows, "input")

    monkeypatch.setattr("hal.eval.h2h.run_matches_vec", fake_run_matches_vec)

    def build_policies(by_port, seed):
        return H2HPolicy({port: _SeededPolicy(seed + index) for index, port in enumerate(by_port)})

    records = run_h2h(
        build_policies,
        name_a="alpha",
        name_b="beta",
        n_configs=3,
        out_dir=tmp_path,
        session_cfg=SessionConfig(iso_path="iso", dolphin_path="dolphin"),
        max_frames=7200,
        max_parallel=2,
        seed=5,
        verify_inputs=False,
    )

    assert len(records) == 6
    assert [r.orientation for r in records] == [0, 0, 0, 1, 1, 1]
    assert [r.config_id for r in records] == [0, 1, 2, 0, 1, 2]
    # Port 1 leads every match, so each model leads exactly its own orientation.
    assert {r.outcome.stock_leader_model for r in records if r.outcome} == {"alpha", "beta"}
    # The orientations use disjoint decode seeds.
    assert seeds == [[5, 6], [100_005, 100_006]]

    # Replays live under a directory named for the port model A sat on, and carry the
    # identity of the model that produced them.
    forward = tmp_path / "replays" / "alpha-on-port1" / "boot_000" / records[0].match_id
    assert records[0].replay_path == str(forward.with_suffix(".slp"))
    assert records[0].identity_stamped is True
    assert replay_display_names(forward.with_suffix(".slp")) == {1: "alpha", 2: "beta"}
    mirror = Path(records[3].replay_path)
    assert mirror.parent.parent.name == "alpha-on-port2"
    assert replay_display_names(mirror) == {1: "beta", 2: "alpha"}

    assert load_records(tmp_path / "matches.jsonl") == records
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["record_schema_version"] == 3
    assert meta["model_a"] == "alpha" and meta["n_matches"] == 6
    assert meta["matches_completed"] == 6 and meta["matches_failed"] == 0
