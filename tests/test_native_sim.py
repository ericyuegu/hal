"""Check stored policy columns and actions through the native simulator boundary."""

from __future__ import annotations

import ctypes
import gc
import hashlib
import json
import math
import os
import struct
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import melee
import numpy as np
import pytest

from hal.data.schema import MDS_PER_FRAME_DTYPES
from hal.data.slippi import slp_stage_to_libmelee
from hal.sim import native
from hal.sim.native_build import build
from hal.training.returns import frame_reward
from hal.wire import ACTION_CHANNELS
from hal.wire import ACTION_DIM
from hal.wire import BUTTON_BITS
from hal.wire import MASK_INT32
from hal.wire import mask_value

_FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fake_sim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data = tmp_path / "data"
    (data / "raw").mkdir(parents=True)
    (data / "raw" / "manifest.json").write_text("{}")
    library = tmp_path / "libmelee_core.so"
    subprocess.run(
        [
            "cc",
            "-std=c11",
            "-shared",
            "-fPIC",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(_FIXTURES),
            str(_FIXTURES / "native_sim_stub.c"),
            "-o",
            str(library),
        ],
        check=True,
    )
    native_source = tmp_path / "native_source"
    (native_source / "src").mkdir(parents=True)
    (native_source / "src" / "api.h").write_bytes((_FIXTURES / "msl_api.h").read_bytes())
    bridge = tmp_path / "libhal_native.so"
    build(native_source, bridge)
    monkeypatch.setenv("HAL_NATIVE_LIBRARY", str(bridge))
    sim = SimpleNamespace(abi_version=lambda: 2, native_library_path=lambda: library, Character=int)
    original_import = native.importlib.import_module
    monkeypatch.setattr(
        native.importlib, "import_module", lambda name: sim if name == "melee_sim" else original_import(name)
    )
    return data


def _test_library(data: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(data.parent / "libmelee_core.so"))
    library.msl_game_data_references.restype = ctypes.c_uint32
    library.msl_test_set_reward.argtypes = [ctypes.c_uint32, *([ctypes.c_uint8] * 4), *([ctypes.c_float] * 4)]
    library.msl_test_disappear_on_step.argtypes = [ctypes.c_uint32]
    library.msl_test_bad_roster_on_reset.argtypes = [ctypes.c_uint32]
    library.msl_test_copy_last_input.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
    return library


def _last_input(data: Path, lane: int, roster: int) -> tuple[int, int, int, int, int, int, int]:
    raw = (ctypes.c_uint8 * 32)()
    _test_library(data).msl_test_copy_last_input(lane, raw)
    buttons, main_x, main_y, c_x, c_y, left, right = struct.unpack_from("<HbbbbBB", raw, roster * 8)
    return main_x, main_y, c_x, c_y, left, right, buttons


def _matches(batch_size: int) -> tuple[native.NativeMatch, ...]:
    return tuple(
        native.NativeMatch(
            stage=25,
            players=(native.NativePlayer(port=2, character=1), native.NativePlayer(port=1, character=22)),
            seed=123 + lane,
        )
        for lane in range(batch_size)
    )


def test_reset_projects_typed_71_fields_with_port_and_item_order(fake_sim: Path) -> None:
    with native.NativeRolloutBatch(fake_sim, 2, chunk_frames=2) as batch:
        assert batch.manifest_sha256 == hashlib.sha256((fake_sim / "raw" / "manifest.json").read_bytes()).hexdigest()
        assert batch.library_sha256 == hashlib.sha256((fake_sim.parent / "libmelee_core.so").read_bytes()).hexdigest()
        frame = batch.reset(_matches(2), np.array([True, False]))
        assert len(frame.columns) == 71
        assert all(frame.columns[name].dtype == np.dtype(MDS_PER_FRAME_DTYPES[name]) for name in frame.columns)
        assert frame.frame_id[0] == -123 and frame.reset.tolist() == [True, False]
        assert frame.columns["stage"][0] == 25
        assert frame.columns["p1_character"][0] == 22 and frame.columns["p2_character"][0] == 1
        assert frame.columns["p1_position_x"][0] == 11 and frame.columns["p2_position_x"][0] == 10
        assert frame.columns["p1_jumps_used"][0] == 1  # HAL's named field means jumps remaining.
        assert frame.columns["p1_hitlag_left"][0] == 1.25
        assert np.isnan(frame.columns["p1_nana_position_x"][0])
        assert frame.columns["p1_nana_stock"][0] == MASK_INT32
        assert [int(frame.columns[f"item{i}_type"][0]) for i in range(4)] == [1, 3, 5, 7]
        assert frame.columns["p1_stock"][1] == MASK_INT32


def test_step_quantizes_all_channels_routes_roster_rewards_and_masks(fake_sim: Path) -> None:
    with native.NativeRolloutBatch(fake_sim, 2, chunk_frames=2) as batch:
        batch.reset(_matches(2), np.array([True, True]))
        actions = np.zeros((2, 2, ACTION_DIM), dtype=np.float32)
        actions[0, 0, 0] = -0.5
        actions[0, 0, 4] = np.float32(42 / 140)
        actions[0, 0, 5] = np.float32(43 / 140)
        actions[0, 0, 6] = 1  # A
        actions[0, 0, 11] = 1  # R
        actions[0, 1, 0] = 0.5
        frame = batch.step(actions, np.array([True, False]))
        assert frame.frame_id.tolist() == [-122, -123]
        assert frame.wire_inputs[0, 0].tolist() == [-40, 0, 0, 0, 42, 140, 0x0120]
        assert frame.wire_inputs[0, 1, 0] == 40
        assert _last_input(fake_sim, 0, 1)[0] == -40
        assert _last_input(fake_sim, 0, 0)[0] == 40
        expected = frame_reward(
            {
                "p1_stock": np.array([4, 4], dtype=np.int32),
                "p2_stock": np.array([4, 3], dtype=np.int32),
                "p1_percent": np.array([0, 0.3], dtype=np.float32),
                "p2_percent": np.array([0, 10.1], dtype=np.float32),
            },
            ego="p1",
            opp="p2",
            damage_shaping=0.0,
            win_reward=50.0,
            stock_value=120.0,
        )[1]
        assert frame.reward[0, 0] == expected and frame.reward[0, 1] == -expected
        assert frame.reward[1].tolist() == [0.0, 0.0]
        assert not frame.reset.any() and not frame.terminated.any()
        frame = batch.step(actions, np.array([True, False]))
        assert frame.terminated.tolist() == [True, False]
        assert frame.reward.tolist() == [[170.0, -170.0], [0.0, 0.0]]
        with pytest.raises(ValueError, match="reset terminal"):
            batch.step(actions, np.array([True, False]))


def test_save_restore_replays_same_branch_and_validates_identity(fake_sim: Path) -> None:
    with native.NativeRolloutBatch(fake_sim, 1, chunk_frames=2) as batch:
        match = _matches(1)
        batch.reset(match, np.array([True]))
        action = np.zeros((1, 2, ACTION_DIM), dtype=np.float32)
        batch.step(action, np.array([True]))
        saved = batch.save(0)
        first = batch.step(action, np.array([True]))
        expected = (first.frame_id.copy(), first.reward.copy(), first.terminated.copy(), first.applied_action.copy())
        restored = batch.restore(0, saved)
        assert restored.frame_id[0] == -122 and restored.reward[0].tolist() == [0.0, 0.0]
        replayed = batch.step(action, np.array([True]))
        for actual, baseline in zip(
            (replayed.frame_id, replayed.reward, replayed.terminated, replayed.applied_action), expected, strict=True
        ):
            np.testing.assert_array_equal(actual, baseline)
        bad = bytearray(saved)
        offset = saved.index(b'"manifest_sha256":"') + len(b'"manifest_sha256":"')
        bad[offset] = ord("0") if bad[offset] != ord("0") else ord("1")
        with pytest.raises(ValueError, match="identity"):
            batch.restore(0, bytes(bad))


def test_rejects_invalid_abi_action_and_match(fake_sim: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sim = native.importlib.import_module("melee_sim")
    monkeypatch.setattr(sim, "abi_version", lambda: 1)
    with pytest.raises(RuntimeError, match="ABI 2"):
        native.NativeRolloutBatch(fake_sim, 1)
    monkeypatch.setattr(sim, "abi_version", lambda: 2)
    with native.NativeRolloutBatch(fake_sim, 1) as batch:
        with pytest.raises(ValueError, match="stage"):
            batch.reset((native.NativeMatch(32, _matches(1)[0].players, 5),), np.array([True]))
        batch.reset(_matches(1), np.array([True]))
        action = np.zeros((1, 2, ACTION_DIM), dtype=np.float32)
        action[0, 0, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            batch.step(action, np.array([True]))


def test_follower_presence_and_frame_horizon(fake_sim: Path) -> None:
    base = _matches(1)[0]
    match = native.NativeMatch(
        stage=base.stage,
        players=(base.players[0], native.NativePlayer(port=1, character=10)),
        seed=base.seed,
        max_frame=0,
    )
    with native.NativeRolloutBatch(fake_sim, 1) as batch:
        frame = batch.reset((match,), np.array([True]))
        assert frame.columns["p1_nana_position_x"][0] == -9
        assert frame.columns["p1_nana_jumps_used"][0] == 2
        action = np.zeros((1, 2, ACTION_DIM), dtype=np.float32)
        for _ in range(123):
            frame = batch.step(action, np.array([True]))
        assert frame.frame_id[0] == 0 and frame.truncated[0] and not frame.terminated[0]


def test_all_buttons_and_half_step_axis_quantization(fake_sim: Path) -> None:
    actions = np.zeros((1, 2, ACTION_DIM), dtype=np.float32)
    actions[0, 0, :4] = np.array([0.00625, -0.00625, 1, -1], dtype=np.float32)
    actions[0, 0, 4:6] = np.array([1, 0], dtype=np.float32)
    actions[0, 0, 6:] = 1
    wire = native._quantize(actions)
    assert wire[0, 0].tolist() == [1, -1, 80, -80, 140, 140, 0x0F78]


def test_supported_stage_ids_match_pinned_libmelee() -> None:
    for hal_stage, slippi_stage in native._STAGE_TO_NATIVE.items():
        assert int(slp_stage_to_libmelee(slippi_stage).value) == hal_stage


def test_quantization_matches_pinned_libmelee_at_threshold_neighbors(fake_sim: Path) -> None:
    stick_values = []
    for boundary in range(-80, 80):
        value = np.float32((boundary + 0.5) / 80)
        stick_values.extend((np.nextafter(value, np.float32(-1)), value, np.nextafter(value, np.float32(1))))
    trigger_values = []
    for boundary in range(0, 140):
        value = np.float32((boundary + 0.5) / 140)
        trigger_values.extend((np.nextafter(value, np.float32(0)), value, np.nextafter(value, np.float32(1))))
    count = max(len(stick_values), len(trigger_values))
    actions = np.zeros((count, 2, ACTION_DIM), dtype=np.float32)
    actions[: len(stick_values), 0, 0] = stick_values
    actions[: len(trigger_values), 0, 4] = trigger_values
    actual = native._quantize(actions)
    for row, value in enumerate(stick_values):
        wire = melee.controller.fix_analog_stick_signed(float(value))
        expected = math.floor((wire - 0.5) * 254)
        assert actual[row, 0, 0] == expected, (row, value, expected)
    for row, value in enumerate(trigger_values):
        wire = melee.controller.fix_analog_trigger(float(value))
        expected = math.floor(wire * 255)
        assert actual[row, 0, 4] == expected, (row, value, expected)


@pytest.mark.parametrize(
    ("old_stock", "new_stock", "old_percent", "new_percent"),
    [
        ((4, 4), (3, 4), (0.0, 0.0), (0.0, 0.0)),
        ((4, 1), (4, 0), (0.0, 0.0), (0.0, 0.0)),
        ((2, 1), (1, 0), (0.0, 0.0), (0.0, 0.0)),
        ((4, 4), (4, 4), (0.1, 0.2), (10.12345, 20.54321)),
        ((3, 4), (2, 4), (85.0, 0.0), (0.0, 0.0)),
        ((4, 4), (4, 4), (float("nan"), 0.0), (10.0, float("nan"))),
        ((4, 4), (4, 4), (-0.0, 0.0), (0.0, -0.0)),
        ((4, 4), (4, 4), (-3e38, 0.0), (3e38, 3e38)),
    ],
)
def test_reward_matches_canonical_float32_bytes(
    fake_sim: Path,
    old_stock: tuple[int, int],
    new_stock: tuple[int, int],
    old_percent: tuple[float, float],
    new_percent: tuple[float, float],
) -> None:
    original = _matches(1)[0]
    quiet_match = native.NativeMatch(original.stage, original.players, original.seed, max_frame=0)
    _test_library(fake_sim).msl_test_set_reward(0, *old_stock, *new_stock, *old_percent, *new_percent)
    with native.NativeRolloutBatch(fake_sim, 1) as batch:
        batch.reset((quiet_match,), np.array([True]))
        with np.errstate(over="ignore", invalid="ignore"):
            frame = batch.step(np.zeros((1, 2, ACTION_DIM), dtype=np.float32), np.array([True]))
            expected = frame_reward(
                {
                    "p1_stock": np.asarray([old_stock[0], new_stock[0]], dtype=np.int32),
                    "p2_stock": np.asarray([old_stock[1], new_stock[1]], dtype=np.int32),
                    "p1_percent": np.asarray([old_percent[0], new_percent[0]], dtype=np.float32),
                    "p2_percent": np.asarray([old_percent[1], new_percent[1]], dtype=np.float32),
                },
                ego="p1",
                opp="p2",
                damage_shaping=0.0,
                win_reward=50.0,
                stock_value=120.0,
            )[1]
        with np.errstate(over="ignore", invalid="ignore"):
            expected_opp = frame_reward(
                {
                    "p1_stock": np.asarray([old_stock[0], new_stock[0]], dtype=np.int32),
                    "p2_stock": np.asarray([old_stock[1], new_stock[1]], dtype=np.int32),
                    "p1_percent": np.asarray([old_percent[0], new_percent[0]], dtype=np.float32),
                    "p2_percent": np.asarray([old_percent[1], new_percent[1]], dtype=np.float32),
                },
                ego="p2",
                opp="p1",
                damage_shaping=0.0,
                win_reward=50.0,
                stock_value=120.0,
            )[1]
        expected_pair = np.asarray([expected, expected_opp], dtype=np.float32)
        np.testing.assert_array_equal(frame.reward[0].view(np.uint32), expected_pair.view(np.uint32))


def test_disappearing_fighter_follower_and_items_clear_without_touching_masked_lane(fake_sim: Path) -> None:
    base = _matches(2)
    matches = tuple(
        native.NativeMatch(
            item.stage,
            (item.players[0], native.NativePlayer(port=1, character=10)),
            item.seed,
            max_frame=0,
        )
        for item in base
    )
    with native.NativeRolloutBatch(fake_sim, 2) as batch:
        frame = batch.reset(matches, np.array([True, True]))
        held = {name: column[1].tobytes() for name, column in frame.columns.items()}
        assert frame.columns["p1_nana_position_x"][0] == -9
        assert [frame.columns[f"item{i}_type"][0] for i in range(4)] == [1, 3, 5, 7]
        _test_library(fake_sim).msl_test_disappear_on_step(0)
        frame = batch.step(np.zeros((2, 2, ACTION_DIM), dtype=np.float32), np.array([True, False]))
        assert frame.columns["p1_stock"][0] == 3
        for prefix in ("p1", "p1_nana"):
            for name in native._PLAYER_FIELDS:
                if prefix == "p1" and name == "stock":
                    continue
                key = f"{prefix}_{name}"
                value = frame.columns[key][0]
                mask = mask_value(MDS_PER_FRAME_DTYPES[key])
                assert np.isnan(value) if isinstance(mask, float) else value == mask, key
        assert frame.columns["item0_type"][0] == 1
        for slot in range(1, 4):
            for name in native._ITEM_FIELDS:
                key = f"item{slot}_{name}"
                value = frame.columns[key][0]
                mask = mask_value(MDS_PER_FRAME_DTYPES[key])
                assert np.isnan(value) if isinstance(mask, float) else value == mask, key
        assert frame.frame_id.tolist() == [-122, -123]
        assert frame.reward[1].tolist() == [0.0, 0.0]
        assert all(column[1].tobytes() == held[name] for name, column in frame.columns.items())


def test_strided_actions_and_mask_reach_the_right_native_roster(fake_sim: Path) -> None:
    with native.NativeRolloutBatch(fake_sim, 2) as batch:
        batch.reset(_matches(2), np.array([True, True]))
        storage = np.zeros((2, 2, ACTION_DIM * 2), dtype=np.float32)
        actions = storage[::-1, ::-1, ::2]
        actions[0, 0, 0] = -0.5
        actions[0, 1, 0] = 0.5
        actions[0, 0, 6] = 1
        mask = np.array([False, True], dtype=np.bool_)[::-1]
        assert not actions.flags.c_contiguous and mask.strides[0] < 0
        frame = batch.step(actions, mask)
        assert frame.frame_id.tolist() == [-122, -123]
        assert frame.wire_inputs[0, 0].tolist() == [-40, 0, 0, 0, 0, 0, 0x0100]
        assert frame.wire_inputs[0, 1, 0] == 40
        assert _last_input(fake_sim, 0, 1) == (-40, 0, 0, 0, 0, 0, 0x0100)
        assert _last_input(fake_sim, 0, 0)[0] == 40
        assert frame.applied_action[0, 0, 0] == -0.5
        assert frame.applied_action[1].sum() == 0


def test_validation_is_atomic_across_selected_lanes(fake_sim: Path) -> None:
    with native.NativeRolloutBatch(fake_sim, 2) as batch:
        matches = _matches(2)
        invalid = native.NativeMatch(32, matches[1].players, matches[1].seed)
        with pytest.raises(ValueError, match="stage"):
            batch.reset((matches[0], invalid), np.array([True, True]))
        frame = batch.reset(matches, np.array([True, True]))
        np.testing.assert_array_equal(frame.frame_id, [-123, -123])
        actions = np.zeros((2, 2, ACTION_DIM), dtype=np.float32)
        actions[1, 0, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            batch.step(actions, np.array([True, True]))
        np.testing.assert_array_equal(frame.frame_id, [-123, -123])
        frame = batch.step(np.zeros_like(actions), np.array([True, True]))
        np.testing.assert_array_equal(frame.frame_id, [-122, -122])
        actions[1, 0, 0] = np.nan
        frame = batch.step(actions, np.array([True, False]))
        np.testing.assert_array_equal(frame.frame_id, [-121, -122])


def test_missing_roster_slot_is_rejected(fake_sim: Path) -> None:
    with native.NativeRolloutBatch(fake_sim, 1) as batch, pytest.raises(RuntimeError, match="unique roster"):
        _test_library(fake_sim).msl_test_bad_roster_on_reset(0)
        batch.reset(_matches(1), np.array([True]))


def test_close_is_idempotent_and_gc_releases_native_batch(fake_sim: Path) -> None:
    library = _test_library(fake_sim)
    assert library.msl_game_data_references() == 0
    batch = native.NativeRolloutBatch(fake_sim, 1)
    assert library.msl_game_data_references() == 1
    batch.reset(_matches(1), np.array([True]))
    batch.close()
    batch.close()
    assert library.msl_game_data_references() == 0
    with pytest.raises(RuntimeError, match="closed"):
        batch.step(np.zeros((1, 2, ACTION_DIM), dtype=np.float32), np.array([True]))
    with pytest.raises(RuntimeError, match="closed"):
        batch.reset(_matches(1), np.array([True]))
    with pytest.raises(RuntimeError, match="closed"):
        batch.save(0)
    batch = native.NativeRolloutBatch(fake_sim, 1)
    assert library.msl_game_data_references() == 1
    del batch
    gc.collect()
    assert library.msl_game_data_references() == 0


def test_random_quantization_matches_dolphin_physical_bytes(fake_sim: Path) -> None:
    rng = np.random.default_rng(7391)
    actions = rng.uniform(0, 1, size=(4096, 2, ACTION_DIM)).astype(np.float32)
    actions[..., :4] = rng.uniform(-1, 1, size=(4096, 2, 4)).astype(np.float32)
    expected = np.zeros((4096, 2, 7), dtype=np.int16)
    sticks = actions[..., :4].astype(np.float64)
    expected[..., :4] = np.rint(((sticks + 1.0) / 2.0 - 0.5) * 160.0).astype(np.int16)
    triggers = np.rint(actions[..., 4:6].astype(np.float64) * 140.0).astype(np.int16)
    expected[..., 4:6] = triggers
    button_masks = np.asarray(
        [BUTTON_BITS[channel.removeprefix("button_")] for channel in ACTION_CHANNELS[6:]], dtype=np.int16
    )
    expected[..., 6] = np.sum((actions[..., 6:] > np.float32(0.5)) * button_masks, axis=-1, dtype=np.int16)
    expected[..., 4] = np.where(expected[..., 6] & BUTTON_BITS["l"], 140, expected[..., 4])
    expected[..., 5] = np.where(expected[..., 6] & BUTTON_BITS["r"], 140, expected[..., 5])
    actual = native._quantize(actions)
    np.testing.assert_array_equal(actual.view(np.uint8), expected.view(np.uint8))


def test_applied_action_can_be_used_as_next_input(fake_sim: Path) -> None:
    with native.NativeRolloutBatch(fake_sim, 1) as batch:
        frame = batch.reset(_matches(1), np.array([True]))
        actions = frame.applied_action
        actions[0, 0, 0] = -0.5
        actions[0, 0, 6] = 1
        actions[0, 1, 0] = 0.5
        stepped = batch.step(actions, np.array([True]))
        assert stepped.applied_action is actions
        assert stepped.applied_action[0, 0, 0] == -0.5
        assert stepped.wire_inputs[0, 0].tolist() == [-40, 0, 0, 0, 0, 0, 0x0100]
        assert _last_input(fake_sim, 0, 1) == (-40, 0, 0, 0, 0, 0, 0x0100)


def test_partial_initial_reset_can_step_active_lane(fake_sim: Path) -> None:
    with native.NativeRolloutBatch(fake_sim, 2) as batch:
        frame = batch.reset(_matches(2), np.array([True, False]))
        assert frame.frame_id.tolist() == [-123, 0]
        assert frame.columns["p1_stock"][1] == MASK_INT32
        frame = batch.step(np.zeros((2, 2, ACTION_DIM), dtype=np.float32), np.array([True, False]))
        assert frame.frame_id.tolist() == [-122, 0]
        assert frame.columns["p1_stock"][1] == MASK_INT32


def test_restore_into_one_lane_of_fresh_multi_lane_batch(fake_sim: Path) -> None:
    actions = np.zeros((2, 2, ACTION_DIM), dtype=np.float32)
    with native.NativeRolloutBatch(fake_sim, 2) as source:
        source.reset(_matches(2), np.array([True, True]))
        source.step(actions, np.array([True, False]))
        saved = source.save(0)
    with native.NativeRolloutBatch(fake_sim, 2) as target:
        frame = target.restore(0, saved)
        assert frame.frame_id.tolist() == [-122, 0]
        assert frame.columns["p1_stock"][1] == MASK_INT32
        frame = target.step(actions, np.array([True, False]))
        assert frame.frame_id.tolist() == [-121, 0]


def test_restore_many_observes_once_and_validates_before_mutation(fake_sim: Path) -> None:
    with native.NativeRolloutBatch(fake_sim, 4) as batch:
        mask = np.ones(4, dtype=np.bool_)
        batch.reset(_matches(4), mask)
        saved = tuple(batch.save(lane) for lane in range(4))
        library = _test_library(fake_sim)
        before = library.msl_test_observe_calls()
        batch.restore_many((3, 1, 2, 0), saved)
        assert library.msl_test_observe_calls() == before + 1
        for lanes, states in (((0, 0), saved[:2]), ((-1,), saved[:1]), ((0,), saved[:2])):
            with pytest.raises(ValueError):
                batch.restore_many(lanes, states)
        bad = saved[1].replace(b'"action_schema":"hal.controller.v1"', b'"action_schema":"hal.controller.v2"')
        with pytest.raises(ValueError, match="identity"):
            batch.restore_many((0, 1), (saved[0], bad))
        assert library.msl_test_observe_calls() == before + 1


def test_output_buffer_validation_and_ownership(fake_sim: Path) -> None:
    records = np.empty(2, dtype=native.FRAME_DTYPE)
    with native.NativeRolloutBatch(fake_sim, 2, records=records) as batch:
        frame = batch.reset(_matches(2), np.ones(2, dtype=np.bool_))
        assert frame.records is records
        assert np.shares_memory(frame.columns["p1_position_x"], records)
    readonly = records.copy()
    readonly.flags.writeable = False
    for bad in (readonly, records[:1], records[::-1], np.empty(2, dtype=np.float32)):
        with pytest.raises(ValueError, match="records"):
            native.NativeRolloutBatch(fake_sim, 2, records=bad)


def test_compressed_snapshot_rejects_corruption_and_trailing_bytes(fake_sim: Path) -> None:
    with native.NativeRolloutBatch(fake_sim, 1) as batch:
        batch.reset(_matches(1), np.ones(1, dtype=np.bool_))
        saved = batch.save(0)
        bad = bytearray(saved)
        bad[-1] ^= 1
        for malformed in (bytes(bad), saved + b"trailing", saved[:-1]):
            with pytest.raises(ValueError, match="snapshot"):
                batch.restore(0, malformed)


@pytest.mark.parametrize(
    "field", ("nonfinite", "axis", "axis_rounding", "button", "string", "wire_fraction", "wire_range", "facing")
)
def test_snapshot_rejects_invalid_action_metadata_before_restore(fake_sim: Path, field: str) -> None:
    with native.NativeRolloutBatch(fake_sim, 1) as batch:
        batch.reset(_matches(1), np.ones(1, dtype=np.bool_))
        saved = batch.save(0)
        size = struct.unpack_from("<I", saved, 8)[0]
        metadata = json.loads(saved[12 : 12 + size])
        if field == "nonfinite":
            metadata["applied_action"][0][0] = float("nan")
        elif field == "axis":
            metadata["applied_action"][0][0] = 1.01
        elif field == "axis_rounding":
            metadata["applied_action"][0][0] = 1.0 + 1e-8
        elif field == "button":
            metadata["applied_action"][0][6] = -0.1
        elif field == "string":
            metadata["applied_action"][0][0] = "0.5"
        elif field == "wire_fraction":
            metadata["wire_inputs"][0][0] = 0.5
        elif field == "wire_range":
            metadata["wire_inputs"][0][0] = 128
        else:
            metadata["config"]["players"][0]["facing"] = True
        changed = json.dumps(metadata, separators=(",", ":")).encode()
        malformed = saved[:8] + struct.pack("<I", len(changed)) + changed + saved[12 + size :]
        library = _test_library(fake_sim)
        before = library.msl_test_observe_calls()
        with pytest.raises(ValueError, match="snapshot"):
            batch.restore(0, malformed)
        assert library.msl_test_observe_calls() == before


def test_match_rejects_boolean_damage_ratio(fake_sim: Path) -> None:
    with native.NativeRolloutBatch(fake_sim, 1) as batch, pytest.raises(ValueError, match="damage_ratio"):
        batch.reset((replace(_matches(1)[0], damage_ratio=True),), np.ones(1, dtype=np.bool_))


def test_shard_wait_callback_failure_preserves_state(fake_sim: Path) -> None:
    import os

    from hal.sim import native_shard

    cpu = min(os.sched_getaffinity(0))
    with native_shard.NativeShard(fake_sim, 1, cpu=cpu) as shard:
        frame = shard.reset(_matches(1), np.ones(1, dtype=np.bool_)).result()

        def failed_copy() -> None:
            raise RuntimeError("copy failed")

        with pytest.raises(RuntimeError, match="copy failed"):
            shard.step(
                np.zeros((1, 2, 14), dtype=np.float32), np.ones(1, dtype=np.bool_), before_write=failed_copy
            ).result()
        assert frame.frame_id[0] == -123
        frame = shard.step(np.zeros((1, 2, 14), dtype=np.float32), np.ones(1, dtype=np.bool_)).result()
        assert frame.frame_id[0] == -122
    shard.close()
    with pytest.raises(RuntimeError, match="closed"):
        shard.save(0)


def test_raw_capture_owns_state_and_keeps_the_saved_format(fake_sim: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with native.NativeRolloutBatch(fake_sim, 1) as batch:
        batch.reset(_matches(1), np.array([True]))
        saved = batch.save(0)

        def reject_compression(payload: bytes) -> bytes:
            raise AssertionError("raw capture compressed simulator state")

        with monkeypatch.context() as capture_patch:
            capture_patch.setattr(native.zlib, "compress", reject_compression)
            captured = batch.capture(0)
        action = np.zeros((1, 2, ACTION_DIM), dtype=np.float32)
        action[0, 0, 6] = 1
        batch.step(action, np.array([True]))
        assert native.encode_snapshot(captured) == saved
        batch.restore(0, native.encode_snapshot(captured))
        assert batch.frame.frame_id[0] == captured.frame_id
        np.testing.assert_array_equal(batch.frame.applied_action[0], captured.applied_action)
        np.testing.assert_array_equal(batch.frame.wire_inputs[0], captured.wire_inputs)


def test_shard_raw_capture_is_independent_and_validates_lanes(fake_sim: Path) -> None:
    from hal.sim.native_shard import NativeShard

    with NativeShard(fake_sim, 2, cpu=min(os.sched_getaffinity(0))) as shard:
        shard.reset(_matches(2), np.ones(2, dtype=np.bool_)).result()
        captured = shard.capture_many((1, 0)).result()
        expected = shard.save_many((1, 0)).result()
        assert tuple(map(native.encode_snapshot, captured)) == expected
        with pytest.raises(ValueError, match="distinct"):
            shard.capture_many((0, 0)).result()
    with pytest.raises(RuntimeError, match="closed"):
        shard.capture_many((0,))
