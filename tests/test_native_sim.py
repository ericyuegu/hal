"""The HAL/native boundary preserves the policy's stored row contract."""

from __future__ import annotations

import ctypes
import gc
import hashlib
import math
import struct
import subprocess
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
        assert frame.wire_inputs[0, 0].tolist() == [-40, 0, 0, 0, 0, 43, 0x0120]
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
            damage_shaping=1.0,
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
    assert wire[0, 0].tolist() == [1, -1, 80, -80, 140, 0, 0x0F78]


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
        assert actual[row, 0, 4] == (0 if expected < 43 else expected), (row, value, expected)


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
                damage_shaping=1.0,
                win_reward=50.0,
                stock_value=120.0,
            )[1]
        expected_pair = np.asarray([expected, -expected], dtype=np.float32)
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


def test_random_quantization_has_old_float32_wire_bytes(fake_sim: Path) -> None:
    rng = np.random.default_rng(7391)
    actions = rng.uniform(0, 1, size=(4096, 2, ACTION_DIM)).astype(np.float32)
    actions[..., :4] = rng.uniform(-1, 1, size=(4096, 2, 4)).astype(np.float32)
    expected = np.zeros((4096, 2, 7), dtype=np.int16)
    sticks = actions[..., :4].astype(np.float64)
    expected[..., :4] = np.rint(((sticks + 1.0) / 2.0 - 0.5) * 160.0).astype(np.int16)
    triggers = np.rint(actions[..., 4:6].astype(np.float64) * 140.0).astype(np.int16)
    triggers[triggers < 43] = 0
    expected[..., 4:6] = triggers
    button_masks = np.asarray(
        [BUTTON_BITS[channel.removeprefix("button_")] for channel in ACTION_CHANNELS[6:]], dtype=np.int16
    )
    expected[..., 6] = np.sum((actions[..., 6:] > np.float32(0.5)) * button_masks, axis=-1, dtype=np.int16)
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
