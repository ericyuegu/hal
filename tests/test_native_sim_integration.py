"""Real MSL boundary checks; fixtures and native libraries are required.

Run with PYTHONPATH pointing at the MSL checkout and zstandard installed.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import TypedDict
from typing import cast

import numpy as np
import pytest

from hal.data import extract
from hal.data import slippi
from hal.sim import native
from hal.sim import native_build
from hal.training import returns

pytestmark = pytest.mark.integration


class ReplayExport(TypedDict):
    abi: int
    config_size: int
    input_size: int
    frame_size: int
    tape: bytearray


@pytest.fixture(scope="module")
def msl_root() -> Path:
    root = Path(os.environ.get("HAL_MSL_SOURCE", "~/src/melee-sim-light")).expanduser().resolve()
    assert (root / "src/api.h").is_file(), f"MSL source fixture missing: {root}"
    assert (root / "data/raw/manifest.json").is_file(), f"MSL game-data fixture missing: {root}"
    return root


@pytest.fixture
def real_msl(msl_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    library = msl_root / "build/melee_core/python/libmelee_core.so"
    assert library.is_file(), f"build real MSL with make python-library: {library}"
    monkeypatch.syspath_prepend(str(msl_root))
    monkeypatch.setenv("MSL_CORE_LIBRARY", str(library))
    adapter = native_build.build(msl_root, tmp_path / "libhal_native.so")
    monkeypatch.setenv("HAL_NATIVE_LIBRARY", str(adapter))
    return msl_root / "data"


def _match(seed: int = 17) -> native.NativeMatch:
    return native.NativeMatch(25, (native.NativePlayer(1, 1), native.NativePlayer(2, 22)), seed)


def test_real_build_buffers_and_cross_batch_restore(real_msl: Path) -> None:
    records = np.empty(3, dtype=native.FRAME_DTYPE)
    selected = np.ones(3, dtype=np.bool_)
    actions = np.zeros((3, 2, 14), dtype=np.float32)
    actions[:, 0, 0] = 0.75
    with native.NativeRolloutBatch(real_msl, 3, records=records) as source:
        frame = source.reset(tuple(_match(seed) for seed in (1, 2, 3)), selected)
        assert frame.records is records
        for _ in range(150):
            source.step(actions, selected)
        saved = tuple(source.save(lane) for lane in range(3))
        expected = []
        for _ in range(20):
            source.step(actions, selected)
            expected.append(records.copy())
        with native.NativeRolloutBatch(real_msl, 4) as target:
            target.restore_many((2, 0, 3), saved)
            target_actions = np.zeros((4, 2, 14), dtype=np.float32)
            target_actions[[2, 0, 3]] = actions
            target_mask = np.array([True, False, True, True])
            for reference in expected:
                actual = target.step(target_actions, target_mask)
                assert actual.records is not None
                assert actual.records[[2, 0, 3]].tobytes() == reference.tobytes()


def _recording(msl_root: Path, name: str) -> Iterator[tuple[Path, native.NativeMatch, ReplayExport]]:
    validation = importlib.import_module("tools.validation.validate_replay")
    slpz = importlib.import_module("tools.validation.slpz")
    admission = importlib.import_module("tools.validation.admission")
    peppi = importlib.import_module("peppi_py")
    path = msl_root / "replays/validation" / name
    assert path.is_file(), f"required replay fixture missing: {path}"
    with slpz.replay_path_for_peppi(path) as replay:
        game = peppi._read_slippi(str(replay), False)
        admission.require_admissible(game, replay)
        tape = validation.load_native().export_replay(game.frames, game.start, game.metadata)
        start = game.start
        players = tuple(
            native.NativePlayer(
                port=index + 1,
                character=int(slippi.slp_character_to_libmelee(int(player["character"])).value),
                costume=int(player["costume"]),
                handicap=int(player["handicap"]),
                start_percent=int(player["damage_start"]),
            )
            for index, player in enumerate(start["players"])
        )
        assert len(players) == 2
        match = native.NativeMatch(
            int(slippi.slp_stage_to_libmelee(int(start["stage"])).value),
            (players[0], players[1]),
            int(start["random_seed"]),
            stocks=int(start["players"][0]["stocks"]),
            damage_ratio=float(start["damage_ratio"]),
        )
        yield replay, match, cast(ReplayExport, tape)


@pytest.mark.parametrize(
    "name",
    (
        "aggregate_recent/BlondHardHippopotamus.slpz",
        "cardinal_1.0_recent/AttachedGoodNaturedGuanaco.slpz",
        "aggregate_recent/Game_20260515T182447_frozenps.slpz",
        "fountain_of_dreams_recent/ElatedWearyTermite.slpz",
        "yoshis_story_recent/CheeryNumbMonkey.slpz",
        "dream_land_recent/FlippantEnchantedHorse.slpz",
        "battlefield_recent/MediumVirtualPig.slpz",
        "icies/auto-falco-2025-10_Game_20251013T073200.slpz",
        "zelda/corpus_fox_20250129T002820.slpz",
        "samus/auto-cptfalcon-2025-05_Game_20250518T000407.slpz",
        "peach/PerfectJollyMagpie.slpz",
        "gamewatch/Game_20260707T202855.slpz",
    ),
)
def test_admitted_replay_columns_and_rewards_are_bitwise(real_msl: Path, msl_root: Path, name: str) -> None:
    for path, match, tape in _recording(msl_root, name):
        sample = extract.extract_replay(str(path))
        assert sample is not None
        assert tape["abi"] == 1
        config_size, input_size, frame_size = (int(tape[key]) for key in ("config_size", "input_size", "frame_size"))
        raw = np.frombuffer(tape["tape"], dtype=np.uint8)
        config = raw[:config_size].reshape(1, config_size)
        previous = raw[config_size : config_size + input_size].reshape(1, input_size)
        frames = raw[config_size + input_size :].reshape(-1, 1, frame_size)
        records = []
        with native.NativeRolloutBatch(real_msl, 1) as batch:
            assert batch.replay_sizes() == (config_size, input_size, frame_size)
            for invalid in (
                replace(match, seed=match.seed ^ 1),
                replace(match, max_frame=0),
                replace(match, players=(replace(match.players[0], facing=1), match.players[1])),
                replace(
                    match,
                    players=(
                        replace(match.players[0], character=0 if match.players[0].character else 1),
                        match.players[1],
                    ),
                ),
            ):
                with pytest.raises(ValueError, match="configuration"):
                    batch.reset_replay((invalid,), config, previous, frames[0])
                assert batch.frame.frame_id[0] == 0
            first = frames[0].copy()
            first[0, 6] = 37  # Replay ABI 1: frame seed + buttons + main_x.
            batch.reset_replay((match,), config, previous, first)
            assert batch.frame.wire_inputs[0, 0, 0] == 37
            batch.reset_replay((match,), config, previous, frames[0])
            assert batch.frame.records is not None
            assert batch.frame.reset[0]
            records.append(batch.frame.records.copy()[0])
            for row in frames[1:]:
                batch.step_replay(row, np.ones(1, dtype=np.bool_))
                records.append(batch.frame.records.copy()[0])
        actual = np.asarray(records, dtype=native.FRAME_DTYPE)
        np.testing.assert_array_equal(actual["frame_id"], sample["frame"])
        mismatches = []
        for column, name in enumerate(native.OBSERVATION_FIELDS):
            expected = sample[name].view(np.uint32)
            different = np.flatnonzero(actual["columns"][:, column] != expected)
            if different.size:
                index = int(different[0])
                mismatches.append(
                    f"{name} at {int(sample['frame'][index])}: "
                    f"0x{int(actual['columns'][index, column]):08x} != 0x{int(expected[index]):08x}"
                )
        for port, opp in ((1, 2), (2, 1)):
            expected_reward = returns.frame_reward(
                sample, ego=f"p{port}", opp=f"p{opp}", damage_shaping=1.0, win_reward=50.0, stock_value=120.0
            )
            different = np.flatnonzero(
                actual["reward"][:, port - 1].view(np.uint32) != expected_reward.view(np.uint32)
            )
            if different.size:
                index = int(different[0])
                mismatches.append(f"p{port} reward at {int(sample['frame'][index])}")
        assert not mismatches, "\n".join(mismatches)


def test_page_locked_output_and_parallel_shards(real_msl: Path) -> None:
    import contextlib
    import threading

    import torch

    from hal.sim import native_shard

    assert torch.cuda.is_available(), "required page-locked CUDA fixture missing"
    cpus = sorted(os.sched_getaffinity(0))[:2]
    assert len(cpus) == 2, "two CPU threads are required"
    buffers = [torch.empty(4 * native.FRAME_DTYPE.itemsize, dtype=torch.uint8, pin_memory=True) for _ in cpus]
    records = [buffer.numpy().view(native.FRAME_DTYPE) for buffer in buffers]
    actions = [np.zeros((4, 2, 14), dtype=np.float32) for _ in cpus]
    masks = [np.ones(4, dtype=np.bool_) for _ in cpus]
    with contextlib.ExitStack() as stack:
        shards = [
            stack.enter_context(native_shard.NativeShard(real_msl, 4, cpu=cpu, records=record))
            for cpu, record in zip(cpus, records, strict=True)
        ]
        for shard, mask in zip(shards, masks, strict=True):
            shard.reset((_match(),) * 4, mask).result()
        # A two-party barrier proves the native calls are dispatched to distinct threads.
        barrier = threading.Barrier(2, timeout=10)

        def synchronize_shards() -> None:
            barrier.wait()

        for _ in range(150):
            futures = [
                shard.step(action, mask, before_write=synchronize_shards)
                for shard, action, mask in zip(shards, actions, masks, strict=True)
            ]
            frames = [future.result() for future in futures]
            assert frames[0].records is records[0] and frames[1].records is records[1]
            assert records[0].tobytes() == records[1].tobytes()
        # The shard waits for an actual H2D event before it overwrites pinned output.
        copied = buffers[0].to("cuda", non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        expected = buffers[0].clone()
        shards[0].step(actions[0], masks[0], before_write=event.synchronize).result()
        assert torch.equal(copied.cpu(), expected)


def test_controller_bytes_match_dolphin_recording(real_msl: Path, tmp_path: Path) -> None:
    import melee
    import peppi_py

    from hal import controller
    from hal import paths
    from hal.sim import loop
    from hal.sim import session
    from hal.sim import sources

    assert Path(paths.ISO_PATH).is_file(), f"required ISO fixture missing: {paths.ISO_PATH}"
    assert Path(paths.EMULATOR_PATH).is_file(), f"required Dolphin fixture missing: {paths.EMULATOR_PATH}"
    rng = np.random.default_rng(741)
    stream = np.zeros((320, 2, 14), dtype=np.float32)
    stream[..., :4] = rng.integers(-80, 81, (320, 2, 4)) / 80
    stream[..., 4:6] = rng.integers(0, 141, (320, 2, 2)) / 140
    stream[..., 6:] = rng.integers(0, 2, (320, 2, 8))
    # Entry suppresses the first controller sample; a neutral prefix moves the
    # comparison past countdown. A neutral tail lets the recording flush.
    actions = np.concatenate(
        (np.zeros((128, 2, 14), dtype=np.float32), stream, np.zeros((16, 2, 14), dtype=np.float32))
    )
    expected = native._quantize(stream)
    matchup = session.ReplayMatchup(
        stage=melee.Stage.FINAL_DESTINATION,
        players=(session.PlayerSetup(1, melee.Character.FOX), session.PlayerSetup(2, melee.Character.FALCO)),
        port_to_mds_prefix={1: "p1", 2: "p2"},
    )
    inputs = {
        port: sources.ScriptedControllerSource(
            sequence=[controller.action_vec_to_controller(row[port - 1]) for row in actions]
        )
        for port in (1, 2)
    }
    with session.Session(
        iso_path=paths.ISO_PATH,
        dolphin_path=paths.EMULATOR_PATH,
        slippi_port=51457,
        replay_dir=str(tmp_path),
        blocking_input=True,
    ) as match:
        loop.drive(match, matchup, inputs, max_frames=len(actions) + 2)
    replays = sorted(tmp_path.rglob("*.slp"))
    assert replays, "Dolphin did not record a replay"
    game = peppi_py.read_slippi(str(replays[-1]))
    assert game.frames is not None
    ids = np.asarray(game.frames.id.to_pylist(), dtype=np.int32)
    for player in (0, 1):
        pre = game.frames.ports[player].leader.pre
        axes = (pre.raw_analog_x, pre.raw_analog_y, pre.raw_analog_cstick_x, pre.raw_analog_cstick_y)
        assert all(axis is not None for axis in axes)
        axis_columns = [np.asarray(axis.to_pylist(), dtype=np.int16) for axis in axes if axis is not None]
        observed = np.column_stack(
            (
                *axis_columns,
                np.rint(pre.triggers_physical.l.to_numpy() * 140),
                np.rint(pre.triggers_physical.r.to_numpy() * 140),
                pre.buttons_physical.to_numpy(),
            )
        ).astype(np.int16)
        # HAL sends an action after observing frame t; Slippi pre[t+1] records it.
        first = int(np.flatnonzero(ids == 6)[0])
        assert observed[first : first + len(stream)].tobytes() == expected[:, player].tobytes()
