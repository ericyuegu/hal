# Native Melee simulator

`hal.sim.native.NativeRolloutBatch` provides HAL's two-port rollout contract
through a compiled adapter over melee-sim-light's public C API. The simulator
owns gameplay. HAL owns controller conversion, policy observations, rewards,
action scheduling and trace collection. Simulation code does not import Torch.

This is not a drop-in libmelee implementation. It produces HAL's flat numeric
fields and starts configured matches directly. It does not provide Dolphin
menus, network play or libmelee `Console` and `Controller` objects.

## Build

Use a melee-sim-light checkout with observation ABI 2 and extracted game data.
ABI 2 adds exact float `hitlag_left` for leaders and followers. HAL rejects older
libraries before reading their observation buffers.

From the HAL checkout, with the simulator next to it:

```sh
make -C ../melee-sim-light python-library
uv run python -m hal.sim.native_build --native-source ../melee-sim-light
```

The simulator's root Makefile provides strict native release flags for
performance builds. Do not add fast-math flags. `MSL_CORE_LIBRARY` selects the
simulator shared library. `HAL_NATIVE_LIBRARY` selects the HAL adapter; its
default is `build/native_adapter/libhal_native.so` in the HAL checkout.
The explicit build command uses the same default destination as the loader.
For an installed package, use `--output /writable/path/libhal_native.so` and
set `HAL_NATIVE_LIBRARY` to that path.

The adapter build reads the selected simulator's public `src/api.h`. It generates
column and mask definitions from HAL's schema, checks the schema hash when
loading, and opens the exact library selected by `melee_sim`. Environment
construction never invokes a compiler. Restart processes after rebuilding
native libraries. The HAL wheel includes the adapter C source and build helper;
it does not contain game assets or a prebuilt simulator.

## API

```python
from pathlib import Path

import numpy as np

from hal.sim.native import NativeMatch, NativePlayer, NativeRolloutBatch

match = NativeMatch(
    stage=25,  # HAL/libmelee Final Destination ID, not native stage ID 32.
    players=(NativePlayer(port=1, character=1), NativePlayer(port=2, character=22)),
    seed=0,
)
with NativeRolloutBatch(Path("../melee-sim-light/data"), 1) as simulator:
    selected = np.array([True])
    frame = simulator.reset((match,), selected)
    actions = np.zeros((1, 2, 14), dtype=np.float32)
    frame = simulator.step(actions, selected)
```

Make the simulator's Python package available on `PYTHONPATH` when running this
example. Use one thread per batch. The batch supports multiple independent
matches through its first array dimension.

- `reset(configs, mask)` resets selected lanes and publishes their entry frame,
  frame -123. `configs` contains one configuration per lane.
- `step(actions, mask)` advances each selected match by one frame. Actions have
  shape `[B, 2, 14]` and dtype float32; masks are bool `[B]`. Strided arrays are
  accepted. Selected actions are validated before any match advances.
- `save(lane)` and `restore(lane, state)` preserve simulator and adapter state.
  They do not capture policy caches, sampling state or action-delay queues.
- A masked lane retains its observation and action history and receives zero
  reward. Inactive lanes must be reset or restored before stepping.
- Final stock loss sets `terminated`. The configured frame limit sets
  `truncated` unless that frame also terminates the match. Reset ended lanes
  explicitly; the adapter does not reset them automatically.

Returned arrays are borrowed views, overwritten by the next step, reset or
restore. `frame.columns` contains 71 fields with the dtypes declared by
`hal.data.schema.MDS_PER_FRAME_DTYPES`. Additional arrays are `frame_id`,
`applied_action`, `wire_inputs`, `reward`, `terminated`, `truncated` and `reset`.
`records` exposes their packed storage for trace collection. The retained
`chunk_frames` argument is accepted for existing callers but no longer controls
native history storage; the caller owns history.

The public native step API observes every slot, even when masked. HAL initializes
private core slots at construction but keeps their public outputs masked until
explicit reset or restore. This permits partial initial resets and restoration
of one lane into a fresh batch.

## Exact conversions

HAL actions contain four signed stick axes, independent L/R triggers, and
buttons A, B, X, Y, Z, R, L and D-up. Packing retains pinned libmelee's float64
intermediate arithmetic and ties-to-even rounding. Do not substitute the native
Python controller helper: its stick range and trigger representation differ.

Observations map configured ports to native roster slots, keep character-select
IDs across transformations, mask absent fighters and followers, and select the
four live items with the lowest spawn IDs. HAL's historical `jumps_used` column
stores jumps remaining. Integer masks and float NaN bytes retain HAL's schema.

The reward for port 1 is `120 * (opponent stocks lost - own stocks lost) +
50 * (opponent final-stock loss - own final-stock loss) +
(opponent positive damage increase - own positive damage increase)`.
Port 2 receives its negative. Percent operations and the final sum preserve the
existing float32 order, including signed zero and nonfinite-value handling.

Snapshots use `HALMSL02` and bind the adapter ABI and binary hash, simulator
binary hash, game-data manifest and HAL schemas. Older `HALMSL01` simulator
snapshots are rejected; model checkpoints are unaffected. A restore error after
native restoration can leave the lane changed; reset it before reuse.

## Run the pinned 059 policy

```sh
PYTHONPATH=../melee-sim-light:. \
uv run python scripts/eval_native_059.py \
  --data-dir ../melee-sim-light/data \
  --output /tmp/hal-native-059 --episodes 2 --max-frames 600
```

The output directory must not exist. The driver uses the local bundle
`runs/netplay/o59-vywk3cih.hal`, verifies its checkpoint hash and declared
transport delay, and retains all 78 trace arrays. It runs two controlled ports,
replays every recorded input and repeats the first model-driven episode.
The maintained timing is a four-frame prediction horizon, two committed actions,
a two-frame replan interval and two frames of physical transport delay.

`hal.eval.native_rollout.run_episode` owns scheduling and trace collection.
`replay_inputs` compares every saved array by bytes, including signed zeros and
NaN payloads. The independent retail-code comparison lives in slippi-cuda's
`tools/profile/hal_native_compare.py`. Only the seed field is excluded from that
comparison; downstream gameplay differences are not masked.

## Validation and performance

Tests in `tests/test_native_sim.py` build a small public-ABI C fixture to test the
actual compiled adapter without requiring game assets. They cover controller
rounding, roster routing, masks, rewards, reset, save/restore and cleanup.
`tests/test_native_rollout.py` checks frame scheduling and trace byte retention.
Real-game parity requires separate runs with extracted assets.

On a Ryzen 5 5600X core, the 600-step Fox/Falco Final Destination input tape
measured 55,500 FPS through the complete compiled adapter, compared with 6,053
through the previous Python adapter. Recording and replay verification measured
45,732 FPS. These measurements exclude model inference and disk writes and do
not establish a rate for every matchup. The native C-only control measured
95,192 FPS. Different hardware, input tapes and batch sizes are separate
benchmarks. Commit descriptions record the validation scope; raw local results
remain under the simulator checkout's ignored `reports/triage/hal-059/`.
