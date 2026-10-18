# Native simulator boundary

Build the real simulator on MSL branch `o65-hal-parity`, then build the HAL
adapter against that checkout. The tested host uses Linux x86-64 and GCC 15.2.
Run builds and tests in sequence. A linker can replace a library that a test
is about to load.

```sh
make -C ~/src/melee-sim-light native python-library validator -j2 \
  PY="$PWD/.venv/bin/python"
uv run python -m hal.sim.native_build --native-source ~/src/melee-sim-light
PYTHONPATH=~/src/melee-sim-light uv run python -m hal.sim.native_benchmark \
  --data-dir ~/src/melee-sim-light/data --cpus 0 1 2 3 4 5
```

MSL needs its extracted `data/raw` and real replay fixtures. A missing fixture
fails the native integration tests. Set `HAL_MSL_SOURCE` to select another MSL
checkout, `MSL_CORE_LIBRARY` to select its compiled library, and
`HAL_NATIVE_LIBRARY` to select a HAL adapter. The adapter checks MSL observation
ABI 2, HAL ABI 2, and HAL's schema hash. Recorded playback also requires MSL
replay ABI 1. Snapshots record both library hashes and the game-data manifest
hash. Restore compares them before loading state. A different build requires a new snapshot.

`NativeMatch.stage` uses libmelee stage ids. MSL's stage ids are Slippi stage
ids; the adapter translates them. Both APIs use **internal fighter kinds** for
characters, which equal libmelee `Character` values. Slippi's external character
select ids need conversion before use. Ice Climbers is kind 10 (Popo), with
Nana as a follower; kind 11 cannot select a leader. Pokemon Stadium is always
frozen in MSL, including free-running matches. The hosted
`grStadium_801D4548` implements Slippi's `FreezePokemon.asm` branch.

## Frame and input rules

A fresh policy reset publishes frame -123 after MSL's first neutral tick.
A policy step advances one frame. `mask=False` leaves simulator state, columns,
and the last applied action unchanged; reward and reset flags clear each call.
A stockout is terminal. `max_frame` is a truncation unless that frame also has
a stockout. Reset a terminal or truncated lane before stepping it again.

The C adapter converts float32 policy actions with the pinned libmelee
`0.47.0+hal.realtime.1` rounding rules. Digital L or R also sets its physical
analog trigger to 140, as Dolphin's GCPad does. `wire_inputs` contains physical
bytes after that digital override. The tests compare these bytes with an
actual Dolphin recording. Remove or revise this conversion only after the
threshold and Dolphin tests pass with the replacement libmelee version.

Recorded playback has different initialization. MSL's public replay exporter
returns a packed config, a previous input, and consecutive recorded frames.
The config carries the initial seed; each frame carries both recorded RNG
boundaries, processed sticks, CPU inputs, and stage events. The HAL replay reset
performs the first recorded tick directly. It does not insert a neutral tick.
HAL metadata must agree with the recorded config. Replay matches require
`facing=None` and `max_frame=-1`; the replay supplies facing and the frame range.
HAL labels use ports 1 and 2
even when the recording used other physical ports; the tape retains those
physical ports for simulator scheduling. Recorded playback reports zero
`applied_action`, the recorded `wire_inputs`, and zero first-frame reward.

## Buffers, threads, and restore

Pass a writable, contiguous, four-byte-aligned `FRAME_DTYPE[B]` array as
`records`. A NumPy view of a pinned host allocation works. Keep its storage
alive through `close()`. The adapter writes 440-byte lane records directly into
that allocation. Returned column arrays
are borrowed views, overwritten by the next write.

`NativeShard` runs a batch on a thread, pinned to an explicit CPU. Select
that CPU from the GPU's socket. Each shard needs separate input and output
storage. Do not modify inputs until its future completes. Pass the previous
host-to-device copy event's `synchronize` method as `before_write` to reset,
step, or restore. It runs on the shard thread before the output changes.
HAL serializes native batch creation and destruction because MSL shares a
process-wide game-data reference count. Stepping uses independent native
batches and releases the GIL. Do not operate on one batch concurrently.

Snapshots use `HALMSL03`, strict identity metadata, SHA-256 of the native
payload, and zlib compression. `restore_many(lanes, states)` checks all HAL
metadata first, restores the selected native lanes, then observes and projects
the batch once. Lanes can move between batches. A native restore failure can
leave a partial restore; discard that batch. Restore clears reward and reset
flags. The next step must match uninterrupted execution bitwise.

Multi-frame native stepping remains optional and is not implemented. Call
`step` for each queued frame so that every intermediate reward and terminal
frame is available.
