# Replay Reproduction Sanity Harness

Standalone harness in `notebooks/replay_reproduction_sanity.py` that takes a
`.slp` file, extracts both players' exact controller inputs, replays them
into Dolphin through libmelee controllers, and verifies that observed game
states match the source replay frame-by-frame. Intentionally independent of
HAL preprocessing, model postprocessing, and `hal/emulator_helper.py`.

Run:
```
HAL_RUN_REPLAY_REPRODUCTION_INTEGRATION=1 \
  uv run pytest hal/test_replay_reproduction_sanity.py::test_replay_reproduction_sanity_dev_prefix -q -s
```
or the CLI:
```
uv run python notebooks/replay_reproduction_sanity.py \
  /path/to/replay.slp --mode normal --start-frame -123 --prefix-frames 300 \
  --debug-dir /tmp/repro --continue-on-mismatch
```

## Status (2026-05-01)

`normal` mode — `Game_20201215T165952.slp`, `start_frame=0 prefix=300`:
21 mismatches across 300 frames, all `hitlag_left` drifts (4-7 frames
extra hitlag) at hit moments (frames 42-45 and 149-155). Action and
position match exactly. The remaining drift looks like an attribute
mismatch on hit (electric vs non-electric, or shield vs body). All
other comparison fields are bit-exact.

`ffw` mode is broken differently: `controller.main_stick.0` diverges
from frame ~−117 onward. EXI input override appears to be using stale
padBuf bytes between our pipe writes — separate timing issue. See the
investigation log.

The previous handoff's "171 frames bit-exact" claim was incorrect: the
prior debug data (`/tmp/repro_debug_neg123_v2/`) actually shows
`controller.l_shoulder` divergence from frame **−4** because of a
trigger inverse bug that was fixed in this revision (see "Fix #1:
mode-aware trigger inverse" below).

## Fix #1: mode-aware trigger inverse (2026-05-01)

Slippi-Ishiiruka's `Pipes::SetAxis("L", v)` sets BOTH
`padBuf[6] = u8(v*255)` AND ControllerInterface's
`Axis L + = max(0, v - 0.5) * 2`. libmelee's GCPadNew.ini binds
`Triggers/L-Analog = Axis L +`, so:

- **Normal mode** (no EXI override) reads triggers via
  ControllerInterface → GCPadEmu → `pad.triggerLeft = u8(triggers[0]*0xFF)`
  where `triggers[0] = Axis L +`. Pipe values < 0.5 land `Axis L + = 0`,
  so triggers register as 0. The previous trigger inverse
  `pipe_value_for_trigger_raw(raw) = (raw + 0.5)/255 = 0.22` for raw=56
  (amount=0.4) — that's < 0.5 so live read trigger = 0. Bug.
- **FFW mode** (EXI override via `Allow Bot Input Overrides` gecko)
  reads `padBuf[6]` directly via `prepareOverwriteInputs`. So the
  raw-byte pipe value that places `padBuf[6] = round(amount*0x8C)`
  works correctly there.

Two-mode trigger inverse:

- `pipe_value_for_trigger_raw(raw) = (raw + 0.5)/255` — for FFW.
  Sets `padBuf[6] = raw`.
- `pipe_value_for_trigger_amount_via_axis(amount) =
   (round(amount*0x8C) + 0.5)/510 + 0.5` — for normal.
  Sets `Axis L + ≈ round(amount*0x8C)/0xFF` so
  `pad.triggerLeft = round(amount*0x8C)`, slp 0x29 logs `amount`.

`ReplayControllerSender(use_exi_inputs=True/False)` switches.

## Open: hitlag_left drift on hit moments

Source has hitlag_left = 0 by frame 42 (post-hitlag); live has 4 (still
in hitlag). Identical action and position. At frame 41-42 in source,
port 2 (Marth) was attacking with FAIR and got hit by port 1 (Falco)'s
laser. Hit-detection landed in both source and live (action transitions
match), but live's hitlag duration was 4 frames longer.

slp 0x3B (raw byte) trace for the relevant frames shows source's bytes
are larger-magnitude than ours (raw=-84 vs our injected -63 for the
same processed=-0.7875). The processed value matches because UCF's
"Pad Buffer + 1.0 Cardinals" or the game's clamp-to-80 logic produces
the same float for both. But the BYTE difference may flip a hit
attribute selection (e.g., which sub-hitbox of the laser). Worth
investigating but not fundamentally blocking.

## Investigation log

See `notebooks/replay_reproduction_sanity.repro_log.md` for the running
notebook of hypotheses, evidence, and dead ends.

## The pipe → game → slp chain (verified from source)

This is the canonical reference. All offsets are byte offsets within a
single Slippi event payload (after the 1-byte command tag).

### libmelee → Dolphin pipe (`libmelee/melee/controller.py`)

`Controller.tilt_analog(button, x, y)` writes `SET MAIN <x> <y>\n` /
`SET C <x> <y>\n` to the named pipe (`x, y ∈ [0, 1]`, center 0.5).
`Controller.press_shoulder(button, amount)` writes `SET L <amount>\n` /
`SET R <amount>\n`. `press_button` / `release_button` write
`PRESS <name>\n` / `RELEASE <name>\n`. `flush()` writes `FLUSH\n` and
flushes the OS buffer.

If `_fix_analog_inputs=True`, `tilt_analog` first applies
`fix_analog_stick(x)` and `press_shoulder` first applies
`fix_analog_trigger(amount)`. We disable that and pre-quantize ourselves
(see "Inverses we apply" below).

### Dolphin pipe parser → padBuf (`slippi-Ishiiruka/Source/Core/InputCommon/ControllerInterface/Pipes/Pipes.cpp`)

```cpp
u8 FloatToU8(double v) { s8 raw = std::floor((v - 0.5) * 254); return reinterpret_cast<u8&>(raw); }
SetAxis: padBuf[2..5] = FloatToU8(value);   // MAIN X, MAIN Y, C X, C Y
         padBuf[6..7] = u8(value * 255);    // L, R analog
ParseCommand: PRESS/RELEASE flip bits in padBuf[0..1]; FLUSH yields the
              SlippiPad for the next polled frame.
```
- `value` is clamped to `[0, 1]` in `SetAxis`.
- `floor(...)·254` ranges over `[−127, 127]`: **the pipe path cannot
  produce padBuf int8 = −128**. Real GC controllers can; this is a 1-step
  quantization gap at full-left.

### Game → slp pre-frame (`libmelee/melee/console.py::__pre_frame`, `slippi-Ishiiruka/Externals/SlippiLib/SlippiGame.cpp::handlePreFrameUpdate`)

```
0x01 (i32)  frame
0x05 (u8)   port (0-indexed)
0x06 (u8)   isFollower flag
0x07 (u32)  random seed
0x0B (u16)  action state id
0x0F (f32)  position X         (post-frame mirrors this; pre-frame is current state)
0x13 (f32)  position Y
0x17 (f32)  facing direction
0x19 (f32)  joystick X         ← post-game-process, in [−1, 1]
0x1D (f32)  joystick Y         ← libmelee remaps both to [0, 1] via (v/2)+0.5
0x21 (f32)  c-stick X          ← post-game-process
0x25 (f32)  c-stick Y
0x29 (f32)  trigger combined   ← raw/0x8C scale (no deadzone applied here)
0x2D (u32)  processed buttons
0x31 (u16)  physical buttons
0x33 (f32)  physical L analog  ← Slippi 1.2.0+
0x37 (f32)  physical R analog  ← Slippi 1.2.0+
0x3B (u8)   joystick X raw     ← Slippi 1.2.0+, declared uint8_t in SlippiLib
0x3C (f32)  damage taken       ← Slippi 2.0.0+
0x40 (u8)   joystick Y raw     ← Slippi 3.15.0+
0x41 (u8)   c-stick X raw      ← Slippi 3.15.0+ (NOT parsed by libmelee)
0x42 (u8)   c-stick Y raw      ← Slippi 3.15.0+ (NOT parsed by libmelee)
```

libmelee reads 0x3B/0x40 as `>b` (signed byte) and exposes them as
`controller_state.raw_main_stick`. **Important nuance**: in
`Game_20201215T165952.slp` we observed `raw_main_stick = (−98, 0)` paired
with `main_stick = (0.00625, 0.5)` (i.e. processed −0.9875). Under the
naive model `processed = clamp(raw, −80, 80) / 80`, raw = −98 should give
processed = −1.0, not −0.9875. The relationship between 0x3B's byte and
0x19's float is **not** that simple model — it's mediated by something
(UCF? game state? deadzone?) we have not yet pinned down. **Don't feed
`raw_main_stick` directly back through the pipe** until this is
understood; doing so risks a second pass of whatever transformation was
applied during recording.

## Inverses we apply (current fix)

In `notebooks/replay_reproduction_sanity.py::ReplayControllerSender.send_frame`:

| Channel       | Source (slp/libmelee) | Pipe value (normal mode)                                            | Pipe value (FFW / EXI mode)                                         |
|---            |---                    |---                                                                  |---                                                                  |
| Main stick X  | `main_stick[0]` ∈ [0,1] | `fix_analog_stick(main_stick[0])`                                 | same                                                                |
| Main stick Y  | `main_stick[1]`       | `fix_analog_stick(main_stick[1])`                                   | same                                                                |
| C-stick X/Y   | `c_stick[i]`          | `fix_analog_stick(c_stick[i])`                                      | same                                                                |
| L analog      | `l_shoulder` ∈ [0,1]  | `pipe_value_for_trigger_amount_via_axis(l_shoulder)` — see Fix #1 | `(round(l_shoulder · 0x8C) + 0.5) / 255`                            |
| R analog      | `r_shoulder`          | same shape                                                          | `(round(r_shoulder · 0x8C) + 0.5) / 255`                            |
| Buttons       | physical bits         | `press_button` / `release_button` on transitions                    | same                                                                |

`Controller(fix_analog_inputs=False)` so libmelee passes the
pre-quantized pipe value through verbatim.

### Why `fix_analog_stick` is correct here (concrete trace)

For `main_stick[0] = 0.00625` (= processed −0.9875, original raw within
the unit circle = −79):
- `fix_analog_stick(0.00625)`: raw_target = `round((0.00625−0.5)·160) = −79`,
  fudged = −78.9, pipe v = `(−78.9 / 254) + 0.5 = 0.18937`.
- Dolphin: `floor((0.18937 − 0.5)·254) = floor(−78.9) = −79`. padBuf = −79.
- Game (simple model): `−79 / 80 = −0.9875`. libmelee = `(−0.9875)/2 + 0.5 = 0.00625`. ✓

Saturated values also reproduce:
- `main_stick[0] = 0.0` (saturated, original raw ≤ −80) → padBuf = −80,
  processed = −1.0, libmelee = 0.0. ✓
- `main_stick[0] = 1.0` (saturated, original raw ≥ +80) → padBuf = +80,
  processed = +1.0, libmelee = 1.0. ✓

### Why `raw = round(processed · 0x8C)` is correct for triggers

slp 0x29 stores `raw / 0x8C` (verified empirically at frame −4 with the
shield-start input `l_shoulder = 0.4`):
- `raw_target = round(0.4 · 0x8C) = round(56.0) = 56`.
- Pipe v = `(56 + 0.5) / 255 = 0.2216`.
- Dolphin: `u8(0.2216·255) = u8(56.5) = 56`. padBuf = 56.
- Game records `56 / 0x8C = 0.4` exactly. ✓

The previous handoff's hypothesis of a deadzone-and-max formula
(`processed = (raw − 0x2A) / (0x8C − 0x2A)`) is wrong for what slp 0x29
records. That deadzone applies to **state transitions** in-game (e.g.,
"is L analog pressed enough to register as shielding") but **not** to the
recorded slp value. This was the original "trigger fidelity always
broken" symptom.

### What `fix_analog_inputs=True` got right and wrong

- Got right: sticks. `fix_analog_stick` is exactly the inverse we need
  inside the unit circle and naturally saturates outside.
- Got wrong: triggers. `fix_analog_trigger` does map to `raw/0x8C` correctly,
  *but* libmelee couples both stick and trigger fix-ups under the same
  `_fix_analog_inputs` flag, so you can't have one and not the other.
  Solution: turn the flag off and apply both inverses ourselves.

## Why "submit raw_main_stick directly" doesn't work

Tempting because slp 0x3B records a byte. But:
1. The byte's relationship to slp 0x19's processed float doesn't fit the
   simple model (raw = −98 ↔ processed = −0.9875, observed in real data).
   So we'd be feeding "the byte that the game read after some unknown
   pre-processing" back through Dolphin's pipe → padBuf, where the live
   game would re-apply that same pre-processing, drifting the result.
2. libmelee parses the byte as `>b` (signed) but SlippiLib stores it as
   `uint8_t`. The semantic frame (centered at 0 vs 128) is unconfirmed.
3. libmelee only exposes raw main stick; c-stick raws at slp 0x41/0x42
   are not parsed (they exist; you'd need to parse the slp manually).

Once UCF behavior is pinned down (next section), feeding raws may become
the cleanest path — but only with UCF disabled on the live emulator and a
verified raw-byte semantic.

## (Resolved) frame-48 drift was the trigger bug

The previous handoff attributed frame-48 drift to UCF and `0x3B`
re-mutation. After fixing the trigger inverse (Fix #1), the original
"frame-48" symptoms went away — port 2's hitstun timing now matches
through frame 47 and beyond. UCF is in fact loaded in our live runs
(the global `[Gecko_Enabled]` `$Required: General Codes` block
embeds UCF 0.84), so the source/live UCF state was equal all along;
the divergence was caused by the trigger `Axis L +` mismatch
masquerading as a downstream physics issue.

The remaining (much smaller) divergence is the hitlag_left drift
described above.

## (Historical) frame-48 drift hypothesis text



After the fix above, frames `−123..47` reproduce exactly. At frame 48 the
state diverges: port 2's `action_frame` is off by 1, `hitstun_frames_left`
off by 1, `position.x` off by ~5.6, `shield_strength` off by ~0.9.
**Inputs** at every frame from −123 through 48 match the source exactly
(verified by JSONL comparison logs in `/tmp/repro_debug_neg123_v2/`).
The post-frame state at frame 47 also matches. So a hit on port 2
registers one frame earlier in live than in source despite identical
inputs — meaning the *game's interpretation* of those inputs differs.

The dominant suspect is **UCF (Universal Controller Fix)**. UCF gecko
codes are enabled by default in Slippi-Ishiiruka via
`Data/Sys/GameSettings/GALEXX.ini` (search for `Universal Controller Fix`,
e.g. line 254). The three logic patches (`UCF DB.asm`, `UCF SD.asm`,
`UCF Tumble.asm` — see GALEXX.ini lines 255, 299, 338) modify how the
game interprets stick movements at decision points (dashback, shield
drop, tumble). UCF doesn't typically rewrite the padBuf bytes but rather
intercepts the game's state-transition checks. Even so, its presence in
live but possibly different state in source (e.g., frame counter offset
on UCF's pad-buffer tracking) could explain the 1-frame attack timing
discrepancy.

### Concrete next investigations

1. **Read the Slippi spec authoritatively.** The repo at
   `https://github.com/project-slippi/slippi-wiki` (`SPEC.md`) defines
   what each pre-frame offset stores. Confirm the semantics of byte 0x3B
   (is it "raw stick byte the game reads", or "UCF-modified", or "UCF
   delta"?). This single answer determines whether feeding raws is
   viable.
2. **Run a calibration sweep on a live emulator.** Boot Dolphin in a
   neutral training-mode state, sweep pipe values v ∈ [0, 1] in fine
   steps, and on each frame log the slp's `joystickX` (post-process) and
   `joystickXRaw`. This produces the empirical
   `pipe → (processed, raw_byte)` table for a UCF-enabled live run.
   Repeat with UCF disabled to isolate UCF's contribution. Comparing the
   two tables to a UCF-enabled source slp directly identifies what UCF
   does to bytes vs. processed values.
3. **Disable UCF in the live emulator and retry.** The cleanest if it
   works: pass a custom Slippi gecko-code list that excludes the UCF
   patches (see `Data/Sys/Slippi/InjectionLists/list_console_*` and
   `melee.Console`'s `setup_gecko_codes` flag — `setup_gecko_codes=False`
   keeps user codes verbatim; you'd need a list that omits UCF). Source
   replays were recorded with UCF on, but if disabling UCF in live makes
   the chain `pipe → padBuf → game → slp` produce values matching a
   UCF-on source... that would tell us UCF's effect is purely
   compensated by our injection path. If it doesn't match, UCF is
   actively rewriting either bytes or decision logic and we need (1).
4. **Examine UCF Dashback behavior in particular.** The frame-48 hit
   is during shield/movement; UCF Dashback is the most likely culprit.
   The asm at GALEXX.ini line 256 onward (`C20C9A44 0000002B`) is the
   patch installed at game-address `0x800C9A44` (the dashback decision
   site). Disassemble that 0x2B-instruction block (e.g. via `gekko-tools`
   or by mapping the hex to PowerPC mnemonics) to see exactly what
   condition it modifies and whether it depends on frame-counter or
   prior-frame state in a way that could be off-by-one between
   reproductions.

### Cleanest unified solution we can imagine

Two designs, in order of preference:

- **Inject post-UCF padBuf bytes with UCF disabled in live.** Requires
  pinning down what slp 0x3B byte actually represents (post-UCF padBuf?
  pre-UCF physical?). If 0x3B is the post-UCF padBuf byte, this is the
  cleanest route: skip UCF on live, feed the byte the game's stick
  processor would have seen, and the game's processor outputs the same
  processed value. No floating-point inverses needed.
- **Empirical calibration table.** Build a per-axis lookup table
  `processed_value → pipe_value` from a sweep, baked into the harness.
  Robust against UCF/deadzone/quantization quirks because it's grounded
  in observation. More code, less elegant, but doesn't require
  understanding game internals.

## Useful artifact paths

- `/tmp/repro_debug_neg123_v2/normal_controller_commands.jsonl` — every
  frame's submitted pipe values + computed raws + digital transitions.
- `/tmp/repro_debug_neg123_v2/normal_frame_comparisons.jsonl` — per-frame
  mismatch reports.

## Test plan checklist (preserved from original spec)

- [x] Source replay reading via `melee.Console(is_dolphin=False)`.
- [x] Dolphin launch and connect from `hal/local_paths.py`.
- [x] Both normal and FFW modes reach in-game.
- [x] Replay metadata extraction (stage, ports, characters, costumes).
- [x] Menu setup with correct characters and costumes.
- [x] Strict comparison: `controller.*`, `action`, `action_frame`,
      `position`, `shield_strength`, `hitlag/hitstun`, speeds, etc.
      Float tolerance 1e-4. Buttons/enums exact.
- [x] Unit tests for the controller-state sender (transitions, holds,
      one flush per frame, pipe quantization round-trip).
- [x] First 171 frames of `~/data/ssbm/dev/Game_20201215T165952.slp`.
- [ ] **Full 300 aligned frames at start_frame=0** — blocked on
      frame-48 drift (open hypothesis: UCF).
- [x] Continue-on-mismatch and stop-on-mismatch CLI controls.
- [x] Structured JSONL debug output.

## Files

- `notebooks/replay_reproduction_sanity.py` — harness.
- `hal/test_replay_reproduction_sanity.py` — unit + gated integration tests.
- `pyproject.toml` — registers the `integration` pytest marker.
