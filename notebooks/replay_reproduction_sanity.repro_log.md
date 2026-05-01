# Replay reproduction — investigation log

## 2026-05-01 update #6 — per-port UCF matches; build-version drift is the leading suspect

Parsed GameStart event in both source slp and a saved live slp:

```
              source slp                      live slp
slp version    3.7.0                          3.19.0
port 1   dashback=UCF shielddrop=UCF    dashback=UCF shielddrop=UCF
port 2   dashback=UCF shielddrop=UCF    dashback=UCF shielddrop=UCF
port 3   dashback=UCF shielddrop=UCF    dashback=UCF shielddrop=UCF
port 4   dashback=UCF shielddrop=UCF    dashback=UCF shielddrop=UCF
```

Per-port controller-fix settings are identical, so UCF DB / UCF SD
will fire equivalently in source and live.

The biggest remaining unaccounted-for difference is the Slippi-Ishiiruka
**build version**: source was recorded under slp 3.7.0
(Slippi-Ishiiruka v2.x era) while our live emulator is the v3.19.0
build. `$Required: General Codes` is a single named gecko block that
the build ships with — its contents change between Slippi-Ishiiruka
releases. If any patch landed between 3.7 and 3.19 that affected hit
attribute resolution or hitlag computation, our 4-7 frame drift on
hit moments is plausibly that.

Verifying this directly would require running the harness under a
slippi-Ishiiruka build closer to the source's vintage (e.g. v2.4.x).
Out of scope for this round; documented as the next path.

## 2026-05-01 update #5 — `blocking_input=True` is the libmelee escape hatch for FFW

Found by reading `~/src/libmelee/test_live.py` (libmelee's own canonical
DolphinTest). It uses `blocking_input=True, polling_mode` not set
(default False). The flag wires through to Slippi-Ishiiruka's
`Pipes::UpdateInput` → `m_blockingPipes`, which makes the input pipe
block on `select()` until the bot writes a complete frame ending in
`FLUSH`. This synchronizes pipe writes with the game's per-frame SI
poll, so EXI override / FFW path can't read stale `padBuf` between our
writes.

Switching to `blocking_input=True` (kept `polling_mode=True` so
`console.step()` still returns promptly): FFW mode's 151 mismatches
collapsed to **4** at frames 42-45 — same `hitlag_left` pattern as
normal mode. Both modes are now equivalent.

Net: `normal` and `ffw` both reproduce
`Game_20201215T165952.slp` with the same residual: a handful of hit
moments where live has 4-7 extra hitlag frames vs source. All other
fields (action, position, controller, shield, hitstun, speeds) match
bit-exact.

## 2026-05-01 update #4 — byte-level diff explained: source has wider effective stick range

Diagnosed by re-running with `save_replays=True` and parsing the live
emulator's slp byte stream directly (the file was unfinalized so libmelee
itself couldn't read it; manual UBJSON-skip + 0x37 walk works).

Around the f=42 hit moment (port 2 = Marth, port 1 = Falco):

```
       source slp 0x19 (jx)  source 0x3B (raw)   live 0x19 (jx)  live 0x3B (raw)
f=41    -0.7875               -84                 -0.7875          -63
f=42    -0.8875               -94                 -0.8875          -71
f=43    -0.9875               -100                -0.9875          -79
f=46    -0.9875               -98                 -0.9875          -79
```

`processed` matches frame-by-frame, but the recorded `raw` byte is
*much wider* in source than in live (e.g. -84 vs -63 at the same
processed value). For source's pairs the relationship is roughly
`processed = (raw + 22) / 80`, NOT `raw / 80`. Equivalently, `processed
= raw / max_raw` where source's effective max ≈ 106 (from
`0.7875·X = 84` and `0.8875·X = 94`).

This is **per-controller stick calibration** — real GC pads have
magnitude up to ~105 raw bytes, and the game (or UCF Pad Buffer + 1.0
Cardinals) calibrates the divisor accordingly. Our pipe injection
goes through Dolphin's GCPadEmu (`MAIN_STICK_RADIUS = 0x7F = 127`,
fix_analog_stick capped at ±80), so live's recorded raw bytes top out
at ±80 while source's are ±105.

This byte-level discrepancy probably feeds the hitlag_left drift: at
hit moments the game inspects stick byte / facing / DI to decide
attack attributes. With identical processed but different bytes, a
sub-hitbox or shield-vs-body decision could flip, changing hitlag
duration by a few frames (electric vs non-electric: +4; shield: +0
or different).

Fixing the byte-level match without breaking processed agreement is
non-trivial. Options:

1. Extend `fix_analog_stick` to push pipe values that put padBuf in
   ±100 range, and reconfigure live's `Main Stick/Radius` so the
   game-side divisor scales correspondingly. Risk: live's stick
   processor doesn't have UCF-Pad-Buffer-style calibration in the
   default flow, so wide bytes saturate to ±1.0.
2. Use the `use_raw_main_stick=True` flag to feed source's slp 0x3B
   byte directly. Already plumbed. Earlier attempt regressed because
   live's stick processor naively clamped/80 for byte=-98 → -1.0
   (vs source's -0.9875). Combining this with reconfigured radius or
   per-port calibration could close the gap.
3. Accept the residual hitlag drift (4-7 frames at hit moments) as
   "good enough" given that all other comparison fields match.

For now, leaving option 3 as the documented status. Going to chase
option 1 next when there's bandwidth.

## 2026-05-01 update #3 — trigger fix landed; residual is hitlag-only

After mode-aware trigger inverse:

- `normal` mode `start_frame=-123 prefix=200` → 4 mismatches total, all
  `port 2 hitlag_left` over frames 42-45 (4,3,2,1 → 0).
- `normal` mode `start_frame=0 prefix=300` → 21 mismatches total. All
  `hitlag_left` on hit events: 4 at frames 42-45 (port 2),
  7 at frames 149-155 (both ports).
- `ffw` mode is broken in a different way — at frame ~64 the live
  emulator's main stick stays stuck at full-right while we send
  centered. Likely an input-cadence issue specific to use_exi_inputs +
  enable_ffw. Not blocking normal-mode progress.

Pattern: every divergence is `live hitlag_left = 4..7` while
`source hitlag_left = 0` — i.e., the live game extends hitlag by a
few frames at moments when a hit lands. Action and position match,
only the per-hit hitlag countdown differs. Falco's laser has an
electric attribute (+4 frames hitlag); a 7-frame delta would be
electric+something else. Possibly our stick byte differs from source
by 1 in a way that flips the hitbox/shield-grid result, changing
the attack's effective attributes.

The previously-suspected "frame-48 dashback drift" was a *symptom*
of the trigger bug, not a separate UCF issue. With triggers fixed,
that drift went away.



Append-only running notebook. Newest at the top of each section. Pair with
`replay_reproduction_sanity.README.md` (which is the *summary* of conclusions);
this log captures the *process*, including dead ends.

## 2026-05-01 update #2 — TRIGGER BUG: pipe → "Axis L +" + binding mismatch

**Reproduced the supposed "171 bit-exact" prefix and discovered it's not.**
Frame -4 actually mismatches in normal mode: source p2 has L=0.4, live p2
has L=0.0. The previous handoff appears to have been wrong about the
prefix matching. Verified directly from the *prior* committed debug data
(`/tmp/repro_debug_neg123_v2/normal_frame_comparisons.jsonl`) — frame -4
already showed the same `controller.l_shoulder 0.4 -> 0.0` mismatch.

Reading slippi-Ishiiruka's GCPadEmu + Pipes:

- `Pipes::SetAxis("L", value)` sets BOTH `m_current_pad.padBuf[6] = u8(value*255)`
  AND `Axis L +` (state=`max(0,value-0.5)*2`) AND `Axis L -` (state=
  `(0.5-min(0.5,value))*2`) — same +/- split as for sticks.
- libmelee's GCPadNew.ini binds `Triggers/L-Analog = Axis L +`. So in
  normal mode (no EXI), the game reads Triggers via ControllerInterface
  → "Axis L +" only.
- Our trigger pipe value `(raw + 0.5)/255 = (56 + 0.5)/255 = 0.2196`
  (for amount=0.4, raw=56). At pipe=0.2196 < 0.5: `hi = 0`, so
  `Axis L + = 0`. Trigger reads as 0. **That's why live frame -4
  records L=0.0.**
- In FFW (`use_exi_inputs=True`): the "Allow Bot Input Overrides"
  gecko code reads padBuf directly via `prepareOverwriteInputs`, which
  bypasses GCPadEmu and uses `padBuf[6] = 56` — that DOES yield L=0.4
  on FFW. So the prior handoff's "bit-exact" claim, if true, was
  almost certainly about FFW, not normal.

For sticks the same +/- split is *self-consistent* because the GC
stick centers at 0x80 and the bindings are split L/R (and U/D), so
`m_main_stick GetState → x = right-left` reconstructs the original
direction. The trigger has a single 0..1 range and the +/- split
loses the lower half. That's the asymmetry that bit only triggers.

**Implication for the reproduction strategy:**

- Normal mode trigger pipe value must be `pipe = (round(amount*0x8C)/0xFF)/2 + 0.5
  = (round(amount*0x8C) + 255) / 510`. That puts `Axis L + = round(amount*0x8C)/255 ≈
  amount*140/255`, so `pad.triggerLeft = u8(0.219*255) = 56`, and slp 0x29
  logs 56/140 = 0.4.
- FFW mode trigger pipe value must be `pipe = (round(amount*0x8C) + 0.5)/255`.
  That sets `padBuf[6] = 56` directly, which the EXI overwrite path uses.
- The two are incompatible: pipe=0.6098 (normal-mode-correct) gives
  padBuf[6] = u8(155) which FFW's EXI path would log as ≈1.107 (clamped).
  pipe=0.2196 (FFW-mode-correct) gives Axis L + = 0 which normal mode
  reads as 0.

So the harness needs **mode-aware trigger inverse**.

For STICKS, both paths happen to land on the same byte for the standard
`fix_analog_stick` mapping, since `(value-0.5)*254 ≈ (right-left)*127`.
This is why sticks reproduce in both modes today.

Frame-48 dashback drift remains a separate problem to investigate after
triggers are correct in normal mode.

## 2026-05-01 update — UCF is loaded in our live runs (not disabled)

I assumed earlier that libmelee's stripped-down `User/GameSettings/GALE01r2.ini`
disabled UCF in live. That was wrong. Reading `GeckoCodeConfig.cpp` in
slippi-Ishiiruka:

- `MergeCodes` loads codes from Sys (global), then User (local) — additive.
- `MarkEnabledCodes` enables names from Sys's `[Gecko_Enabled]` and
  ALSO enables names from User's `[Gecko_Enabled]` — additive.
  Disabling requires `[Gecko_Disabled]` in the user file.

So Sys/GameSettings/GALE01r2.ini has `$Required: General Codes` enabled,
and that *single* gecko block embeds **UCF 0.84 Dashback (0x800C9A44),
SDI, Shield Drop, Shield Drop Extended, Tumble, DBOOC, AND
"Pad Buffer + 1.0 Cardinals" (0x806B460)**. All ride along. The user-side
ini libmelee writes only ADDS Extract Menu Info + Instant Match.

Implication: live runs and source recordings both have UCF active. So a
simple "UCF on in source, off in live" framing isn't the discrepancy.

But UCF could still produce frame-48 drift if its decisions depend on
stick-byte HISTORY, since we feed `padBuf` bytes derived from the slp's
*processed* float (typically magnitude ~78 for full-tilt) while source's
recorded `0x3B` was wider (e.g. byte=-98). UCF DB's threshold checks
might fire differently for byte=-78 vs byte=-98 even when the resulting
processed value is identical. **Frame 48-49 in the source is a literal
dashback** (stick goes from full-left to full-right): the exact
decision-point UCF DB intercepts.

Frame trace (source replay) around the divergence:

| f  | p1.main_x  | p1.raw_x | p1.action | p2.main_x | p2.raw_x | p2.action     |
|----|------------|----------|-----------|-----------|----------|---------------|
| 46 | 0.00625    | -98      | DASHING   | 0.01875   | -80      | DAMAGE_AIR_1  |
| 47 | 0.01875    | -87      | DASHING   | 0.1625    | -54      | DAMAGE_AIR_1  |
| 48 | 0.25625    | -39      | DASHING   | 0.5       | -28      | DAMAGE_AIR_1  |
| 49 | 0.9875     | +101     | TURNING   | 0.5       | -2       | DAMAGE_AIR_1  |
| 50 | 0.9875     | +101     | DASHING   | 0.5       | +4       | DAMAGE_AIR_1  |

p1's stick reverses from full-left to full-right between f48 and f49 —
classic dashback. UCF DB intercepts on the f49 transition. UCF DB
inputs include the recent byte history. Our `fix_analog_stick`
injection at f46-49 fed bytes ~(-78, -78, -39, +78), while source had
(-98, -87, -39, +101). At the dashback site UCF DB might evaluate the
"how long was stick at saturated negative" predicate with different
results.

## Open hypotheses (ranked)

1. **UCF state-machine drift on stick-byte history.** UCF DB / SD / Tumble
   read the raw stick byte (slp 0x3B) at decision sites and compare to past
   frames. Even if our injection produces the same `processed` per frame
   (171 frames bit-exact), it produces a different `padBuf` *byte sequence*,
   so UCF's history-based checks resolve differently and offset attack/hit
   timing by ±1 frame around frame 48. Falsifiable by disabling UCF in
   live and checking whether frame 48 drift disappears.
2. **Netplay vs local-pipe input pipeline divergence.** The source replay
   was a netplay match. Slippi-Ishiiruka's netplay code may pre-process
   stick bytes differently than the local Pipes path before they reach
   the game's stick processor — so the same `padBuf` byte yields different
   `processed` floats on netplay-source vs local-pipe-live. This explains
   why slp 0x3B = -98 ↔ slp 0x19 = -0.9875 (not -1.0) cannot be reproduced
   on local pipe (raw=-98 → live processed = -1.0). Falsifiable by
   mass-grepping netplay packet handling for stick byte transforms.
3. **Per-controller stick calibration baked into source.** Real GC pads
   have varying physical ranges (often ±100..110, not ±80). Melee may
   calibrate per-pad on first poll. If source calibrated against a wider
   range and the slp logs the post-calibration byte, we can't reproduce
   without replicating the calibration table. Likely subsumed by (2).

## Key empirical mismatches (slp Game_20201215T165952.slp)

slp version 3.7.0 (raw_y not present, raw_c-stick not present).

| Frame | Port | slp 0x19 (processed, libmelee 0..1) | slp 0x3B (raw int8) | naive raw/80 | Comment |
|---|---|---|---|---|---|
| -15 | 1 | 0.5 (=0.0)            | -19  | -0.2375 | deadzone snapped to 0 |
| -15 | 2 | 0.9875 (=0.975)       |  106 | clamp 1.0 | clamped *but to 0.975, not 1.0* |
| -14 | 1 | 0.24375 (=-0.5125)    | -41  | -0.5125 | naive matches |
| -14 | 2 | 0.9875 (=0.975)       |  106 | clamp 1.0 | clamped to 0.975 |
| -13 | 1 | 0.0125 (=-0.975)      | -98  | clamp -1.0 | clamped to -0.975 |
| -13 | 2 | 0.99375 (=0.9875)     |  105 | clamp 1.0 | clamped to 0.9875 |
| 48  | 2 | (0.5, 0.01875) = (0, -0.9625) | (-28, ?) | X non-zero | cardinal-snapped X→0 due to large Y |

Naive `processed = clamp(raw, -80, 80)/80` holds for raw=-41 but breaks
above the clamp boundary. The over-saturation values (raw=±98, ±105, ±106)
all yield processed slightly *less* than 1.0, not *exactly* 1.0. This is
inconsistent with a simple in-game clamp.

Live emulator under our processed-based injection (`fix_analog_stick`)
reproduces 171 consecutive frames bit-exact under strict comparison —
including these "non-naive" frames. So when we inject pipe values that
yield `padBuf = round((processed-0.5)*160)` (i.e. -78 for processed=-0.975),
the live game produces processed=-0.975 too. **Live's stick processor
does match naive clamp/80** when fed via the local pipe. The source
replay's processor did NOT (raw=-98 → processed=-0.975 ≠ -1.0).

Conclusion: source's recorded `0x3B` is *not* the byte that the live
emulator's stick processor reads from. Either Slippi's netplay packet
decoder writes a different byte to padBuf than what it logs at 0x3B, or
the game's processor differs between netplay and local-pipe paths.

## What we tried

### A. Inject slp 0x3B raw byte directly via padBuf (committed and reverted)

Hypothesis: 0x3B is the pre-UCF physical byte; feeding it makes UCF's
history match. Result: regressed at frame -13 onward — live processed
saturates to ±1.0 while source processed is at ±0.975. Disproved the
"raw byte injection alone reproduces processed values" idea. Feature
left in as opt-in flag (`use_raw_main_stick=True`) for combined
experiments later (e.g. raw byte + UCF disabled).

### B. Read SPEC.md for 0x3B semantics

Confirmed: int8, comment "Used by UCF dashback code", added in 1.2.0.
Spec confirms it's *raw* in the sense of "pre-UCF", but doesn't pin
down whether netplay decode writes the same byte to padBuf.

### C. Read GALEXX.ini UCF gecko code asm

UCF DB hooks 0x800C9A44, UCF SD hooks 0x800998A4, UCF Tumble hooks
0x800908F4. Original instruction at 0x800C9A44 = `stfs f0, 0x2C(r31)`
(store *processed* stick X float, not raw byte). UCF intercepts at
the post-processed stage and reads adjacent state, including raw bytes
loaded from elsewhere in player struct. Not yet disassembled
end-to-end.

## Next steps (in order)

1. **Test UCF disabled in live.** Start dolphin with `setup_gecko_codes=False`
   plus a manually-configured GALEXX.ini that omits the three UCF gecko
   blocks. Re-run the harness in normal mode at start_frame=-123,
   prefix_frames=300. If frame 48+ matches, UCF is confirmed.
2. **Calibration sweep.** Boot the emulator into a stable in-game state,
   sweep pipe values 0..1 for port 1 main X with port 2 idle, log the
   live emulator's resulting (padBuf byte ← we can probe by reading
   slp 0x3B during live run, processed at slp 0x19) per pipe value.
   Cross-reference to source's (0x3B, 0x19) pairs to localize the
   netplay-vs-pipe divergence.
3. If both fail, look at Slippi-Ishiiruka netplay packet handler
   (`SlippiNetplay.cpp` / `SlippiSavestate.cpp`) for any stick-byte
   transform applied on the receive side before injection into padBuf.
