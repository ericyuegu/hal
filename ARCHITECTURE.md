# Architecture

A working developer's reference for the data-and-emulator stack: how a `.slp`
file becomes an MDS shard, becomes a per-frame controller view, becomes a
Dolphin pipe write, becomes a captured gamestate that we can `diff` against
the original. Everything below is implemented in `hal/data/`, `hal/scripts/`,
and `hal/emulator/`, with cross-layer vocabulary centralized in `hal/wire.py`.

## Elevator pitch

```
.slp  ──peppi-py──▶  hal/data/extract.py  ──▶  MDS shard (per-frame ndarrays)
                                                    │
                                                    ▼
                                          MdsControllerView (zero-copy)
                                                    │
                                                    ▼
                                       apply_inputs (libmelee Controller)
                                                    │
                                                    ▼
                                               Dolphin pipe
                                                    │
                                                    ▼
                                          Trajectory.from_capture
                                                    │
                                                    ▼
                                     diff(live, Trajectory.from_slp)
```

The same `drive()` loop powers four compositions: round-trip validation,
online eval vs CPU, RL rollouts, and human exhibitions. The only thing that
changes is which `ControllerSource` plugs into each port.

## Three-stage data pipeline

All three scripts live in `hal/scripts/`. Run them in order; the artifact
each emits is the input of the next.

| Stage | Script | Reads | Writes |
|---|---|---|---|
| 1 | `stage1_build_index.py` | `.slp` tree (or `.7z` archive) | `index.jsonl` (one `ReplayIndexEntry` per replay; start/end/metadata only — no frame iteration) |
| 2 | `stage2_filter_replays.py` | `index.jsonl` | `paths.txt` (newline-delimited absolute or `archive://...!member` paths) |
| 3 | `stage3_process_replays.py` | `paths.txt` + `index.jsonl` | MDS shards under `train/`, `val/`, `test/` and `manifest.jsonl` (subset of index with `Stage3Annotation` populated) |

`paths.txt` is self-describing: each line is either an absolute filesystem
path or `archive://<abs-archive>!<member>`. A single `paths.txt` may mix
both. Splits are deterministic by `replay_uuid` (md5 of the path).

## Source-of-truth diagram

```
                       ┌────────────────────────┐
                       │      hal/wire.py       │
                       │  (slp-native lexicon)  │
                       └───────────┬────────────┘
                                   │
              ┌────────────────────┼─────────────────────┐
              ▼                    ▼                     ▼
       hal/data/schema.py   hal/data/extract.py   hal/emulator/controller_io.py
       hal/data/manifest.py hal/data/...          hal/emulator/trajectory.py
                                                  hal/emulator/session.py
```

`hal/wire.py` owns every concept the data and emulator layers both reference:
button bits, mask sentinels, raw-byte ↔ wire math, port translation, stage /
character bridges, post-frame field naming. No other module re-defines what
lives there; downstream files derive their local shapes at import time.

## Glossary (terminology)

Used consistently in code, comments, and docs. Extend this list rather than
inventing parallel terms.

| Term | Meaning |
|---|---|
| **logical** | Post-processed float on peppi's side. Sticks `[-1, 1]` neutral 0; triggers `[0, 1]` neutral 0. A lossy view of `raw`. |
| **raw** | Signed int8 byte from the slp record (`-128..127`). Bit-exact ground truth where the slp version recorded it. |
| **wire** | Float in libmelee's pipe-protocol form (`[0, 1]` neutral 0.5 for sticks). Computed from `raw` or `logical`; consumed by Dolphin's parser. |
| **physical** | Slp-native trigger value (`pre.triggers_physical`). Distinct from peppi's smoothed `pre.triggers` (which we call `logical`). |
| **field** | A peppi dotted attribute, e.g. `pre.joystick.x`. Never an MDS column. |
| **column** | An MDS shard column name, e.g. `p1_main_stick_x`. The on-disk vocabulary. |
| **frame_id** | Slp signed-int frame counter; `-123` at first in-game frame. Not an array index. |
| **frame_index** | 0-indexed array position into a column / per-replay sequence. |
| **port** | Libmelee 1..4. Say `peppi_idx` explicitly when referring to peppi's 0..3 indexing. |

## Conventions

- **Peppi-native int IDs on disk.** Stage, character, slp-version, end-method,
  costume, frame counts are stored as the bytes peppi-py reads off the wire.
  Libmelee enums appear only at the controller-injection boundary, via the
  bridges in `hal/wire.py`.
- **Ports are libmelee-1-indexed (`1..4`)** in every external-facing API,
  manifest field, and MDS column prefix (`p1`, `p2`). Peppi's 0..3 is
  internal to `extract.py` only and is named `peppi_idx` there.
- **Sticks**: MDS stores peppi-native logical floats in `[-1, 1]` neutral 0;
  the emulator converts to libmelee's `[0, 1]` neutral 0.5 wire format at
  `apply_inputs` time. The libmelee `Controller` is constructed with
  `fix_analog_inputs=False` so our composed wire float passes through
  unmodified.
- **Triggers**: `_logical` is peppi's smoothed analog value, stored as the
  training-side analog-shoulder feature. `_l_physical` / `_r_physical` are
  the slp-native per-shoulder bytes used by the emulator wire path.
- **Frame range**: peppi's `frames.id` includes pre-game (negative) frames.
  Extract trims to `frame_id >= wire.GAME_START_FRAME = -123` (Slippi
  standard "first in-game frame"). That trims the 2-second countdown but
  preserves the negative ids exactly as peppi yields them.

## Footguns

The list of things that have bitten us once and would silently bite us again
without a checklist.

- **Slp stage IDs ≠ `melee.Stage` enum values.** Fountain of Dreams is slp 2
  but `melee.Stage.FOUNTAIN_OF_DREAMS.value` is 8. Always go through
  `wire.slp_stage_to_libmelee`.
- **Character IDs happen to coincide today** (verified in
  `notebooks/peppi_vs_libmelee.py`) but treat them as distinct value spaces.
  Use `wire.slp_character_to_libmelee` — explicit beats lucky.
- **Raw stick bytes are slp-version-gated.** Main-x ≥ 1.2.0, main-y ≥ 3.15.0,
  c-stick ≥ 3.17.0. Older replays fall back to `wire.MASK_INT8` (= -128);
  the emulator wire path then takes the lossy logical-float fallback, which
  drifts ~0.0025 stick-units per frame.
- **Float mask is `NaN`** — detect with `np.isnan`, not `==`, because
  `nan != nan` is always true. Int masks (`wire.MASK_INT8`, `wire.MASK_INT32`,
  etc.) round-trip through equality normally.
- **Action–state alignment is shaped for training, not playback.** MDS row
  `t` stores `(state[t], action[t])` where `action[t]` is the controller
  input that *produced* `state[t+1]`. The training objective is "predict
  the next action conditioned on the current gamestate," so `action[t]` is
  the model's target given `state[t]`. The replay-time consequence is the
  `+1` shift in `MdsControllerSource`: at `frame_index = t` the input to
  inject lives at column index `t + 1`. Off-by-one between record and
  replay is a silent bit-exact failure invisible on neutral-stick
  sequences but real on varying inputs.
- **`Console.stop()` SIGKILLs Dolphin** and leaves Slippi's recorded `.slp`
  truncated (no GameEnd footer → peppi can't parse it). `Session._teardown`
  sends SIGTERM first and waits briefly so the file-write thread finalizes.
- **Frozen Stadium menu toggle**: libmelee defaults it ON; older tournament
  `.slps` were typically recorded with it OFF. A mismatch is the most
  plausible source of post-spawn physics drift even on non-PS stages.

## Library boundaries

- **peppi-py** is the offline parser. We rely on its dotted field names
  directly (`pre.joystick.x`, `pre.cstick.x`, `pre.buttons_physical`,
  `frame.id`, etc.). The fork renames the ergonomic post-frame attribute
  names (`post.action`, `post.stock`, `post.jumps_used`, `post.hitlag_left`)
  via dataclass metadata while preserving the underlying Arrow / JSON
  field names from the slp wire format. MDS column suffixes
  (`wire.POST_FIELD_SUFFIXES`) match these renamed attributes 1:1, so the
  same suffix addresses both sides without a translation map.
- **libmelee** owns the Dolphin process lifecycle, the Enet/spectator
  protocol, and per-frame controller pipe injection. `apply_inputs` calls
  libmelee's `tilt_analog` / `press_shoulder` / `press_button` /
  `release_button` directly; `fix_analog_inputs=False` keeps our composed
  wire bytes from being re-processed. The fork's canonical `Post` dataclass
  uses the same renamed attribute names as peppi-py's (`action`, `stock`,
  `jumps_used`, `hitlag_left`), so `Trajectory.from_capture` and
  `Trajectory.from_slp` produce identically-keyed dicts. Wire-format math
  (`fix_analog_stick`, `fix_analog_stick_signed`, `raw_byte_to_wire`) lives
  in `melee.controller`; `hal/wire.py` delegates rather than duplicating.
- **Slippi-Ishiiruka exi-ai-rebase** is the emulator build. Released
  AppImage and our local source build are byte-equivalent on the round-trip
  test. Era-mismatched 2020 replays still diverge — that's build-version
  drift, not a code bug.

## When in doubt

If a comment in the codebase says "see ARCHITECTURE.md", look here. If
something belongs on this page but isn't, add it — this is the place where
the right answer is supposed to live.
