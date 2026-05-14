# HAL Project Guidelines

## About the project

The goal of this project is to train Transformer models on Super Smash Bros. Melee using imitation learning & RL.

We're rebooting the project after a long hiatus, and we're in the midst of rewriting everything from scratch.

For context, here is how we did it previously:
- We preprocess human .slp replays using libmelee and stored them as MDS shards following the schema in schema.py
- We sample trajectories from the dataset by choosing a random episode, random starting frame, and preprocessing seq_len subsequent frames to predict controller inputs as next-token prediction
- Preprocessing and target feature discretization are defined as functions in configs: input_configs.py, target_configs.py, postprocess_configs.py
    - The previous best config was `fine_main_analog_shoulder`, which discretizes the analog main stick into 37 joint x, y positions, predicts analog shoulder presses (no digital button L/R), all as single-label classification problems
- Model definitions are under `hal/training/models/gpt.py`
- We have a closed loop eval harness that runs Dolphin emulator and batches inputs on GPU in eval/eval.py. This is a very precise script that writes directly to shared memory buffers, be careful when touching it.

Going forward, I would like to:
- simplify & rewrite the data preprocessing pipeline from .slp to .mds for reliability, scalability & speed
- establish sanity checks for closed loop gamestate reproducibility from .slp to nparrays/tensors back through the melee.Controller interface into Dolphin
- modeling
    - use receding horizon control with flow matching action heads instead of classification
    - action chunk predictions should directly regress on continuous (float) values, either in time or frequency domain (i.e. DCT)
- revisit training loop
- revisit use of tensordicts (need to profile speed)
- revisit eval harness interface
- investigate resuming from arbitrary frames in replay and forking/performing controller takeover in Dolphin to perform efficient rollouts for RL

## Principles

- Be concise.
- Existing code is not precious. Code is tech debt. Delete liberally. The marginal cost of rewriting code rounds to zero, but the benefit of cleaner, better abstractions is high.
- Don't make references to "existing convention" from other parts of the repo in your comments unless asked.
- Invalid states should be impossible to represent.
- Don't re-implement library helpers. If a dependency (libmelee, peppi-py, streaming, torch, ...) already exposes the function you need, call it directly. Local re-implementations drift away from upstream over time and turn library upgrades into silent behavioral changes. The exception is when the upstream function genuinely doesn't exist for your case — then write the smallest primitive that fills the gap and reuse the library for everything else.

## Code Style
- **Formatting**: ruff with line_length=119, isort
- **Types**: Use type annotations everywhere. Return types required.
- **Imports**: Group order: stdlib, third-party, first-party (hal). Single line imports.
- **Naming**: snake_case for functions/variables, CamelCase for classes, UPPERCASE for constants
- **Error Handling**: Use descriptive exception messages, contextmanager for resources
    - Never swallow exceptions (i.e. just `pass`), never use bare `except`
    - Don't catch exceptions just to log and rethrow—only wrap an exception if that part of the stack can add helpful context for debugging
    - Always name the exceptions being caught, ideally with extremely specific clauses; do not write `except Exception` unless it is a crucial runtime code path that must never crash—these cases are uncommon but readily apparent
    - Avoid fallback logic or fallback values that silently change behavior or configuration
- **Type Annotations**: All functions, classes, and variables must specify explicit type annotations. Always include return types for functions. This ensures complete static type safety and clarity throughout the codebase.
    - We are on py314, do not use `from __future__ import annotations`

### Suggested Libraries
- Use `loguru` for logging
- Use MosaicML Streaming `streaming` and MDS format for datasets: https://docs.mosaicml.com/projects/streaming/en/stable/index.html
- Use `libmelee` to handle the Dolphin (emulator) lifecycle, Enet/spectator protocol, and blocking controller injection
- Use `peppi-py` to batch read .slp files offline
- Use `tyro` for CLIs
- Use `@dataclass(frozen=True, slots=True)` for value objects, prefer functional programming patterns over in-place mutations

## Project Structure

### Architecture

A working developer's reference for the data-and-emulator stack: how a `.slp`
file becomes an MDS shard, becomes a per-frame controller view, becomes a
Dolphin pipe write, becomes a captured gamestate that we can `diff` against
the original. Everything below is implemented in `hal/data/`, `hal/scripts/`,
and `hal/emulator/`, with cross-layer vocabulary centralized in `hal/wire.py`.

### Elevator pitch

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

### Three-stage data pipeline

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

### Source-of-truth diagram

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

### Glossary (terminology)

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

Tensor-shape vocabulary used in preprocess/training/models:

| Symbol | Meaning |
|---|---|
| **B** | Batch size |
| **T** | Trajectory length (partially preprocessed sequence sampled from dataset or closed-loop eval buffer size) |
| **L** | Sequence length (training sample sequence length) |
| **D** | Model dimension (a.k.a. `n_embd`, `d_model`, `embedding_dim`) |
| **G** | Preprocessed gamestate / input size — analogous to vocabulary size |
| **C** | Controller input / target size — number of classes |

### Conventions

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
- **Triggers**: `_logical` is peppi's smoothed/fused (across L+R) analog
  value, stored as the training-side analog-shoulder feature.
  `_l_physical` / `_r_physical` are the slp-native per-shoulder f32 values
  used by the emulator wire path. Unlike sticks, the slp spec never stored
  a raw int8 byte for triggers — `physical` *is* the lowest-level signal
  available, and the emulator path goes logical-float → `fix_analog_trigger`
  → wire. `apply_inputs` calls `fix_analog_trigger` explicitly because
  `fix_analog_inputs=False` on the Controller (the flag is stick-driven —
  see Footguns; triggers tag along).
- **Frame range**: peppi's `frames.id` includes pre-game (negative) frames.
  Extract trims to `frame_id >= wire.GAME_START_FRAME = -123` (Slippi
  standard "first in-game frame"). That trims the 2-second countdown but
  preserves the negative ids exactly as peppi yields them.

### Footguns

The list of things that have bitten us once and would silently bite us again
without a checklist.

- **Slp stage IDs ≠ `melee.Stage` enum values.** Fountain of Dreams is slp 2
  but `melee.Stage.FOUNTAIN_OF_DREAMS.value` is 8. Always go through
  `wire.slp_stage_to_libmelee`. Pinned by `tests/test_wire_bridges.py`.
- **Character IDs happen to coincide today** but treat them as distinct
  value spaces. Use `wire.slp_character_to_libmelee` — explicit beats lucky.
  Pinned by `tests/test_wire_bridges.py`.
- **Raw stick bytes are slp-version-gated.** Main-x ≥ 1.2.0, main-y ≥ 3.15.0,
  c-stick ≥ 3.17.0. Older replays fall back to `wire.MASK_INT8` (= -128);
  the emulator wire path then takes the lossy logical-float fallback, which
  drifts ~0.0025 stick-units per frame.
- **Triggers have no raw-byte equivalent in any slp version.** The Slippi
  spec encodes `triggers_physical.{l,r}` as f32 in every version — there
  is no analog of `raw_analog_x/y` for shoulders to extend the emulator
  path with. Don't go looking for one; the logical-float → `fix_analog_trigger`
  → wire path *is* the bit-exact path. Also note `pre.triggers` (peppi's
  logical scalar) is `max(physical_l, physical_r)` — it fuses both
  shoulders, so it cannot distinguish "L only" from "L+R both pressed"
  and is unsafe to feed back through the wire as a single value. Train
  on physical, not logical, if per-shoulder semantics matter.
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

### Library boundaries

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

### When in doubt

If a comment in the codebase says "see CLAUDE.md → Architecture", look here.
If something belongs on this page but isn't, add it — this is the place
where the right answer is supposed to live.

## Where documentation lives

The repo holds exactly **two** markdown files plus inline docstrings. No
module READMEs, no ARCHITECTURE.md, no fragmented doc surface.

- **README.md** — *Strictly* how a new contributor gets the project running.
  Setup, install, environment, and the CLI invocations of the public entry
  points (stage1/2/3, trainer, eval). Nothing else. If a reader who just
  installed the project doesn't need it to run their first command, it
  does not belong in README.
- **CLAUDE.md** — The operating manual for anyone editing this code (human
  or agent). Project goals, principles, code style, library choices, the
  wire/glossary vocabulary, footguns, and library-boundary conventions.
  This is the file you read *after* you've installed and *before* you
  touch code.
- **Docstrings / inline comments** — *Why* this specific code does what it
  does the way it does. Non-obvious invariants, hidden constraints,
  workarounds for specific bugs, behavior that would surprise a reader.
  If removing the comment would not confuse a future reader, do not
  write it. Never restate what well-named identifiers already say.

Tie-breaker when a fact fits multiple homes: docstring > CLAUDE.md > README.
Setup/invocation is the exception — always README, regardless of how local.
