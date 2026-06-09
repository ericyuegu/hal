# HAL Project Guidelines

## About the project

The goal of this project is to train Transformer models on Super Smash Bros. Melee using imitation learning & RL.

The offline data pipeline (`.slp` → MDS shards) lives in `hal/data/` and is driven by the CLI stages under `hal/scripts/`.
The closed-loop driver (Dolphin + libmelee) lives in `hal/sim/`: `Session` owns the emulator process, `ControllerSource` implementations produce per-port inputs, and `drive()` runs the step loop that powers round-trip validation, online eval vs CPU, self-play, and RL rollouts.
Cross-layer wire conventions (button bits, mask sentinels, deadzones, port and stage/character bridges, post-frame field naming) are the single source of truth in `hal/wire.py`.
Project policy (included characters/stages, player port conventions) lives in `hal/policy.py`.
Integration fixtures (dev archive, MDS bundle, ISO, Dolphin) are declared in `hal/fixtures.py` and fetched into `<repo>/fixtures/` via `python -m hal.scripts.fetch`; see `README.md`.
Cloud GPU training runs in a Docker image (`docker/`, vast.ai CUDA base) carrying code+deps only; `docker compose -f docker/compose.yaml run --rm hal …` mounts `data/`, reserves the GPU, bumps `--shm-size` (StreamingDataset uses `/dev/shm`), and runs Xvfb so the closed-loop eval gets a GL context. The instance is stateless: datasets/emulator are fetched at runtime (never baked) and checkpoints stream to R2 in the background (`hal/training/checkpoints.py`), with `--resume <run>` pulling them back. R2 client/creds are shared via `hal/r2.py` (one source for both `fetch` downloads and checkpoint uploads); checkpoints deliberately bypass the immutable, sha-pinned `Fixture`/`fetch` path.

## Controller data model

One controller representation end-to-end: the **logical** (game-causal) values peppi reads from a .slp are what the MDS stores, what the model predicts, and what `apply_inputs` feeds back. The only wire conversion is `fix_analog_stick_signed` / `fix_analog_trigger` in our libmelee fork — nothing else translates.

- **Sticks** (`main_stick_*`, `c_stick_*`): slp-logical, [-1, 1] on the 1/80 grid. Melee zeroes the dead-band (gate 23/80 = 0.2875 per axis) and clamps at ±80, so logical is post-deadzone; raw byte jitter is game-inert and not stored.
- **Triggers** (`trigger_l/r`): per-shoulder, [0, 1] on the 1/140 grid (Melee saturates at byte 140; physical = byte/140), zeroed below `wire.TRIGGER_DEADZONE` (43/140) at extract. slp's fused `pre.triggers` scalar (max of both shoulders) is lossy and unused.
- **Buttons**: multi-label bitmask (`wire.BUTTON_BITS`) — all combinations co-record; ~19% of pressed human frames hold ≥2 buttons. The digital L/R click is a distinct causal channel from the trigger analog: click ⇒ analog = 1.0, but not vice versa (lightshield). Keep both; exclude START from the action space (pause).
- **Wire protocol**: stock Dolphin pipe semantics — `SET {L,R} t` means trigger byte `u8(t·255)`; stick floats are 0.5-centered. Our exi-ai Dolphin ≥ 0.2.1 matches stock on both the GCPad and EXI input paths (0.2.0 mangled GCPad-path triggers; the EXI fast-forward path vladfi trains on was always correct).
- **Round-trip guarantee**: every stored value reproduces its exact byte through the pipe, and pipe→slp latency is a constant +1 frame. Same-build record→replay is bit-exact on post-frame gamestate. Replaying era-mismatched 2020 dev slps diverges in spawn descent from build drift — known, not a wire bug.

## Principles

**Posture**
- Be concise. Don't be lazy — fix smells you encounter en-route; >30min, leave a TODO and flag.
- Delete liberally. Code is tech debt — rewrites are cheap, better abstractions compound. Versioning is git's job: no `*_v0.py`, `stage1_*`, `*_old.py`.
- Invalid states should be impossible to represent. Fail loud, fail early — no fallback values that silently change behavior or configuration.
- Don't re-implement library helpers (libmelee, peppi-py, streaming, torch). Local copies drift and turn upstream upgrades into silent behavior changes. If the upstream genuinely doesn't fit, write the smallest primitive that fills the gap and reuse the library for everything else. Fork-dep fixes (libmelee, peppi-py) go upstream, not into a local translation layer.
- Don't reference our conversations or "existing convention from elsewhere in the repo" in code comments.
- Follow the 3-tier codebase layout for organizing shared infra: https://www.moderndescartes.com/essays/research_code/

**Architecture**
- One source of truth per cross-cutting vocabulary. `wire.py` owns slp/wire conventions (button bits, ports, mask sentinels); `policy.py` owns included character/stage tuples. No second source.
- Schema is versioned; consumers fail loud on mismatch. Extend `SCHEMA_VERSION` discipline to any future shared artifact (action tokenizer, observation builder).
- Round-trip is the contract. No PR touching `extract`, `wire`, `sim/inputs`, or `sim/session` lands without a green `pytest tests/test_roundtrip.py` (wire-format faithfulness, same-build bit-exact record→replay, analog sweep on the full trigger/stick grids).
- Value objects: frozen dataclasses. Behavior surfaces: Protocols. Transforms: free functions. Composition over inheritance and over generators (a `Source` Protocol + explicit step loop beats a yielding generator that receives inputs). Classes only own genuine resources (`Session` owns a Dolphin process).
- Policies are pure `obs → action`. The model never touches libmelee; the simulator never touches torch. Glue lives in the eval driver.
- Hot path is zero-allocation. No per-frame `dict()`, `torch.tensor(...)`, or Python loop over button names. Pre-resolve at import time (see `_BUTTON_DISPATCH` in `sim/inputs.py`).

**Organization**
- `hal/__init__.py` is a curated public API facade. Explicit re-exports with `__all__`; no side-effecting imports; no `import *`.
- No utility grab-bags. `utils.py`, `helpers.py`, `common.py` are forbidden. Name files by what they own.
- `hal/data/` owns single-replay primitives, value objects, pure transforms, and shared schema. `hal/scripts/` owns per-stage CLI entry points AND the cross-replay orchestration that drives the stage (incremental walks, pool plumbing, batched IO, archive dispatch). One module per stage; no cross-script helpers.
- `hal/experiments/` holds single-file experiments (data preprocessing, model arch, training objective, optimizer, training loop). These must not cross-import from notebooks and vice versa.
- Three "scripts" homes, by tier — don't conflate: `hal/scripts/` is package code (data-pipeline stage CLIs, shipped in the wheel, run as `python -m hal.scripts.X`); root `scripts/` is host ops that does NOT import `hal` (dev-box `setup.sh`, the `launch_vast.py` cloud launcher); `docker/` is the cloud image and what runs *inside* it (`Dockerfile`, `compose.yaml`, `entrypoint.sh`, `on-start.sh`). The launcher lives in `scripts/`, not `docker/`, because it runs on the host and is never baked into the image.

## Code Style

- **Formatting**: ruff with `line_length=119`, isort.
- **Types**: Type annotations everywhere; return types required. Types over primitives — encode domain semantics and invariants in the type system. Prefer static over dynamic checks. py3.14; do not use `from __future__ import annotations`.
- **Imports**: Group order: stdlib, third-party, first-party (hal). Single-line imports.
- **Naming**: idiomatic Python.
    - Shortest unambiguous name in context — the same value can take different names in different functions. `def clean_company_name(name: str)` beats `def clean_company_name(company_name: str)`. When you do need specificity, keep going: `noun_adj_adj_other_modifiers` is fine.
    - Capitalize acronyms in CamelCase (`MDSWriter`, not `MdsWriter`).
    - Order args intuitively. Prefer explicit kwargs at call sites when ordering or types aren't obvious. Use lone `*` to force keyword-only.
- **Control flow**: keep the happy path at the lowest indentation level.
- **Error handling**: descriptive messages; `contextmanager` for resources.
    - Never swallow exceptions (lone `pass`); never use bare `except`.
    - Don't catch just to log and rethrow — only wrap if this layer can add helpful context.
    - Always name the exception classes caught, ideally with specific clauses. Avoid `except Exception` unless this is a crucial runtime path that must never crash.

## Suggested libraries
- `loguru` for logging
- MosaicML `streaming` + MDS format for datasets: https://docs.mosaicml.com/projects/streaming/en/stable/index.html
- `libmelee` for the Dolphin lifecycle, Enet/spectator protocol, blocking controller injection
- `peppi-py` for batch reads of .slp files
- `tyro` for CLIs
- `@dataclass(frozen=True, slots=True)` for value objects; prefer functional patterns over in-place mutation
