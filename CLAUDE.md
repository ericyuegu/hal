# HAL Project Guidelines

## About the project

The goal of this project is to train Transformer models on Super Smash Bros. Melee using imitation learning & RL.

HAL is a 1v1-Melee imitation/RL training rig. The offline data pipeline (`.slp` → MDS shards) lives in `hal/data/` and is driven by the CLI stages under `hal/scripts/`. The closed-loop driver (Dolphin + libmelee) lives in `hal/sim/`: `Session` owns the emulator process, `ControllerSource` implementations produce per-port inputs, and `drive()` runs the step loop that powers round-trip validation, online eval vs CPU, self-play, and RL rollouts. Cross-layer wire conventions (button bits, mask sentinels, raw↔wire math, port and stage/character bridges, post-frame field naming) are the single source of truth in `hal/wire.py`. Project policy (included characters/stages, player port conventions) lives in `hal/policy.py`.

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
- Follow the 3-tier codebase for organizing shared infra and utilities, allowing experiments to re-use core components: https://www.moderndescartes.com/essays/research_code/

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

## Operating principles

1. **One source of truth per cross-cutting vocabulary.** `wire.py` owns button bits, port mappings, mask sentinels. No second source. `policy.py` owns the included character/stage tuples. No second source.
2. **Schema is versioned; consumers fail loud on mismatch.** Extend the `SCHEMA_VERSION` discipline to any future shared artifact (action tokenizer config, observation builder).
3. **Round-trip diff is the contract.** No PR touching `extract`, `wire`, `sim/inputs`, or `sim/session` lands without a green `python -m hal.scripts.roundtrip` on the dev MDS.
4. **Hot path is zero-allocation.** No `dict()` per frame, no `torch.tensor(...)` per frame, no Python-level loop over button names per frame. Pre-resolve at import time (see `_BUTTON_DISPATCH` in `sim/inputs.py`).
5. **Value objects: frozen dataclasses. Behavior surfaces: Protocols. Transforms: free functions.** Classes only for things that own genuine resources (`Session` owns a Dolphin process).
6. **No utility grab-bags.** `utils.py`, `helpers.py`, `_helper.py`, `common.py` are forbidden. Name files by what they own.
7. **Folder names are nouns, not adjectives.** `data/`, `sim/`, `scripts/`. Not `core/`, `common/`, `infra/`, `legacy/` (legacy is git history).
8. **Number-suffixed names are a smell.** `*_v0.py`, `stage1_*`, `*_old.py`. Versioning is git's job.
9. **`hal/__init__.py` is a curated public API facade.** Explicit re-exports with `__all__`; no side-effecting imports; no `import *`.
10. **Policies are pure `obs → action`.** The model never touches libmelee directly; the simulator never touches torch. Glue lives in the eval driver.
11. **Composition over generators for staged work.** A `Source` Protocol plus an explicit step loop beats a yielding generator that receives inputs.
12. **Fail loud, fail early.** Assert at load, raise (not log-and-continue) on inconsistent state. No fallback values that silently change behavior.
13. **CLIs are ≤80 lines of `tyro` glue.** The work is in importable functions in the library.
14. **Fork-dep fixes go upstream**, not into a local translation layer. (libmelee, peppi-py)
15. **Delete liberally.** Transitional code that "might be useful one day" rarely is. Git remembers.
