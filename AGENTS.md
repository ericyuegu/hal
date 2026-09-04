# HAL

HAL trains Melee policies from Slippi replays and evaluates them in Dolphin.

## Priorities

Use this order:

1. Preserve correctness and experimental validity.
2. Make the main path and data flow obvious.
3. Improve measured performance.

## Work

- Read the affected code, callers, tests, schemas, and persisted formats before you edit.
- Make the smallest coherent change.
- Fix an adjacent defect only when the fix is local, obvious, behavior-preserving, and covered by a focused test. Report other defects.
- Reject invalid configuration, unsupported data, schema mismatches, and artifact mismatches. Do not silently use a fallback.
- Prefer direct code. Prefer duplication to the wrong abstraction.
- Preserve unrelated user changes.
- Comments explain a reason, constraint, or invariant. Do not narrate an edit or conversation.
- Use clear, concise prose.

## Boundaries

- `hal/` owns stable, reusable code.
- `experiments/` contains standalone research programs.
- An experiment can import `hal`. It must not import another experiment.
- Duplication between experiments is intentional.
- Move code to `hal/` only when its contract is stable and it has two real consumers or a production boundary.
- Do not compose experiments with dynamic imports, wildcard exports, module `__getattr__`, or mutation of another module.
- Freeze a completed experiment. Reproduce it from its recorded Git commit, environment, data manifest, and artifacts.
- A CLI parses and dispatches. Put reusable behavior in the module that owns it. Import only public names across modules.
- `hal/wire.py` owns wire conventions.
- `hal/policy.py` owns project policy.
- `hal/data/schema.py` owns stored columns.
- The model must not import `melee`.
- Simulation must not import `torch`.
- Evaluation joins the model and simulation.
- Treat persisted formats as APIs. Version them and validate their identity.

## Research

- State the treatment, control, and invariant inputs before a run.
- Do not hide a scientific change inside a refactor.
- Record the Git SHA, resolved configuration, seed, environment, data manifests, and artifact hashes.
- Exact resume restores the model, optimizer, scheduler, loader cursor, and every random-number generator.
- Test the next batch and the next optimizer update after resume.
- Offline training and live evaluation must use the same representation.

## Python

- Write simple, direct Python.
- Keep the main path at the lowest practical indentation level.
- Use a class only for identity, lifecycle, learned parameters, or real mutable state.
- Use frozen, slotted dataclasses for value objects.
- Add a protocol only when there are multiple implementations or a useful test seam.
- Type all maintained code. The active type-check target has zero diagnostics.
- An experiment config owns experiment choices. A library API accepts only the values that it uses.
- Avoid untyped mappings, dynamic attributes, and `**kwargs` at domain boundaries.
- Pass dependencies explicitly. Do not patch module globals.
- Keep worker functions and serialized callbacks at module scope.
- Use tensor shape suffixes when they remove ambiguity.
- Acquire resources inside their cleanup scope. Clean up partial initialization.
- Catch specific exceptions. Do not swallow exceptions or catch only to log and raise again.
- Write a docstring only when it adds a contract, reason, or invariant that the signature cannot show.

## Compatibility

- Keep a workaround at the dependency or format boundary that owns it.
- State the affected dependency and tested version.
- State the failure, regression test, and removal condition.
- Pin a dependency when HAL calls its private API.
- Reject dependency versions that have not passed the compatibility tests.
- Prefer a wrapper or subclass.
- If a global patch is unavoidable, centralize it, make it idempotent, and test it.
- Do not add compatibility for a hypothetical consumer.

## Performance

- Profile before you optimize.
- Keep static work out of frame and batch loops.
- Measure the complete path.
- Report emulator FPS, frame latency, throughput, or MFU before and after a performance change.

## Validation

Run the smallest relevant test while you work.

Every behavior change needs a focused regression test. Every refactor needs parity coverage.

Before you hand off a code change, run:

    uv run ruff format --check .
    uv run ruff check .
    uv run ty check --python-version 3.14 --error-on-warning \
      hal \
      experiments/051_muon_parameterization.py \
      scripts/cache_modal_fixtures.py \
      scripts/launch_gce.py \
      scripts/launch_modal.py \
      scripts/launch_vast.py \
      scripts/replay_policy_fault.py
    uv run pytest -q --ignore=tests/experiments -m "not integration"

If a change touches replay extraction, wire format, controller input, session stepping, or offline/live parity, also run:

    HAL_REQUIRE_INTEGRATION=1 uv run pytest -q \
      tests/test_roundtrip.py \
      tests/test_session_cleanup.py \
      -m integration

A missing required fixture is a failure. Report each command, failure, and skip.
