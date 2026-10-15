# Independent stream overlay, October 15, 2026

The owner requested deployment of the committed website/overlay restyle, then
approved separating overlay deployment from gameplay and restoring physical frame
delays 2 and 3.

## Release

- Runtime source: `ffed25c3c944d55ed37bb5564d8956418171f193`.
- Host deployment fix: `b5b8dd08a910d2ae136633d486197fa564259695`.
- Runtime digest: `sha256:32fddfcd3085cb512f4f1e55a04f4c27471d862a0b27bd20ab12304731fd0652`.
- Registry: `us-west1-docker.pkg.dev/centering-star-502613-k3/hal-netplay/hal-netplay-runner`.
- Website Worker: `hal-netplay-web`, version `fad5795b-68c8-4206-b429-cc65bc51ff15`, route `20xx.xyz/*`.
- Existing VM: `hal-netplay-g4-1015-us-west4-a`, `us-west4-a`.
- The API Worker, policy weights, account objects, and experimental inputs were unchanged.
  Only the published policy metadata was changed to offer delays `[2, 3]`.

The migration image derives from the exact prior deployed image, adds IBM Plex
and the committed Schibsted fonts, and reinstalls HAL with `--no-deps`. Its build
recipe is saved with the ignored release evidence. No inference Python dependency
was upgraded. Native OBS uses Schibsted Grotesk and IBM Plex Mono, confirmed by
`fc-match` in the built image.

## Operation

The runner retains Dolphin, display management and PulseAudio. A separate
`hal-netplay-overlay.service` runs OBS, sharing the runner's network namespace,
Xorg socket and audio socket. Its mounted source/fonts use an atomic `current`
symlink and a source revision file. It does not receive the runner environment
or cloud/queue credentials. Queue control writes an owner-only versioned stream
lease with a 30-second expiration. Capacity updates cannot extend lease authority
without health acknowledgements; storage failure does not disconnect gameplay.

Use `deploy/netplay/gce-overlay.sh NAME --project PROJECT --zone ZONE` after
committing changes. It validates imports, switches releases, restarts only OBS,
and checks the new service heartbeat and live output. Failed activation restores
the previous release. Twitch reconnects briefly; games continue.

The host service bootstrap is supplied as VM metadata alongside the startup
script, so service configuration does not require a runtime image rebuild.
Future full image builds use the maintained Dockerfile.

Both physical delays share horizon 8, fixed prefix 4, and replan interval 4.
Delay 2 retains its two-frame inference allowance; delay 3 allows one frame.
One shared prediction profile keeps mixed-delay games in a single inference
batch. Each match receives a RuntimeConfig with its selected physical delay.
Full-capacity qualification uses delay 3's stricter 16.667 ms p99 limit; its
measurement is recorded against both schedules.

## Validation

- `uv run ruff format --check .`: passed, 330 files.
- `uv run ruff check .`: passed.
- `uv run ty check --python-version 3.14 --error-on-warning hal experiments/059_muon_action_sequence.py scripts`: passed, zero diagnostics.
- `uv run pytest -q -m "not integration"`: final run 1,854 passed,
  eight skipped, 25 deselected, 26 warnings in 161.36 seconds.
- `HAL_REQUIRE_INTEGRATION=1 uv run pytest -q tests/test_roundtrip.py tests/test_session_cleanup.py -m integration`: eight passed, six deselected, zero skips.
- Focused overlay/runner/admin/GCE/control tests: 129 passed. Updated GCE/overlay
  tests after the host-path fix: 30 passed.
- Frontend test, lint, TypeScript check, build and Worker deployment passed.
- Bash parsing and `git diff --check` passed.

The eight unit skips are the existing six optional local v7 dataset fixtures and
two opt-in production GPU tests. The live runner's own qualification is recorded
separately. Warnings are the existing Torch/websockets/fork warnings.

Two test runs were interrupted by session restarts and were rerun. An initial
OBS socket test needed network escalation; it passed with socket access. A test
synchronization error in the new restart test was fixed before release. A scratch
operator script incorrectly used AdminClient as a context manager; it failed
before making a mutation and was corrected to use `contextlib.closing`.

## Migration findings

The first seed used an absolute host release link, which did not resolve at the
container mount path. The link and installer were corrected to use relative
paths. A regression test executes the seed script and moves its root to a different
mount path. The live link was corrected and only the overlay was restarted.
Initial display/capture startup attempts also retried before Openbox and the
shared display were ready. NVIDIA GLX and OBS capture subsequently initialized.
No active human games were present during the migration; admissions were paused.

## Live verification

Eight-stream complete-path qualification passed with 200 samples: p50 10.164 ms,
p95 14.767 ms, p99 15.177 ms. This is below both the delay-2 allowance (33.333 ms)
and the delay-3 allowance (16.667 ms). These are one shared measurement, not two
independent experiments. All eight slots became ready at 22:09:48 UTC.

The actual native OBS scene was captured and inspected: new fonts/layout,
no difficulty label. OBS reported 60.0000024 fps, active output, zero network
or encoder drops, zero congestion, and no reconnect after startup. Initial render
skips were four. No claim is made about live emulator FPS without a human game.

The new overlay-only deploy command successfully activated release
`b5b8dd08a910d2ae136633d486197fa564259695` and verified fresh active output.
The gameplay container ID remained
`2bc06ef7f450c1ca6073ef7e26a1e6855c023f5e073fe2d9eb2bb6669328579a`,
its host PID remained `34207`, and its start time remained
`2026-10-15T22:06:07.196295523Z`. All eight slots stayed healthy; `/healthz`
returned `ok`. This demonstrated the source-bundle rollout without rebuilding an
image, restarting gameplay, or re-running model preparation.

The runtime image remains pinned to source `ffed25c3`; the independently deployed
overlay and host bootstrap use source `b5b8dd08`. The boot metadata supplies the
corrected host bootstrap directly, rather than reading its older image copy.
Evidence is in ignored `runs/netplay/release-20261015-overlay/`.

Admissions were reopened after these checks. Public options returned `[2, 3]`
and public capacity reported eight healthy slots with service status `ready`. A live human game/rematch
and reboot were not exercised in this release.
