# Protocol 4 integration and cleanup — October 3, 2026

Scope: integrate the retained protocol-4 commits, fix the audited deployment and
operator debt, consolidate documentation and memory. The owner explicitly stopped
this work before G4 activation. No cloud deployment, VM start, image push, R2 write,
Twitch stream or Git push ran during this integration.

## Source reconciliation

The common base was `3219c86c`. Main added two native-simulator commits through
`700165f3`. `netplay-edge-queue` added four shutdown/design/protocol-4/deployment
commits through `f0d79b40`. The old worktree was absent; the branch was intact.

The restored `~/src/hal-edge-queue` worktree merges main with commit `da7a123f`.
There were no conflicts. Cleanup was validated on that branch, then main was fast-forwarded to `62f54764`.
The working diff, staged diff and untracked-file list were identical immediately
before and after the fast-forward. Unrelated work changed concurrently since the
initial snapshot; its latest state was preserved. No stash or reset of main was used.

Milestones:

- `7942ffa1`: align Compose sockets/driver, remove replay expiry, allow later-game
  uploads, fix continuous-game overlay text, default maintenance on, and add CI.
- `5fea487b`: track the encrypted operator helper and exact historical campaign
  source. No credentials or recordings are copied into Git.
- Documentation separates current status/runbook from historical evidence.

## Changes

Production Worker config defaults to maintenance on. `deploy-api.sh` requires an
explicit `--maintenance on|off` and runs Worker tests/typecheck before deploying.
Local launchers and Worker tests explicitly opt out. A regression proves that
maintenance routes return without obtaining a Durable Object stub.

Compose shares only the named Xorg socket and the NVIDIA Vulkan manifest. Other
Xvfb sockets stay private and writable. Bind mounts cannot create missing host
paths. A real `docker compose config --format json` test checks the rendered mounts.

The admin expiry command and its storage helper are removed. Replay metadata
accepts every positive integer game number, including games six and 100. The
serialized metadata format and object-key convention are unchanged. Tests verify
later-game uploads, metadata, local deletion only after success, and invalid
numbers. The direct overlay no longer claims a five-game cap.

CI removes the nonexistent netplay-server extra and checks Worker tests/types and
frontend lint/types/build. The manual integration workflow also tests the local
Worker/runner wire. It installs the locked Worker dependencies first.

`CredentialStore` preserves the existing systemd-creds format and requires private,
owner-held files/directories. Decrypted values go only into memory and the admin
child environment. Dummy subprocess tests cover encryption roundtrip, replacement,
file modes, symlinks, error redaction and the existing admin environment contract.
No real credential was read, re-encrypted or moved.

Old x_pilot controls and the roster-only deployment helper are frozen under
`archive/netplay-2026-09-operator-tools/`, with SHA-256 values. They are incompatible
with continuous protocol-4 sessions and are not advertised as runnable tools.
Porting the historical campaign is a separate feature, not an implicit restart.

## Commands and results

Inspection used `git status/log/worktree/diff`, `cat`, `sed`, `rg`, and selected
Python file reads. Source edits used temporary Python scripts and heredocs inside
the isolated worktree. All source commits use short messages without attribution.

| Command | Result |
| --- | --- |
| `git diff --binary` / `git status --porcelain=v1` | Saved unrelated worktree state before integration under `/tmp/hal-netplay-integration-user.*`. |
| `git worktree add /home/ericgu/src/hal-edge-queue netplay-edge-queue` | Restored isolated worktree. |
| `git merge main -m 'Merge main into netplay integration'` | Passed; no conflicts. |
| `uv sync --locked` | Passed in the isolated worktree. |
| Focused `uv run pytest -q tests/test_netplay_gce.py tests/test_netplay_replays.py tests/test_netplay_admin.py tests/test_netplay_stream.py tests/test_netplay_credentials.py` | 57 passed. |
| `uv run ruff format --check .` | Passed; 303 files. |
| `uv run ruff check .` | Passed. |
| `uv run ty check --python-version 3.14 --error-on-warning hal experiments/059_muon_action_sequence.py scripts` | Passed. |
| `uv run pytest -q -rs -m 'not integration'` | Initial run: 1,681 passed, 29 skipped, 24 deselected. Fixture links were absent at collection; final rerun result follows below. |
| `HAL_REQUIRE_INTEGRATION=1 uv run pytest -q -rs tests/test_roundtrip.py tests/test_session_cleanup.py tests/test_netplay_queue_integration.py -m integration` | Initial run: eight missing-fixture failures, four passed. After fixture link restoration: 12 passed, six deselected, no skips. |
| `npm ci`, `npm test`, `npm run typecheck` in `web/netplay-api` | Passed; 103 tests in 13 files. |
| `npm ci`, `npm run lint`, `npx tsc --noEmit --incremental false`, `npm run build` in `web/netplay` | Passed; static root prerendered. |
| `uv run python scripts/netplay_admin.py --help` | Passed; no credentials loaded. |
| `uv run python scripts/upload_ranked_replays.py --help` | Passed; documented `--root` option confirmed. |
| `bash -n deploy/netplay/deploy-api.sh deploy/netplay/run-local.sh deploy/netplay/gce-startup.sh` | Passed. |
| Python archive-hash and current-doc-link checks | Four archived files match originals; current links resolve. |
| `git diff --check` | Passed before commits. |

### Failures, warnings and skips

- One source search named missing `tests/test_netplay_local.py`; the actual helper
  tests are `tests/test_netplay_local_worker.py`. A documentation check named the
  old `hal/eval/play.py`; the maintained module is `hal/eval/netplay.py`.
- The first Compose regression expected an explicit false JSON field. Compose
  omits false `create_host_path`; the corrected test accepts its normalized form.
  That run had 51 passes and one failure; the rerun passed all 57 focused tests.
- Missing local fixture links caused the first integration failures and the first
  full suite's data-dependent skips. An ignored fixture directory now links the
  existing local fixtures. No fixture upload or download was needed.
- The full integration run emits seven existing multiprocessing/fork warnings.
- npm ci reports blocked install scripts for esbuild/workerd and frontend sharp;
  both builds/tests succeeded. The frontend audit reports 22 findings (one low,
  four moderate, 17 high), including direct vinext, RSC, Vite and build tooling.
  The Worker install separately reports five high-severity dependency findings.
  `npm audit --json` returns exit 1 and suggests framework/toolchain upgrades;
  it is not a clean security audit. Those dependency upgrades were not part of
  the seven audited integration defects and have not been applied blindly.
- The specific Wrangler deployments documentation URL failed to load. The official
  [configuration reference](https://developers.cloudflare.com/workers/wrangler/configuration/)
  loaded; the change uses the already verified `--var` CLI contract.
- No live G4 qualification, FPS comparison, stream check, image build/push or
  production activation was run. These remain owner-held release steps.

## Final validation and memory

Final `uv run pytest -q -rs -m 'not integration'`: **1,708 passed, two skipped,
24 deselected**, 26 warnings, 155.95 seconds. Both skips are the explicitly gated
production GPU qualification tests in `tests/test_netplay_hardware.py`. All local
data tests ran after fixture restoration. The required integration rerun passed
12 tests with six deselections and no skips. No missing required fixture remains.

The active memory index and netplay, harness, emulator/version-gate and direct-mode
notes are consolidated around the current repository runbook/status. Old binary
patch instructions and ready-to-merge harness claims are retired. The local build
recipe stays explicitly historical for debugging. Original memory files are backed
up outside the repository before edits. No unrelated memories or private credentials
are changed. Main receives a fast-forward only after the validated source is committed.

The final handoff does not claim any new deployment.

### Completed integration

`git merge --ff-only netplay-edge-queue` advanced main from `700165f3` to
`62f54764`. A Python check compared working/staged diffs and the untracked-file list
across that operation; all matched. Working diff SHA-256 at that boundary:
`8a92a38e2616d9c7eab9c1331b6b7008c46643678851d79b4853aa5f84382630`.

Seven local memory files were updated: MEMORY, netplay edge queue, old harness,
old version gate, Dolphin setup, direct-mode behavior and local build history.
The two obsolete harness/version-gate entries were removed from the active index;
their files now explain their historical status. The original notes are retained
in `/tmp/hal-netplay-memory-before-20261003-92z0joai/`. All other memory content is
untouched. The canonical state is the tracked runbook and deployment ledger.

The fixture link was converted to an ignored `data/` directory with links only to
existing emulator/raw/processed fixtures. The integration worktree is clean.
The documentation commit hooks skip source checks because no source changed.
No dependency version or production service changed during the final handoff.
