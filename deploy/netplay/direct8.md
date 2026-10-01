# Eight-slot direct play — October 1, 2026 UTC

## Configuration

The owner approved eight slots sharing one Slippi account, a 60-second initial
connection deadline, and delay-2-only admission. Rematches retain 600 seconds.
The owner then renamed the account from HAL#647 to **HAL#9000**.

The queue leases one account per session. A persistent pairing row permits one
new search at a time. Playing games and their rematches continue concurrently.
Cancellation, failed connection, and lease expiry keep that row until Dolphin
cleanup finishes. Cleanup includes the attempt number, so an old retry cannot
release a newer pairing. The supervisor handles dead slots and engine recovery.

Runner protocol is 2. Storage schema is 3, at instance `global-v3`. Deployment
will use a fresh queue and republish the existing policy and account references.
The former `global` instance retains its historical state. There is no migration.
No reservation was active or queued at preparation time; admissions were paused.

## Inference measurement

Control and treatment used the same G4, one RTX PRO 6000 Blackwell Server
Edition, policy, seed 120647, BF16, compiled KV cache, eight request streams,
0.5 ms batching window, replan 4, and horizon 8. Prediction measurement includes
IPC, batching, validation, and model execution. Each run took 200 samples.

| Schedule | Prefix | Inference allowance | p99 | Result |
| --- | ---: | ---: | ---: | --- |
| Delay 2, previous schedule | 3 | 16.667 ms | 22.409 ms | Failed |
| Delay 2, approved schedule | 4 | 33.333 ms | 20.97 ms | Passed |

The approved schedule measured 18.90 ms p50, 20.48 ms p95, and 21.13 ms maximum.
It has no spare horizon beyond transport, inference allowance, and replan.
This establishes eight-stream inference capacity. It does not establish eight
simultaneous live games or their frame rate.

The control used image/source `8831869a7b315fc2755f895b399c4f84a968a84a`.
The treatment used a source archive based on `fb4316cb`, with the pending
timing and pairing changes. Archive SHA-256:
`237b75c9d6c5ad36db0e3a9fb8f8fb9e193fbbeebe906c6bda6829126ee26325`.
The production image will repeat qualification from its committed source.

Policy SHA-256:
`0ff1daf80caa36a94a713c4ccba9223db8d7ba7c1379b5865bbc40b8a8c2f3ec`.
Checkpoint SHA-256:
`52b5233ed506f59f514f7e90a6a6111206152db7413f451dc35be5c30d1e671b`.
Environment: Python 3.14.3, Torch 2.11.0+cu130, Linux 7.0.0-1011-gcp.
Treatment preparation and measurement took 188.21 seconds.

The [parallel pairing experiment](parallel-pairing.md) failed by self-pairing.
The earlier [sequential pairing experiment](account-reuse.md) established two
concurrent neutral-input games. These are separate from model qualification.

## Account rename

Only `connectCode` changed in the existing local account export. The display
name remains HAL. Authentication fields were preserved. A mode-600 private
backup records the old export. The renamed account was uploaded and published
with `hal-netplay-admin accounts upload`. Its SHA-256 is
`9af49baaaee4f9767ec3dfb7db4fc8904e54056643730fb1b51cca54373d61ae`.

The page uses `job.connect_code`; it contains no fixed HAL code. The runner
reads the code from the verified account and reports it on connection.
New ranked replay paths use the account's code from each actual replay.
Old replays and their paths retain HAL#647. Current operator instructions and
the x_pilot chat controls use HAL#9000. Replay validation in the campaign
control reads the code recorded on that job.

## Validation and command results

Evidence is under `runs/netplay/direct8/`. Private account files stay in its
mode-700 `private/` directory and are not committed.

- `pwd`, `cat AGENTS.md`, `git status --short`, `git log`, `git diff`,
  `git diff --stat`, `git diff --check`, `rg`, selected `sed`/`cat` reads,
  and Python JSON summaries inspected source, callers, tests, docs, and state.
  Exploratory reads of an absent `/var/empty/no-such-file`,
  `hal/inference/qualification.py`, `web/netplay-api/test/README.md`, and
  `tests/conftest.py` failed. Correct searches found the maintained files.
- `uv run pytest -q tests/test_netplay_runner.py tests/test_netplay_queue_client.py`
  initially passed 108 tests after an error-message expectation was corrected.
  Added regressions found one wrong test expectation: retry starts at 0.25 s,
  not 0.5 s. Correcting it passed the expanded focused suite.
- `uv run pytest -q tests/test_netplay_admin.py tests/test_netplay_queue_client.py tests/test_netplay_runner.py`
  passed **133** tests.
- `uv run ruff format --check .` passed for 287 files.
  `uv run ruff check .` initially found one import-order error in a new test.
  `uv run ruff check --fix tests/test_netplay_queue_client.py` corrected it.
  Both checks then passed.
- `uv run ty check --python-version 3.14 --error-on-warning hal experiments/059_muon_action_sequence.py scripts`
  passed with zero diagnostics.
- `uv run pytest -q -rs -m 'not integration'` passed **1,626** tests,
  with **eight skips**, 21 deselections, and 26 warnings in 149.40 seconds.
  Skips were two opt-in GPU qualification tests and six optional local v7
  subset tests. G4 timing was measured separately above.
- `HAL_REQUIRE_INTEGRATION=1 uv run pytest -q tests/test_roundtrip.py tests/test_session_cleanup.py tests/test_netplay_queue_integration.py -m integration`
  passed **ten** tests, with six deselections and six warnings in 97.42 seconds.
  No required fixture was missing.
- API `npm test` passed **118** tests in twelve files. Earlier runs failed
  expectations for protocol version, parallel claims, and missing pairing
  cleanup in old traces. The traces now contain explicit cleanup steps and
  check the new connection deadline at 59 and 60 seconds. Their rematch
  deadline remains unchanged. An unrelated proposed code check was removed.
  Workerd printed its existing WebSocket shutdown diagnostic.
- API `npm run typecheck` and `npx wrangler deploy --dry-run` passed.
- `python3 -m py_compile` passed for the two updated x_pilot operator controls.
  No Twitch chat message was sent and that campaign remains paused.
- A credential-backed `AdminClient.status()` read confirmed zero active and
  waiting jobs, one free account, and paused admissions.
- `uv run hal-netplay-admin accounts upload runs/netplay/direct8/private/user.json`
  succeeded with the existing encrypted credentials. A subsequent admin read
  confirmed that HAL#9000 is the only published account. Admissions stayed paused.
- `tar --exclude=__pycache__ -czf /tmp/hal-direct8-source.tgz hal`,
  `sha256sum`, and `scp` staged the exact qualification source on G4.
  The first audit-script edit failed because its copied file was root-owned.
  Repeating that edit with `sudo python3` succeeded.
- Remote `docker run -d --name hal-netplay-capacity8-delay2 --gpus all --ipc=host --network=none ... python /audit/qualify.py --capacity 8 --bundle /policy.halpolicy --output /audit/qualification-8.json`
  exited zero. `docker inspect`, `docker logs`, `nvidia-smi`, and `scp`
  verified and retained the result. This run accepted no public matches.
- Remote `Xvfb` and `glxinfo -B` showed Mesa llvmpipe on a private display.
  `vulkaninfo --summary` failed with no Vulkan driver. The image has no
  NVIDIA Vulkan ICD registration. This blocks use of GPU rendering for
  non-streamed slots until repaired; the streamed Xorg display already uses NVIDIA.
- Process inspection confirmed the test jobs completed. A broad first `ps`
  filter also matched old unrelated shell wrappers; they were left untouched.

## Deployment status

The existing one-GPU G4 is running at 8.229.68.10. The game runner is stopped;
the health container remains up. Eight-slot source is validated, but its image
and schema cutover are not deployed yet. Ranked and local player loops remain
paused. No Git push was made.

## Automatic GPU setup

The G4 driver is NVIDIA 580.173.02. Its host manifest declares Vulkan 1.4.312
and `libGLX_nvidia.so.0`. The container already had that library, but lacked
the manifest. NVIDIA documents this
[driver registration](https://download.nvidia.com/XFree86/Linux-x86_64/570.124.04/README/installedcomponents.html).
A read-only mount of the host manifest made `vulkaninfo --summary` identify
the RTX PRO 6000 as a discrete NVIDIA GPU. The unmodified image could not
create a Vulkan instance.

`gce-startup.sh` now installs matching graphics and video packages, checks
that the driver manifest exists, mounts it read-only, and sets
`VK_DRIVER_FILES`. It runs `vulkaninfo` inside the selected runner image and
requires an NVIDIA device before starting services. The streamed slot retains
NVIDIA OpenGL on Xorg :90. Managed non-streamed slots use Vulkan on Xvfb.

Reapplying startup metadata now drains existing runner and health services
before changing Docker. It restarts Xorg after writing its configuration and
authorization, then restarts both containers from the new unit definitions.
This also covers a restart of the existing VM; no service-file edit is needed.

Validation:

- Exact remote `docker run --rm --gpus all ... vulkaninfo --summary` preflight
  passed, with and without a test Xvfb display. Results are
  `vulkan-after.log` and `vulkan-preflight.log`.
- `bash -n deploy/netplay/gce-startup.sh` passed.
- Focused `uv run pytest -q tests/test_netplay_runner.py tests/test_netplay_gce.py`
  passed **92** tests. The tests cover fresh and existing host setup, draining
  before Docker changes, the matching driver mount, renderer selection, and
  clean drain when the last worker exits during the supervisor loop.
- Ruff format, Ruff check, and ty passed. One test formatting check failed
  first; `uv run ruff format tests/test_netplay_gce.py` corrected it.
- The full non-integration suite passed **1,627** tests with the same eight
  skips, 21 deselections, and 26 warnings in 148.79 seconds. The final shell
  restart changes also passed the focused suite.
- The mandatory combined integration command passed **ten** tests with six
  deselections and six warnings in 97.18 seconds.
- A source search first named absent `tests/test_netplay_host.py`; the
  maintained host tests are in `tests/test_netplay_gce.py`.
- Reading `/etc/vulkan/icd.d/nvidia_icd.json` on the host failed; the installed
  host manifest is `/usr/share/vulkan/icd.d/nvidia_icd.json`.
- API deployment `npx wrangler deploy` succeeded, version
  `11b4ac42-3ca3-4f83-abfa-00efb53c1b02`, on the existing route.
  The new queue was paused, the same policy was published with only
  `online_delays` changed to `[2]`, and HAL#9000 was registered.
  A temporary session received eight grants for that one account and ended
  cleanly. Public options report delay 2, no-show 60 seconds, rematch 600.
  This check did not report fabricated healthy capacity.
- `docker build --file deploy/netplay/Dockerfile --build-arg HAL_GIT_SHA=9817205a --tag hal-netplay-runner:direct8-prebuild .`
  prepares the build cache. That temporary tag is not a release or deployment.

Separate frontend wording and spec edits appeared during this work. They are
outside this GPU setup commit.
