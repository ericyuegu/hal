# Eight-slot direct play — October 1, 2026 UTC

Historical record. Status and commands below apply to the recorded release.
See [current deployment status](status.md) and [the runbook](README.md).

## Configuration

The owner approved eight slots sharing one Slippi account, a 60-second initial
connection deadline, and delay-2-only admission.
The owner then renamed the account from HAL#647 to **HAL#9000**.

The queue leases one account per session. A persistent pairing row permits one
initial search at a time. Connected reservations continue concurrently.
Cancellation, failed connection, and lease expiry keep that row until Dolphin
cleanup finishes. Cleanup includes the attempt number, so an old retry cannot
release a newer pairing. The supervisor handles dead slots and engine recovery.

Runner protocol is 3. Storage schema is 5, at instance `global-v5`. Deployment
uses a fresh queue and republishes the existing policy and account references.
The former instances retain their historical state. There is no migration.
No reservation was active or queued at preparation time; admissions were paused.

## Continuous reservations

A direct reservation has no set or game cap, and Dolphin keeps the player
connected between games. If the queue is non-empty after 15 minutes assigned,
the current game is the last before the reservation yields. Its end reason is
player canceled, page left while queued, player disconnected, no show, idle
timeout, yielded, or service failure.

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
The production image repeated qualification from its committed source; see
the final deployment result below.

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
  check the new connection deadline at 59 and 60 seconds. These protocol-2
  traces also record the former rematch deadline, which protocol 3 removes.
  An unrelated proposed code check was removed.
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

The existing one-GPU G4 is running at 8.229.68.10. The new image and schema
are deployed. The runner reports eight healthy slots, zero recoveries, and
zero service restarts after the socket fix. Admissions are open. OBS is live.
Ranked and local player loops remain stopped. No Git push was made.

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
  cleanly. At that release, public options reported delay 2, no-show 60 seconds,
  and the former 600-second rematch deadline.
  This check did not report fabricated healthy capacity.
- `docker build --file deploy/netplay/Dockerfile --build-arg HAL_GIT_SHA=9817205a --tag hal-netplay-runner:direct8-prebuild .`
  prepares the build cache. That temporary tag is not a release or deployment.

Separate frontend wording and spec edits appeared during this work. They are
outside this GPU setup commit.

## Private display socket fix

The first eight-slot launch failed before inference preparation. The runner
mounted the complete host X11 socket directory read-only. Xvfb could start
an abstract listener but could not create its filesystem socket. The runner
therefore timed out on display :101 and restarted.

The launch script now mounts only the host Xorg socket, X90. The remaining
socket directory belongs to the container. A G4 reproduction with the former
mount logged a Unix listener failure. The corrected mount created X151 and
passed an actual `xsetroot` request. Both runs emitted nonfatal xkbcomp
warnings about unsupported key names.

- Remote `systemctl stop hal-netplay-runner.service` stopped the empty restart
  loop. No game was active and admissions remained paused.
- Remote `docker run --rm --network none ... Xvfb :151 ... xsetroot`
  reproduced the failure and verified the corrected socket mount.
  Evidence: `/var/lib/hal-netplay/direct8-check/socket-{before,after}.log`.
- `uv run pytest -q tests/test_netplay_gce.py` passed **20** tests.
- `bash -n deploy/netplay/gce-startup.sh` and `git diff --check` passed.

## Release and final checks

Image source is `6c79d130582f52495f226d261a48877049981175`.
The complete release image was built and pushed to the existing Artifact
Registry repository. Its manifest digest is
`sha256:b00e4c9911d8183891e777744c47c55316759ba75da8ca0ec62f121430212c4a`.
The startup script adds the socket fix from `6860af82`. This host script
change does not require another image build.

- `docker build --file deploy/netplay/Dockerfile --build-arg HAL_GIT_SHA=<full-sha> --tag <registry-image> .`
  and `docker push <registry-image>` succeeded. An isolated image check
  verified installed source hashes, protocol 2, the approved timing profile,
  and the absence of bundled ISO and emulator fixtures.
- `gcloud compute instances add-metadata hal-netplay-g4 --project=centering-star-502613-k3 --zone=us-west1-a --metadata=hal-netplay-image=<image>,hal-netplay-git-sha=<sha>,hal-netplay-slots=8 --metadata-from-file=startup-script=deploy/netplay/gce-startup.sh`
  succeeded. The second metadata update changed only the startup script.
- Remote `systemctl restart google-startup-scripts.service` completed with
  exit zero on both attempts. The first runner then failed on Xvfb :101.
  The corrected launch created X90 and X101 through X107, with zero runner
  restarts during preparation.
- `uv run ruff format --check .`, `uv run ruff check .`, the required
  `uv run ty check ...`, and `git diff --check` passed after the socket fix.
- A repeated `uv run pytest -q -rs -m 'not integration'` failed with
  `ValueError('I/O operation on closed file.')` and `lost sys.stderr`.
  Read-only sandbox calls also failed with mount quota errors at that time.
  The precise cause of the pytest capture failure was not established.
  The earlier complete run had passed.
- `TMPDIR="$PWD/runs/tmp8" uv run pytest -q -x -rs -m 'not integration' --basetemp="$PWD/runs/tmp8/handoff"`
  passed **1,627** tests, with eight skips, 21 deselections, and 27 warnings
  in 332.79 seconds. The separate 20-test GCE run includes the new socket
  regression. Skip reasons were unchanged; no required fixture was missing.
- An initial `vkcube --c 120 --width 640 --height 480` check failed with
  reduced driver capabilities. Repeating it with production capabilities
  `compute,graphics,utility,video,display` passed on NVIDIA under Xvfb.
- Automatic approval review rejected a proposed private account-file copy.
  Instead, the already-present VM account was updated only in its public
  connect code. Its exact SHA-256 matched the published R2 object. No new
  credential transfer was needed.
- Automatic approval review initially rejected the live Vulkan test because
  the original instructions required owner approval for live G4 verification.
  The owner then explicitly approved one 60-second test game.
- The test container downloaded and verified its fixtures. Its first path
  diagnostic incorrectly called `is_file()` on a string. Wrapping paths in
  `Path` fixed the diagnostic and confirmed HAL#9000 and both fixtures.
- An admin status diagnostic incorrectly used `AdminClient` as a context
  manager. Repeating it with the explicit `close()` lifecycle succeeded.
- Remote `journalctl`, `systemctl show`, `docker ps/top/logs`,
  `nvidia-smi`, and selected JSON reads tracked startup. A status-file read
  before preparation failed because the file did not yet exist. An old budget
  file still named the former image and was not treated as new evidence.
- Local `df -h`, `free -h`, and process inspection checked the test host.
  A broad process filter matched old unrelated wrappers; they were left alone.
- Commits `9817205a`, `6c79d130`, and `6860af82` passed their pre-commit
  hooks. Separate frontend wording and spec changes remain unstaged.

## Final deployment result

The production image qualified all eight inference streams while OBS was
running. Its 200 measurements gave 16.617 ms p50, 20.807 ms p95,
**21.148 ms p99**, and 21.434 ms maximum. Percentiles use linear interpolation.
The allowance is 33.333 ms. The exact report is
`runs/netplay/direct8/production-budget.json`; it records the image SHA,
policy, checkpoint, environment, timing, and generated sampling seed.
This is an additional qualification run, not the fixed-seed control comparison.

The owner-approved test paired G4 **HAL#9000** with local **CRYO#610**.
Both processes exited zero after the observation window. G4 advanced 3,597
frames in 60.009 seconds: **59.941 FPS**. Its five-second samples ranged from
59.865 to 59.967 FPS. Both sides used neutral input; this test checked the
renamed account and renderer, not model gameplay or eight-game throughput.

The running Dolphin mapped `libvulkan.so`, the NVIDIA 580.173.02 libraries,
and NVIDIA device nodes. `nvidia-smi pmon` showed Dolphin on GPU 0.
No llvmpipe or swrast library appeared in the recorded renderer mappings.
The test used the release image, Vulkan, an isolated Xvfb display, and delay 2.

Both replays are retained locally:

- `runs/netplay/direct8/render-check/g4/live-1/replays/Game_20261001T193637.slp`
- `runs/netplay/direct8/render-check/local-live-1/replays/Game_20261001T123638.slp`

The G4 copy also remains under `/var/lib/hal-netplay/direct8-check/live-1/`.
These diagnostic replays were not uploaded to R2. Production queue jobs use
the normal replay uploader.

OBS reported 1920×1080 at 60 FPS, an active stream, no reconnect, zero output
skips, and zero congestion. It recorded four rendering skips out of 16,921
frames during startup and qualification. No game occupied the production
stream slot during this check; the separate Vulkan test was not broadcast.

Final commands:

- `scp ... probe.py ...:/var/lib/hal-netplay/direct8-check/probe.py`
  staged only public test code. `scp ... runner-status.budget.json ...`
  saved the production measurements.
- Remote `docker run -d --name hal-netplay-render-check ...` and
  `docker exec ... python -c '... ensure(ISO); ensure(NETPLAY_EMULATOR)'`
  prepared the approved test. Its private account was a read-only mount of
  the existing VM cache.
- On each machine, `timeout --signal=TERM --kill-after=5s 210s xvfb-run -a ... probe.py ...`
  ran the synchronized 60-second game. Both commands exited zero.
  The G4 used port 51460 and Vulkan; the local peer used port 52460 and OGL.
- Remote `awk ... /proc/<dolphin-pid>/maps` and `nvidia-smi pmon -c 1`
  recorded the active driver and GPU process.
- Remote OBS `GetStats`, `GetStreamStatus`, and `GetVideoSettings`
  requests succeeded through its authenticated local socket. No key or
  control password was printed. The result is `obs-ready.json`.
- Remote `tar -czf - ...` through SSH and local `tar --no-same-owner -xzf ...`
  saved the game and renderer evidence. `docker rm -f hal-netplay-render-check`
  removed only the completed diagnostic container.
- `AdminClient.status()` verified eight real healthy slots and delay 2.
  `AdminClient.set_paused(False)` reopened admissions.
  `curl --fail --silent --show-error https://20xx.xyz/v1/capacity`
  confirmed public capacity 8, healthy slots 8, state ready, and zero recoveries.
- The final source and documentation diff was checked before commit.
  No frontend deploy, Git push, VM replacement, or new billable resource
  was needed for this final verification.

The Cloudflare billing-plan query from the previous deployment was denied
with HTTP 403. Its plan remains unverified. See `direct16.md` for the request
quota estimate; the SQL read-cost fix alone does not raise request quotas.
