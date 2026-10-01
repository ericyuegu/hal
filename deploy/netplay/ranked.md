# Ranked play on G4

**Current status:** Ranked remains offline. The former VM and its boot disk
were deleted on October 1, 2026 UTC. All 106 saved replay files were verified
in R2; see the [teardown report](g4-teardown.md). A replacement G4 now serves
[direct play](direct16.md) and its Twitch stream. The ranked run directories
below describe the former host and its preserved backup.

Run the maintained entry point from the checkout:

```sh
uv run python scripts/play_ranked.py \
  --bundle /path/to/policy.halpolicy \
  --account /path/to/user.json \
  --output /var/lib/hal-netplay/ranked \
  --twitch-key /run/hal-netplay/ranked-twitch-key \
  --source-revision "$(git rev-parse HEAD)"
```

The script requires exactly one validated RTX PRO 6000 Blackwell GPU, Slippi
Online 3.6.4, OBS 30.2.3, Tesseract, PulseAudio, and an NVIDIA Xorg display
(default `:90`). The Dockerfile includes the script and OCR dependency.
Use Docker `--init`, `--gpus all`, `--ipc=host`, and the existing Xauthority,
fixture, artifact-cache, and output mounts. This VM has one physical GPU.
Run Python from `/tmp` in the current patched image so the old source tree
at `/opt/hal` cannot shadow installed modules.

The owner confirmed permission for this ranked account. The account has a
ranked subscription, verified email, and accepted rules. Official Slippi
Launcher 2.15.1 was installed on the VM to refresh the account state. The
launcher uses a private display; gameplay uses hardware OpenGL on NVIDIA
Xorg. The stream contains only the Dolphin render window.

## Policy and timing

- Policy bundle: `0ff1daf80caa36a94a713c4ccba9223db8d7ba7c1379b5865bbc40b8a8c2f3ec`.
- Fox; player conditioning `IBDW#0`; advantage 120; temperature 1.
- Seed 120647. Network delay 2; inference allowance 1; fixed prefix 3.
- Replan interval 4; prediction horizon 8 (five new actions).
- Compile and capture CUDA graphs once. Skip the qualification benchmark on
  this previously validated GPU, as requested by the owner.
- The first signal finishes the current game. A second signal stops at once.
  Stop at a set boundary when an operator can do so.

This is an operational menu validation, not a controlled strength evaluation.
The opponents, stages, network latency, and driver differ across games. Do
not treat the results below as a performance treatment comparison.

## Menu path

`hal/sim/ranked.py` chooses Ranked and Fox. It uses the displayed prompt to
lock Fox and search because libmelee reports `coin_down=False` even when the
online character is already locked.

Between games, read the Dolphin screenshot at 960 by 720. OCR reads only
the prompt, not the player names or connect codes. Select the highlighted
legal stage. If two bans are required, move right before selecting the second
stage. Each press lasts 65 ms and has at least 250 ms between actions.

After a ban or stage choice, press A on the highlighted **OK** button.
Then press A again on **OK** to confirm Fox. These are distinct screens.
The gold button border pulses, so both dim and bright recordings are regression
fixtures. Never infer OK just because the stage cursor disappeared. Each
action requires a new screenshot captured after the previous action settled.
Wait for the opponent when that prompt is shown.

The menu helper stops at the in-game state. The inference scheduler starts
during the countdown, before frame zero. It sends controller inputs through
the same netplay path used by direct play. Menu actions never enter that path.

A countdown quit releases the inference generation and returns to menu
navigation without restarting Dolphin. A lost emulator connection records
the failure and restarts Dolphin; three consecutive failures stop the script.
Normal replay end and NO_CONTEST are both recorded. An invalid artifact or
replay still fails explicitly.

OBS runs on its own thread. A screenshot or stream failure restarts OBS with
backoff and does not stop the model. The WebSocket accepts screenshots up to
8 MiB; a gameplay screenshot exceeded the previous 1 MiB default and caused
the first supervised stream failure. OCR is disabled while the game is live.
The overlay contains the public model label and advantage; it never contains
a connect code. Emulator menus remain visible.

## Records

Each maintained run writes a new UTC directory under `--output`:

- `manifest.json`: Git revision, actual source hashes and copies, artifact
  hashes, seed, resolved policy settings, timing, GPU, driver, and Torch.
- `replays/*.slp` and `game-NNNN.json`: replay hash, end method, ports, stage,
  frame count, FPS, frame latency, inference latency, and transport corrections.
- `status.json`, `screen.json`, `actions.jsonl`, and `obs-stats.json`.
- `latest.png` and at most 300 saved screenshots, one every two seconds.
- Separate records for countdown quits, connection losses, OCR errors, and
  stream errors. A fatal failure leaves status `failed`.

The supervised prototype records are on G4 at
`/var/lib/hal-netplay/ranked-cody120/`. They include source copies under
`control/` and `control-v2/`, timestamped manifests, `result-NNN.json`,
screenshots, actions, and replays. The first manifest is preserved.
Local inspection and validation logs are under
`runs/netplay/ranked-cody120/`. Completed ranked replays now upload to R2;
see the upload section below.

## Observed run — 2026-09-30

The first completed set against Dr. Mario ended **2–1 for HAL**. Replay end
placements confirm all three outcomes. The first game used the ban timer;
games two and three needed operator OK presses while the helper was being
fixed.

| Replay UTC | Result | Average FPS | Frame p95 | Inference p95 |
| --- | --- | ---: | ---: | ---: |
| 19:20:44 | Win | 58.99 | 18.12 ms | 6.46 ms |
| 19:24:06 | Loss | 59.09 | 18.02 ms | 6.43 ms |
| 19:27:38 | Win | 59.05 | 18.00 ms | 6.93 ms |

After the helper fix, HAL won the next set against Falco **2–0**. Replay end
placements confirm both wins. The helper entered game one, selected a ban,
confirmed the ban, confirmed Fox, played game two, and queued for the next
set without operator input. Game one averaged 59.10 FPS with frame p95
18.01 ms; game two averaged 58.44 FPS with frame p95 18.11 ms. OBS reported
60 FPS, zero render skips, zero encoder skips, and zero network drops.

Live LRAS and disconnect recovery have not yet been observed. Regression tests exercise countdown quits, replay
NO_CONTEST, connection loss, and independent OBS restart. More live screen
coverage is needed before claiming that every ranked dialog is handled.

## Validation log

See `runs/netplay/ranked-cody120/` for the local command output. Required
handoff checks are recorded below.

Failures found and fixed during development:

- OBS screenshots exceeded the WebSocket limit. Increase the limit and keep
  OBS recovery independent from game control.
- A dim gold OK outline failed the first detector. Add recorded dim and
  character-confirmation fixtures and check both sides and the bottom edge.
- The online character lock flag was false. Read the displayed prompt.
- A new dead-process check exposed a mock with no running-process state.
  Correct that fixture; retain the stalled-read regression.
- A new test used nonexistent `EndMethod.QUIT`. The replay API uses
  `NO_CONTEST`. Another fixture used a non-SHA checkpoint string. Correct both.

Earlier environment failures: the G4 kernel lacked its matching NVIDIA
module; install the matching 580 server module. Docker without `--init`
triggered the parent-death guard; use `--init`. The local shared temporary
directory reached its quota; use a private test temporary directory. The
required suites then passed before the current changes.

Skipped deployment actions: no registry push, R2 upload, Git push, new VM,
or Cloudflare deployment. The existing G4 and live Twitch stream are used.

Commands and results for this milestone:

| Command | Result |
| --- | --- |
| `uv run ruff format --check .` | Pass; 275 files |
| `uv run ruff check .` | Pass |
| `uv run ty check --python-version 3.14 --error-on-warning hal experiments/059_muon_action_sequence.py scripts` | Pass; zero diagnostics |
| `TMPDIR="$PWD/runs/netplay/ranked-cody120/test-tmp" uv run pytest -q -m "not integration" --basetemp="$PWD/runs/netplay/ranked-cody120/test-tmp/ranked-full"` | 1537 passed, 8 skipped, 21 deselected; 145.21 s |
| `HAL_REQUIRE_INTEGRATION=1 TMPDIR="$PWD/runs/netplay/ranked-cody120/test-tmp" uv run pytest -q tests/test_roundtrip.py tests/test_session_cleanup.py -m integration --basetemp="$PWD/runs/netplay/ranked-cody120/test-tmp/ranked-integration"` | 7 passed, 6 deselected; 54.98 s |
| `uv run pytest -q tests/test_ranked_runner.py tests/test_ranked_menu.py tests/test_netplay_session.py tests/test_netplay_obs.py` with the private temporary directory | 85 passed; initial fixture failures are listed above |
| `uv run pytest -q tests/test_netplay_session.py tests/test_netplay_driver.py` with the private temporary directory | 57 passed after the process mock correction |
| `uv run pytest -q tests/test_ranked_menu.py tests/test_netplay_obs.py` with the private temporary directory | 31 passed after the recorded screen regressions |
| `uv run python scripts/play_ranked.py --help` | Pass |
| `git diff --check` | Pass |

The eight pytest skips are two opt-in hardware qualification tests and six
tests that need the optional local v7 training subset. All required emulator
fixtures were present. Warnings include the existing WebSocket connection
API deprecation and Python's multithreaded fork warning.

Operational commands used SSH with the existing GCE key and pinned host key.
`docker logs`, JSON status/action reads, `scp` of Dolphin screenshots, and
`view_image` verified menu transitions. Remote `python3` wrote atomic
controller commands; the first write lacked root permission and failed, then
`sudo python3` succeeded. The manual command acknowledgment and subsequent
game screenshot confirmed OK was accepted. No controller command was sent
while status was playing.

`scp` staged the helper, then `docker kill --signal SIGTERM`,
`docker wait`, and `docker start` loaded it after the deciding game.
`docker exec ... python` read replay end placements to verify the two set
results. `tar` and `scp` staged the maintained source; `docker cp` preserved
107 MiB of compiled kernels for its next start. No image was published.

`systemctl stop hal-slippi-launcher.service hal-slippi-launcher-display.service`
closed the private launcher and debug port after account setup. Ctrl-C closed
its local SSH tunnel (SSH returned 255 on interruption). The account profile
is retained on the VM. Searches and source reads used `rg`, `sed`, `cat`,
and `git diff`; two optional path searches failed because the paths did not
exist. Formatting used `uv run ruff format`. Image inspection used Pillow
and NumPy. `uv lock` succeeded with an existing yanked-zstd warning.

Additional checks passed: `npm test` ran 110 tests in 11 files;
`npm run typecheck` passed. The test worker logged a WebSocketPipe teardown
message but reported no failures.
`HAL_REQUIRE_INTEGRATION=1 uv run pytest -q tests/test_netplay_queue_integration.py -m integration`
with the private temporary directory passed all 3 tests in 42.76 s.

Source milestone: `d4822adb` (`Automate ranked play and menu confirmations`).
Commit hooks passed Ruff format, Ruff check, and ty.

## Maintained deployment

Container `hal-ranked-player-v3` ran `scripts/play_ranked.py` from source
commit `d4822adb`. Its run was:

`/var/lib/hal-netplay/ranked/20260930T193913.999585Z/`

The Docker argument list is saved on G4 as
`/var/lib/hal-netplay/ranked-cody120/player-command-v3.json`. Source mounts
come from `maintained-v3/` and the existing `hotfix/obs-v1/` directory.
This launch still uses the local OBS image. It is not a published image or a
new boot service. The public runner remains disabled by the ranked marker.

The handoff stopped the prototype at character select, after the set.
It started the maintained player with `docker run`. Model preparation took
about 49 seconds, and OBS resumed at about 19:40:04 UTC. The restart caused
a brief Twitch outage. OBS then reported an active output at 60 FPS, no
network or encoder drops, and three startup render skips. A screenshot
confirmed the Dolphin menu, and the helper started the next ranked search.

The maintained run uses the requested fixed timing and skips qualification.
The actual source files and hashes are copied into its run directory.


The maintained player completed its first game and entered game two without
operator input. Its first record is `game-0001.json`: 58.72 average FPS,
18.02 ms frame p95, 6.64 ms inference p95, one transport correction, and a
normal GAME end. The replay hash is
`3ecd2c80ba6e6bca0b9d6473592ba1b64f0e044a98273705c8c3d403f939e73c`.
No stream or OCR error record was present at that check.

The prototype also completed a second full set without operator input and
re-queued. HAL won both games. Completed prototype replays and manifests were
copied with SSH/tar to `runs/netplay/ranked-cody120/recordings/`.
A local `uv run python` check verified all eight replay hashes and read the
match IDs, game numbers, and placements with peppi.

## Ranked replay uploads — 2026-09-30

Completed games upload to bucket `hal` under:

```text
netplay/v1/replays/YYYY/MM/DD/HAL#647/ranked-<set-and-tiebreaker-hash>/game-NN.slp
netplay/v1/replays/YYYY/MM/DD/HAL#647/ranked-<set-and-tiebreaker-hash>/game-NN.json
```

List this run's objects with:

```sh
rclone lsf 'r2:hal/netplay/v1/replays/2026/09/30/HAL#647/' --recursive
```

The first manual batch uploaded **21 replays, 47,955,696 replay bytes**, plus
small JSON metadata records. Each upload checked size, SHA-256 metadata,
and ETag for both objects. The local replays remain on disk.

There is **no automatic replay expiration**. A 30-day rule was briefly
installed during the manual batch, then removed when the owner corrected
the retention choice. The existing seven-day abort rule applies only to
unfinished multipart uploads. No completed replay is subject to that rule.

`hal/eval/ranked_replays.py` reuses the public runner's verified uploader.
The object key is stable across retries. It includes the Slippi set identity
and tiebreaker, so rematches and tiebreakers cannot replace another game.
The metadata records the local player's result, stage, policy hash, source
revision, and game times. The replay itself retains the Slippi match ID.

`play_ranked.py` starts an upload worker and wakes it after atomically
writing each completed game record. The worker also scans earlier run
directories every 15 seconds. It selects committed game records, never a
replay that Dolphin is still writing. It rejects an unknown schema, a changed
file hash, a mismatched stage or end, and a receipt for different input.

The network work runs outside controller input. Each R2 connection and read
has a five-second timeout; the upload worker owns retries. A failed upload
leaves the local files and writes `upload-error.json`. Successful uploads
write `uploads/game-NNNN.json` receipts. A restart reads those receipts and
skips completed uploads. No video, screenshots, or detailed frame arrays
are uploaded.

To retry completed runs manually:

```sh
uv run python scripts/upload_ranked_replays.py --root /var/lib/hal-netplay/ranked
```

Add `--watch` to run the same upload worker beside an existing player. The
first deployment used that mode at a set boundary before game 19. It kept
the model and Twitch stream online. Uploader source and deployment times
are recorded under `/var/lib/hal-netplay/ranked-cody120/upload-v4/`.
The resumed player starts the upload worker itself.

Validation and command results:

- The manual dry run parsed completed replays, checked their recorded hashes,
  and printed the planned records. It passed.
- The first remote log redirection lacked write permission. It failed before
  starting the uploader. Moving the redirection into `sudo sh -c` fixed it.
- `docker exec ... upload-backlog.py --upload` uploaded the 21 verified pairs.
- A remote R2 client removed only `hal-netplay-v1-replays-30d` and verified
  that the unrelated multipart rule remained. The one-time script was then
  changed so it cannot reinstall replay expiration.
- `uv run pytest -q tests/test_ranked_replays.py tests/test_ranked_runner.py tests/test_netplay_replays.py`
  passed 28 tests with the private temporary directory.
- `uv run ruff format --check .` passed for 278 files.
- `uv run ruff check .` and
  `uv run ty check --python-version 3.14 --error-on-warning hal experiments/059_muon_action_sequence.py scripts`
  passed.
- `git diff --check` passed. Targeted source reads used `rg`, `cat`, and
  `sed`; one shell glob for absent lifecycle files failed and was replaced
  by a repository search.
- `tar` and `scp` staged the uploader source on G4. No registry image,
  Cloudflare Worker, GPU allocation, or Git remote was changed.

- `TMPDIR="$PWD/runs/netplay/ranked-cody120/test-tmp" uv run pytest -q -m "not integration" --basetemp="$PWD/runs/netplay/ranked-cody120/test-tmp/ranked-upload-full"`
  passed 1550 tests, with 8 optional skips and 21 deselections, in 145.65 s.
  Skips: two opt-in GPU qualification tests and six tests for the optional
  local v7 subset.
- `HAL_REQUIRE_INTEGRATION=1 TMPDIR="$PWD/runs/netplay/ranked-cody120/test-tmp" uv run pytest -q tests/test_roundtrip.py tests/test_session_cleanup.py -m integration --basetemp="$PWD/runs/netplay/ranked-cody120/test-tmp/ranked-upload-integration"`
  passed 7 tests, with 6 deselections, in 55.54 s. No required fixture was missing.
- `uv run python scripts/upload_ranked_replays.py --help` passed.
- Worker/npm checks were not repeated for this upload change; Worker source
  did not change. They passed in the preceding ranked milestone.

## Resumed deployment — 2026-09-30

At the owner's request, two SIGINT signals stopped Ranked play and OBS at
20:29:29 UTC. The immediate stop recorded `KeyboardInterrupt` and exit code
130. The owner then requested an immediate restart.

Container `hal-ranked-player-v4` started at 20:31:43 UTC. It runs source
`86c5d627` with the integrated replay uploader. Its Docker arguments are in
`/var/lib/hal-netplay/ranked-cody120/player-command-v4.json`. Its run is:

`/var/lib/hal-netplay/ranked/20260930T203153.297761Z/`

The manifest confirms Cody Fox (`IBDW#0`), advantage 120, network delay 2,
inference allowance 1, fixed prefix 3, replan interval 4, and horizon 8.
The same single RTX PRO 6000 Blackwell runs the model.

At 20:32:15 UTC, OBS reported active output at 60 FPS, no reconnect, no
network or encoder drops, and two startup render skips. A fresh screenshot
confirmed the Dolphin window and Ranked opponent search. Twitch is
[hal_20xx](https://www.twitch.tv/hal_20xx). The public direct-play runner
remains disabled by the Ranked marker.

All 26 completed games from before the stop have upload receipts: eight
prototype games and 18 maintained games. The interrupted game has no
completed game record and is not included. The new process scans completed
runs and uploads each new completed game. Replay expiration remains disabled.

Command results for this deployment:

- SSH `docker kill --signal SIGINT hal-ranked-player-v3` ran twice and
  stopped the player. `docker inspect` confirmed exit code 130.
- The pending-launcher check completed. The upload watcher had already
  started at the set boundary, before game 19.
- SSH `sudo python3 -c ...` loaded and executed the saved v4 Docker
  argument list with `subprocess.run(..., check=True)`. `docker run`
  succeeded and the deployment record was updated.
- `docker inspect` and `docker top` confirmed the new player, Dolphin,
  PulseAudio, and OBS were running.
- Remote Python checks read status, OBS statistics, the manifest, upload
  receipts, and the direct-runner guard. They passed. No upload or stream
  error file was present in the new run at the check.
- SSH `sudo cat .../latest.png` saved a local screenshot. Visual inspection
  confirmed the Ranked search and the Dolphin image.
- Local `cat`, `sed`, `git status`, `git log`, and `git diff --check`
  checks passed.
- The first documentation edit used unavailable `python` and failed before
  writing. Repeating it with `python3` succeeded.
- This handoff changes documentation only. The code validation results
  above apply to the deployed source; no test suite was repeated.


## Value meter — 2026-09-30

Every action prediction now includes a finite ego state value. The value
head uses the serving weights in BF16 on CUDA, as requested by the owner.
The response scalar and EMA use FP32/Python floats. The head estimates
future reward in training reward units. Positive values favor HAL.

The Ranked stream shows a bar and signed value in the right margin.
The old Cody Fox title is removed. The bar saturates at ±120; the number
shows the full estimate. The EMA half-life is six game frames:
alpha = 1 - 2 ** (-elapsed_frames / 6). At replan interval four, inference
publishes about 15 values per second. Duplicate or old samples cannot move
the meter. A new game or generation resets it. Menus, countdown, and
estimates older than one second hide it.

### Independent overlay process

The model callback updates only memory. The stream supervisor writes an
atomic, schema-1 snapshot to value.json. A separate process runs:

    python -m hal.scripts.ranked_overlay --run-dir <run-directory>

It reads that snapshot and updates four OBS sources at up to 30 Hz. The
Ranked supervisor starts this process and restarts it after failure with
backoff. It does not restart Ranked, Dolphin, or OBS when the overlay exits.

To reload presentation code, update the mounted overlay file **in place**
and send SIGTERM to the PID in <run-directory>/overlay.pid. The supervisor
then launches a fresh Python process. An atomic replacement of the host
file does not update an existing Docker file bind mount.

The file obs-control.json holds only the local OBS control password. It has
mode 0600 and is removed when OBS closes. It is not uploaded. The overlay
receives no Slippi account data or connect code. Replay upload continues to
select only completed replay records and their small metadata files.

Screenshot capture and OCR use a separate thread and OBS connection.
They cannot block value publication or share response IDs with the overlay.
OBS still owns Dolphin window capture and audio; the Ranked process owns
inference and controller input.

### Matched inference measurement

Control and candidate use the same bundle, GPU, software, seed 120647,
IBDW#0 identity, advantage 120, and timing 2/1/3/4/8. Each run measures 300
request-to-response samples after 20 warm-up calls, with synthetic fixed
observations. These are inference measurements, not emulator FPS.

| GPU | Version | p50 ms | p95 ms | p99 ms |
| --- | --- | ---: | ---: | ---: |
| RTX 3060 | Control | 9.802 | 11.147 | 11.727 |
| RTX 3060 | BF16 value output | 9.048 | 9.519 | 10.661 |
| G4 RTX PRO 6000 Blackwell | Control | 5.051 | 5.085 | 5.095 |
| G4 RTX PRO 6000 Blackwell | BF16 value output | 4.678 | 4.743 | 5.049 |

All 320 sampled action sequences matched the control exactly on each GPU.
There were no compilation starts or new CUDA graph captures during sampling.
Both candidates met the one-frame inference allowance. A single matched
pair does not establish a speed improvement.

On the 3060 synthetic state, the provisional FP32 head produced 15.177 and
the selected BF16 head produced 15.688, a difference of 0.510 reward units.
This is a precision spot check on one state, not a calibration result.
The BF16 G4 value was 16.125; exact values across different GPU architectures
are not required. The production bundle hash remains
0ff1daf80caa36a94a713c4ccba9223db8d7ba7c1379b5865bbc40b8a8c2f3ec.

Evidence is local under runs/netplay/value-meter/, and on G4 under
/var/lib/hal-netplay/ranked-cody120/value-meter/. It includes the benchmark
script, control/candidate commands, source hashes, timing samples, and action
hashes. Benchmark and source artifacts stay on disk. Completed Ranked games
continue to upload their replay and small result metadata.

### Checks and failures

- cat, sed, rg, and Git status/diff reads inspected the affected code and
  callers. An initial cache path was absent; the production bundle was found
  at /tmp/hal-o59-production.halpolicy.
- Sandboxed nvidia-smi could not access the driver. The authorized host
  command succeeded and identified the RTX 3060.
- uv run python runs/netplay/value-meter/benchmark.py ran the control,
  provisional FP32 candidate, and final BF16 candidate. The first candidate
  failed because FP32 head input met BF16 weights. The owner chose BF16;
  the head now uses its serving weight dtype. The final runs passed.
- The first G4 benchmark finished inference but failed to read Git metadata
  inside the image. Supplying the recorded source revision fixed it.
- An intermediate ty check found four indentation errors while extracting
  the OBS connection class. The correction passed ty.
- The first focused run had ten failures: three direct lifecycle test
  constructors needed the optional callback default, and seven whole-plan
  equality checks needed a separate floating-point tolerance for the value.
  Action and identity comparisons remain exact.
- A later focused run had one failure because its new callback assertion
  lacked callback registration. The test was corrected.
- Final focused API/stream checks passed 81 tests. Additional lifecycle
  tests cover overlay spawn failure and screenshot setup failure.
- uv run ruff format --check ., uv run ruff check ., and the full
  AGENTS.md ty check passed. git diff --check passed.
- HAL_REQUIRE_INTEGRATION=1 uv run pytest -q tests/test_roundtrip.py
  tests/test_session_cleanup.py -m integration, with the private temporary
  directory, passed 7 tests, with 6 deselections, in 57.14 seconds.
  No required fixture was missing.
- SSH docker kill --signal SIGINT hal-ranked-player-v4 stopped Ranked
  cleanly with exit code zero. All 24 completed games in that run have upload
  receipts. The G4 VM stayed running.
- SSH mkdir, tee, and tar staged isolated candidate source. Python executed
  the saved Docker benchmark argument lists. No registry push, Cloudflare
  deployment, VM allocation, or Git push occurred.
- One documentation tool request had a JavaScript syntax error before any
  command ran. The corrected request wrote the notes.

- The final full command was:

      TMPDIR="$PWD/runs/netplay/ranked-cody120/test-tmp" uv run pytest -q -m "not integration" --basetemp="$PWD/runs/netplay/ranked-cody120/test-tmp/value-full-final"

  It passed 1582 tests, with 8 skips and 21 deselections, in 150.02 seconds.
  Skips were two opt-in hardware qualification tests and six optional v7
  subset tests. The preceding full run passed 1580 tests; the repeat covered
  two added lifecycle tests and the final cleanup changes.
- The final lifecycle test command passed all 26 tests. The independent
  OBS capture-connection test passed. Overlay CLI --help passed.
- Worker/npm checks were not repeated; this change does not modify Worker
  code or its network protocol. A concurrent roster commit (86b0e534)
  included the staged meter implementation in this shared worktree. Commit
  1d9e1e6a added the final validation record. History was left intact.


### Live verification

The tested source runs in hal-ranked-player-v5 from revision 1d9e1e6a.
Its run directory is:

    /var/lib/hal-netplay/ranked/20260930T220118.832519Z/

The first launch failed before Dolphin started. The temporary staging
directory retained an unrelated domain module from concurrent roster work;
its dependency was absent from the image. The corrected launch reads an
explicit source manifest and verifies each file hash. It does not discover
deployment files by scanning the staging directory.

The first game completed and uploaded its replay. The menu helper started
game two automatically. No stream, OCR, or upload error record was present.

| Live observation | Previous run, first game | Meter run, first game |
| --- | ---: | ---: |
| Average emulator FPS | 58.626 | 58.914 |
| Frame p95 ms | 18.101 | 18.053 |
| Inference p95 ms | 6.820 | 6.658 |

These games had different opponents and stages. They show operational
health; the matched synthetic measurements above provide the inference
comparison. OBS output stayed at 60 FPS with zero encoder/network drops
and two startup render skips.

An authenticated GetSourceScreenshot request captured the full HAL program
scene at 1920 by 1080. Visual inspection confirmed Dolphin gameplay, the
right-hand bar and signed value, and removal of the old title. The meter
sources contained no connect code.

During game one, SIGTERM restarted only the overlay process. Its PID changed
from 576 to 1007. Ranked PID 7, OBS PID 531, and Dolphin PID 586 kept the same
process start times. Inference sequence advanced from 777 to 798. OBS stream
duration increased, remained active without reconnect, and added zero dropped
frames. The EMA continued across the overlay restart.

Evidence in the run directory and local runs/netplay/value-meter/:

- meter-before.png and meter-after.png show the actual OBS program output.
- overlay-restart-check.json records PIDs, start times, inference sequence,
  stream counters, and meter samples before and after restart.
- game-0001.json records 6555 frames, normal GAME end, no transport
  corrections, 111.248 seconds, and the performance figures above.
- uploads/game-0001.json confirms the verified replay upload.

Deployment and verification commands:

- SSH tee and tar transferred the source manifest and eleven selected Python
  files. Remote Python verified hashes and ran player-command-v5.json.
- docker logs identified the initial import failure. docker inspect confirmed
  that this candidate had exited. docker rm removed only that failed
  container; the corrected docker run succeeded.
- docker exec -w /tmp hal-ranked-player-v5 python -c ... queried OBS,
  captured the composed scene, sent SIGTERM to the verified overlay PID,
  and checked process identities and stream continuity. All assertions passed.
- SSH cat copied the two PNGs, restart evidence, and benchmark results
  locally. Local image inspection confirmed the display.
- Remote Python read status, game metrics, OBS statistics, and upload
  receipts. The first completed game had a receipt and game two was playing.
- Git add, diff --check, and commit recorded implementation and validation.
  The final documentation commit ran no code tests; its Python hooks skipped
  because no Python files changed.

The G4 VM, Ranked player, and Twitch stream remain running. No VM was stopped
or recreated. The previous v4 container remains available for rollback.
To roll back, drain v5 with one SIGINT, wait for its exit, then start v4.
Replay files and receipts are on the shared persistent volume.


## Stream and memory monitor — September 30, 2026

`hal.scripts.ranked_monitor` runs in a separate container. It reads existing
`obs-stats.json` and `value.json` once per second. It records:

- Five-second game frame progress, estimated from inference source frames.
- OBS FPS, rendering time, bitrate, reconnects, and congestion.
- New render, encoder, and network drops since the previous snapshot.
- Per-process RSS, anonymous memory, CPU time, threads, PID, and start time.
- Explicit warnings for stale or invalid telemetry and slow frame progress.

Game frame progress is an estimate, not a direct measurement of displayed
frames. A paused telemetry writer can also cause a stale warning. Menus and
countdowns do not count as slow games. New generations reset the rolling
window. OBS counter resets do not produce negative drops or bitrate.
Thresholds are 55 FPS for five-second game progress, 58 FPS for OBS, two
seconds for stale predictions, and three seconds for stale OBS data.

The monitor has no GPU, network, credentials, or writable player files. It
runs as UID 65534 in the player's PID namespace and cannot signal the
root-owned player. It does not poll OBS, capture screenshots, or change the
stream. Its two inputs remain owned by Ranked. Only its output directory is
writable. The launcher refuses to start against a stopped player.

After an authorized Ranked start, use the actual new run directory:

```sh
sudo /var/lib/hal-netplay/stream-monitor/deploy/netplay/run-ranked-monitor.sh \
  hal-ranked-player-v5 /var/lib/hal-netplay/ranked/ACTUAL_RUN_DIRECTORY
sudo docker logs --follow hal-ranked-player-v5-monitor
```

The launcher uses the player's exact local image. The staged source is under
`/var/lib/hal-netplay/stream-monitor/hal`. It makes no registry request.
Remove an exited monitor container before reusing the same monitor name.
The monitor exits when the player's PID namespace ends. It never restarts
Ranked. A monitor failure leaves the player running.

Read `ACTUAL_RUN_DIRECTORY/monitor/status.json` for the latest sample.
`history.jsonl`, `.1`, and `.2` retain at most 24 MiB in total. Docker alert
logs retain at most 2 MiB. These records stay on disk and are not uploaded
to R2. Alerts appear in Docker logs only when the warning set changes.
No external alert destination is configured.

OBS counter definitions follow the official
[obs-websocket protocol](https://github.com/obsproject/obs-websocket/blob/5.5.6/docs/generated/protocol.md#getstats).

### Slow final set

The run `20260930T220118.832519Z` completed 32 games. All 32 have replay
upload receipts. The first 29 games had mean FPS near 59 in each group:
58.946 for games 1–10, 59.025 for 11–20, and 58.852 for 21–29. Games 30–32
belonged to the next set and averaged 54.863, 55.438, and 49.257 FPS.
Inference p95 stayed at 6.71–6.87 ms during that set. Frame interval p95
stayed near 18.1 ms, so the aggregates do not locate the longer stalls.

Game 32 ended with NO_CONTEST and an LRAS initiator of P2, the opponent's
port. No HAL transport-disconnect record was written. The owner then sent
the stop signal. Docker recorded the signal and a clean exit at
23:29:11 UTC, with exit code zero and OOMKilled=false. No agent command
stopped or restarted Ranked or Twitch during this investigation.

OBS stayed at 60 FPS, with zero encoder/network drops and the same two
startup render skips. OBS RSS was about 542–544 MiB. A live sample showed
168 GiB of available RAM, no CPU/memory/I/O pressure, about 3.9 GiB of
container memory, and 2.4 GiB of GPU memory at 42 degrees Celsius. Kernel
logs showed no GPU fault or OOM event in the inspected interval. These
samples do not establish a long-term memory trend.

Saved screenshots show 26 ms ping in the preceding set and 60 ms in the
last set; both screenshots display Dolphin FPS 60. A transient connection
or opponent-side issue is plausible. Packet loss and short stalls were not
recorded, so the cause remains unproven. No performance fix or gameplay
setting change was made.

Evidence is in local `runs/netplay/stream-monitor/`: `incident.json`, four
screenshots, the full test log, and the source archive. G4 retains the original
run and `stream-monitor/check-result.json`. The new monitor was attached
after the authorized Ranked restart below. It was not running during the incident.

### Monitor checks and commands

- `cat`, `sed`, `rg`, Git status/log, and Python reads inspected the runtime,
  telemetry formats, callers, process lifecycle, and deployment configuration.
  Some exploratory reads named absent files; corrected reads found the owners.
- Read-only SSH `docker ps`, `inspect`, `top`, `stats`, `logs`, and `events`
  confirmed the runtime, resource samples, signal, and clean exit. `nvidia-smi`,
  `/proc/meminfo`, `/proc/pressure/*`, and kernel journal reads found no resource
  exhaustion. One read was interrupted by a user message and repeated safely.
- SSH Python reads and `cat`/tar transfers saved existing metrics and four
  screenshots. A network-disabled, GPU-free temporary container parsed replay
  headers with peppi. No game, account, or replay was changed.
- The first local file-writing command used unavailable `python` and failed
  before writing. The corrected command used `python3`.
- `uv run pytest -q tests/test_stream_monitor.py` passed all 16 tests.
- `uv run ruff format --check .`, `uv run ruff check .`, and
  `uv run ty check --python-version 3.14 --error-on-warning hal
  experiments/059_muon_action_sequence.py scripts` passed. Formatting checked
  287 files. `bash -n deploy/netplay/run-ranked-monitor.sh` and CLI `--help`
  passed.
- `TMPDIR="$PWD/runs/netplay/ranked-cody120/test-tmp" uv run pytest -q
  -m 'not integration' --basetemp="$PWD/runs/netplay/ranked-cody120/test-tmp/stream-monitor-full"`
  passed 1598 tests, with eight skips, 21 deselections, and 26 warnings in
  144.89 seconds. Skips remain two opt-in GPU tests and six optional v7 tests.
- SSH mkdir/tar staged the monitor. A 25-second synthetic Docker producer
  verified the actual launcher without GPU or network access. Eight monitor
  samples reported 59.97 FPS, correct memory counters, and no warnings. The
  actual Ranked container stayed exited. The fixture exited zero; its monitor
  exited 137 when the shared PID namespace ended. Both test containers were
  removed after exit. This expected monitor termination does not restart play.
- Emulator integration tests were not needed for this monitor-only change;
  it does not touch controllers, session stepping, replay extraction, or
  inference. Live gameplay verification followed the authorized start below.
  The later deployment checks also passed all ten required emulator and queue
  integration tests; see [the deployment report](public-relaunch.md).


### Ranked restart with live monitoring

The owner requested Ranked instead of the pending public direct-play restart.
`sudo docker start hal-ranked-player-v5` succeeded at 23:56:25 UTC on
September 30. The existing player retains Cody Fox, advantage 120, BF16,
network delay 2, inference allowance 1, replan interval 4, and horizon 8.
The new run is `/var/lib/hal-netplay/ranked/20260930T235630.598726Z`.

The menu helper selected the character, searched, struck two stages, and
confirmed without manual input. The first game entered IN_GAME at
23:57:08 UTC. The first verified live monitor window estimated 59.88 game
FPS. OBS reported 60 FPS with zero render, encoder, and network drops,
no reconnect, and no congestion. There were 41 monitor samples and no alerts
at that check. These short observations do not establish a memory trend.

The stream is live at <https://www.twitch.tv/hal_20xx>. The value overlay,
recording, and replay uploader remain part of the unchanged player. Both
player and monitor stay running. The public runner remains inactive and the
`ranked-active` exclusion marker remains present.

Restart commands and results:

- SSH `test`, `systemctl is-active`, and `docker inspect` confirmed the marker,
  stopped public runner, and stopped Ranked container before startup.
- `sudo docker start hal-ranked-player-v5` succeeded.
- The monitor launcher succeeded:

      sudo /var/lib/hal-netplay/stream-monitor/deploy/netplay/run-ranked-monitor.sh hal-ranked-player-v5 /var/lib/hal-netplay/ranked/20260930T235630.598726Z

- SSH `docker logs`, `docker inspect`, and Python reads of status, actions,
  OBS telemetry, and monitor samples confirmed both containers running,
  automatic menu progression, active Twitch output, and live inference.
- SSH Python output saved the existing telemetry locally as
  `runs/netplay/stream-monitor/ranked-restart.json`. No secret was read.
- Dolphin logged an Adwaita pixbuf warning during startup. It still entered
  the game and OBS output remained active.
- Git diff checks and the documentation commit recorded this restart.
  Code tests were not repeated for documentation-only edits. The completed
  test results and skips are recorded above and in the deployment report.
