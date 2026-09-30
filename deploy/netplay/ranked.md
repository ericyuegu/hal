# Ranked play on G4

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
`runs/netplay/ranked-cody120/`. No ranked records have been uploaded to R2.

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
