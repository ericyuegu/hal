# Ranked play

Ranked is a separate single-player process. It does not use the public queue or
share the public runner's inference process. Do not run both against the same
account/display without a new concurrency check. Current host state is in
[status.md](status.md). This integration does not restart ranked or Twitch.

Use the maintained entry point from an exact committed checkout or its complete
runner image. No source bind mounts or patched-image import workaround is needed:

```sh
uv run python scripts/play_ranked.py \
  --bundle /path/to/policy.halpolicy \
  --account /path/to/user.json \
  --output /var/lib/hal-netplay/ranked \
  --twitch-key /run/hal-netplay/ranked-twitch-key \
  --source-revision "$(git rev-parse HEAD)"
```

The image contains the script, Slippi dependencies, OBS and Tesseract. It requires
one validated RTX PRO 6000 Blackwell, NVIDIA Xorg :90, its X90 socket and
Xauthority cookie, GPU access, shared memory, and persistent output/asset mounts.
Use the GCE startup script for host driver/display preparation. Mount only X90,
not the whole X11 socket directory. Keep the Twitch key in a private runtime file.
Run only after owner authorization; the account needs its ranked subscription,
verified email and accepted rules. Those account steps were completed for the
prior host, but the launcher install was removed with that host.

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
The ranked overlay shows the smoothed state value; it never contains a connect code. Emulator menus remain visible.

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

## Independent overlay and monitoring

Inference emits state values through `ValueMeter`; the player writes `value.json`.
The overlay process reads that file and updates OBS independently. Changing its
UI must not signal or restart the player. OBS and OCR failures restart their
stream work without stopping model inference. The stream shows Dolphin menus.

`deploy/netplay/run-ranked-monitor.sh PLAYER_CONTAINER RUN_DIRECTORY` starts an
isolated monitor of OBS counters, frame progress and process memory. It does not
have credentials, GPU access or permission to signal the player. Its source mount
supplies the monitor only, not a player patch.

The first player signal finishes the current game; a second stops immediately.
Stop the player container with a sufficient grace period when authorized.
Completed replays upload automatically. Retry a stopped run with:

```sh
uv run python scripts/upload_ranked_replays.py --root /path/to/run
```

Check that command's `--help` for required options before use. There is no bucket
expiry operation. Preserve local records and receipts until uploads are verified.

## Evidence

[Ranked history](ranked-history.md) retains measurements, failures, commands and
results. [Teardown](g4-teardown.md) records the 106 verified replay files and local
backup. These are historical results, not qualification of the integrated release.
