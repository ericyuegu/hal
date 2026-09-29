# x_pilot matchup run

## Status

Prepared on 2026-09-29 from `96ea6218`. No x_pilot games have started.
Twitch chat authentication is required. The existing Twitch stream key cannot
authenticate chat. Keep the current local opponent running until chat is ready.

The G4 runner is ready at `HAL#647`. Its stream is
[hal_20xx](https://www.twitch.tv/hal_20xx). The opponent channel is
[x_pilot](https://www.twitch.tv/x_pilot).

The live check reported 59.95 game FPS and 60.00 OBS FPS. OBS reported zero
encoder skips and zero network drops. These figures describe the existing
local opponent match, not a match against x_pilot.

## Schedule

[x-pilot-96.csv](x-pilot-96.csv) preserves the order and both characters from
`matchups_for_vs_cpu(96)`. Its indices start at one. The supported rows remain
pending until a completed replay confirms the requested pairing.

The SHA-256 of the compact JSON list of numeric character pairs is
`a2202b353e3e769f2ab25e673226ef29fb6f949f4391c2b9f3003afdc7ce3c15`.

The supplied `gm-v2` roster covers 91 games, 53 distinct ordered pairings,
13 HAL characters, and 11 opponent characters. Five rows are unsupported:

| Schedule row | HAL character | Missing opponent |
| --- | --- | --- |
| 37 | Fox | Ganondorf |
| 48 | Falco | Ganondorf |
| 53 | Fox | Dr. Mario |
| 77 | Fox | Ness |
| 96 | Marth | Ganondorf |

Other supplied agent families include these characters. Do not substitute
another family in the `gm-v2` results. The CPU schedule excludes opponent
Sheik, although `gm-v2-sheik` is available. HAL still plays Sheik in eight rows.

| Opponent agent | Games |
| --- | ---: |
| `gm-v2-falco` | 28 |
| `gm-v2-fox` | 23 |
| `gm-v2-marth` | 14 |
| `gm-v2-cptfalcon` | 6 |
| `gm-v2-jigglypuff` | 6 |
| `gm-v2-peach` | 6 |
| `gm-v2-samus` | 4 |
| `gm-v2-popo` | 1 |
| `gm-v2-yoshi` | 1 |
| `gm-v2-luigi` | 1 |
| `gm-v2-pikachu` | 1 |

## Run protocol

This run tests the deployed HAL policy against the requested Phillip agents.
It reuses the CPU evaluation's character schedule. It is a separate opponent
treatment; its scores do not reproduce the CPU evaluation. Netplay stages,
ports, random seeds, and remote agent artifacts are not controlled by HAL.
Record their observed values where the replay exposes them.

Keep the current online settings fixed: imitation `IBDW#0`, delay 2,
desired return 20, and temperature 1.0. Change HAL's character for each row.
The loaded policy SHA-256 is
`0ff1daf80caa36a94a713c4ccba9223db8d7ba7c1379b5865bbc40b8a8c2f3ec`.
The running source includes the OBS changes from `a978dbf3`; the base image
label still reports `728d9601`. See the deployment README for the patch record.
Record the source and runtime configuration again when the run starts.

1. Validate the Twitch user token with `GET https://id.twitch.tv/oauth2/validate`.
   Require `user:write:chat` for sending and `user:read:chat` for reading replies.
   Read the token from a private file. Do not print it or commit it.
2. Read replies through EventSub and send commands through
   `POST https://api.twitch.tv/helix/chat/messages`. Check `is_sent` and
   `drop_reason`; HTTP success alone does not confirm delivery.
3. Send `!help` to x_pilot and confirm its current Slippi connect code from the
   bot's reply. Historical references to `PHAI#591` are not a live check.
4. Stop the local opponent loop after its current game. Let its cleanup cancel
   its reservation. Confirm that the G4 slot is free.
5. Select the next supported row with `!agent gm-v2-<character>`. Wait for the
   selection reply. Never change agents during a game: the published bot
   implementation restarts an active session when `!agent` changes.
6. Create one HAL reservation through `POST /v1/jobs`. Set `player_code` to the
   confirmed x_pilot code and `character` to this row's HAL character. Use the
   fixed settings above. Save the reservation token in a mode-600 local file.
   Confirm that the reservation reaches `connecting` and reports `HAL#647`.
7. Send `!play HAL#647` if the bot has no active session for this Twitch user.
   Confirm its reply and the HAL transition to `playing`. A full remote server,
   failed connection, or rejected agent is a failed attempt, not a played row.
8. Wait for the game to finish and for its replay upload. Record the job ID,
   game number, chat message IDs, selected agent, actual characters, stage,
   timestamps, result, replay key, and SHA-256 in a local run manifest. Count a
   row only after checking the replay's completed end record and both players.
9. Cancel the reservation in `rematch_wait`, then start the next row. This uses
   one game per reservation and preserves the schedule order. The next agent
   command may restart x_pilot's session; read its reply before sending another
   play request. Stop the remote session with `!stop` when the run ends.

The queue's `last_result` and replay metadata `result` use the joining player's
perspective. Here that player is x_pilot. Invert win/loss when reporting HAL's
result; keep ties unchanged.

Command behavior was checked in the public
[Phillip bot source](https://github.com/vladfi1/slippi-ai/blob/275c07270b5aa5f7b22aabff79dfe7f7b10a385f/scripts/twitchbot.py).
Check live replies because the deployed version may differ. Twitch documents
[chat messages](https://dev.twitch.tv/docs/chat/send-receive-messages/) and
[token validation](https://dev.twitch.tv/docs/authentication/validate-tokens/).

## Recordings

The existing runner uploads each Slippi replay and its small metadata JSON to
`r2:hal/netplay/v1/replays/YYYY/MM/DD/<opponent-code>/<job-id>/game-01.{slp,json}`.
The R2 rule expires these objects after 30 days. Download each verified pair
to the local run directory so the requested games remain available. Keep the
run manifest beside them. Do not add video uploads to R2.

The replay sidecar records the policy hash and a Git SHA, but the deployed base
image's Git SHA does not describe its source patches. The run manifest must
also include the effective source record above. A Twitch stream is not proof
of a saved VOD. Slippi replay recording is the confirmed recording path.

## Preparation checks

| Command or inspection | Result |
| --- | --- |
| `cat AGENTS.md`, `pwd`, `git status --short`, `git branch --show-current`, `git log -3 --oneline` | Confirmed the requested worktree and branch. It was clean before preparation. |
| `sed`, `cat`, and `rg` over the matchup allocator, tests, CPU evaluation callers, queue, runner, replay uploader, admin client, and deployment README | Confirmed the schedule, result perspective, one-game reservation procedure, and replay path. Initial searches named absent `hal/eval/vs_cpu.py` and `web/netplay-api/src/jobs.ts`; the existing callers and `store.ts` were read instead. |
| Python credential filename and environment-key scans; targeted `rg -l` for Twitch token settings | Found a stream key and no Twitch chat token. One targeted search included absent configuration paths and returned exit 2. No secret values were printed. |
| `gcloud compute ssh hal-netplay-g4 --project centering-star-502613-k3 --zone us-west1-a` with a read-only Python status check | Runner ready; one healthy slot; active 1080p60 OBS stream. Credential key names contained no Twitch chat token. |
| Python `/proc` inspection | Found the active local peer loop. It was left running while authentication was pending. |
| Python `urllib.request` GET of the public queue | HTTP 403. Repeated with the project's `httpx` client. Both `/v1/capacity` and `/v1/options` returned HTTP 200. |
| Python GET of the public `vladfi1/slippi-ai` GitHub tree and raw `scripts/twitchbot.py` | The first tree request used `master` and returned HTTP 404. The repository reports `main`; the corrected request and pinned source download succeeded. |
| Python generation from `matchups_for_vs_cpu(96)` | Exactly 96 rows: 91 supported and 5 explicitly unsupported. |
| `cp /tmp/hal-x-pilot-96.csv deploy/netplay/x-pilot-96.csv`, then Python CSV validation | Verified all 96 pairings, agent names, unsupported row indices, and the schedule hash against the allocator. |
| `git diff --check` | Passed. |
| `uv run pytest -q tests/test_matchups.py` | 10 passed; no failures or skips. |
| `rclone lsf 'r2:hal/netplay/v1/replays/2026/09/29/CRYO#610/' --max-depth 2 --files-only` | Existing local-peer matches have both `.slp` and `.json` objects. This does not claim any x_pilot recording. |
| Twitch API validation and chat sends | Not run: no user access token was available. |
| Full Python, Worker, and Dolphin handoff suites | Not rerun: this preparation changes only documentation and a schedule CSV. Runtime code is unchanged. |
