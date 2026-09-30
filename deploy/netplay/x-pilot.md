# x_pilot matchup run

## Status

Started on 2026-09-29 with effective runner source `1dd743b2`.
The first two completed games used `MASTER` player conditioning and raw desired
return 120. Their verified Slippi replays record HAL Fox against `gm-v2-falco`
and `gm-v2-marth`. HAL lost both. The run continues through the supported schedule.
See the local result files for current progress; this document is a checkpoint.

The G4 runner is ready at `HAL#647`. Its stream is
[hal_20xx](https://www.twitch.tv/hal_20xx). The opponent channel is
[x_pilot](https://www.twitch.tv/x_pilot).

The first x_pilot game averaged 58.7 game FPS across 5,810 frames. Frame
interval p95 was 18.0 ms, Dolphin step p95 was 17.9 ms, and policy round-trip
p95 was 7.4 ms. The runner reported one missed deadline and four transport
correction frames. OBS reported 60.00 FPS, one encoder skip, four render
skips, and zero network drops across roughly 18,500 output frames.
The earlier local opponent check averaged 59.95 game FPS. Different opponents
and network paths prevent treating these figures as a controlled comparison.
During the Marth game, the live sample reported 59.95 game FPS and 60.00 OBS
FPS, with no new encoder skips or network drops.

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

Keep the requested online settings fixed: imitation `MASTER`, delay 2,
raw desired return 120, and temperature 1.0. Change HAL's character for each row.
The loaded policy SHA-256 is
`0ff1daf80caa36a94a713c4ccba9223db8d7ba7c1379b5865bbc40b8a8c2f3ec`.
The first game used the OBS changes from `a978dbf3` and the shared return
validation from `1dd743b2`. The local image is `hal-netplay-runner:obs-local`;
its registry base is still `728d9601`. The local runtime snapshot records
the effective source, image ID, checkpoint hash, environment, and sampling
seeds. Record subsequent source changes at game boundaries.

After game two, `e942072e` adds bounded libmelee receiver shutdown. The old
receiver blocked on a full pipe after Dolphin exited and prevented the next
reservation from starting. The first instance was released manually after its
recording was verified. The fix changes cleanup only; gameplay settings stay
fixed. The G4 uses the same source bind mount as the prior OBS deployment.

1. Validate the Twitch user token with `GET https://id.twitch.tv/oauth2/validate`.
   Require `user:write:chat` for sending and `user:read:chat` for reading replies.
   Decrypt the token in memory from the local credential store described below.
2. Read replies through EventSub and send commands through
   `POST https://api.twitch.tv/helix/chat/messages`. Check `is_sent` and
   `drop_reason`; HTTP success alone does not confirm delivery.
3. Use the owner's confirmed opponent code `PHAI#591`. The first completed
   replay also confirms this code and HAL's `HAL#647` code.
4. Stop the local opponent loop after its current game. This was completed
   before the x_pilot run; its reservation and local Dolphin were cleaned up.
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

The local run directory is `runs/netplay/x-pilot-master120/`:

- `result-NNN.json` records each verified schedule row and HAL's result.
- `NNN-<job-id>/game-01.{slp,json}` holds the downloaded recording and sidecar.
- `runtime-snapshot.json` records the initial G4 environment and sampling seeds.
- `events.jsonl` and `chat.jsonl` record progress and relevant bot replies.
- `private/` contains reservation tokens with mode 600.
- `control/` retains the local operator scripts used for this run.

The initial excluded attempts were the canceled `IBDW#0`/20 reservation
`oNHh0qZ14F2pg27ZOQc5cZpQ`, and `D54iIcqY1Z7RCUIQ-S9kigOz`, which failed
before completing a game because inference still enforced the old return cap.
The fix shares `[-20, 140]` across the inference, evaluation, and queue
boundaries. Tests now perform actual predictions with `MASTER`/120.

During the deployment cleanup, 47 games had been verified through schedule
row 49. The first attempt at row 50 received no x_pilot acknowledgment for
`!play HAL#647` and stopped the scheduler. Its reservation
`w_yrF_Mg8NlTpMhjy1lkGFgN` was confirmed canceled with zero completed games.
The failed attempt remains in the event log and private reservation records.
The same row was resumed and reached `playing`, with `MASTER`/120 unchanged.

## Local operation and credentials

Two local user services manage the run. They use no new cloud resources:

```sh
systemctl --user status hal-xpilot-chat hal-xpilot-games
journalctl --user -u hal-xpilot-games -n 30 --no-pager
```

To stop after the current game and preserve its recording:

```sh
touch runs/netplay/x-pilot-master120/stop-after-game
```

The scheduler skips verified rows and stops on a failed or mismatched game.
Inspect a failure before restarting it. The chat listener restarts after a
network failure. These are transient local services; they do not start after
a reboot. The G4 runner and stream have their own system services.

Operator credentials are encrypted with `systemd-creds --user --with-key=host`
under `runs/netplay/credentials/`. The directory has mode 700; files have
mode 600. Encryption binds each credential to this host, user, and credential
name. It protects copied files, but not a compromised local user or root.
Do not use these files as portable backups.

The store contains Twitch chat and refresh tokens, the Twitch stream key,
runner environment, runner/admin tokens, and Cloudflare deployment credentials.
The old temporary plaintext copies were removed after verifying the encrypted
copies. Original Slippi account files were left in place. The G4 still reads
its existing Secret Manager secret into root-only files under `/run`.

The operator helper decrypts secrets in memory. It locks refresh operations,
stores replacement Twitch tokens atomically, validates the account and scopes,
and uses only official Twitch endpoints. A live refresh and the next bot
commands passed after the move. Twitch's
[refresh protocol](https://dev.twitch.tv/docs/authentication/refresh-tokens/)
describes the token replacement requirement. Never print decrypted secrets,
put them in command arguments, or commit them.

## Preparation checks

These entries describe the initial preparation. The execution checks below
supersede the authentication and local-peer status in this table.

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

## Execution checks — 2026-09-29

All local commands used `/home/ericgu/src/hal-edge-queue`; npm commands used
its `web/netplay-api` directory. No command changed another worktree.

| Command or operation | Result |
| --- | --- |
| `cat AGENTS.md`; `git status --short`; targeted `rg`, `sed`, and `cat` reads of the queue, inference validators, evaluation settings, session cleanup, tests, and installed libmelee | Read the affected boundaries before editing. Some initial searches named absent files; corrected searches found the real callers. A later unquoted `tests/test_v7*` glob also matched no files; a recursive `rg` found the skip definitions. |
| Twitch device authorization, `GET /oauth2/validate`, EventSub subscription, and `POST /helix/chat/messages` | Authenticated `hal_20xx` through official Twitch endpoints with `user:read:chat` and `user:write:chat`. The bot confirmed agent selection and `!play HAL#647`. The initial device authorization tool review timed out; retry succeeded. |
| Local peer stop and reservation cleanup | Waited for its active game, then stopped the parent loop. It left the G4 slot available for x_pilot. |
| Publish the existing policy's approved `desired_return_range` of `[-20, 140]` | Passed. Policy bytes, hash, and default return were unchanged. |
| First `MASTER`/120 reservation | Failed before completing a game: inference still enforced `[0, 40]`. Commit `1dd743b2` fixes the remaining inference/evaluation validators and adds actual prediction coverage. This attempt is excluded. |
| `uv run pytest -q tests/test_policy_api.py tests/test_policy_adapter.py tests/test_netplay_domain.py tests/test_action_sequence_policy_batch.py -m 'not integration'` | 105 passed, 2 deselected, 14 warnings after the return fix. |
| `uv run pytest -q tests/test_session.py tests/test_netplay_session.py` | 54 passed. Covers a full receiver pipe, ignored SIGTERM, shutdown before connect, repeated cleanup, and rejection of an untested libmelee version. |
| `uv run ruff format hal/sim/session.py tests/test_session.py` | Both files already formatted. |
| `uv run ruff format --check .` | 270 files passed. |
| `uv run ruff check .` | Passed. |
| `uv run ty check --python-version 3.14 --error-on-warning hal experiments/059_muon_action_sequence.py scripts` | Passed with zero diagnostics. |
| `uv run pytest -q -m 'not integration'` | Final source: 1,491 passed, 8 skipped, 21 deselected, 24 warnings in 141.12 s. Earlier return-fix checkpoints passed 1,470 and 1,487 tests. |
| `npm test` | 105 tests passed in 10 files. workerd printed a WebSocketPipe disconnect diagnostic; no tests failed. |
| `npm run typecheck` | Passed. |
| `HAL_REQUIRE_INTEGRATION=1 uv run pytest -q tests/test_netplay_queue_integration.py -m integration` | 3 passed in 41.60 s. Includes `MASTER`/120 creation, live update to 140, and rejection of 141. |
| `HAL_REQUIRE_INTEGRATION=1 uv run pytest -q tests/test_roundtrip.py tests/test_session_cleanup.py -m integration` | 7 passed, 6 deselected, 6 warnings in 54.16 s. Required fixtures were present. |
| `.venv/bin/python /tmp/hal_xpilot_run.py --first-job runs/netplay/x-pilot-master120/private/first-game.json --max-new-games 1` | Downloaded and verified the first replay's hash, size, completed end record, player codes, characters, policy, and result. Automatic approval review initially rejected this read-only collector for a suspected R2 upload. Inspection found only list/download methods; the reviewed retry passed. |
| `systemd-creds --user --with-key=host --name=... encrypt - -`; `systemd-creds --user --name=... --refuse-null decrypt - -`, through the local credential helper | Non-secret round-trip and wrong-name rejection passed. Sandbox access to the system credential service failed; host execution passed. All nine real credentials passed encrypted round-trip checks. Temporary plaintext copies were then removed. |
| Direct Twitch `POST /oauth2/token` refresh, followed by validation and bot commands | Passed with the encrypted store. No third-party token service was used. |
| `systemd-run --user --unit=hal-xpilot-chat --property=UMask=0077 --property=Restart=on-failure --property=RestartSec=5 ... hal_xpilot_chat.py listen` | Listener subscribed successfully with encrypted credentials. |
| `systemd-run --user --unit=hal-xpilot-games --property=UMask=0077 ... hal_xpilot_run.py --first-job .../private/first-game.json` | Continued the schedule, verified game two, then honored `stop-after-game` for deployment. |
| `journalctl --user -u hal-xpilot-games -n ... --no-pager`; `systemctl --user show/status ...` | Confirmed agent replies, connection, gameplay, recording, and boundary stop. |
| `gcloud compute ssh hal-netplay-g4 --project centering-star-502613-k3 --zone us-west1-a --command ...` | Read status, OBS statistics, source identity, logs, process wait states, and runtime budget. Saved the initial environment and seeds locally. No credentials were printed. One status read during startup found no file; the readiness check was corrected to handle preparation. |
| G4 process inspection and targeted SIGTERM | Found the completed game's Slippi receiver blocked in pipe write, with its parent waiting for exit. Releasing that receiver let game two start. No active Dolphin game was terminated. |
| `gcloud compute scp hal/sim/session.py hal-netplay-g4:/tmp/hal-session-e942072e.py --project centering-star-502613-k3 --zone us-west1-a` | Staged the committed cleanup fix. SHA-256: `07d087f60e6eb51d73bc902e68e607b2256dad344a53d0132cf51a6305c96286`. |
| G4 source installation, `systemctl daemon-reload`, and runner stop/start at the completed-game boundary | Installed the cleanup fix and updated the effective source label to `e942072e`. |
| Python CSV validation and manifest generation | All 96 pairings match `matchups_for_vs_cpu(96)`. Exactly five remain unsupported. Saved settings, source, runtime, and operator script hashes. |
| `git check-ignore` on the credential, reservation, and operator-script paths | All are ignored. |
| `git diff --check`; focused `git add`; `git commit` | Whitespace passed. Commits contain no attribution trailers. Source milestones are `2f56cd18`, `1dd743b2`, and `e942072e`; the first was incomplete and is superseded by the second. |

The eight full-suite skips comprise two production GPU qualification checks
that require `HAL_REQUIRE_NETPLAY_HARDWARE_QUALIFICATION=1`, and six tests
whose optional local v7 subset is absent. Integration deselections are tests
outside the requested marker. Warnings concern Python 3.14 TorchScript,
uncompiled flex attention, and multiprocessing fork from a threaded process.
No required integration fixture was missing.

Read-only documentation research used official Twitch and systemd sources.
The freedesktop manual URL returned HTTP 403; the official systemd GitHub
manual and the installed manual supplied the credential details instead.

No new VM, instance group, Secret Manager entry, registry push, R2 asset
upload, Worker deployment, or Git push was performed during this run setup.
The existing runner continues its configured replay uploads. The local
collector only downloads them.
