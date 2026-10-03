# Concurrent Slippi account test — October 1, 2026 UTC

Historical record. Status and commands below apply to the recorded release.
See [current deployment status](status.md) and [the runbook](README.md).

A later [parallel-pairing test](parallel-pairing.md) made the two G4
emulators connect to each other. Sequential pairing remains the tested way
to reuse this account for independent concurrent games.

## Result

Two independent games used the same `HAL#647` and `CRYO#610` accounts at
the same time. We started the second pairing after the first was in game.
The second game ran for 90 seconds. During that interval, the first pair
finished a game normally and started its next game.

This establishes account reuse for the tested Slippi Online 3.6.4 build.
It does not establish two production queue slots or sixteen model streams.
The second pair used neutral controller inputs and bypassed the HAL queue.
The production runner still advertises one slot.

| Endpoint | Observation time | Last frame | Frames / second |
| --- | ---: | ---: | ---: |
| Local secondary | 90.015 s | 5,396 | 59.945 |
| G4 secondary | 90.012 s | 5,398 | 59.970 |

Fifteen samples had fresh, in-game status from both local Dolphin processes.
The primary model game reported 59.932–59.990 FPS in those samples.
Its highest sampled policy p95 was 12.116 ms. These are short steady-play
samples. The primary's complete second game reported 59.031 FPS.

## Setup and timeline

The control was the ongoing Cody Fox model match between the local RTX 3060
and the existing G4. Both policies used advantage 120, BF16 inference,
network delay 2, inference allowance 1, replan 4, and horizon 8.
The treatment added another Dolphin at each endpoint, using the same account
identities but distinct ports, user directories, displays, and replay paths.
There was no second policy engine and no additional GPU or VM.

The second pair used G4 Falco and local Marth on Final Destination.
The different characters make the two peer connections distinguishable.
Both secondary replays confirm these characters and account codes.
The probe ended both emulators after 90 seconds. These unfinished replays
have no game-end event and must not enter win-rate calculations.

| UTC time | Event |
| --- | --- |
| 05:35:21.120 | Primary game 2 started |
| 05:37:00 | Secondary pairing began |
| 05:37:25.542 | Local secondary reached frame 0 |
| 05:37:25.581 | G4 secondary reached frame 0 |
| 05:37:39.674 | Primary game 2 ended normally; G4 won |
| 05:37:46.027 | Primary game 3 started |
| 05:38:55.557 | Local secondary completed its observation |
| 05:38:55.593 | G4 secondary completed its observation |
| 05:38:56.604 | Both secondary sessions had closed |

Afterward, UDP ports 52442 and 51442 were closed and both probe processes
had exited. The primary model game continued. G4 reported one healthy slot
and zero runner recoveries. The production display and stream stayed in place.

G4 used image commit `8831869a7b315fc2755f895b399c4f84a968a84a`;
the local worktree was at `84f55db1`.
The primary policy SHA-256 was
`0ff1daf80caa36a94a713c4ccba9223db8d7ba7c1379b5865bbc40b8a8c2f3ec`.
This was an operational compatibility test, not a policy evaluation.
The probe sent deterministic neutral inputs. We did not control a game RNG seed.

## Why it can work

The pinned client disconnects from matchmaking after receiving an opponent
assignment. It then creates a separate peer connection with the assigned
addresses and ports. See the upstream
[matchmaking implementation](https://github.com/project-slippi/Ishiiruka/blob/v3.6.4/Source/Core/Core/Slippi/SlippiMatchmaking.cpp)
and [netplay client](https://github.com/project-slippi/Ishiiruka/blob/v3.6.4/Source/Core/Core/Slippi/SlippiNetplay.cpp).
The live test supplies evidence for account reuse; client source alone does
not establish what the private matchmaking server permits.

## Proposed HAL change

The spec currently requires a distinct account for every slot. The proposed
replacement is:

1. Keep the account leased exclusively to one runner session.
2. Let that runner's slots reuse the account, with a separate Dolphin and
   Slippi port for each connection.
3. Admit only one matchmaking attempt per account at a time. Release this
   gate only after peer connection succeeds or failure cleanup finishes.
4. Preserve established peer connections and their rematches while another
   slot pairs. Apply the same gate to reconnection after a disconnect.
5. Test two real model slots, failure recovery, cancellation, and rematching
   before increasing advertised capacity.

This proposal requires owner approval under the original work instructions.
The canonical spec, queue schema, deployed Worker, and slot count are unchanged.
If the persisted account model changes, bump the schema guard as required.
The failed sixteen-stream inference qualification remains a separate limit.

## Evidence

Local evidence is under
`runs/netplay/account-reuse/20261001T053325Z/`:

- `probe.py` and `probe-ogl.py`: exact temporary test sources.
- `local-ogl/status.json`, `g4-ogl/status.json`: all timing samples.
- `overlap-ogl.jsonl`: five-second samples of both local games.
- `summary.json`: timings, overlap check, and replay hashes.
- `local-ogl/replays/` and `g4-ogl/replays/`: both secondary replays.
- `local.log`, `g4.log`, and the corresponding status files: failed first attempt.

The remote originals are under
`/var/lib/hal-netplay/account-reuse/20261001T053325Z/`.
The primary model run is under
`runs/netplay/local3060-cody120/20261001T053223Z/`.
These ignored operator files contain local paths and are not deployment code.
No new replay upload was made for this experiment.

## Commands, failures, and skips

- Read `AGENTS.md`, Git status/history, the spec, deployment reports,
  runner and simulation code, and test helpers with `cat`, `sed`, and `rg`.
  Searches for absent `web/netplay-api/src/runner.ts` and
  `hal/trajectory.py` failed; corrected paths located the code.
  A local Slippi source search found no source; upstream reads supplied it.
  An optional upstream `SlippiUser.cpp` fetch failed and was not needed.
- The old local operator run stopped after its first completed game:
  `int(NaN)` failed when its result logger read the trailing menu frame.
  A new run reads final stocks from the finalized replay. A Python
  `compile(...)` check and a focused replay assertion passed.
  That run then completed games and rematched successfully.
- The model launcher used
  `APPIMAGE_EXTRACT_AND_RUN=1 xvfb-run -a .venv/bin/python runs/netplay/local3060-cody120/20261001T053223Z/play.py`,
  with temporary and Torch cache directories inside this worktree.
- Both probe endpoints used
  `timeout --signal=TERM --kill-after=10s 200s xvfb-run -a <python> <probe> --account <existing-private-path> --opponent <peer-code> --port <port> --character <character> --output <run-dir>`.
  G4 used the installed Python inside `hal-netplay-runner` via
  `docker exec -d -w /tmp`; local used `.venv/bin/python`.
  G4 parameters were CRYO#610 / 51442 / FALCO.
  Local parameters were HAL#647 / 52442 / MARTH.
- The first attempt used Vulkan on both private displays. G4 showed a
  warning window before producing menu observations and hit a frame timeout.
  Local hit its 90-second connection timeout. This did not show an account
  rejection. The exact warning text was not captured.
- The retry used `probe-ogl.py --graphics-backend OGL` on G4 and the
  default Vulkan backend locally. Both completed the 90-second assertion.
- Remote `xwininfo` first failed with the wrong Xauthority, then succeeded
  with the probe display's authority. A later Pillow screenshot failed
  because the timed-out display had closed; viewing its empty file failed.
- Python overlap samplers completed for both attempts. SSH `cat` and
  `tail` checked statuses and logs. `docker exec ... pgrep -af 'probe|dolphin'`
  returned 1: it found no matching names. A later `docker top` confirmed the
  main Dolphin under its actual process name.
- SSH `tar -C <remote-run> -cf - ... | tar -xf - -C <local-run>` copied
  test evidence. A local Python check parsed both replays with
  `peppi_py.read_slippi(..., skip_frames=True)`, checked both passed
  statuses, and required more than ten fresh overlapping samples. Passed.
- `ps -p 2020522,2020537,2006318 -o pid,stat,comm` showed only the main
  player's launcher. Local `ss -lunp 'sport = :52442'` and remote
  `ss -lunp 'sport = :51442'` showed no test sockets. Cleanup passed.
- `git diff --check` and local Markdown-link checks passed.
- Ruff, ty, Python unit/integration suites, and Worker tests/typecheck were
  not rerun for this documentation-only milestone. Maintained runtime code
  did not change. The live experiment and replay checks are reported above.
- No schema change, deployment, image push, new VM, or Git push was made.
