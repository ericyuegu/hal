# Parallel Slippi pairing test — October 1, 2026 UTC

## Result

Parallel searches did not connect the requested opponents.

Both G4 Dolphins used HAL#647 and requested CRYO#610. They started together
at 18:42:29 UTC. They connected to **each other** at about 18:42:48, before
either local Dolphin started at 18:42:49. Both replay headers contain
HAL#647 versus HAL#647, with Fox and Falco.

The two local Dolphins then connected to each other as CRYO#610 versus
CRYO#610, with Marth and Sheik. HAL rejected all four games because its
local account did not identify exactly one player.

This is evidence of unsafe parallel pairing in the tested Slippi stack.
It does not identify whether the client or the private matchmaking server
caused the incorrect pairing. It also does not prove that every parallel
attempt fails. The earlier [sequential-pairing test](account-reuse.md)
established two concurrent games without this failure.

## Treatment and controls

- Treatment: two new searches use the same account at the same time.
- Prior control: a second search starts after the first connection is live.
- Two existing accounts: HAL#647 on G4 and CRYO#610 locally.
- G4: the existing single RTX PRO 6000 host, image commit 8831869a.
- Local: RTX 3060 machine, source commit 3a1cbb07 plus uncommitted queue work.
  The simulation and menu helper source were unchanged.
- Emulator: verified Slippi Online 3.6.4 fixture.
- Separate Dolphin user directories, Xvfb displays, replay paths, and
  observation ports: G4 51451/51452; local 52451/52452.
- G4 used OpenGL; local used Vulkan. Both sent neutral controls.
- Each probe had a 90-second connection timeout, a planned 60-second game
  observation, and a 210-second process limit. No game reached the observation.
- Local probes started 20 seconds after the G4 probes. This separates the
  incorrect G4 pairing from the duplicate local account searches.
- No model, Twitch stream, new VM, account upload, or R2 replay upload.
  Random game seeds were not controlled.

The production queue remained paused. The shared-account implementation was
not deployed for this test.

## Evidence

Local originals and logs:

```text
runs/netplay/parallel-pairing/20261001T183946Z/
```

Remote originals:

```text
/var/lib/hal-netplay/parallel-pairing/20261001T183946Z/
```

`launch.json` contains both schedules and the local launch commands.
`probe.py` is the exact operator script. The four status files report the
identity failure. `header-players.json` records the replay player identities.

The original replay files stopped before a complete frame was flushed.
Normal Peppi parsing failed. The diagnostic extracted each complete Game
Start event into a separate, valid header-only file under `headers/`.
Peppi read those files with `skip_frames=False`. These derived files are
header evidence, not playable games or evaluation results. Originals remain
unchanged.

## Commands, failures, and skips

- `rg --files`, targeted `sed`/`rg`, `cat`, and `ls` read the existing
  probe, fixture paths, account paths, timeout constants, and menu code.
  A Python read printed only account codes. A search named absent
  `hal/sim/replay.py`; a shell glob for absent `hal/slippi*` also failed.
  The maintained repair code was found in `hal/data/slp_finalize.py`.
- The preceding eight-stream qualification finished before this experiment.
  `docker inspect` and `docker logs` showed exit 1: delay-2 prediction p99
  was 22.409 ms against the current 16.667 ms allowance. Its receipt was
  copied to `qualification-8.json`. Delay 3 was not reached.
- Python created the ignored probe and passed `compile(...)`. SSH
  `install -d` and `tee` copied it to G4.
- `docker run -d --name hal-netplay-pairing-test ... sleep 900` created
  the temporary test container. `docker exec ... python -c 'from hal.fixtures
  import ISO, NETPLAY_EMULATOR, ensure; ensure(ISO); ensure(NETPLAY_EMULATOR)'`
  downloaded and verified both fixtures successfully.
- Each G4 endpoint used `docker exec -d -w /tmp ... sh -c ...` to run
  `timeout --signal=TERM --kill-after=5s 210s xvfb-run -a python /audit/probe.py`
  with its account, opponent, port, character, output, and scheduled start.
  Local Python `subprocess.Popen` launched the same probe in its virtual
  environment. The first local launcher resolved the Python symlink outside
  the virtual environment and failed to import melee. Corrected launchers
  started before the scheduled local search time. Both failure logs remain.
- SSH/local `tail` and status reads observed all four identity failures.
  SSH `tar ... | tar ...` copied remote replay evidence locally.
- Direct Peppi reads failed on incomplete originals. The first header-only
  read with `skip_frames=True` also failed. Reads with
  `skip_frames=False` succeeded and confirmed both incorrect pairs.
- `docker top`, `ps`, and `ss -lunp` confirmed the four probe processes
  and ports had closed. `docker stop --timeout 3 hal-netplay-pairing-test`
  and `docker rm hal-netplay-pairing-test` succeeded.
- `git diff --check` passed for this documentation milestone. No maintained
  runtime code was added by the experiment. Unit/integration suites and
  Worker checks are pending for the separate, uncommitted queue changes.
