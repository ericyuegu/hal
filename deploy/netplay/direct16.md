# Direct-play capacity — October 1, 2026 UTC

The owner requested restoration of direct play with up to 16 simultaneous
matches. Commit `8831869a` raises the per-host admission limit from eight to
sixteen in the runner CLI, Worker, and GCE scripts. Each slot keeps its own
account, Slippi port, Dolphin process, and inference stream. One shared
inference engine serves the host. Storage remains schema 2.

## Launch requirements

- Sixteen distinct Slippi account JSON files for sixteen matches. The current
  production account list contains only `HAL#647`. The owner's local peer
  account is not added to the bot pool.
- Qualification on the actual GPU at the requested capacity for both delay
  profiles. The existing gate requires p99 prediction latency below one frame
  and refuses admission if either profile fails. Raising a configuration limit
  is not evidence that the GPU can sustain sixteen matches.
- A Cloudflare plan that supports the traffic. Sixteen idle slots poll every
  five seconds: 276,480 claims/day. A two-second session report adds 43,200
  requests/day, before browsers, games, and alarms. The total exceeds the
  Workers Free limit of 100,000 requests/day. Workers Paid has a $5/month
  minimum; see [Cloudflare's current pricing](https://developers.cloudflare.com/durable-objects/platform/pricing/).
  The owner's confirmation is pending. The current token gets HTTP 403 from
  the subscriptions API, so it cannot establish the billing plan.

The initial recovery uses one account and exactly one RTX PRO 6000 Blackwell
GPU (`g4-standard-48`: 48 vCPUs and 180 GiB RAM). The additional accounts are
required before sixteen slots can be opened. No account or queue capacity is
fabricated to advertise unqualified slots.

`hal-netplay-admin accounts upload` replaces the account list. Supply the
complete intended list, including the existing bot account. Do not supply
only the fifteen additions. The command validates distinct codes, uploads
private account objects by hash, and publishes their references.

## Queue cost check

A real local Durable Object measured these calls with sixteen leased accounts
and one active game. Adding 10,000 completed jobs and sessions changed none
of the counts:

| Call | SQL rows read | SQL rows written | Alarm writes |
| --- | ---: | ---: | ---: |
| Empty claim | 8 | 0 | 0 |
| Session report | 27 | 1 | 0 |
| Capacity poll | 22 | 0 | 0 |

The report also reads the alarm once. These are per-call measurements, not
an estimate of a full sixteen-game workload. Browser polling, active-game
heartbeats, state transitions, and alarms add requests and reads.

## Validation and command results

- Read `AGENTS.md`, affected runner/client code, persisted formats, host and
  image scripts, Worker routes, and existing tests. Git was clean at start.
- Used the Durable Objects and Wrangler skills and official Cloudflare
  pricing/testing documentation. No package versions were changed.
- `uv run python runs/netplay/public-relaunch/operations.py status` succeeded.
  It found one bot account, zero active games, zero capacity, and an old
  session lease left from the quota incident.
- Gcloud listed no running instances. Service account
  `hal-netplay-runner@centering-star-502613-k3.iam.gserviceaccount.com` and
  secret `hal-netplay-runner-env` still exist. The initial registry lookup
  used absent repository `hal` and returned NOT_FOUND. Listing repositories
  located the actual `hal-netplay` registry and the retained image.
- `docker image ls`, `rclone`-independent local file searches, and selected
  deployment reads found the image cache and the two known Slippi account
  paths. No credential value was printed. Some exploratory file searches
  used absent filenames; corrected searches found the maintained modules.
- `gcloud compute machine-types describe g4-standard-48` confirmed exactly
  one GPU. `gcloud compute images describe-from-family` resolved boot image
  `common-cu129-ubuntu-2404-nvidia-580-v20260909` in
  `deeplearning-platform-release`.
- `uv run pytest -q tests/test_netplay_runner.py tests/test_netplay_gce.py -m 'not integration'`
  passed 81 tests. New cases cover sixteen distinct account paths/ports and
  rejection before resource creation for unsupported slot counts.
- The initial focused Worker run failed one test because its IP range reused
  the separate rate-limit test's exhausted address. Giving the new test a
  distinct range fixed it. The first type check found an untyped result array;
  adding its explicit type fixed it. The focused rerun passed all 25 tests.
- `npm test -- test/costs.test.ts -u` recorded the SQL-cost regression snapshot;
  six tests passed. `npm run typecheck` then passed.
- `uv run ruff format --check .` passed for 287 files. `uv run ruff check .`
  and `uv run ty check --python-version 3.14 --error-on-warning hal experiments/059_muon_action_sequence.py scripts`
  passed with zero diagnostics. `git diff --check` passed.
- `TMPDIR="$PWD/runs/netplay/ranked-cody120/test-tmp" uv run pytest -q -rs -m 'not integration' --basetemp="$PWD/runs/netplay/ranked-cody120/test-tmp/direct16-full"`
  passed 1,610 tests with eight skips, 21 deselections, and 26 warnings in
  149.21 seconds. Skips: two opt-in GPU qualification tests and six tests
  requiring the optional local v7 subset.
- `HAL_REQUIRE_INTEGRATION=1 TMPDIR="$PWD/runs/netplay/ranked-cody120/test-tmp" uv run pytest -q tests/test_roundtrip.py tests/test_session_cleanup.py tests/test_netplay_queue_integration.py -m integration --basetemp="$PWD/runs/netplay/ranked-cody120/test-tmp/direct16-integration"`
  passed all ten tests with six deselections and six warnings in 104.93 seconds.
- API `npm test` passed 114 tests in eleven files. `npm run typecheck` passed.
  Workerd printed its existing WebSocket shutdown diagnostic; no test failed.
- `npx wrangler whoami` confirmed the existing account and Worker-write scope.
  A path search returned status 2 because one optional auth directory was
  absent; the actual credential file was found and no token was printed.
- A read-only Cloudflare subscriptions request returned HTTP 403. The Workers
  account-settings request returned HTTP 200 with the standard usage model;
  this does not establish a paid subscription.
- `npx wrangler deploy --dry-run` passed. The subsequent deployment succeeded
  on the existing `20xx.xyz/v1/*` route, version
  `7e885dd1-1ab8-4e04-8518-241e181b0876`. It preserved the schema and data.
- The code commit's Ruff and ty hooks passed. No Git push was made.

Local command logs and deployment evidence are under `runs/netplay/direct16/`.

## Recovery preparation

The approved expanded player list was published with
`uv run python runs/netplay/public-relaunch/operations.py publish-policy`.
Its guard confirmed that only `imitations` changed. The existing policy
object was reused. Admissions were paused through the admin API while the
new host was prepared.

The old lease did not expire after the quota incident, even after a new
policy publication. The former session last reported more than a day before
this launch and its VM was confirmed deleted. Calling the normal idempotent
session-end API released `HAL#647` and failed zero jobs. The receipt is
`runs/netplay/direct16/stale-session-cleanup.json`. Recovery of overdue
alarms after free-tier exhaustion remains a separate issue; this launch
uses the explicit operator cleanup and does not wipe queue state.

## Replacement image

The current Dockerfile built successfully. A first local build populated the
cache; the release build used the exact committed SHA and tagged image:

```text
us-west1-docker.pkg.dev/centering-star-502613-k3/hal-netplay/hal-netplay-runner:8831869a7b315fc2755f895b399c4f84a968a84a
```

A network-disabled, GPU-free preflight verified the installed runner against
its source SHA-256, checked the image revision, imported the runtime, and
found all 54 policy choices. Runtime versions are Python 3.14.3, Torch
2.11.0+cu130, Pillow 12.2.0, and OBS 30.2.3.1-3~bpo24.04.1. The preflight ran
from `/tmp` to verify installed modules without source-tree shadowing.
The image contains no ISO, policy, account JSON, or stream key.

Build and preflight logs are `image-build.log`, `image-release-build.log`,
and `image-preflight.json` under the local evidence directory. Docker build
used `--build-arg HAL_GIT_SHA=8831869a7b315fc2755f895b399c4f84a968a84a` for
the release tag. The first cache build was never deployed.


The private registry push succeeded. Its manifest digest is
`sha256:b3f341cd60e15df2c899d6b2bf6bce50afe0297d3b8d03762d40252e59d9a501`.
The existing registry reused 31 layers. The replacement was requested with:

```sh
deploy/netplay/gce-up.sh hal-netplay-g4   --project centering-star-502613-k3 --zone us-west1-a   --machine-type g4-standard-48   --image us-west1-docker.pkg.dev/centering-star-502613-k3/hal-netplay/hal-netplay-runner:8831869a7b315fc2755f895b399c4f84a968a84a   --git-sha 8831869a7b315fc2755f895b399c4f84a968a84a   --secret hal-netplay-runner-env   --service-account hal-netplay-runner@centering-star-502613-k3.iam.gserviceaccount.com   --slots 1
```

The launch uses one slot because only one bot account is available. The
configured maximum of sixteen does not change that advertised live capacity.

## Fresh-host startup repairs

The VM was created successfully as instance `8731743256809856141`,
address `34.83.210.75`, in `us-west1-a`. Its only GPU is an NVIDIA
RTX PRO 6000 Blackwell Server Edition, driver 580.173.02. The 100 GB boot
disk expanded to a 96 GiB filesystem with 79 GiB free after the image pull.
The create command's disk-resize warning did not indicate a failed resize.

The selected Deep Learning VM image lacked Docker. The first startup exited
before creating the runner service. Commit `a3617973` installs `docker.io`
when absent, configures the NVIDIA container runtime, enables Docker, and
checks it before continuing. A regression test covers both fresh and already
installed Docker. `uv run pytest -q tests/test_netplay_gce.py` passed 19 tests.
The full unit suite passed 1,612 tests, with the same eight skips and 21
deselections, in 151.10 seconds. Ruff format, Ruff check, ty, and diff checks
passed. The commit hooks passed.

The first OBS attempt then found no `libnvidia-encode.so.1` on the host.
OpenGL worked on NVIDIA, but NVENC could not start. The startup script now
installs encode and decode libraries at the exact installed 580 server driver
version, alongside the matching GLX libraries. The regression test checks
those pins and confirms that CDI refresh follows installation.

Commands for these repairs:

- `gcloud compute instances add-metadata hal-netplay-g4 --project centering-star-502613-k3 --zone us-west1-a --metadata-from-file=startup-script=deploy/netplay/gce-startup.sh`
  succeeded after each repair.
- Remote `sudo systemctl restart google-startup-scripts.service` completed
  host setup and pulled the verified image.
- Remote `sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libnvidia-encode-580-server=580.173.02-0ubuntu0.24.04.1 libnvidia-decode-580-server=580.173.02-0ubuntu0.24.04.1`
  installed both packages without changing the running driver.
- Remote `sudo systemctl restart nvidia-cdi-refresh.service` passed.
- `systemctl is-active`, `docker ps`, `docker logs`, `nvidia-smi`,
  `dpkg-query`, `apt-cache policy`, and selected status-file reads diagnosed
  each stage. The first status-file read failed because preparation had not
  yet written the file. Later reads reported one healthy slot.
- One edit command used absent `python`; rerunning with `python3` applied
  the change. The first formatting check requested one test-file reformat.
  `uv run ruff format tests/test_netplay_gce.py` fixed it; all static checks
  then passed. Exploratory reads of absent play-script/test paths were
  corrected with `rg --files`.

No live match was admitted during host repair or capacity measurement.

The NVIDIA video repair is committed as `6ec246ff`. Its focused host tests
passed 19 tests. The final unit command used
`--basetemp="$PWD/runs/netplay/ranked-cody120/test-tmp/direct16-video-full"`
and passed 1,612 tests with eight skips, 21 deselections, and 26 warnings in
144.40 seconds. Logs are in `video-python-tests.log`. All static checks and
commit hooks passed. The already-passing Worker and integration suites were
not repeated for these host-only package changes.

The local smoke-test setup found that this worktree did not yet have the
pinned netplay emulator. `uv run python -c 'from hal.fixtures import NETPLAY_EMULATOR, ensure; print(ensure(NETPLAY_EMULATOR))'`
downloaded and verified it. Its log is `local-emulator.log`.

## Sixteen-stream result

A separate container ran the production qualification code on the replacement
G4 with the same policy hash, compiled BF16 inference, a 0.5 ms batch wait,
and seed 120647. It had no network or account credentials. The paused
one-slot runner remained idle. No additional GPU or VM was created.

The delay-2 profile failed its 200-sample gate:

| Capacity | Network delay | Inference allowance | Replan | Horizon | Prediction p99 | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 16 | 2 frames | 1 frame (16.667 ms) | 4 frames | 8 frames | 33.812 ms | Fail |

This measures request delivery, batching, validation, and prediction. It is
not a measurement of sixteen live Dolphin games. The failure occurred on
the first profile, so the delay-3 profile was not measured. Full sixteen-game
rendering, audio, network, and CPU load remain untested.

The command ran `/audit/qualify.py --bundle /policy.halpolicy --output /audit/qualification-16.json --capacity 16`
inside the exact `8831869a` image with `--gpus all --ipc=host --network=none`.
Only the verified policy and audit directory were mounted. `docker wait`
reported container exit code 1. The receipt is preserved locally at
`runs/netplay/direct16/qualification-16.json` and on G4 under
`/var/lib/hal-netplay/capacity16/`. Start-to-finish time was 261.9 seconds,
including cold compilation.

The configuration limit is sixteen; the validated live capacity is one.
Do not set the production runner to sixteen until the inference path passes
both profiles, sixteen accounts exist, Cloudflare traffic limits are resolved,
and actual simultaneous Dolphin load is tested. Gameplay delays were not
relaxed to pass this check.

## Restored service and live connection check

The runner restarted with the NVIDIA video libraries and passed both
one-slot profiles:

| Network delay | Prediction p95 | Prediction p99 | Allowance |
| --- | ---: | ---: | ---: |
| 2 frames | 5.532 ms | 5.727 ms | 16.667 ms |
| 3 frames | 5.015 ms | 5.129 ms | 16.667 ms |

The full receipt, including image revision, bundle and checkpoint hashes,
sampling seeds, runtime versions, and all samples, is
`runs/netplay/direct16/qualification-1.json`. These are operational
qualification runs, not a controlled performance treatment comparison.

`uv run python runs/netplay/public-relaunch/operations.py resume` succeeded.
The local test then queued Cody Fox (`IBDW#0`), advantage 120, temperature 1,
and network delay 2. It connected from the owner's existing peer account
`CRYO#610` to `HAL#647`, sent neutral inputs for 1,800 frames, and canceled
its own reservation. It did not run the ranked player or contact x_pilot.

The command was:

```sh
APPIMAGE_EXTRACT_AND_RUN=1 xvfb-run -a uv run python runs/netplay/direct16/smoke.py > runs/netplay/direct16/smoke.log 2>&1
```

The first attempt failed in the local test harness. Python 3.14's forkserver
reimported its unguarded main code; the extra reservation request correctly
returned HTTP 409. The original reservation was canceled with HTTP 200.
Adding a main guard and the existing libmelee test convention of plain fork
fixed the harness. The second attempt passed and its cancellation returned
HTTP 200.

Measured results:

- 1,800 frames in 30.018972 seconds: **59.962 FPS**.
- During play, G4 reported 59.964 FPS, prediction round-trip p95 7.350 ms,
  and model p95 6.018 ms.
- The runner recorded one missed prediction deadline and four neutral
  fallback frames. There were no prefix mismatches or transport corrections.
- OBS reported 60 FPS, zero network drops, zero encoder drops, and zero
  congestion. Its three render skips occurred before the game; the count
  did not increase during the observed game.
- OBS selected only the `Dolphin / AppRun.wrapped` render window, with
  cursor and border disabled. Both a menu screenshot and gameplay screenshot
  were checked. The encoded output remains 1920 by 1080 at 60 FPS.
- The deliberate disconnect ended as no contest. G4 retained the 698,707-byte
  replay at `/var/lib/hal-netplay/replays/slot-0/Game_20261001T045018.slp`.
  This check does not claim a completed competitive game or an R2 upload of
  that unfinished test replay.

Evidence: `smoke-result.json`, `smoke-game-stream.json`, `smoke.png`, and
`smoke-game.png` under the local evidence directory. An initial OBS
inspection called a method that exists only on the studio owner. Repeating
the inspection with the public `GetStats` and `GetStreamStatus` requests
succeeded. A local `uv` JSON-summary command hit the read-only cache sandbox;
plain `python3` read the saved receipt successfully.

Final checks succeeded:

- Remote `curl --fail --silent --show-error http://127.0.0.1:9101/healthz`
  returned `ok`.
- `docker ps` showed only the runner and health containers running.
- Public `GET /` returned HTTP 200.
- Public `GET /v1/options` returned HTTP 200 with 53 visible player/rank
  choices.
- Public `GET /v1/capacity` returned HTTP 200 with capacity 1, healthy slots
  1, service `ready`, active 0, and queued 0. The receipt is
  `final-capacity.json`.
- The stream reported active output after the game. The site is
  [20xx.xyz](https://20xx.xyz); video is
  [hal_20xx on Twitch](https://www.twitch.tv/hal_20xx).

Sixteen concurrent matches remain blocked. The next work is to profile the
sixteen-stream prediction path, verify accelerated rendering for every
additional slot, provide fifteen more bot accounts, confirm Cloudflare's
traffic allowance, and then test the full concurrent workload. No extra VM,
GPU, account upload, plan upgrade, or Git push was performed.

The documentation commit skipped Ruff format, Ruff check, and ty hooks
because it changed no checked source files. The required full checks above
had already passed. The worktree is clean after the deployment commits.
