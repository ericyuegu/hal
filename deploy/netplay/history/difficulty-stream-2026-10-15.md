# Difficulty and continuous stream release — October 15, 2026

The owner requested Easy/Medium difficulty on the existing 059 checkpoint,
a stream layout based on the hillclimbing article, continuous video through
menus and rematches, review, and deployment to the existing GCE host.
This request supersedes the October 3 activation hold.

Final source: `ba6ca2e3cd7b1c9abb1394c431cd4be59869bbd4`.
Base release source: `a25f9225f2d8611e78bc293f536b327bdc4f290a`.

Easy sends desired return 20. Medium sends desired return 120. Keyboard shortcuts
`[` and `]` select those exact values. The browser tool uses the same two values.
The numerical API and policy artifact contracts remain unchanged.

OBS uses native color and text sources with the reference page's white, dark ink,
and green palette. Live Dolphin capture stays above the waiting background.
The stream grant belongs to the host session, so rematches do not restart OBS.
The sidebar shows connection status, selected player, character, and game number.
At the owner’s follow-up request, difficulty is omitted from the stream overlay. Stale slot health still clears game labels. Ranked layout is unchanged.

## Validation

Commands used `UV_CACHE_DIR=/tmp/hal-uv-cache` because the sandbox cannot write
the default uv cache. Socket, emulator, and Worker runtime checks ran with the
required host access.

| Command | Final result |
| --- | --- |
| `uv run ruff format --check .` | Passed; 327 files. |
| `uv run ruff check .` | Passed. |
| `uv run ty check --python-version 3.14 --error-on-warning hal experiments/059_muon_action_sequence.py scripts` | Passed; zero diagnostics. |
| `uv run pytest -q -m 'not integration'` | 1,840 passed, eight skipped, 25 deselected; final rerun 162.76 seconds. |
| `uv run pytest -q tests/test_netplay_obs.py tests/test_netplay_stream.py` | 21 passed. |
| `HAL_REQUIRE_INTEGRATION=1 uv run pytest -q tests/test_roundtrip.py tests/test_session_cleanup.py -m integration` | Eight passed, six deselected; no skips; 61.18 seconds. |
| `npm test` in `web/netplay-api` | 103 passed across 13 files. |
| `npm run typecheck` in `web/netplay-api` | Passed. |
| `npm test` in `web/netplay` | One difficulty mapping regression passed. |
| `npm run lint` in `web/netplay` | Passed. |
| `npx oxlint components/sentence.tsx` in `web/netplay` | Passed. |
| `npx tsc --noEmit` in `web/netplay` | Passed. |
| `npm run build` in `web/netplay` | Passed; static routes rendered. |
| `npx wrangler deploy --config dist/server/wrangler.json --dry-run` | Passed. |
| `git diff --check` and source review | Passed. |

The eight non-integration skips were six tests in `test_policy_world_schema.py`
whose optional local v7 data subset is absent, and two explicitly gated production
GPU tests in `test_netplay_hardware.py`. Separate `-rs` checks confirmed these
reasons. The idle-runner fault integration test was also inspected with `-rs`;
its opt-in production-GPU gate skipped it. It was not in the non-integration count.

The initial focused run found three expectations for the old overlay behavior
and one socket sandbox failure. Expectations were updated for the requested
behavior; the host-access rerun passed. An initial type check found an optional
float passed to an integer-key dictionary; direct comparisons fixed it. The first
new toggle lint found a group-role accessibility issue and a `never` value in an
error template; semantic fieldset markup and a fixed error message corrected them.
An extra lint of all components also found the pre-existing label-control warning
in `components/ui/label.tsx`; that unrelated component was left unchanged.

Warnings remain for Torch's Python 3.14 JIT deprecation, uncompiled flex attention
in a test, direct websocket connection ownership, and forking a process with
multiple threads. These warnings did not fail the tests.

## Deployment preparation

The original GCE instance was confirmed terminated, with its original ID
`8731743256809856141` and retained disk. No replacement VM or disk was created.
The existing image registry and gcloud credential helper were verified.

The encrypted `cloudflare-api` token is valid but lacks Worker access. The existing
Wrangler OAuth login has Worker write access to account
`988a9ae658afe0b81fbd3af9e86eb7de` and was used for deployment. The R2 account ID
is different and must not be used as the Worker account ID.

Website version: `f4b57671-fcd4-47a7-ba9a-7054ed9b153a`.
API version: `f4d45a2c-9bdf-464c-a9f8-c53a7c80ed2d`.
API maintenance was disabled to restore admin access, then admissions were paused.
An initial post-deploy pause used the wrong working directory; the correct
repository-root invocation succeeded before any runner started.

Automatic approval review rejected re-uploading the local Slippi account JSON.
Read-only R2 HEAD checks then proved that the exact policy and account objects
already existed, including matching SHA-256 metadata. Only their object references
were restored with `AdminClient.put_policy` and `put_accounts`. No account file
or Slippi authentication fields were uploaded. The queue was confirmed paused,
with the expected policy and one account.

Policy bundle SHA-256:
`0ff1daf80caa36a94a713c4ccba9223db8d7ba7c1379b5865bbc40b8a8c2f3ec`.
Account SHA-256:
`9af49baaaee4f9767ec3dfb7db4fc8904e54056643730fb1b51cca54373d61ae`.
The local bundle passed `policy_config_for`, which validates the 059 artifact and
vocabulary. No model weights, sampling temperature, delay, or action schedule changed.

## Scope limits

The game settings in the stream state are sampled when each game starts.
The owner removed the difficulty label after reviewing its delayed publication.
The play-page toggle still changes return conditioning during the current game.
The release does not claim eight simultaneous human games or a human rematch test.

## Native OBS layout check

The exact release image imported the new installed overlay and difficulty code.
A native OBS 30.2.3 preview rendered the idle and rematch states at 1920×1080.
Both images were inspected: text fits, status remains visible, and the absent-window
background is light. Images are under `runs/netplay/release-20261015/`.
The temporary preview's first retry loop ended OBS too early; the second attempt
identified the missing X11 window manager. Starting Openbox, as the production
runner already does, completed both screenshots. No preview connected to Twitch.

The deployed API still has the existing `ADMIN_TOKEN_SHA256`,
`RUNNER_TOKEN_SHA256`, and `TWITCH_STREAM_KEY` secret bindings.
Source review also found that the opt-in hardware test still expects two timing
profiles and a 16.667 ms p99 limit, while the maintained runner declares one
33.333 ms delay-2 profile. This unrelated test was not changed. Live readiness
must use the runner's current complete-path qualification and its recorded budget,
not those stale assertions.

## Image release

`docker build --file deploy/netplay/Dockerfile --build-arg
HAL_GIT_SHA=a25f9225f2d8611e78bc293f536b327bdc4f290a --tag
us-west1-docker.pkg.dev/centering-star-502613-k3/hal-netplay/hal-netplay-runner:a25f9225f2d8611e78bc293f536b327bdc4f290a .`
completed. The installed image passed an import/label smoke check.
The existing gcloud Docker helper pushed the image to the existing registry.

Registry digest:
`sha256:90cc4eb3489e26610480cb59de164b791dad302f7b5e95e37182e799b647ab81`.

`gcloud compute instances add-metadata` updated the existing stopped VM's image,
source SHA, eight-slot count, and `startup-script` from `gce-startup.sh`.
The VM start was requested only after the registry push completed at about
21:04 UTC. The retained instance and disk were reused.


## Final overlay revision

The owner requested removal of difficulty from the overlay and a preview.
`overlay_text` now emits only activity, selected player, character, and game number.
Both return values have focused regressions that require the same label output.
The revised native OBS preview was rendered and shown to the owner. The layout
code is unchanged from the validated base image.

The code-only revision used the exact full-release digest as its base and
reinstalled HAL with `--no-deps`, preserving the verified environment:

```dockerfile
FROM us-west1-docker.pkg.dev/centering-star-502613-k3/hal-netplay/hal-netplay-runner@sha256:90cc4eb3489e26610480cb59de164b791dad302f7b5e95e37182e799b647ab81
COPY hal /opt/hal/hal
RUN uv pip install --python /opt/venv/bin/python --no-deps /opt/hal
ARG HAL_GIT_SHA
RUN test -n "$HAL_GIT_SHA"
ENV HAL_GIT_SHA="$HAL_GIT_SHA"
LABEL org.opencontainers.image.revision="$HAL_GIT_SHA"
```

`docker build --file /tmp/hal-overlay-release.Dockerfile --build-arg
HAL_GIT_SHA=ba6ca2e3cd7b1c9abb1394c431cd4be59869bbd4 --tag
us-west1-docker.pkg.dev/centering-star-502613-k3/hal-netplay/hal-netplay-runner:ba6ca2e3cd7b1c9abb1394c431cd4be59869bbd4 .`
passed; the final image was pushed. This recipe is also saved with local evidence.
A comparison of both images confirmed identical Python and all 184 installed
package versions. An import check from `/tmp` forced use of the installed package.
Its `stream.py` hash matches the committed source:
`b2fd20143ec25460f667ec992c56bcd31b6d445259bcb65e7654c9eb8c9e7f3b`.
The check required the exact output `Playing · Master rank · Fox · Game 1`.
Ruff format/check, ty, the 21 focused stream tests, and the complete non-integration
suite passed again after the revision. Frontend and Worker sources were unchanged
from their validated deployments; emulator stepping was unchanged.

Final registry digest:
`sha256:b6a76e60cdd34a7bc34ccf2aaf43a2fd4e3bfe62fb7cce50751c0b610b593286`.

## GCE activation result

Three `gcloud compute instances start hal-netplay-g4 --project
centering-star-502613-k3 --zone us-west1-a --quiet` attempts failed with
`ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS`, reporting no available
`g4-standard-48` / one `nvidia-rtx-pro-6000` in that zone. The final Google API
check still reports the original instance `8731743256809856141` as `TERMINATED`.
The third attempt followed the metadata update to the final overlay revision.
No new VM, disk, or snapshot was created, and no guest image was installed.

The website and API are deployed, the final runner image is pushed, policy and
account references are restored, and the VM metadata is ready. Admissions remain
paused and Twitch is offline. Live NVIDIA checks, eight-stream inference
qualification, OBS/NVENC measurements, and a human rematch check could not run.
Do not resume admissions until the runner starts and passes readiness.

The neighboring `us-west1-b` zone supports the same machine type and GPU; this
was verified with `gcloud compute machine-types describe`. That confirms product
support, not available stock. Creating a second VM and boot disk would change the
resource plan, so the owner was asked before that action. The existing stopped
VM and retained disk must remain intact during any alternative activation.

The owner then explicitly authorized trying any region with GPU capacity.
The first alternative is `hal-netplay-g4-b` in `us-west1-b`, using the same
`g4-standard-48`, one RTX PRO 6000, eight slots, final image, existing service
account, and Secret Manager reference. The original VM and disk remain intact.

## Training and evaluation shutdown authorization

The owner explicitly requested stopping the existing eight-RTX-6000 training job,
then authorized stopping evaluation containers too, with only gameplay running.
Google API and live GPU/process inspection identified `hal-o62-dev` in `us-east1-b`,
instance `2183030840477497619`, label `workload=training`, as an eight-GPU
`g4-standard-384`. Its `hal-job` container was running
`torchrun --standalone --nproc-per-node=8 experiments/062_production_bc_action_sequence.py train --defer-gameplay-evaluations --comment production`.
All eight GPUs were RTX PRO 6000 Blackwell and at 100% utilization.
The local production `latest.pt` was last modified at 21:12:14 UTC, size
1,579,299,676 bytes. The generic `/var/log/hal-training.log` path did not exist;
checkpoint metadata was captured from the running container instead.

`gcloud compute instances stop` was requested for `hal-o62-dev` in `us-east1-b`
and the one-GPU `hal-o62-eval-rtx` in `us-west4-a`. These are stops, not VM/disk
deletions. The evaluation stop completed. The background capacity-search loop
was terminated and cloud state reconciled. Its transient `STAGING` server in
`us-east1-b` rolled back after another capacity failure; no new VM survived it.
A new gameplay allocation was then requested in the evaluator's freed zone.

Both requested stops completed. A final project-wide active-instance query showed
only `hal-netplay-g4-1015-us-west4-a`, one `g4-standard-48` / RTX PRO 6000.
The stopped trainer's production `latest.pt` was verified in R2 by a read-only
HEAD: 1,579,299,676 bytes, last modified 21:12:32 UTC, matching the local file size.
The key is recorded in local evidence; no checkpoint was deleted.

After the evaluation VM released its GPU, the same-zone gameplay allocation
succeeded in `us-west4-a`: instance `7185265734144818248`, address `34.125.176.210`.
The fresh 100 GB boot disk expanded automatically; `/` reported 96 GiB with 79 GiB
available before the image pull. GPU driver: `580.178.04`. The NVIDIA display
service became active. The final release image then began downloading/unpacking.
The original Oregon VM and all training/evaluation disks remain intact.

Regional attempts before that success failed for capacity in `us-west1-b`,
`us-west1-c`, `us-west4-a/b/c`, `us-central1-b/c/f`, `us-east4-b`, and `us-east1-b`.
`us-west3-a` and `us-east4-c` failed for GPU-family quota. The documented Cloud
Quotas REST API confirmed RTX PRO 6000 quotas of 32 in `us-west1`, `us-west4`,
`us-east1`, and `us-east5`. The CLI's beta quota component was absent; it was not
installed. Failed creation attempts rolled back their transient VM allocations.

## Guest and live stream verification

The guest installed the exact final digest. Startup exited successfully at
21:21:02 UTC. The renderer preflight used NVIDIA Vulkan; the display and runner
services are active. The runner downloaded and verified the recorded ISO,
Slippi Online 3.6.4, policy SHA, and account SHA. The actual guest's installed
`stream.py` hash matches the final committed source.

An authenticated OBS connection captured the actual live scene at 1920×1080.
It was visually checked: white/green layout, readable waiting state, no difficulty,
and a light absent-window background. Screenshot evidence is
`runs/netplay/release-20261015/live-overlay.png`.
OBS reported version 30.2.3, active FPS 60.0000024, active Twitch output, no
reconnection, zero congestion, and zero network-dropped stream frames.
Initial render/encode skip counters were seven/five during startup; they were
unchanged in the next sample at 84 seconds. Render time was 0.112 ms in that sample.
These are idle-stream measurements, not an emulator gameplay FPS measurement.
The model's horizon-8/prefix-4 qualification began at 21:21:49 UTC while streaming.


## Final activation

The runner completed full-load qualification and became ready at 21:24:51 UTC.
The recorded source, policy, and checkpoint identities match this release.
Python 3.14.3; Torch 2.11.0+cu130; RTX PRO 6000 Blackwell; driver 580.178.04.
Boot image: `common-cu129-ubuntu-2404-nvidia-580-v20261001`, image ID
`1749640761673227869`; new 100 GB Hyperdisk Balanced boot disk.

The one delay-2 profile measured 200 complete-path samples with eight concurrent
request streams, including IPC, batching, validation, and model execution:

| p50 | p95 | p99 | Allowed inference time |
| --- | --- | --- | --- |
| 13.673 ms | 15.003 ms | 15.624 ms | 33.333 ms |

The schedule remains delay 2, inference allowance 2, fixed prefix 4, replan 4,
and horizon 8. This is a fresh qualification, not a controlled before/after
performance comparison. All eight slots reported healthy and the host health
endpoint returned `ok`. OBS logs confirmed the NVIDIA OpenGL adapter and active
`obs-nvenc` H.264 encoder. Network-dropped frames were zero; output remained
active without congestion or reconnection. Startup render/encode skip counters
remained seven/five through the later three-minute sample.

The admin status check required the expected policy SHA, eight healthy slots,
and service state `ready` before `AdminClient.set_paused(False)`. It then verified
admissions open, eight healthy slots, and exactly one active runner session.
The final project-wide GCE inventory showed only
`hal-netplay-g4-1015-us-west4-a` running, with exactly one GPU. Training and
evaluation VMs are terminated; their disks and the original Oregon disk remain.

The public `/v1/capacity` and `/v1/options` checks confirm readiness and return
support. Live human gameplay, audio during a game, a human rematch, and eight
simultaneous human games remain untested. Existing focused visibility tests cover
capture through game/rematch status changes; the deployed stream remains active
while idle. Local emulator roundtrip/cleanup tests passed separately.

Evidence is under `runs/netplay/release-20261015/`: the final budget, fresh health,
stream statistics, package environment comparison, preview screenshots, queue
activation status, and one-active-VM inventory. This directory contains operator
state and is not committed or uploaded as public source.
