# Netplay deployment status

Updated October 15, 2026. Website restyle and the independent overlay service are
live. The one-GPU gameplay VM in `us-west4-a` has eight healthy slots and admissions
are open. Physical delays 2 and 3 are offered; shared-profile inference p99 passed
at 15.177 ms against delay 3's 16.667 ms limit. An overlay-only source-bundle
rollout preserved the gameplay container, process, and readiness. All training
and evaluation VMs remain stopped. See the [overlay release record](history/overlay-service-2026-10-15.md)
and the [initial release record](history/difficulty-stream-2026-10-15.md).

| Component | Last verified state |
| --- | --- |
| Website | `hal-netplay-web`, route `20xx.xyz/*`; version `fad5795b-68c8-4206-b429-cc65bc51ff15`. Easy = return 20; Medium = return 120. |
| API | `hal-netplay-api`, route `20xx.xyz/v1/*`; version `f4d45a2c-9bdf-464c-a9f8-c53a7c80ed2d`; maintenance off, admissions open. |
| Queue | Protocol 4, schema 7, `global-v7`. Verified 059 policy and HAL#9000 account references restored; one active runner session; no reservations at verification. |
| Gameplay G4 | `hal-netplay-g4-1015-us-west4-a`, project `centering-star-502613-k3`, zone `us-west4-a`; one `g4-standard-48`, one RTX PRO 6000, eight-slot target, fresh 100 GB boot disk. No MIG. |
| Gameplay state | `RUNNING`; instance ID `7185265734144818248`; address `34.125.176.210`. NVIDIA driver `580.178.04`, Vulkan preflight passed. |
| Original G4 | `hal-netplay-g4` in `us-west1-a`, instance `8731743256809856141`, remains `TERMINATED` with its 100 GB disk preserved. Three starts failed before the regional move. |
| Original disk image | Source `6c79d130582f52495f226d261a48877049981175`, protocol 2. Its metadata points to the new release; do not run the old guest image against the current queue. |
| Installed release | Runtime source `ffed25c3c944d55ed37bb5564d8956418171f193`, protocol 4; independent overlay release `b5b8dd08a910d2ae136633d486197fa564259695`. |
| Twitch | Live at `hal_20xx`; OBS 30.2.3, 1080p60 NVENC, new layout, no difficulty label. Actual scene screenshot and output status checked. |
| Training/evaluation | `hal-o62-dev` (eight GPUs, `us-east1-b`) and `hal-o62-eval-rtx` (one GPU, `us-west4-a`) are stopped. All other old HAL VMs were already stopped. Only gameplay remains active. |

Registry prefix:
`us-west1-docker.pkg.dev/centering-star-502613-k3/hal-netplay/hal-netplay-runner`.
Current runtime digest:
`sha256:32fddfcd3085cb512f4f1e55a04f4c27471d862a0b27bd20ab12304731fd0652`.
It derives from the prior deployed `ba6ca2e3` image and adds the new fonts and HAL
source without changing inference Python dependencies. Overlay source and fonts
are mounted from `/var/lib/hal-overlay/current`; update them with
`deploy/netplay/gce-overlay.sh`. The host service bootstrap is supplied as VM
metadata alongside the startup script, independent of the runtime image.

The former ranked VM used the same name but instance ID `4269454647821964449`.
Its VM and disk were deleted October 1. That deletion does not describe the
replacement host's retained disk. See [teardown](history/g4-teardown.md) and the later
[shutdown](history/g4-stop-2026-10-02.md).

## Assets and credentials

- Current Slippi code: **HAL#9000**. Historical HAL#647 replay paths stay unchanged.
- Policy bundle SHA-256: `0ff1daf80caa36a94a713c4ccba9223db8d7ba7c1379b5865bbc40b8a8c2f3ec`.
- Account SHA-256: `9af49baaaee4f9767ec3dfb7db4fc8904e54056643730fb1b51cca54373d61ae`.
- R2 bucket `hal`: content-addressed policy/account objects and replay recordings.
  `hal/fixtures.py` owns the ISO and official Slippi Online 3.6.4 download.
- Secret Manager `hal-netplay-runner-env`: runner environment, read by the VM's
  service account. Operator credentials remain encrypted with systemd-creds in
  the original local directory. No secret values moved during integration.
- Historical teardown verified 105 ranked replays and one unfinished direct replay
  in R2. This is not a fresh audit of every later upload. No expiration rule is
  authorized; retained replay storage can exceed the R2 free allowance.

## Verification and operating boundaries

The built-in full-load qualification passed with 200 samples: p50 10.164 ms,
p95 14.767 ms, p99 15.177 ms. Both delays share prefix 4, replan 4 and horizon 8.
Delay 2 allows two inference frames; delay 3 allows one. Qualification uses
the stricter delay-3 limit of 16.667 ms at all eight streams. All eight slots reported fresh healthy state; the host
health endpoint returned `ok`. Admissions were resumed after these checks.
OBS uses NVIDIA OpenGL and NVENC H.264, at 1080p60. Actual scene and output
status were verified; no network drops, congestion, or reconnection were observed.
Initial startup render/encode skips stopped increasing after warmup.

Keep the original Oregon VM and all training/evaluation VMs stopped. Preserve
their disks. The stopped production trainer's latest checkpoint was verified in
R2 at 21:12:32 UTC. Do not restart training or evaluation compute without a new
owner instruction. Only the one-GPU gameplay server should remain active.

A human rematch and eight simultaneous human games were not tested. No new
emulator FPS claim is made from idle-stream FPS. The opt-in hardware test now
checks each declared allowance; this release used the live runner's built-in full-capacity qualification. Reboot and managed-instance
replacement remain unverified.
