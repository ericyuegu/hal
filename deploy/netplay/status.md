# Netplay deployment status

Updated October 17, 2026. The live overlay and idle self-play are active.
Eight human slots are healthy, both delay budgets qualified at p99 15.924 ms,
and admissions are open. The existing Cloudflare human queue and stream lease
remain in use; demos create no human reservations. A temporary internal budget
allowance expires October 18 at 00:00 UTC. Training and evaluation VMs remain
stopped. See the [acceptance record](history/live-overlay-self-play-2026-10-16.md).

| Component | Last verified state |
| --- | --- |
| Website | `hal-netplay-web`, route `20xx.xyz/*`; version `fad5795b-68c8-4206-b429-cc65bc51ff15`. Easy = return 20; Medium = return 120. |
| API | `hal-netplay-api`, route `20xx.xyz/v1/*`; version `b150c6c2-dc9e-42d1-a0ad-b304c9a71b92`; maintenance off, admissions open; 36,088 / 37,000 internal credits at activation. Normal 35,000 limit resumes October 18, 00:00 UTC. |
| Queue | Protocol 4, schema 7, `global-v7`. Verified 059 policy and HAL#9000 account references restored; one active runner session; no reservations at verification. |
| Gameplay G4 | `hal-netplay-g4-1015-us-west4-a`, project `centering-star-502613-k3`, zone `us-west4-a`; one `g4-standard-48`, one RTX PRO 6000, eight-slot target, fresh 100 GB boot disk. No MIG. |
| Gameplay state | `RUNNING`; instance ID `7185265734144818248`; address `34.125.176.210`. NVIDIA driver `580.178.04`, Vulkan preflight passed. |
| Original G4 | `hal-netplay-g4` in `us-west1-a`, instance `8731743256809856141`, remains `TERMINATED` with its 100 GB disk preserved. Three starts failed before the regional move. |
| Original disk image | Source `6c79d130582f52495f226d261a48877049981175`, protocol 2. Its metadata points to the new release; do not run the old guest image against the current queue. |
| Installed release | Matching runtime and independent overlay source `37e7ec0d2b2f3188197b219babd585435997a5ef`, protocol 4, Docker init enabled (private stream-state version 2). |
| Twitch | Live at `hal_20xx`; OBS 30.2.3, 1080p60 NVENC, native game image without pillarboxes, exact header/footer alignment, character sprite, applied difficulty, physical controller inputs, and State value. Actual scene screenshot and output status checked. |
| Training/evaluation | `hal-o62-dev` (eight GPUs, `us-east1-b`) and `hal-o62-eval-rtx` (one GPU, `us-west4-a`) are stopped. All other old HAL VMs were already stopped. Only gameplay remains active. |

Registry prefix:
`us-west1-docker.pkg.dev/centering-star-502613-k3/hal-netplay/hal-netplay-runner`.
Current runtime digest:
`sha256:9a2c7c0b1b68703f65dbabebb1ef55e6f3f2e7df83e15d402179aa5d2a0411f9`.
It derives from the prior deployed image and updates HAL source without
changing inference Python dependencies. Overlay source and fonts
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

Both delay budgets passed full-load qualification with 200 samples: p50
10.690 ms, p95 15.304 ms, p99 15.924 ms. Both share prefix 4, replan 4, horizon 8.
Delay 2 allows 33.333 ms; delay 3 allows 16.667 ms at all eight streams.
The final public canary released demo resources within 403 ms from POST start
and reached the connection menu in 3.431 seconds. It was canceled without a peer;
self-play resumed. The monitor observed at most one Dolphin and one unchanged
OBS process throughout the handoff. Docker init reaps adopted AppImage children.

With the final controller and state-value display active, self-play measured
59.940 median FPS; the final sample had frame-interval p95 18.021 ms and client
round-trip p95 7.247 ms. OBS uses NVIDIA OpenGL and NVENC H.264 at 1080p60.
All 79 final handoff samples had active output, zero output skips, zero congestion,
and no reconnect. The model process was unchanged. Overlay activation restarted
OBS only. The later Sheik correction restarted matched services; maintenance
continuity is distinct from handoff continuity. Sheik now retains the required
A hold through stage loading; the character identity check remains enabled.

Keep the original Oregon VM and all training/evaluation VMs stopped. Preserve
their disks. The stopped production trainer's latest checkpoint was verified in
R2 at 21:12:32 UTC. Do not restart training or evaluation compute without a new
owner instruction. Only the one-GPU gameplay server should remain active.

A connected human rematch and eight simultaneous human games were not tested.
Performance above is two-stream self-play; the built-in qualification tests
all eight inference streams. Reboot and managed-instance replacement remain
unverified.
