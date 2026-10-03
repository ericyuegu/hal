# Netplay deployment status

Updated October 3, 2026. Owner instruction: integrate and clean up; **do not
activate G4**. This file owns current deployment state. Other deployment reports
are historical evidence. No live infrastructure changed during integration.

| Component | Last verified state |
| --- | --- |
| Website | `hal-netplay-web`, route `20xx.xyz/*`; version `f18e9cf3-1e35-4c85-adaa-3f0e48028044`, verified October 2. |
| API | `hal-netplay-api`, route `20xx.xyz/v1/*`; version `8358d716-7b37-4e4d-a36b-9af5759337a7`; maintenance on, verified October 2. |
| Queue | Protocol 4, schema 7, `global-v7`. Policy/account initialization and activation are pending. Old objects retain history. |
| G4 | `hal-netplay-g4`, project `centering-star-502613-k3`, zone `us-west1-a`; one `g4-standard-48`, one RTX PRO 6000, eight-slot target, 100 GB retained disk. No MIG. |
| G4 shutdown | Guest accepted poweroff October 2 at about 14:44 UTC. Instance ID `8731743256809856141`; last address `8.229.68.10`. Google API status is unverified because login expired. |
| Last installed image | Source `6c79d130582f52495f226d261a48877049981175`, protocol 2. It cannot serve the deployed protocol-4 queue. |
| Built local image | Source `7b33980d9b239ce158b72e252c96743393c673c1`, protocol 4. Built and import-tested, not pushed or installed. It predates this cleanup; build the next release from the integrated commit. |
| Ranked and Twitch | Last recorded stopped. No restart authorized in this integration. |

Registry prefix:
`us-west1-docker.pkg.dev/centering-star-502613-k3/hal-netplay/hal-netplay-runner`.
The prior installed image digest is
`sha256:b00e4c9911d8183891e777744c47c55316759ba75da8ca0ec62f121430212c4a`.
The prepared local image ID is
`sha256:7a49140137e0321ccfac218425183d9b1a98311154bb226a5950005c63a3856f`.

The former ranked VM used the same name but instance ID `4269454647821964449`.
Its VM and disk were deleted October 1. That deletion does not describe the
replacement host's retained disk. See [teardown](g4-teardown.md) and the later
[shutdown](g4-stop-2026-10-02.md).

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

## Before a future activation

1. Obtain owner authorization and restore `gcloud auth login`. Confirm the retained
   instance and disk; do not create another VM based only on the old address.
2. Build and push an image from the clean integrated commit. Record its full SHA
   and registry digest here. Worker, frontend and runner must all use protocol 4.
3. Keep admissions paused while restoring the exact policy and account references
   to the fresh queue. Check current quota health; an old reset time is not proof.
4. Update the existing VM's image, SHA, eight-slot count and startup-script metadata.
   Do not use `gce-up.sh` to update it: that script creates resources.
5. Start the existing one-GPU host, qualify it, verify NVIDIA rendering and shared
   control, and complete the runbook's live checks before opening admissions.

Do not deploy from an earlier protocol branch or start the old image as a recovery
shortcut. Reboot and managed-instance replacement remain unverified. Eight-stream
inference was measured at 21.148 ms p99; eight simultaneous model games were not.
