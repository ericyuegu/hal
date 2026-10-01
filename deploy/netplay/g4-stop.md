# G4 stop — October 1, 2026 UTC

The owner requested a VM stop. The G4 instance and its disk remain available
for a later restart. Direct play, the local test player, and Twitch are stopped.
The Cloudflare page and API remain deployed; queue admissions are paused.

The current game finished at 07:02:42 UTC. Its replay upload completed at
07:02:43 UTC:

```text
netplay/v1/replays/2026/10/01/CRYO#610/a2Tx1-QSb3eD8ge-_ibcICrv/game-02.slp
```

The upload log reported 1,841,704 bytes and ETag
`b4d576d009cc06f3b0837b5ef641f4be`. No pending replay sidecars remained in
the production replay directory. This stop did not repeat a full R2 audit.
The separate account-reuse probe replays remain on disk and in the local
evidence directory.

## Shutdown defect

The slot exited with code zero after draining. The supervisor then reached
`RuntimeError("all netplay slots are unavailable")` and exited with code one.
Systemd marked the runner failed. The log also reported one leaked semaphore
at process shutdown. These defects remain to be fixed before the next release.

The runner's final cleanup still ended its queue session. The admin API
confirmed zero active games, zero queued jobs, zero capacity, and no account
lease. The local player released its reservation with HTTP 200 and exited.

## Commands and results

- `cat AGENTS.md`, `sed` reads of deployment/client/player code, and
  `rg` searches established the shutdown contract. `gce-down.sh` deletes
  the VM, so this stop used the Compute Engine stop command.
- `gcloud compute instances describe hal-netplay-g4 --project centering-star-502613-k3 --zone us-west1-a --format='json(name,status,id,disks.deviceName,disks.autoDelete)'`
  confirmed instance 8731743256809856141 was running.
- Remote `systemctl show`, status-file reads, and `docker logs --tail 6`
  confirmed an active game and the configured graceful stop.
- Local Python loaded the existing encrypted credentials and called
  `AdminClient.set_paused(True)`. Succeeded.
- Local Python checked PID 2006334's command line and sent `SIGTERM`.
  The player stopped after its game. Event/status reads confirmed
  `reservation_released` with HTTP 200 and `state: stopped`.
- Remote `sudo systemctl stop --no-block hal-netplay-runner.service`
  accepted the request. A bounded `systemctl show` loop observed completion
  with `ActiveState=failed` and `Result=exit-code`; see the defect above.
- Remote `journalctl -u hal-netplay-runner.service -n 45 --no-pager`,
  `find /var/lib/hal-netplay/replays -name '*.slp.upload.json'`, and
  `docker ps --format '{{.Names}}'` confirmed the final replay upload,
  no pending uploads, and removal of the runner container. Docker warned
  that its `--time` flag is deprecated; it did not prevent shutdown.
- `gcloud compute instances stop hal-netplay-g4 --project centering-star-502613-k3 --zone us-west1-a --quiet`
  succeeded. A subsequent `instances describe` confirmed `TERMINATED`.
- `AdminClient.status()` confirmed the paused, empty queue and released
  account. `ps -p 2006334,2006318 -o pid,stat,comm` showed neither local
  process. A focused source read identified the supervisor error path.
- `git diff --check` passed. The deployment note was committed.
  Ruff, ty, pytest, Worker tests, and typecheck were skipped because this
  operation changed only deployment documentation. Commit hooks had no
  applicable source files. No runtime fix or deployment was made.
