# G4 teardown — October 1, 2026 UTC

The owner requested verification of all R2 replay uploads, then deletion of
`hal-netplay-g4`. The scope includes its single RTX PRO 6000 Blackwell GPU
and its 100 GB boot disk. Both were confirmed deleted at 01:12:09 UTC.
Ranked and Twitch are offline. The page, API Worker, R2 bucket, registry image,
and Secret Manager entry remain available.

## Replay verification

The Ranked player had already stopped cleanly at 00:58:55 UTC, with exit
code zero and OOMKilled=false. This teardown sent no player stop signal.
The latest run completed 23 games. The public runner was inactive and the
independent monitor had exited with the player.

| Local replay group | Files |
| --- | ---: |
| Supervised Ranked prototype | 8 |
| Ranked run `20260930T193913.999585Z` | 18 |
| Ranked run `20260930T203153.297761Z` | 24 |
| Ranked run `20260930T220118.832519Z` | 32 |
| Ranked run `20260930T235630.598726Z` | 23 |
| Unfinished direct-play replay | 1 |
| Total | 106 |

All 105 Ranked replays already had upload receipts. The audit recomputed
each local replay hash and checked its game record and receipt. It then
downloaded every R2 replay and checked the actual bytes, size, SHA-256
metadata, and ETag. Each small result JSON also passed its hash and replay
identity checks. Existing uploads were not duplicated.

The Ranked replays remain in bucket `hal` under:

```text
netplay/v1/replays/2026/09/30/HAL#647/ranked-*/game-*.slp
netplay/v1/replays/2026/10/01/HAL#647/ranked-*/game-*.slp
```

The remaining 61,451-byte file is an unfinished direct-play replay from
September 30 at 00:17 UTC. Peppi found a direct match ID, no end event, and
no metadata. It had no matching R2 object. Its original bytes and a small
recovery note were uploaded and downloaded again for exact verification:

```text
netplay/recovered/hal-netplay-g4/751bc108c9a4c9acfd5b9282543d0f0f7f651915e28367078d81eb83f74121c9.slp
netplay/recovered/hal-netplay-g4/751bc108c9a4c9acfd5b9282543d0f0f7f651915e28367078d81eb83f74121c9.json
```

The note marks this replay incomplete. It is not a completed-game result.
No replay expiration rule was added. Total verified replay bytes are
280,084,675, excluding the small JSON objects.

## Local backup

`runs/netplay/g4-teardown/runtime-records.tar.gz` preserves all 106 replays,
run manifests, game results, upload receipts, status snapshots, and the new
FPS and memory history. Every archived member passed its source SHA-256
and size check: 341 files in a 75,400,068-byte archive.

Archive SHA-256:

```text
6822a7b0a8d92d46d099f23d6952970a6edb5848540fe61ea76395ba1ea23f45
```

`r2-verification.json` maps each replay to its verified object key.
`local-backup-verification.json` records the archive checks. Both are in the
same local directory. The archive excludes credentials, stream profiles,
screenshots, video, model assets, and the Docker image. The host-only image
and source mounts are removed with the disk; a future host must be rebuilt
from the deployment source. The local backup is not uploaded to R2.

## Commands and results

- `cat AGENTS.md`, the teardown script, uploader modules, and deployment
  notes established the stop and upload contracts. `git status` was clean.
  An initial `rg` named absent `hal/scripts/ranked.py` and exited 2; the
  corrected search found `hal/eval/ranked.py` and `scripts/play_ranked.py`.
  A search for signal handling in the CLI returned no matches because the
  maintained behavior is in `hal/eval/ranked.py`.
- Read-only SSH `docker ps`, `docker inspect`, `docker logs`, and Python
  inventory checks confirmed the clean stop and all local replay paths.
  The scan checked `/root`, `/home`, `/tmp`, `/opt`, the HAL cache, and HAL
  state. It excluded build caches. All replay files were in HAL state.
- `gcloud compute instances describe hal-netplay-g4 --project=centering-star-502613-k3 --zone=us-west1-a`
  confirmed instance ID `4269454647821964449`, one GPU, and one 100 GB disk
  with automatic deletion enabled. The selected fields are saved in
  `instance-before.json`.
- `rclone version` and `rclone check --help` inspected available checks.
  Verification used the existing Python R2 client instead. Local `ls`
  confirmed the older prototype recording backup.
- `gcloud compute instances delete --help` documented disk deletion behavior.
  Its sandboxed help call warned that the normal debug log directory was
  read-only; the help command returned successfully.
- Local Python wrote two one-time operator scripts, retained under
  `runs/netplay/g4-teardown/`. SSH `mkdir` and `tee` staged them on G4.
- A temporary Docker container ran `verify-replays.py` with no GPU, the
  committed HAL R2 client, read-only replay files, and only R2 credentials.
  It downloaded and verified all 105 Ranked replay/result pairs. The first
  report correctly marked the unrecorded direct replay as missing from R2.
- A network-disabled Docker Peppi check confirmed that the unrecorded file
  was incomplete. `recover-replay.py` uploaded only this replay and its note,
  verified both downloads, and marked all 106 files verified. Both temporary
  containers exited successfully and were removed automatically.
- SSH Python and tar copied the selected records into the local archive.
  Local Python verified all member hashes and the compressed archive hash.
  SSH copied the final R2 report and rechecked that both players were stopped.
- `gcloud compute addresses list --project=centering-star-502613-k3 --filter='address=136.67.57.119'`
  returned no reserved address. Gcloud warned that the empty result had no
  `address` filter key. The VM used an ephemeral external IP.
- Local Python required successful R2 and archive verification and the exact
  instance identity before the deletion command.
- The existing teardown script was run without `--force`:

  ```sh
  deploy/netplay/gce-down.sh hal-netplay-g4 \
    --project centering-star-502613-k3 --zone us-west1-a
  ```

  It stops the already inactive public runner and deletes the VM with its
  auto-delete boot disk. Its output is saved in `delete.log`.
- A filtered `gcloud compute instances list` observed `STOPPING` during
  deletion. An unprivileged local `ps` check saw only its own sandbox and
  could not inspect the elevated teardown process.
- No application code changed. Ruff, ty, pytest, and npm suites were not
  repeated for this operational teardown and documentation update.

- The teardown script exited zero and reported the VM deleted. Final filtered
  `gcloud compute instances list` and `gcloud compute disks list` both returned
  empty arrays. The empty project results produced harmless missing-filter-key
  warnings. Local Python asserted both results and wrote `teardown-result.json`.
- `git diff --check` passed. Git staged only the three deployment documents
  and committed the teardown record on `netplay-edge-queue`. Ruff and ty commit
  hooks skipped because no Python files changed. No Git push was made.
