# Netplay Edge Queue — Plan 4: Host Bring-up

**Goal:** Build one immutable runner image and provide one-command Google Cloud G4 creation, graceful deletion, and optional single-VM managed-instance-group recovery without putting secrets or runtime assets in the image or instance metadata.

**Architecture:** `gce-up.sh` creates either one `g4-standard-48` VM or a size-one zonal managed instance group from a GPU-ready Deep Learning VM image. A metadata startup script reads a single dotenv file from Secret Manager through the VM service account, pulls the exact runner image, and installs runner and health services in systemd. `gce-down.sh` asks the runner to drain before deleting the VM or group. A host health endpoint reads the runner status file and lets Google Cloud replace a dead managed instance.

**Tech Stack:** Bash, Docker, systemd, Google Compute Engine G4, Artifact Registry or another OCI registry, Secret Manager, Python 3.14 HTTP server, pytest.

**Spec:** `docs/superpowers/specs/2026-09-27-netplay-edge-queue-design.md`

The requested `superpowers:writing-plans` and `superpowers:subagent-driven-development` skills are not installed in this session. The owner also directed implementation without subagents. This plan follows the established plan format and is executed directly.

## Global Constraints

- Work only in `/home/ericgu/src/hal-edge-queue` on `netplay-edge-queue`.
- Do not create a VM, instance template, managed instance group, firewall rule, health check, secret, or registry image during implementation.
- The image tag is the full Git SHA. The image contains code and dependencies, but no ISO, policy bundle, account JSON, or secret.
- Secret values come only from Secret Manager at VM boot. Instance metadata contains the secret name, project, image URI, Git SHA, and non-secret runner settings.
- The VM service account needs `roles/secretmanager.secretAccessor` on the dotenv secret and registry read access. The launcher does not grant IAM roles.
- The default machine is `g4-standard-48`, with one attached RTX PRO 6000 Blackwell GPU. The default image is the repository's current GPU-ready Deep Learning VM family.
- `gce-down.sh` must stop the runner and allow its drain deadline before deletion. `--force` is the explicit escape hatch after a failed drain.
- A size-one managed instance group uses the same image, metadata, startup script, and service account as a standalone VM.
- Live G4 checks are owner actions because they cost money. Scripts and a complete checklist are the handoff.
- Commit messages have no attribution trailer. Do not push.

## File Structure

```text
deploy/netplay/Dockerfile             runner image labeled with the Git SHA
deploy/netplay/gce-startup.sh         VM boot and systemd setup
deploy/netplay/gce-up.sh              standalone VM or size-one managed group creation
deploy/netplay/gce-down.sh            graceful standalone VM or group removal
deploy/netplay/README.md               setup and live G4 verification checklist
hal/netplay_service/host_health.py     managed-group health endpoint
tests/test_netplay_gce.py              health and shell command tests
```

---

### Task 1: Pin and inspect the runner image

**Files:** `deploy/netplay/Dockerfile`, `tests/test_netplay_gce.py`.

- [ ] **Step 1: Add a test that reads the Dockerfile.** Assert it declares and validates `HAL_GIT_SHA`, labels the image with it, keeps the runner entry point, and does not copy `data`, `fixtures`, a policy bundle, or account JSON.
- [ ] **Step 2: Run `uv run pytest -q tests/test_netplay_gce.py -k dockerfile`.** Expected: the new test fails.
- [ ] **Step 3: Add `ARG HAL_GIT_SHA`, reject an empty value during build, set `org.opencontainers.image.revision`, and expose the SHA to the runner. Keep runtime libraries in the image and runtime assets out.**
- [ ] **Step 4: Run `uv run pytest -q tests/test_netplay_gce.py -k dockerfile`.** Expected: pass.

### Task 2: Add a host health endpoint

**Files:** `hal/netplay_service/host_health.py`, `tests/test_netplay_gce.py`.

**Interface:** `python -m hal.netplay_service.host_health --status-path PATH --host 0.0.0.0 --port 9101 --max-age 10`. `GET /healthz` returns `200` only when the runner status is valid, fresh, and has at least one healthy slot. Other paths return `404`.

- [ ] **Step 1: Test missing, invalid, stale, recovering, and ready status files, plus exact HTTP paths.** Use an injected clock for the status predicate and a local ephemeral port for one endpoint test.
- [ ] **Step 2: Run `uv run pytest -q tests/test_netplay_gce.py -k health`.** Expected: fail before implementation.
- [ ] **Step 3: Implement a small `ThreadingHTTPServer`.** Suppress request logs. Set `Cache-Control: no-store`. Keep status parsing in `hal.netplay_service.health`.
- [ ] **Step 4: Run `uv run pytest -q tests/test_netplay_gce.py -k health`.** Expected: pass.

### Task 3: Boot a runner from Secret Manager

**Files:** `deploy/netplay/gce-startup.sh`, `tests/test_netplay_gce.py`.

**Metadata keys:** `hal-netplay-project`, `hal-netplay-image`, `hal-netplay-git-sha`, `hal-netplay-secret`, `hal-netplay-slots`, `hal-netplay-drain-timeout`.

- [ ] **Step 1: Add static tests for shell syntax, metadata reads, `gcloud secrets versions access`, mode `0600`, GPU checks, the exact image pull, persistent cache/state mounts, `--gpus all`, `--ipc=host`, `NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video`, and both systemd units.**
- [ ] **Step 2: Run `uv run pytest -q tests/test_netplay_gce.py -k startup`.** Expected: fail.
- [ ] **Step 3: Write `gce-startup.sh`.** Refuse a missing Docker daemon, NVIDIA runtime, GPU, metadata value, or secret. Authenticate the registry only when the image host ends in `.pkg.dev`. Write the dotenv secret directly to `/run/hal-netplay/runner.env`. Install and start `hal-netplay-health.service` and `hal-netplay-runner.service`; give runner stop enough time to drain.
- [ ] **Step 4: Run `bash -n deploy/netplay/gce-startup.sh` and `uv run pytest -q tests/test_netplay_gce.py -k startup`.** Expected: pass.

### Task 4: Create and remove a G4 host

**Files:** `deploy/netplay/gce-up.sh`, `deploy/netplay/gce-down.sh`, `tests/test_netplay_gce.py`.

**Commands:**

```bash
deploy/netplay/gce-up.sh NAME --project PROJECT --zone ZONE \
  --image REGISTRY/hal-netplay-runner:GIT_SHA \
  --secret RUNNER_DOTENV --service-account ACCOUNT
deploy/netplay/gce-down.sh NAME --project PROJECT --zone ZONE
```

- [ ] **Step 1: Use a stub `gcloud` to test the rendered standalone commands.** Assert `g4-standard-48`, GPU-ready image family, Hyperdisk, `TERMINATE`, restart on failure, cloud-platform scope, service account, startup metadata, and no secret value.
- [ ] **Step 2: Test that down sends `systemctl stop hal-netplay-runner.service`, waits for the configured drain interval, refuses deletion if that fails, and accepts explicit `--force`.**
- [ ] **Step 3: Run `uv run pytest -q tests/test_netplay_gce.py -k 'standalone or down'`.** Expected: fail.
- [ ] **Step 4: Implement strict option parsing and validation in both scripts.** `gce-up.sh` requires project, image, secret, and service account. It derives and validates the full SHA from the image tag unless `--git-sha` is supplied. It only prints follow-up commands after successful creation.
- [ ] **Step 5: Run `bash -n deploy/netplay/gce-up.sh deploy/netplay/gce-down.sh` and `uv run pytest -q tests/test_netplay_gce.py -k 'standalone or down'`.** Expected: pass.

### Task 5: Add optional managed-instance-group recovery

**Files:** `deploy/netplay/gce-up.sh`, `deploy/netplay/gce-down.sh`, `tests/test_netplay_gce.py`.

- [ ] **Step 1: Test `--managed` with stub `gcloud`.** Up creates an HTTP health check on port 9101, a health-check-source firewall rule, an instance template, and a zonal group of size one with autohealing. Down drains every current group instance before resizing to zero and deleting the group resources.
- [ ] **Step 2: Run `uv run pytest -q tests/test_netplay_gce.py -k managed`.** Expected: fail.
- [ ] **Step 3: Add the managed path.** Use names derived from `NAME`. Set an initial delay that covers fixture download and qualification. Make cleanup commands idempotent when a resource is already absent.
- [ ] **Step 4: Run `uv run pytest -q tests/test_netplay_gce.py`.** Expected: pass.

### Task 6: Document owner setup and live G4 verification

**Files:** `deploy/netplay/README.md`.

- [ ] **Step 1: Document the exact image build and registry push commands, but do not run either push.** The build command passes the full SHA and tags `hal-netplay-runner:<sha>`.
- [ ] **Step 2: Document the dotenv secret fields and service-account permissions.** Do not show secret values or a command that writes them.
- [ ] **Step 3: Add the first-G4 checklist from the spec:** driver and `torch==2.11.0` on `sm_120`; Vulkan; standard driver versus RTX Virtual Workstation; Slippi UDP direct connect; realtime budget at the configured slot count; NVENC next to model work after Plan 5. Include commands and fields for measured results.
- [ ] **Step 4: Mark live G4 verification as not run and owner-gated.** No performance result is invented.

### Task 7: Handoff checks and Plan 4 status

**Files:** `docs/superpowers/specs/2026-09-27-netplay-edge-queue-design.md`, `deploy/netplay/README.md`.

- [ ] **Step 1: Run `uv run ruff format --check .`.** Expected: pass.
- [ ] **Step 2: Run `uv run ruff check .`.** Expected: pass.
- [ ] **Step 3: Run `uv run ty check --python-version 3.14 --error-on-warning hal experiments/059_muon_action_sequence.py scripts`.** Expected: zero diagnostics.
- [ ] **Step 4: Run `uv run pytest -q -m "not integration"`.** Expected: pass.
- [ ] **Step 5: Run `npm test` and `npm run typecheck` in `web/netplay-api`.** Expected: pass.
- [ ] **Step 6: Run `HAL_REQUIRE_INTEGRATION=1 uv run pytest -q tests/test_netplay_queue_integration.py -m integration`.** Expected: pass.
- [ ] **Step 7: Record every result, failure, and skip in `deploy/netplay/README.md`. Update only the Plan 4 row in the spec to `done`. Run `git diff --check`, review the diff, and commit with `git commit -m "Add netplay G4 bring-up"`. Do not push.**
