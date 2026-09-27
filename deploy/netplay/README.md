# HAL netplay deployment

The runner starts one prepared inference process and one Dolphin worker per Slippi account. Workers using the same checkpoint share model weights. The service admits matches only after it prepares and checks both delay profiles. Publishing a deployment remains a separate step after hardware qualification.

## Local run

Install `uv`, `npm`, and `xvfb-run`. Copy the environment template, set the policy, ISO, emulator, account, and R2 values, then run:

```sh
cp deploy/netplay/.env.example deploy/netplay/.env
deploy/netplay/run-local.sh
```

The local page is at `http://127.0.0.1:3000`. The host launcher uses `HAL_NETPLAY_USER_JSON_A` and, when set, `HAL_NETPLAY_USER_JSON_B`. It assigns distinct default Slippi ports 51441 and 51442 and sets API queue capacity to the worker count. Override ports with `HAL_NETPLAY_SLIPPI_PORT_A` and `HAL_NETPLAY_SLIPPI_PORT_B`. The runner rejects duplicate credentials or ports.

Export a validated 059 checkpoint with `uv run hal-policy export /path/to/checkpoint.pt /path/to/policy.halpolicy`. The runner requires a capability-v2 artifact for the declared netplay profiles; an older bundle keeps its original delay meaning and does not gain new capabilities when loaded.

## Hosted run

Set the public origins, API URL, and tunnel token in `.env`, then run `deploy/netplay/run-host.sh` and `deploy/netplay/deploy-frontend.sh` in separate terminals. An empty tunnel token disables Cloudflare. The tunnel sends the API hostname to `http://127.0.0.1:8080`, and the frontend origin must match `HAL_NETPLAY_ALLOWED_ORIGINS` exactly.

Docker Compose uses `deploy/netplay/compose.yaml` for one session. After two-session Ada qualification, add the second account and use the two-worker override:

```sh
cd deploy/netplay
docker compose -f compose.yaml -f compose-ada.yaml up --build -d
```

The override gives the API capacity two and binds two distinct credentials and ports. Do not publish that capacity until the matched Ada timing, failure, and match/rematch gates pass. Both Compose configurations keep queue capacity equal to their Dolphin worker count.

The runner selects the NVIDIA container runtime because Dolphin uses Vulkan.
CUDA access alone does not ensure that Vulkan's driver and graphics libraries are
available. Before starting the service, confirm that the runner image sees the
expected GPU:

```sh
docker compose run --rm --no-deps --entrypoint vulkaninfo runner --summary
```

The command must list the NVIDIA device. A missing driver is a deployment failure;
do not switch to software rendering or another Dolphin backend to hide it.

## Health and timing

Check `/health/ready` and `/v1/capacity` before admitting users. The runner writes a schema-4 preparation and timing record beside its status file. Slot health uses schema 4; aggregate runner health uses schema 5. A live inference process that stops responding is detected within one second. The supervisor invalidates old streams and prepares one replacement before new admission; failed recovery leaves the service unavailable.

Initial preparation can compile kernels and has a separate 30-minute limit, set with `--preparation-timeout-seconds`. Recovery still has a 120-second deadline with cached artifacts. Neither path admits a match before preparation finishes.

The fixed delay-2 profile uses physical delay 2, fixed prefix 3, one frame of inference allowance, replan interval 4, and horizon 8. Delay 3 uses prefix 4 with the same allowance, interval, and horizon. The engine prepares them separately, uses at most 0.5 ms to coalesce ready requests, and does not wait for admitted idle streams. There is no automatic timing selection in production.

The API accepts `desired_return` in `[0, 40]` or `null` and `temperature` in `[0.8, 1.1]`. During a game, `PATCH /v1/jobs/{id}/policy` changes either setting at the next replan after the runner receives it.

## Qualification

Use a capability-v2 artifact and run the opt-in hardware tests on the target GPU:

```sh
HAL_REQUIRE_NETPLAY_HARDWARE_QUALIFICATION=1 \
HAL_NETPLAY_POLICY=/absolute/path/to/policy.halpolicy \
HAL_NETPLAY_QUALIFIED_CAPACITY=2 \
uv run pytest -q tests/test_netplay_hardware.py -m integration
```

On a 3060, use capacity 1 and qualify delay 2 independently. This preparation check is only one gate; the required 2,400-frame trials, 30-minute soak, ten matches/rematches, fault injection, and Ada capacity sweep must also pass before production publication. Record those results in [the refactor evidence](../../docs/refactor-059.md). The runtime API and detailed timing rules are in [Inference](../../docs/inference.md).
