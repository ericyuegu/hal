# HAL netplay deployment

The runner starts one prepared inference process and one Dolphin worker per Slippi account. Workers using the same checkpoint share model weights. The service admits matches only after it prepares and checks both delay profiles. Publishing a deployment remains a separate step after hardware qualification.

## Local run

Install `uv`, `npm`, and `xvfb-run`. Copy the environment template and set the policy, ISO, emulator, one Slippi account, and R2 values. Leave `HAL_NETPLAY_USER_JSON_B` empty for one worker. Then run:

```sh
cp deploy/netplay/.env.example deploy/netplay/.env
deploy/netplay/run-local.sh
```

The launcher also accepts an environment-file path, for example
`deploy/netplay/run-local.sh /absolute/path/to/play.env`. It starts the local
API and page together; stop it with Ctrl-C.

The local page is at `http://127.0.0.1:3000`. The host launcher uses `HAL_NETPLAY_USER_JSON_A` and, when set, `HAL_NETPLAY_USER_JSON_B`. It assigns distinct default Slippi ports 51441 and 51442 and sets API queue capacity to the worker count. Override ports with `HAL_NETPLAY_SLIPPI_PORT_A` and `HAL_NETPLAY_SLIPPI_PORT_B`. The runner rejects duplicate credentials or ports.

Set `HAL_NETPLAY_GRAPHICS_BACKEND=OGL` in the environment file to select OpenGL
for the netplay worker. The default is `Vulkan`; the runner also accepts
`--graphics-backend OGL`. Vulkan showed periodic pauses in local 3060 controls.
OpenGL passed a short neutral-player control, but a complete model-service game
has not yet been measured on that setting.

Export a validated 059 checkpoint with `uv run hal-policy export /path/to/checkpoint.pt /path/to/policy.halpolicy`. The runner requires a capability-v2 artifact for the declared netplay profiles; an older bundle keeps its original delay meaning and does not gain new capabilities when loaded.

## Hosted run

Set the public origins, API URL, and tunnel token in `.env`, then run `deploy/netplay/run-host.sh` and `deploy/netplay/deploy-frontend.sh` in separate terminals. An empty tunnel token disables Cloudflare. The tunnel sends the API hostname to `http://127.0.0.1:8080`, and the frontend origin must match `HAL_NETPLAY_ALLOWED_ORIGINS` exactly.

Docker Compose uses `deploy/netplay/compose.yaml` for one session. After measuring
two-session capacity on the target GPU, add the second account and use the
two-worker override (whose filename is historical):

```sh
cd deploy/netplay
docker compose -f compose.yaml -f compose-ada.yaml up --build -d
```

The override gives the API capacity two and binds two distinct credentials and
ports. Publish that capacity only after timing, failure, and match/rematch checks
pass on the target GPU. Both Compose configurations keep queue capacity equal to
their Dolphin worker count. These Compose commands use the runner's Vulkan
default; the local host launcher carries `HAL_NETPLAY_GRAPHICS_BACKEND`.

The Compose runner selects the NVIDIA container runtime for its default Vulkan
backend. CUDA access alone does not ensure that Vulkan's driver and graphics
libraries are available. For that configuration, confirm that the runner image
sees the expected GPU:

```sh
docker compose run --rm --no-deps --entrypoint vulkaninfo runner --summary
```

For the Vulkan profile, the command must list the NVIDIA device. Record the
selected graphics backend and its actual driver when checking another profile;
a working CUDA inference device does not establish the renderer's behavior.

## Health and timing

Check `http://127.0.0.1:8080/health/ready` and
`http://127.0.0.1:8080/v1/capacity` before play. The first endpoint checks that
the API can read a current runner heartbeat. The second reports the worker
state; for one-account play, check `service_status: "ready"` and
`healthy_slots: 1`. The runner writes a schema-4 preparation and timing record
beside its status file. Slot health uses schema 4; aggregate runner health uses
schema 5. A live inference process that stops responding is detected within one
second. The supervisor invalidates old streams and prepares one replacement
before new admission; failed recovery leaves the service unavailable.

Initial preparation can compile kernels and has a separate 30-minute limit, set with `--preparation-timeout-seconds`. Recovery still has a 120-second deadline with cached artifacts. Neither path admits a match before preparation finishes.

The fixed delay-2 profile uses physical delay 2, fixed prefix 3, one frame of inference allowance, replan interval 4, and horizon 8. Delay 3 uses prefix 4 with the same allowance, interval, and horizon. The engine prepares them separately, uses at most 0.5 ms to coalesce ready requests, and does not wait for admitted idle streams. There is no automatic timing selection in production.

The API accepts `desired_return` in `[0, 40]` or `null` and `temperature` in `[0.8, 1.1]`. During a game, `PATCH /v1/jobs/{id}/policy` changes either setting at the next replan after the runner receives it.

## Qualification

Use a capability-v2 artifact and run the opt-in hardware tests on the target GPU:

```sh
HAL_REQUIRE_NETPLAY_HARDWARE_QUALIFICATION=1 \
HAL_NETPLAY_POLICY=/absolute/path/to/policy.halpolicy \
HAL_NETPLAY_QUALIFIED_CAPACITY=1 \
uv run pytest -q tests/test_netplay_hardware.py -m integration
```

Set the capacity to the intended number of concurrent accounts and run this
check on the target GPU. This preparation check is one gate; the 2,400-frame
trials, 30-minute soak, ten matches/rematches, fault injection, and capacity
checks still need evidence before production publication. They are not a
prerequisite to start a local one-account game. Record results in
[the refactor evidence](../../docs/refactor-059.md). The runtime API and timing
rules are in [Inference](../../docs/inference.md).
