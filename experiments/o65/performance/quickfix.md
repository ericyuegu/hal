# O65 quick performance fixes

The treatment skips reference evaluation and factor KL during critic warmup,
and removes the per-epoch Python shuffle from full-batch critic fitting.
Warmup logs `policy/kl_per_frame` and its lambda bins as zero. The warmup exit
update still fits only the critic. Reference scoring resumes on the next update,
before the first actor step. PPO rewards, masks, detach points, and optimizers
retain their definitions.

The control is `9ab5b9b0`, before these fixes. Both arms use the recorded 17M
O63 bootstrap30 checkpoint, sampler, seed 65, initial weights, initial simulator
states, D=8/R=4 and D=10/R=2, one group, four native shards on CPUs 0..3,
256 frames per match per update, 32 windows per actor step, four critic epochs,
and compiled sampling. Critic warmup is either disabled or set to 10,000 updates.
All timing runs use default Torch algorithms. Checkpoints and evaluation stay
outside update timing. No run exceeds 64 matches.

Runs use an RTX 3060 12 GB and Ryzen 5 5600X, Python 3.14.4,
Torch 2.11.0+cu130, CUDA 13.0, and NumPy 2.1.3. The trunk and stored history
retain BF16; the decoder retains FP32. TF32 matmul is disabled. Each GPU job
runs under the shared `gpu.lock` and `systemd-run --user --scope -p MemoryMax=10G`,
with `OMP_NUM_THREADS=2`. Heavy jobs run sequentially. Logs, complete metrics,
commands, and resolved settings are in local `runs/quickfix/`.

| Input | SHA-256 |
| --- | --- |
| O63 checkpoint | `65e7e43e5cbb18a365a005922ec1cfd026ad807c045eae13b331f487b4ec82e6` |
| Feature statistics | `6870bda6c0970826f6467407647cf9bb5901ef246b6947d87518db67252cdde3` |
| Training data manifest | `b97eab90e761bcf2bf03b48981f0ab6acc1ac3057157c58ae0c5a72c76c43bd8` |
| Sampler | `c5ffb1b5f75aa0d9245192304871c2979de9a3e59ddf0c5d55e6877994ab661e` |
| MSL game-data manifest | `4c2b074e872a7ad55176b63a3971fd6a3bbf09dd1988240d1dbdaf61a63616cb` |
| MSL library | `9f16aefd23377abf4a94a185b702bc34033cf890d63b88e79f410ef5efb7c78a` |
| HAL adapter | `e9e2bf2ee60a3cf31034f65a12505f29e06af010971d20d569c44b2849c5feb2` |

MSL source is `2dbca5e3214d88859ac80333cc19caa328d4be4a`. The treatment
and control fix all inputs above. This measures local cost, not learning or
production throughput. Full-batch reordering can change floating-point sums.
Critic loss and gradient parity are checked within float tolerance. The removed
`critic_shuffle` keys have no other consumer: counter RNG draws are stateless
and each domain has its own keys. Exact resume remains tied to source and
configuration identity.

## Pilot determinism impasse

Independent pool qualifications now use default algorithms. The connected
training segments still use deterministic algorithms. `_configuration_digest`
binds both `TrainConfig.deterministic` and the runtime deterministic flag, plus
`CUBLAS_WORKSPACE_CONFIG`; `restore_checkpoint` rejects any mismatch.
The verifier itself needs deterministic execution only for its two next-update
comparisons. Switching the pilot segments requires an explicit resume-contract
change. No implicit mode transition or relaxed checkpoint validation is added.
The protocol now states this constraint.

## Parity and review

`tests/test_batched_self_play_integration.py` now runs shared weights and two
weight sets. The second actor copies the proxy and adds a deterministic
-0.001..0.001 perturbation to the button output weights, with a distinct policy
hash. Each case compares batched CUDA graphs against independent per-frame
`SelfPlayPlayer` instances at mixed cadence, both port orientations, cache
handoffs, asynchronous resets, and timeout/budget stops. Sampled factors,
submitted actions, native frame bytes, and outcomes match exactly through seven
stockouts per case. Maximum log-probability gaps are 3.81469727e-6 nats for
shared weights and 3.57627869e-6 nats for distinct weights. Both cases pass in
337.37 seconds, with no skips. No evaluator defect was found.

Focused tests check bit-identical student/reference distributions before the
first actor step, both before and after critic fitting; zero warmup KL metrics;
reference omission; full-batch critic loss and gradient parity; rewards and GAE
after warmup exit; and resume during warmup, at its exit, and during PPO.
An invalid PPO batch with no reference is rejected before either optimizer or
model state changes. The warmup exit verifier has identical batch and optimizer
hashes and zero parameter error, with no actor step on that update.

Review covered `git diff o65-selfplay-ppo-plan...HEAD` and the stream delta from
`9ab5b9b0`, including masks, offsets, detach points, RNG domains, reference
readers, and persisted identity. Review strengthened CUDA distribution parity,
warmup-exit resume coverage, and rejection before optimizer mutation. A second
pass found no further code defects. `_batch_hash` is the only reader outside
`Trainer` changed for the optional reference. No columnar-owned structure or
method changed.

## Complete-update measurements

Each run has three updates. This table averages updates 2 and 3; update 1
includes cold preparation. [Raw measurements](quickfix-measurements.json) retain
all three updates, commands, source hashes, resolved configuration and inputs.
Warmup omits `systems/actor_seconds`; the table uses zero because no actor step
runs. Phase values are the existing logged wall times. Diagnostics and cleanup
outside those phase timers explain the difference from the update total.

| Matches | Phase | Arm | Rollout s | No-grad s | Actor s | Critic s | Update s |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 16 | Warmup | Before | 6.760 | 0.686 | 0.000 | 0.048 | 7.535 |
| 16 | Warmup | After | 7.132 | 0.503 | 0.000 | 0.017 | 7.704 |
| 16 | PPO | Before | 6.772 | 0.697 | 0.514 | 0.047 | 8.072 |
| 16 | PPO | After | 6.907 | 0.703 | 0.514 | 0.016 | 8.184 |
| 64 | Warmup | Before | 9.687 | 2.607 | 0.000 | 0.164 | 12.581 |
| 64 | Warmup | After | 9.901 | 1.567 | 0.000 | 0.045 | 11.637 |
| 64 | PPO | Before | 9.697 | 2.619 | 1.853 | 0.164 | 14.459 |
| 64 | PPO | After | 9.803 | 2.655 | 1.856 | 0.044 | 14.482 |

At 64 matches, warmup update time falls 7.5%, from 12.581 to 11.637 s.
No-grad time falls 39.9%, and critic time falls 72.9%. At 16 matches,
no-grad and critic times fall, but slower rollout samples leave the full update
2.2% slower. PPO totals are 1.4% slower at 16 matches and 0.2% slower at 64.
These two steady samples per arm do not establish a PPO throughput gain.
Actor time is unchanged; the removed reference pass affects warmup only.

The treatment uses source `6ee34dd3`; the experiment source is unchanged since
`de389442`. Every arm has identical resolved configuration and input hashes
to its control except for the recorded source commit. No timing run resets or
times out a match. The pilot determinism change affects qualification commands,
and is not part of this default-algorithm timing comparison.

## Validation

All pytest basetemps are under `$TMPDIR`. GPU commands use the shared lock,
`systemd-run --user --scope -p MemoryMax=10G`, and `OMP_NUM_THREADS=2`.
No pytest worker pool is used. The full suite sets
`HAL_O65_PROXY_CHECKPOINT=runs/proxy/final.pt` and `HAL_O65_COMPILE_CUDA=1`,
so the real-proxy checks execute.

| Command | Result |
| --- | --- |
| `uv run ruff format --check .` | Passed; 377 files formatted |
| `uv run ruff check .` | Passed |
| `uv run ty check --python-version 3.14 --error-on-warning hal experiments/059_muon_action_sequence.py scripts experiments/065_selfplay_ppo.py experiments/o65/pilot.py tests/test_065_loop.py tests/test_batched_self_play_integration.py tests/test_o65_pilot.py` | Passed; zero diagnostics |
| `uv run pytest -q -ra -m 'not integration' --basetemp "$TMPDIR/handoff"` | 2,501 passed, 8 skipped, 86 deselected, 27 warnings; 548.21 s |
| `HAL_REQUIRE_INTEGRATION=1 uv run pytest -q -s tests/test_batched_self_play_integration.py --basetemp "$TMPDIR/batched-integration"` | 2 passed; 337.37 s; no skips |
| `HAL_REQUIRE_INTEGRATION=1 uv run pytest -q -ra tests/test_roundtrip.py tests/test_session_cleanup.py -m integration --basetemp "$TMPDIR/roundtrip-integration"` | 8 passed, 6 deselected, 7 warnings; 57.54 s; no skips |
| `uv run pytest -q tests/test_o65_pilot.py --basetemp "$TMPDIR/pilot"` | 10 passed; 2.52 s |
| `uv run pytest -q tests/test_065_loop.py -k 'resume_diagnostic' --basetemp "$TMPDIR/warmup-exit-resume"` | 3 passed, 98 deselected; 11.65 s |
| `uv run pytest -q tests/test_065_loop.py::test_actor_fit_rejects_missing_reference --basetemp "$TMPDIR/rejected-update"` | 1 passed; 2.01 s |
| `git diff o65-selfplay-ppo-plan...HEAD --check` and `git diff 9ab5b9b0...HEAD --check` | Passed |

The eight full-suite skips are explicit:

- Two tests in `tests/test_netplay_hardware.py` require
  `HAL_REQUIRE_NETPLAY_HARDWARE_QUALIFICATION=1`, the production GPU and a
  qualified 059 policy bundle. This stream does not qualify those profiles.
- Six tests in `tests/test_policy_world_schema.py` lack
  `data/processed/ranked-anonymized-1/mds-v7-sub4/train`. The supplied processed
  data tree is read-only. **This is an unqualified fixture failure under
  AGENTS.md, despite pytest's zero exit status.** No replacement data or schema
  fallback is used.

Earlier focused checks and failures are retained in `runs/quickfix/`:

- The initial CPU identity/critic proof had 1 failure and
  1 pass: its randomly sampled prefix contained only R=2. The fixture now
  explicitly covers both 8/4 and 10/2, and the identity repeat passed before
  the warmup skip. The locked CPU/CUDA proof passed all 3 cases in 3.82 s.
- A CPU-only warmup command accidentally selected the CUDA critic case:
  7 passed, 1 failed because CUDA was disabled. The test now marks unavailable
  CUDA explicitly. Repeated warmup/critic checks each had 7 passes, 1 intentional
  CUDA skip and 91 deselections; the last took 8.99 s. Both CUDA parity tests
  execute in the final full suite.
- The first commit hook failed type checking because the new local environment
  lacked CairoSVG. Installing CairoSVG 2.9.1 locally resolved it. No dependency
  manifest or lockfile changed; all subsequent hooks passed. The existing
  undeclared dependency remains outside this stream's code scope.
- A local summary helper initially used an unavailable `python` shell command.
  Repeating it with `uv run python` passed. Evidence publication also verified
  each recorded source checksum against its Git commit and matched all resolved
  configuration and input identities across arms.

All eight three-update timing runs passed. No run exceeded 64 matches, and
none had a reset or timeout. No cloud job, VM action, push or PR was performed.
