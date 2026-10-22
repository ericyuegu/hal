# O64 outcome-reward run

Treatment: PPO from O64 `step-0098304.pt` for 256 updates with damage 0.1,
stock 120 and deciding-stock 120. Control: frozen copies of that checkpoint.
Keep seed 65, the full-corpus training identity counts, ranked-1 matchup prior,
stages, optimizer, three snapshot slots and held-out start grid fixed.

[production-256.json](production-256.json) records run choices. It is a planning
manifest, not trainer input. Capacity fields are required placeholders. The
memory stream completed no O64 PPO update on the 3060, including at four matches
and one window. Its 180-GiB envelope applies only to the proxy. Measure full PPO
on B200 before filling these fields; no O64 match/window default is qualified.
The DDP stream supplies multi-GPU execution.

Checkpoint SHA-256:
`e6d6303fbf25993c768b1e86fecbb350433a2fb229235b556334e9eb6ae868cf`.
The O64 offset embedding has 31 rows of width 16: two more rows (32 parameters)
than O63. Supervised offsets are 1..24, 27 and 30. Keep all eleven admitted
production cadences. Padded decoder positions can reach 25; only supervised
executed positions enter policy and KL losses.

Prepare the full-corpus sampler with the existing builder, using retained O64
manifests and the pinned professional identity sidecar:

```sh
uv run scripts/build_o65_sampler.py \
  --checkpoint ~/data/runs/o64-step-0098304/step-0098304.pt \
  --manifests-root runs/o64-check/training-sources \
  --output experiments/o65/production-sampler.json \
  --provenance experiments/o65/production-sampler-source.json \
  --evaluation experiments/o65/production-evaluation.json
```

After capacity measurement, use the native CLI settings below with the measured
CPU allocation, active matches and actor-window dimensions. The trainer rejects
incomplete or invalid settings; this command requires explicit shell variables.

```sh
uv run experiments/065_selfplay_ppo.py train \
  --checkpoint ~/data/runs/o64-step-0098304/step-0098304.pt \
  --sampler experiments/o65/production-sampler.json \
  --data-dir ~/src/melee-sim-light/data --output runs/o65-o64-256 \
  --cpus ${O65_CPUS:?set available CPU ids} \
  --active-matches-per-gpu "${O65_MATCHES:?measure on B200}" \
  --cadence production --rollout-frames 512 --groups 4 --updates 256 \
  --training.windows-per-step "${O65_WINDOWS:?measure on B200}" \
  --training.actor-microbatches "${O65_MICROBATCHES:?measure on B200}" \
  --training.seed 65 --training.critic-warmup-updates 32 \
  --training.head-start-frames 1200 --training.checkpoint-interval 32 \
  --training.snapshot-interval 64 \
  --training.reward-weights.damage 0.1 \
  --training.reward-weights.stock 120 --training.reward-weights.game-win 120 \
  --wandb-project hal-o65
```

Training logs `behavior/student_*` across all student exposure,
`behavior/student_pi_ref_*` for the student in reference lanes, and
`behavior/pi_ref_*` for the reference. Rates divide event counts by minutes in
which both players are active. Offstage uses the walkable stage edge. Shield
counts entries into the shield-state set. Zero-damage self-destructs mean stock
losses without positive damage since respawn, not native KO attribution.
Counts retain state across update and exact-resume boundaries (training v11).
Held-out match events include the same student and reference counts and rates.

`flops/*` estimates dense matrix and attention work from the loaded architecture,
including padded slots. `throughput/*_tflops` divides phase work by measured phase
time. `mfu/*` uses the hardware peak in `hal/training/mfu.py`; an unknown peak
logs `mfu/bf16_peak_known=0` and omits utilization. B200 uses 2,250 dense BF16
TFLOP/s. Small FP32 heads, nonlinearities, optimizer and simulator work are
outside the BF16 interpretation described in the FLOP estimator.
