# Historical run forensics

## Result

The named runs are a close optimizer comparison, but they are not a strict
one-variable experiment. They use the same architecture, data, seeds, update
budget, evaluation schedule, AWR objective, and value objective. The Muon run
used Git commit `cdf84f904d322d2a8c342fe3ea75d926e10346ba`; the AdamW run used
`d67c68cb36c830bb4100ba8ef1b4cefc452539de`. Every file in each W&B code
artifact matches its stated commit. The later source adds the all-AdamW mode
and passive diagnostics.

| Role | W&B run | Experiment ID | Optimizer |
| --- | --- | --- | --- |
| Treatment | `p1fyyp1z` | `050_scaled_temporal_awr_v4` | Muon for eligible matrices; AdamW for the remainder |
| Control | `tgpuo1be` | `050_scaled_temporal_awr_v6` | AdamW for all parameters |

Both runs have `d_model=1024`, 16 layers, 16 heads, context length 256, four
512-wide temporal blocks, batch size 512, 131,072 updates, seed 0, evaluation
seed 0, and gradient clipping at 1. They use the same exact 44-source
`policy-world-v8` mixture, MDS schema 7, source manifests, player artifacts,
and source-selection SHA-256
`2593361352b92e705be3fbeae1b4e9bb1a3c9f1787cd713014a7a95b7df62477`.

The W&B state is `crashed` for both runs. This state does not mean that
training stopped early. Each history reaches update 131,072, each final
evaluation completed 96 of 96 games without a crash, and each immutable log
records a successful `final.pt` upload. The W&B teardown did not record a
normal finish.

## Exact chronology

The Muon run learns gameplay faster through update 24,576. It first trails
AdamW at update 32,768. The regression persists through update 73,728. Stock
rate later exceeds the Muon run's early peak, but damage rate does not. At the
endpoint, Muon remains behind AdamW on both gameplay measures.

| Update | Muon stock/min | AdamW stock/min | Difference | Muon net damage/min | AdamW net damage/min | Difference |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | -0.217529 | -0.541562 | +0.324033 | 9.53852 | -2.52756 | +12.06608 |
| 16,384 | 0.058385 | -0.132615 | +0.191001 | 45.02695 | 24.88319 | +20.14376 |
| 24,576 | 0.164392 | 0.068948 | +0.095445 | 50.31414 | 31.26474 | +19.04940 |
| 32,768 | 0.095532 | 0.275962 | -0.180430 | 25.05997 | 34.25638 | -9.19641 |
| 40,960 | -0.154012 | 0.238872 | -0.392884 | 39.58321 | 43.56869 | -3.98548 |
| 49,152 | 0.000000 | 0.164503 | -0.164503 | 21.42847 | 42.35011 | -20.92164 |
| 57,344 | -0.175099 | 0.387395 | -0.562494 | 27.19126 | 51.77112 | -24.57986 |
| 65,536 | -0.079584 | 0.334251 | -0.413835 | 19.31362 | 44.17720 | -24.86357 |
| 73,728 | -0.074292 | 0.435394 | -0.509686 | 24.60988 | 47.44115 | -22.83128 |
| 81,920 | 0.079584 | 0.425075 | -0.345492 | 39.05418 | 54.95006 | -15.89588 |
| 90,112 | 0.116728 | 0.403181 | -0.286453 | 33.16912 | 53.76519 | -20.59607 |
| 98,304 | 0.137845 | 0.350106 | -0.212261 | 31.63141 | 54.17752 | -22.54611 |
| 106,496 | 0.249297 | 0.382229 | -0.132932 | 38.28512 | 51.75290 | -13.46779 |
| 114,688 | 0.259892 | 0.408654 | -0.148762 | 37.51010 | 53.47822 | -15.96812 |
| 122,880 | 0.291648 | 0.493379 | -0.201731 | 41.47655 | 51.32147 | -9.84492 |
| 131,072 | 0.344959 | 0.493671 | -0.148711 | 37.59019 | 52.71965 | -15.12946 |

The logged gradient norm and clip fraction are means over each 25-update
logging window. They are not instantaneous samples. In the Muon run, the
first post-warmup nonzero clip fraction is at update 28,950. The first mean
gradient norm above 1 is at update 41,550. The largest displayed value is a
25-update mean of 17.959 at update 99,875. Gameplay first falls at update
32,768, before the large gradient growth.

Training NLL is in bits and includes the AWR-weighted policy objective; this
is not plain behavior cloning. Muon training NLL crosses behind AdamW near
update 32,768, then continues to improve despite poor gameplay. Muon NLL is
1.41748 bits at update 32,768, 1.42348 near update 57,344, and 1.35779 at the
endpoint. AdamW is 1.40738, 1.36073, and 1.30764 at those points.

Muon learning-rate annealing is a major confound in the late recovery. Its
rate falls from 0.013126 near update 24,576 to 0.008784 near 57,344,
0.002247 near 98,304, and the 0.00008235 floor at 131,072. The history does
not establish an unassisted escape from a bad basin.

## Checkpoints

The immutable W&B training logs confirm all 16 numbered checkpoint uploads
for each named run. `artifacts/forensics/checkpoint_inventory.json` records
the exact R2 URI and upload time for each object. The primary Muon sequence is:

- 24,576: early gameplay peak, before sustained regression.
- 57,344: stock-rate trough during sustained regression.
- 98,304: last durable checkpoint before the update-99,875 gradient event.
- 131,072: recovered endpoint at the learning-rate floor.

AdamW checkpoints at 57,344 and 131,072 provide matched controls. Model-only
exports preserve the raw checkpoint config, checkpoint step, W&B ID, source
URI, source checkpoint SHA-256, R2 ETag, loader data protocol, source-selection
hash, and every source-manifest hash. The export rejects a boundary-step,
experiment-ID, or loader-provenance mismatch.

The six complete weight files are at the root of Modal Volume
`hal-mechinterp-20260911`. Their names are
`{run_id}-step-{seven_digit_step}.model.pt`. The local artifact directory keeps
the adjacent metadata JSON and aggregate `model_exports.json`; it ignores the
5.2 GB of weight files. Each sidecar records the expected exported byte count
and SHA-256 for verification after transfer.

The current portable O50 implementation accepts both recorded experiment IDs.
Functional probes must still validate the exact mixture statistics, schema,
player vocabulary, and feature representation before inference.

No ready serialized training batch exists for this pair. The checkpoints have
exact loader state, but the named runs did not save a fixed diagnostic batch.
A functional comparison must freeze and hash one newly reconstructed v8 batch
or one portable input trace, then use it for every checkpoint. Old local v7
shards are not a substitute.

## Controlled supporting evidence

The O51 16-layer parameterization grid completed nine Muon/Adam-remainder
learning-rate combinations on the full 44-source v7 corpus. Validation chose
Muon 0.014 and Adam 0.0002125 (`0m55am9g`), while closed-loop gameplay chose
Muon 0.014 and Adam 0.000425 (`cg4rwqnr`). This is direct evidence that offline
NLL can mis-rank gameplay. It is not evidence about the later instability.

The related O52 all-AdamW learning-rate sweep completed five rates from
0.00010625 through 0.0017. It used only the rank-1 v8 source, whereas O51 used
all 44 v7 sources. Do not use cross-sweep cells as an optimizer comparison.

The cleanest existing intervention is W&B run `7dgv85x8`, a half-Muon-LR fork
from parent `p1fyyp1z` at update 24,576. It restores the parent model,
optimizer, scheduler, loader cursor, and RNG state, then changes only the Muon
learning-rate multiplier to 0.5. Its mean gradient norm remains near 0.27
through update 71,225. From updates 24,576 to 57,344, parent stock rate changes
by -0.33949; the half-rate branch changes by -0.09035, a +0.24914
difference-in-differences benefit. At update 65,536, the relative benefit is
+0.26499.

The branch's repeated baseline evaluation at update 24,576 differs from the
parent evaluation despite the same checkpoint and evaluation seed. Therefore,
raw branch levels are not directly comparable; use changes from each branch's
own update-24,576 baseline. The branch also spans two logged code versions: an
initial `0c5540163d9900771a7a450aa229df9b8736bec5` version and the
mixed-device diagnostic fix at `ab92ee037a4f9399ac020167b1f54271d7ac8c4f`.

## Reproducibility artifacts

Each named run directory under `artifacts/forensics/` contains the exact W&B
config, metadata, summary, immutable output log, requirements, run record, and
separate `scan_history` JSONL files keyed by `global_step` for training,
validation, evaluation, and optimization-health metrics. Sparse metric
families are separate so W&B row sparsity cannot shift their chronology.
`artifacts/forensics/manifest.json` records byte counts and SHA-256 hashes.
