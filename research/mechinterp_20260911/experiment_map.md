# Prior-experiment map for mechanistic research

Updated 2026-09-11. This map uses the 2026-08-27 session transcript as a search index, not as
authority. Claims below were checked against the experiment reports, source, and the stored O39 and
local evaluation artifacts. Paths under `/home/ericgu/src/hal` are in the original workspace; the
research worktree has no copy of the old `results/` or `runs/` trees.

## Short conclusion

O37 is the best existing causal matrix. Its four final checkpoints keep model size, data, optimizer,
seed, and evaluation fixed while changing future-action and within-frame group conditioning. It
contains both a finite teacher-forced/rollout exposure diagnostic and H1/H2/H4/H6 gameplay results.

O39 is useful for boundary conditions, not for a mechanistic model-size claim. The capacity grid uses
the current controller and a dense 36-offset decoder, but its delay sweep ties `control_delay` to
`replan_interval` and its iso-FLOP figure fits validation NLL rather than gameplay. It shows that
capacity, processed data, and timing interact, with no monotone model-size ordering.

The strongest local qualitative artifact is the O26 final checkpoint evaluated at H4 and H6 with the
same checkpoint, matchup schedule, seed, and 12 boots. It has SLPs and per-game rows. Historical
artifacts do not contain semantic hidden states or action logits. The `.live.npy` captures contain
state/controller observations; `inference_trace.json` is an operation-level performance trace.

## Experiment table

| Experiment | Controlled change and useful result | Mechanistic value and limitation |
|---|---|---|
| **O37 factorization matrix** (`docs/experiments/037_factorization_matrix.md`) | Four same-size, same-data BC cells: D0 future-independent/group-independent; D1 future-independent/group-AR; D2 future-AR/group-independent; D3 future-AR/group-AR. All final checkpoints are step 16,384, 7,147,504 parameters, 8,388,608 examples, seed 0, and 96-boot H1/H2/H4/H6 evaluation. | Best matched intervention. Future AR makes the teacher-to-rollout gap large but gives the most stable H4/H6 play. Group AR is a second, separable intervention. Exact masked rollout NLL can be `+Inf` when a sampled trigger makes the target button illegal; use the finite-support audit, not the exact value. |
| **O39 capacity/data/delay scaling** (`experiments/039_capacity_scaling.py`) | L3/L4/L5/L7/L10/L13/L16/L18 plus frozen 026 baseline, processed positions `2^26`…`2^30`, delays 1…16, and local latency. At `D=2^28`, d1/d4 net-stock rates are, for example, L5 −3.435/−1.320, L7 −2.188/−1.350, L10 −3.340/−1.465, L16 −0.820/−1.597, and L18 −0.713/−1.487 stocks/min. | Shows timing-dependent, non-monotone capacity behavior. The rows tie d=R, so they do not isolate replanning. Evaluation entries are gameplay evidence but not latency-valid deployment points; use the separate latency frontier. The fit is an NLL model and should not select checkpoints. |
| **O44 fixed prediction delay** (`docs/experiments/044_fixed_prediction_delay.md`) | O44 trained fixed delay d3/d12/d18, H6/R1, against an O43 d1 anchor. The transcript reports O43 d1 +0.244 ± 0.078 versus O44 d12 −0.861 ± 0.136 stocks/min. | Suggests delayed targets can change control, but the comparison is not sampler-matched: O43 requested 20 future frames and O44 requested 37, and the recipes differ in H6 details. Treat it as a hypothesis generator, not a causal estimate. |
| **O46 autoregressive RTC** (`docs/experiments/046_ar_rtc.md`) | Planned dense offsets 1…20, training-delay samples 0…4, H20, and execution stride 4. The clean comparison is dense AR control versus RTC at the same evaluation delay. | Useful design for an exact chunked-AR test. No completed production endpoint was found; an early U4096 row has no matched dense control. |
| **O43 architecture arms** | Nonlinear heads, zero-init FiLM, linear heads, and trunk-logit-skip removal were early/stopped arms around 15.05M parameters. H1 versus H4 on the same O43 anchor was +0.244 versus +0.005 stocks/min in the transcript. | Candidate head/trunk pairs for probing, but the stopped arms do not support architecture claims. The H1/H4 pair is an execution-horizon control only; recover exact R2 identities before use. |
| **O13 / E0-E2 head and factorization work** | O13 stores periodic and final replays plus diagnostics for horizon gradients, cosine conflicts, action changes, persistence, A-B-A flipbacks, and click mass. E1 compares a linear head (`obx3o3az`) with a state MLP (`q3aojgfm`); the transcript reports a paired H2H preference for linear, mean stock difference −0.195 for MLP-minus-linear. E2 was canceled before a closed-loop endpoint. | Good offline probing vocabulary and old head-capacity control. Protocol and representation differ from O26/O37; do not pool results. |
| **O17 / O32 chunked or flattened AR** | Earlier flattened frame/group-token designs (O17 16×4 tokens; O32 10×4 tokens) have source and tests but no production result that isolates the mechanism. | Historical implementation context only. Prefer O37/O46 for a new test. |
| **O47/O48 capacity follow-ups** | O47 attempted fixed-codec capacity brackets but the new arms stopped around 5.5–7.9k updates. O48 planned a deduplicated iso-FLOP matrix and was canceled before launch. | No comparable result. Their designs identify the missing controls: fixed codec, independent timing, and a complete compute-matched matrix. |

## O37: the main matched mechanistic evidence

The audited final finite rollout diagnostic and main gameplay values are:

| Cell | Future / group conditioning | Finite teacher NLL (bits/joint frame) | Finite rollout NLL (bits/joint frame) | Gap (bits) | Net stocks/min at H1 / H4 / H6 |
|---|---|---:|---:|---:|---:|
| D0 | independent / independent | 7.336620 | 7.336662 | 0.000041 | +0.095 / −0.702 / −0.905 |
| D1 | independent / autoregressive | 5.950180 | 6.107087 | 0.156908 | +0.133 / −0.313 / −0.467 |
| D2 | future autoregressive / independent | 5.233901 | 7.486019 | 2.252117 | −0.186 / −0.196 / −0.234 |
| D3 | future autoregressive / autoregressive | 4.120091 | 6.589160 | 2.469070 | +0.053 / −0.096 / −0.021 |

The finite NLL values are **bits**, not nats: `scripts/audit_037_results.py` divides the summed
natural-log cross-entropies by `math.log(2.0)`. Each joint value sums the four group losses and
averages over the model's ten prediction offsets `(1, 2, 3, 4, 5, 6, 9, 12, 16, 20)`.

D1 is best at H1/H2; D3 is best at H4/H6. The finite gap is not itself a failure: D2/D3 are
deliberately exposed to sampled earlier offsets. The useful next **offline** probe is to replay the
same fixed validation windows with teacher-forced ancestors, sampled ancestors, and an identity
condition, then connect per-offset/group errors to H-specific outcomes. It is not a closed-loop
gameplay treatment.

Audited checkpoint identities are in the O37 report:

| Cell | W&B run | R2 prefix | `final.pt` SHA-256 prefix |
|---|---|---|---|
| D0 | `98r9smrj` | `runs/037-D0-future-independent-group-independent-bc-seed0/` | `50da28a2…f12a23` |
| D1 | `a117chkw` | `runs/037-D1-future-independent-group-ar-bc-seed0/` | `665a2185…2bf88` |
| D2 | `50q39o9j` | `runs/037-D2-future-ar-group-independent-bc-seed0/` | `eede17db…d3f7dc` |
| D3 | `5wfk2esf` | `runs/037-D3-future-ar-group-ar-bc-seed0/` | `0d62a529…bc5c` |

The report also records 96-boot replay/evaluation artifacts for every cell and H1/H2/H4/H6. These
are the first artifacts to recover for semantic logit or activation traces.

## O39 scaling artifacts and what they support

The audited result slab is:

- `results/039_capacity_scaling/experiment_039_raw_outcomes.csv` — 47,095 rows: 62 run rows, 315
  evaluation rows, 63 latency rows, and 46,655 match rows.
- `results/039_capacity_scaling/figures/09_isoflop_fit.png` — the requested plot.
- `results/039_capacity_scaling/figures/07_delay_sensitivity_latency.png`, `11_match_distributions.png`,
  `12_closed_loop_controlled.png`, and `13_closed_loop_iso_data.png`.
- `results/039_capacity_scaling/latency_capacity_frontier.csv` and
  `deployable_frontier_median.csv`.
- `notebooks/039_scaling_viz.py` — result selection and fit code; `scripts/analyze_039_capacity.py`
  — artifact builder.

The fit is

```text
L(N_eff, D) = E + A / N_eff^alpha + B / D^beta,
N_eff = training_FLOPs / (6D),  C = 6 N_eff D.
```

It uses robust Huber fitting over non-diverged scaled-family cooldown checkpoints. The figure
extrapolates `C = 4.2e19` to `N_eff ≈ 1.34e9`, `D ≈ 5.24e9` (2.2 corpus epochs), predicted NLL
1.703, and `N_opt ∝ C^0.58`. The notebook explicitly warns that predictions past the corpus size
enter a repeated-data regime and are optimistic. L7's exact endpoint diverged near `D=2.61e9`
(target `3.80e9`); an optimizer-failed L10 endpoint is excluded. These facts make the plot useful for
planning a matched study, not for claiming a causal compute law.

Representative O39 gameplay rows at `D=2^28` (the two columns are the tied d=R conditions):

| Model | d=1 net stocks/min | d=4 net stocks/min | 96 boots? |
|---|---:|---:|---:|
| 026 baseline | −2.738 | −1.698 | yes |
| L5 | −3.435 | −1.320 | yes |
| L7 | −2.188 | −1.350 | yes |
| L10 | −3.340 | −1.465 | yes |
| L16 | −0.820 | −1.597 | yes |
| L18 | −0.713 | −1.487 | yes |

At `D=2^30`, L5 is −3.617/−1.071 and L7 is −3.648/−0.936 for d1/d4; L10 is −2.704/−1.044.
The ordering changes with timing and data. Do not call these iso-FLOP comparisons: the displayed
models have different processed data and several exact endpoints are missing or excluded. The d1/d4
rows also have `eval_valid_latency_bucket=0`; use the local latency artifact for deployment claims.

Measured RTX-3060 latency gives a separate frontier: at d1, L10 (51.15M parameters, p95 11.66 ms)
is the largest eligible model; at d2, L13 (111.72M, 21.67 ms); at d4, L18 (293.13M, 45.78 ms).
All listed frontier points have zero deadline misses. This is a systems constraint, not a gameplay
selection result.

## Delay, replanning, and chunked decoding

The existing evidence leaves one important control absent:

- O39 supports independent `control_delay` and `replan_interval`, but the stored sweep uses
  `eval_control_delay == eval_replan_interval`. Its `decode_horizon` is `delay + replan − 1`, so d4
  means a seven-frame internal decode horizon, not the O37 H4 condition.
- O44 changes the trained absolute delay and fixes H6/R1. Its d1 anchor is external and requests a
  different future buffer. It is not a matched delay-only estimate.
- O46 defines the right dense-AR/RTC control, but no completed matched endpoint was found.
- O26 offers a local same-checkpoint execution control (below), but not a full R sweep.

A useful new matrix is therefore: one frozen checkpoint, fixed H, fixed observation and random
streams, `R ∈ {1,2,3,6}`, and an explicit trained-delay condition. Save both sampled plans and
the per-offset teacher/sampled errors. This separates action-horizon effects from replanning.

## Local checkpoints and replay candidates

### Strong local pair: O26 H4 versus H6

Root:

```text
/home/ericgu/src/hal/runs/260810-071709_026_temporal_mtp_mtp026-d384-L8-h6-Lc128-t128x2-o1-2-3-4-5-6-9-12-16-20-s4-base_ranked-anon-1_production-seed0-d384-b512/
```

Use these two directories:

```text
eval_rerun_h4_n12_w12/
eval_rerun_h6_n12_w12/
```

Both `match_rows.json` files record checkpoint SHA `22333d1d…`, schedule SHA
`a060fac1…`, seed 0, six ego/CPU characters, 12 scheduled boots, 7,200 frames, and CPU level 9.
Only `exec_horizon` differs (4 versus 6). The H4 summary is net stocks/min −0.255 and net damage/min
16.94; H6 is −0.042 and −0.665. The H4/H6 SLPs are in `boot_000`…`boot_011` below each directory.

Illustrative rows, not independent paired samples: boot 0 Fox–character 22 is H4 2–2 stocks with
198.9 dealt/343.6 taken, versus H6 2–1 with 278.6/239.7. H4 boot 9 Fox–character 15 is a favorable
3–1 row (281.7/223.6); H6 boot 5 Fox–character 2 is a poor 1–3 row (223.7/331.2). These rows are
good targets for replay/state alignment, but separate Dolphin runs can diverge; boot identity is a
schedule match, not frame-level pairing.

The O26 checkpoint root also has `final.pt` and periodic `step_004096.pt`, `step_008192.pt`,
`step_012288.pt`, and `step_016384.pt`. Periodic H4 evaluations share schedule SHA `d3446326…` and
32 scheduled matchups; the final 96-matchup H4 evaluation uses schedule SHA `a2202b35…`, so it is not
a one-to-one continuation of the periodic rows. `native_slippi_accepted_protocol/` is a valid
single-protocol integration artifact. `native_slippi_all_matchups_run03/` has a recorded
always-neutral failure for one port and should not be used as clean policy evidence.

### Learner-visited replay feasibility

An SLP can supply a valid offline model context, but not the complete live decoder state. The
bounded path is:

```text
hal.data.extract.extract_replay
  -> add masked p1_player_id/p2_player_id (historical identity is player 0)
  -> hal.training.dataloader.make_window(..., ego_prefix="p1", length=256,
       projection=features.ITEM_PLAYER_PROJECTION)
  -> collate_windows
  -> train_batch_from_columns(..., stats, L_ctx=256,
       extra=features.ITEM_PLAYER_COLUMNS,
       projection=features.ITEM_PLAYER_PROJECTION)
  -> O50Model.context_tokens / forward_dense / trace_last_attention
```

This uses the same canonical frame flattening, action grouping, item masking, and feature
normalization as training. `tests/test_extract.py::test_extract_matches_canonical_flatten_column_for_column`
is the relevant wire-parity guard. Before selecting a window, require a finalized SLP, at least
256 consecutive rows, no decrease in `frame`, finite required state/action columns, and matching
stage/character fields. Split at instant-restart frame decreases; never pad across a restart.

The replay does not record the pending action queue, sampled latent prefix, decoder RNG, or model
hidden states. It therefore supports a same-context attention/logit trace, but not reconstruction
of the exact closed-loop action-plan state. Port and identity also need explicit protocol control:
the historical runner uses port 1 and masks the model identity to player 0; SLP names or ranks are
not a substitute. A partially written SLP must be rejected: `extract_replay` can otherwise fail
with a truncated-buffer error or expose an incomplete episode.

The current short pilot confirms the path on `early_24576/boot_000`: 1,788 extracted rows, 168
schema-7 columns, and a 256-row context with 114 model features. After teardown, three finalized
boots from each of `early_24576` and `slump_57344` were traced through both historical checkpoints
on identical offline inputs. The bounded artifact is
`research/mechinterp_20260911/artifacts/live_attention/live_attention_heatmaps.png` with its
provenance and raw summaries in the adjacent `live_attention.json`. The trace is FP32 and its
`trace_last_attention` versus `forward_dense` hidden-state parity error is zero. Across the six
contexts, mean last-query entropy was 0.100 nats for the early model and 0.029 nats for the slump
model (mean of per-example/head entropy); mean effective context was 1.15 and 1.04 positions.
These are offline traces of observed contexts, not evidence that the live decoder retained the
same latent prefix or that sharper attention caused the gameplay change.

#### Trace provenance and port audit

The attention trace ran once on CPU with `CUDA_VISIBLE_DEVICES=''` and an explicit
`torch.device("cpu")`; command wall time was about 20 seconds (05:27 local time). The exact
source is preserved as `artifacts/live_attention/original_live_replay_attention_probe.py.txt`,
SHA-256 `36abbd73071cac44d7a25fa109e3700f074918fa39af16556d7dbfb8f69a8f05`.
The `.txt` suffix identifies frozen source data; the contents are unchanged. Its output hashes
are recorded in `trace_manifest.json`. This CPU trace overlapped the final portable gameplay
condition; it issued no CUDA work. The condition wall times are not controlled performance
benchmarks. All model arithmetic in this trace used CPU FP32, whereas the main GPU attention
audit used BF16 projections followed by FP32 QK scores and softmax. Do not pool their means.
The durable audit-only reconstruction is
`research/mechinterp_20260911/live_replay_audit.py`, with focused tests in
`tests/test_mechinterp_live_replay.py`; it performs no model loading or forward pass. Its fixed
selection rule is boots `(0, 1, 2)` in both conditions and raw context row `700` with length `256`,
chosen before any attention output was inspected.

The historical runner calls `cross_stage.sweep_vs_cpu_prior_with_rows` without an `ego_port`
override, so the public default is controlled libmelee port 1. `PolicyBatchAdapter` passes each
slot's port as `controlled_port`, and the pilot protocol records masked player identity 0. The
audit re-read game-start metadata for all six selected SLPs: p1 is peppi port 0 → libmelee port 1,
p2 is peppi port 1 → libmelee port 2, and extracted p1/p2 characters match the fixed pilot schedule
(FOX–FALCO, FOX–MARTH, FOX–FOX). Thus p1 is verified as the controlled port for these replays;
this is not inferred from SLP names or rank.

### Other local candidates

- O13 contiguous-normalized:
  `/home/ericgu/src/hal/runs/260801-184628_013_multi_token_gpt-d256-L8-h4-Lc256-o1.2.3.4_ranked-anon-1_013-ablation-C-contiguous-normalized/`
  has `step_002048.pt` through `step_030720.pt`, periodic SLPs, and a final 96-matchup replay set.
  Periodic files use 16 matchups while final uses 96; use it for state/action trajectories, not a
  paired gameplay estimate.
- O13 validity baseline has a final 96-matchup path, but its final rows are empty in the stored
  `replays/final` file; the `eval_replays` directory is the usable final artifact.
- O28, O34, O35, O36, and O38 local smoke runs have only 2–5 games. They are integration checks,
  not evidence for a circuit or architecture claim.

## Data, evaluation, and probing interfaces

| Need | Interface/artifact | Use | Boundary |
|---|---|---|---|
| Reconstruct a trajectory | `hal/sim/trajectory.py`: `Trajectory.from_slp`, `from_capture`, `from_mds_rows`; `tests/test_trajectory.py` | Align SLP, captured state, and stored MDS fields by frame; compare state and controller observations. | No model activations are stored. Capture fields are NaN-masked when unavailable. |
| Rebuild training inputs | `hal/data/schema.py`, `hal/data/policy_schema.py`, `hal/wire.py`, `hal/training/features.py` | Preserve MDS-v7 columns, wire names, feature transforms, action grouping, and context alignment. | O13/O26/O37 use different schemas/codecs in places; record the checkpoint config and data identity. |
| Audit gameplay | `match_rows.json` + `metrics.json` beside each replay directory | Read protocol schema, checkpoint SHA, matchup schedule SHA, boot index, active frames, stocks, damage, characters, and stage. | Games, not frames, are the independent units. Require schedule and checkpoint identity before pairing. |
| O37 exposure probe | `scripts/audit_037_results.py` | Compute finite teacher-forced versus sampled-rollout logits/NLL by offset/group and button-support rate. | Exact legality-masked NLL may be infinite; finite support diagnostics are the intended comparison. |
| Offline temporal probe | `docs/experiments/013_multi_token.md` and its experiment diagnostics | Gradient norms/cosines by horizon, action-change rate, persistence, A-B-A flipback, and click mass. | Observational metrics. They do not prove a gameplay circuit. |
| Decode implementation parity | `docs/experiments/p2_temporal_kv_ablation.md` | Compare FP32/FP16 logits and sampled action bytes under recompute versus KV paths. | Tests the decoder implementation, not why a representation controls well. |

Before a new semantic trace, rerun the frozen checkpoint on fixed replay windows and export, per
frame and per predicted offset: trunk state, group logits before/after legality masking, sampled
group IDs, target IDs, support validity, and plan age. Keep the exact feature/config/data hashes.
The old `inference_trace.json` and `inference_profile.txt` cannot supply these fields.

## Three testable hypotheses

1. **Future AR helps through temporal dependence among sampled actions, not access to expert futures.**
   This remains an unrun closed-loop test. At each fixed observation, compare the normal jointly
   sampled AR tail with a tail assembled from independently drawn full plans: draw one independent
   plan for each executed offset, then take that plan's action at the corresponding offset. This
   preserves each frame's action marginal while removing cross-offset dependence. If that dependence
   matters, D2/D3 should lose much of their H4/H6 advantage while per-offset marginals stay matched.
   This is not forcing a demonstration suffix (true future expert actions are unavailable in a valid
   closed-loop treatment), and it is not making the model future-independent: every independent plan
   still uses the trained decoder and its own within-plan conditioning.

2. **Within-frame group AR acts mainly at sampled conditional/support boundaries.** D0→D1 and D2→D3
   change H4 gameplay more than their small group-accuracy change suggests. In an offline diagnostic,
   swapping a sampled earlier group for the recorded target should reduce later-group logit/support
   errors in group-AR cells; it should have little effect on first-group logits. The identity
   intervention must reproduce the unmodified decoder exactly.

3. **The useful capacity is deadline-conditional.** At fixed processed data and a fixed trained
   delay, larger models should help only when the execution budget permits their latency and when
   the horizon/replanning schedule matches training. A complete L5/L10/L16/L18 matrix with
   independent `R` and fixed H should break the O39 d=R tie; per-offset error and latency should
   predict the H1 versus H4/H6 ranking better than validation NLL alone.

## Non-claims and risks

- The session transcript is coverage, not provenance. Source reports, protocol-bearing artifacts,
  and raw result rows take precedence.
- O39's fit is a validation-NLL extrapolation. Its predicted 2.2-epoch point is explicitly in the
  repeated-data regime; it is not a gameplay optimum.
- O44's d1/d12 comparison is confounded by future-buffer and target-layout changes.
- Small local smoke evaluations and different matchup schedules cannot establish architecture or
  mechanistic effects.
- A parameter or activation swap can change compatibility with the rest of the model. A successful
  rescue localizes a dependency; a failed rescue does not prove that the component is irrelevant.
