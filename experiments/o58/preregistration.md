# O58 return conditioning

Registered before the O58 treatment launch.

## Question

Does conditioning O55's selected `add-history` action decoder on a requested next-60-frame return improve closed-loop play?

## Treatment, control, and invariants

The treatment is one seed-0 proxy run from `experiments/058_return_conditioning.py`. The historical control is W&B run `kahz2d77`, run name `260914-014038_055_history_cross_attention_attention055-d256-L16-h4-Lc256-t128x4-o1-2-3-4-5-6-9-12-16-20-d2r2-add-history-final-prefix-all-adamw-alr0.0017-g0.99855_ranked-anon-1_o55-b-add-history`.

The control has W&B state `crashed`, but remote verification found a finite final checkpoint at update 16,384 and complete final evaluation rows for all 96 registered boots. Its B200 run metadata, configuration, checkpoint, archived source, and evaluation protocol are recorded in `control-verification.json`.

- Control final checkpoint SHA-256: `e8ef7228d0fb0197e31e328fee6b0667b768878b7f0cb3d803da0abdb35c5136`.
- Control archived source SHA-256: `71f59435ae8c85921dd05c948310d91e2cd6b283c58f37285c5e04542291b514`.
- The archived source is byte-identical to the frozen local `experiments/055_history_cross_attention.py`.

The corpus, physical-shard selection, replay-ring sample order, action targets, identity dropout, shared parameter initialization, seed, AWR objective and full-match returns, optimizer, schedule, batch size 512, 16,384 updates, and `2^23` sampled windows are invariant. Training uses one B200. No control training is permitted.

## Return label and decoder

For the final context frame `t`, the label is

`G_t = sum(k=1..60, 0.99855^(k-1) * r[t+k])`.

The reward is opponent damage minus ego damage, plus 120 when the opponent loses a stock, minus 120 when ego loses a stock, plus or minus 50 on the deciding stock. The label does not bootstrap from a value estimate. It is computed from the complete replay row before window sampling. A known terminal event pads later rewards with zero. A truncated tail without all 60 future frames is unavailable.

Unavailable labels disable the conditioning vector and retain the window's original policy and AWR losses. The separate full-match return remains the value target and AWR weight source.

The decoder divides the scalar by 120, applies a learned bias-free `1 -> temporal_d_model` projection, and adds the result to each action token before the temporal blocks. The projection uses an isolated RNG. Future returns do not enter observation features, trunk states, the direct trunk-to-logit path, or the value head. Teacher forcing and live decoding use the same explicit scalar and availability inputs. Each live replan receives the fixed requested return again.

Checkpoint format version 1 records the conditioning protocol, calibration values and identity, all RNG states, loader cursor, optimizer, scheduler, source hashes, Git SHA, environment, and dependency versions. Resume rejects any mismatch.

## Evaluation

Before gameplay evaluation, freeze three targets from the first 65,536 training windows: zero, and the median and 90th percentile of strictly positive valid next-60-frame returns. Save the ordered calibration rows, availability, replay identities, targets, and SHA-256 in every later checkpoint.

Evaluate the final treatment at all three targets on the registered 96-boot schedule with prediction 4, delay 2, replan 2, seed 0, and the control's sampling semantics. Select the target with the highest net stocks per active minute; a tie prefers zero, then the median, then the 90th percentile. Offline validation metrics cannot select the target.

The saved seed-0 control rows may be reused only if schema, final checkpoint hash, all protocol fields, boot identities, and the full registered schedule validate. Otherwise reevaluate the verified control checkpoint.

After target selection, evaluate the selected treatment target and verified control checkpoint on another matched 96-boot schedule with decode seed 1. Keep seed-0 selection and seed-1 confirmation files separate. Report treatment-minus-control net stocks per active minute and a paired 2,000-resample boot-cluster 95% percentile interval for each comparison.

Report actual-return and valid-label-shuffled validation NLL, label availability, valid return quantiles, damage, and decode latency as diagnostics. These do not change the target selection.

## Commands

```text
uv run experiments/058_return_conditioning.py train --proxy --comment o58-return60
uv run experiments/058_return_conditioning.py suite --checkpoint final.pt --run <treatment-run> --saved-control experiments/o58/control-selection-rows.json
```

The experiment does not change the deployment adapter and cannot adopt the treatment automatically.
