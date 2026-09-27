# Production 059 cached-batch numerical check

The control used the immutable `d7454f9d1a136f7c745d2d4478af22e19cd65faf`
checkout's batch-one streaming cache. The candidate used the maintained
batch-two cache with prepared observation rows. Both used the same 246,862,205
parameter checkpoint and replay on an RTX 3060 with PyTorch `2.11.0+cu130`.
The checkpoint bundle SHA-256 was
`13a2a2922ea3a015d583f3efa345368ca41ee81adc614d5fd182700337a926fa`;
the replay SHA-256 was
`81af02a949c3c436d5e527b5dd83600d42338887d4445de4dc515e489e6cbe94`.

Two independent streams consumed 280 frames each, starting at replay frames
0 and 512. At 15 matched source positions, including positions after the
256-frame context began evicting observations, the decoder was teacher-forced
with the replay's next eight controller actions. This produced 960 matched
conditional categorical distributions across the four controller groups.
The old streaming cache is the control after eviction; a newly cropped dense
window is not an exact oracle for its hidden states.

| Cohort | Count | Mean KL, nats | p99 KL, nats |
|---|---:|---:|---:|
| All | 960 | 0.00007230 | 0.00114066 |
| Before eviction | 576 | 0.00005188 | 0.00103042 |
| After eviction | 384 | 0.00010293 | 0.00137659 |

Every cohort passed the proposed mean `<= 5e-4` and p99 `<= 5e-3` limits.
All logits and KL values were finite. Tiny negative computed KL minima, down
to `-1.6e-7`, are floating-point rounding.

The durable capture script `tests/fixtures/o59/production_kl_capture.py`
SHA-256 is
`5a9b4c8c52f8602480138fdcfcbb72b9025d3c62c99210aea8af848371779cb4`.
The comparison script `tests/fixtures/o59/production_kl_compare.py` SHA-256
is `a21649daa16dd7f7ab57407604e92455052676f41c6aa70e498b7203cad67d4d`.
The candidate policy, cache, GPU observation, and model source SHA-256 values
were respectively `2031757d0abcc2931b7924e3558246b0d9710933340d04c36dd95a2cdaac8868`,
`6d75628c4688d610c1bed93c94873d960ba25572d3104af61bc5a128b2d8dadc`,
`e2d417c56344fec88c9f5ba21f77d5b701b5d84ac118e6d3499f3a2a2e101e2c`,
and `4b33191ce26c44313a9e948e501432dc36003549b116492e7ee9edcc93dfc7d2`.

Run the capture in each checkout with `CHECKOUT` set to that checkout's root,
and set `MODE` to `control` or `candidate` respectively:

```bash
PYTHONPATH="$CHECKOUT" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  TMPDIR=/home/ericgu/src/hal/runs/refactor-059/tmp \
  /home/ericgu/src/hal/.venv/bin/python \
  /home/ericgu/src/hal/tests/fixtures/o59/production_kl_capture.py "$MODE" \
  /home/ericgu/src/hal/runs/netplay/o59-vywk3cih.hal \
  /home/ericgu/src/hal/runs/netplay/selfplay-o59-20260924-retry1/replays-a/Game_20260924T152607.slp \
  "/home/ericgu/src/hal/runs/refactor-059/production-kl-$MODE.pt"
```

Run `production_kl_compare.py` on the two captures to recreate the summary.
The control and candidate raw captures are
`runs/refactor-059/production-kl-control.pt` (SHA-256
`f08eb10c4c203144f33b2beb0e14a39274df1c1ec928c0c9dbc34dcbf2402033`)
and `runs/refactor-059/production-kl-candidate.pt` (SHA-256
`1f98cae93a9837f5b766a43e1b18bfe79cb8d5d32fb020f76991ccb8ecd2e656`).
The full distribution summary is `runs/refactor-059/production-kl-result.json`
(SHA-256 `7eacd4748ad66ed03532929e1b9c122e071a7ed8e96cc43b9ba161c7204d775c`).
