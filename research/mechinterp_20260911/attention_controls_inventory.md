# Attention-control run inventory

The supplied timestamped all-AdamW name is not a second control run. It is the
exact W&B display name and R2 directory for `tgpuo1be`.

## Identity and compatibility

- W&B: `ericyuegu/hal/tgpuo1be`;
- source commit: `d67c68cb36c830bb4100ba8ef1b4cefc452539de`;
- logged code: `source-hal-experiments_050_scaled_temporal_awr.py:v12`;
- experiment: `050_scaled_temporal_awr_v6`;
- optimizer: all AdamW, learning rate 0.0008, weight decay 0.0001;
- representation: schema 7, policy-world-v8, d1024/L16/h16/context256,
  temporal d512/L4, prediction frames 4;
- resolved-config SHA-256:
  `5d4d851c9e009ecd237da093326c691fbe7eec392e91ecfdf957271b1f3eaeec`;
- source-selection SHA-256:
  `2593361352b92e705be3fbeae1b4e9bb1a3c9f1787cd713014a7a95b7df62477`.

Both selected exports have identical config and source-manifest hashes. They
are compatible with the frozen O50 attention audit. No conversion or new
checkpoint download is necessary.

## Selected snapshots

| Boundary update | Local model-only export SHA-256 | Source checkpoint SHA-256 |
| ---: | --- | --- |
| 57,344 | `2584c1664f48750bf39391ce682c8d9d58249d957ad1d0f4c3f51e21153375b4` | `1a2b159184e0e2e19616d9b5e62cfbf9c8d9d326c72ff1fc3cff5ed2c6a6848a` |
| 131,072 | `15e58bfc14c69151ec296422b9d6d44d55bcd5d4c9ca6f551dba7cb688f63175` | `4c5d43e2b5f9351d13845051b30b0ad4871c76975ec350c7cbd85ed73f60c0e1` |

The local files are in
`/home/ericgu/src/hal/runs/mechinterp-20260911-scratch/`. Update 131,072 is the
latest numbered saved endpoint; update 57,344 is the existing earlier
same-run comparison. This repeat does not select a checkpoint by loss or
attention. The R2 run contains numbered checkpoints every 8,192 updates from
8,192 through 131,072, plus `final.pt`.

## Recorded health context

The frozen `scan_history` extracts reach `global_step=131072`. All 5,248
recorded training rows have finite loss, NLL, gradient norm, and clip fraction.
After warm-up, gradient norm stayed in [0.1846, 0.7550], and the 25-update
window clip fraction stayed at zero. Validation NLL improved from 1.6927 at
update 4,096 to 1.3826 at 57,344 and 1.3301 at 131,072. The 96-boot gameplay
evaluations had zero crashes and net stock rates of +0.3874/min at 57,344 and
+0.4937/min at 131,072.

This is evidence that the recorded optimizer and loss series did not blow up.
It does not override the W&B process state, which is `crashed`, or establish
operational completion from that label. The durable numbered endpoint and
`final.pt` are separately recorded in the immutable upload log.

Machine-readable paths, ETags, byte counts, hashes, and frozen-history source
hashes are in `artifacts/attention_controls/run_inventory.json`.
