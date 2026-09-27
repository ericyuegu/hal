# RTX 3060 functional CUDA checks

The first full CUDA run used the installed Python 3.14/PyTorch 2.11.0 CUDA
environment, one Torch CPU thread, and the workspace TorchInductor cache.
The exact command, warnings, failure traceback, and process resource summary
are in `runs/refactor-059/cuda-functional-3060.log`.

Selected files: the 059 experiment test; action-sequence model, artifact,
checkpoint, attention, chunks, prefill, KV reference/batching, timing, and
policy-batching tests; GPU observations; Muon; window and CUDA-graph
preparation; inference sampling and engine tests. The command used
`pytest -q -ra -m 'not integration'` with the listed test paths.

Result: **211 passed, 1 failed, 1 integration test deselected, 15 warnings**
in 147.18 seconds. No required fixture was skipped. The failure was
`test_evaluation_persists_emulator_and_inference_metrics_separately`: the
experiment evaluation factory passed the global CUDA default to `make_policy`
although this test supplied a CPU model and window executor on a CUDA host.
`make_policy` correctly rejected the mismatch. The factory needs to pass its
model's actual device. This is an evaluation-wrapper defect, not a numerical
disagreement in the 059 update or cached batch path. The failed run remains
part of the evidence. The fix removed the redundant `make_policy(device=...)`
argument so the supplied `WindowPolicy` takes its device from its model. The
CUDA-visible focused rerun of the affected evaluation/factory cases passed:
**8 passed, 56 deselected in 3.32 seconds**, with the exact command and output
in `runs/refactor-059/evaluation-device-seam.log`. A full final-source CUDA
rerun remains outstanding.

This suite uses proxy training geometry for next-update resume. It does not
replace the required representative production-checkpoint resume comparison.
The opt-in complete-path netplay hardware tests are separate and have not run
in this record.
