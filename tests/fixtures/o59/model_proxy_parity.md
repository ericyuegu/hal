# 059 model extraction control

Control: `d7454f9d1a136f7c745d2d4478af22e19cd65faf` in `/tmp/hal-059-control-d7454f9`. Candidate: working tree after the canonical model extraction. Both used `/home/ericgu/src/hal/.venv/bin/python` (Python 3.14.4, PyTorch version in `model_proxy_parity.json`) with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. The proxy uses seed 29 for construction, seed 73 for inputs, batch one, CPU FP32, the dense trunk, and full conditioned rollout logits.

Capture each side with the checked-in script:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/ericgu/src/hal/.venv/bin/python tests/fixtures/o59/capture_model_proxy.py /tmp/hal-059-control-d7454f9 /tmp/hal059-baseline
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/ericgu/src/hal/.venv/bin/python tests/fixtures/o59/capture_model_proxy.py /home/ericgu/src/hal /tmp/hal059-candidate --candidate
```

The control script path in the first command is the current checkout's script. Compare both JSON files for exact parameter count, names, shapes, state keys, parameter bytes, and post-construction Torch RNG. Compare the saved tensors elementwise for the dense hidden states, rollout actions, and each group's conditional logits. All matched exactly. The control identities and output hashes are in `model_proxy_parity.json`. The default 059 model has 246,862,205 parameters, checked with meta allocation.

This fixture establishes mechanical model parity. It does not establish real-checkpoint resume, GPU BF16 numerical limits, or hardware throughput; those remain separate acceptance gates.

The extracted training loop was also checked against the frozen source. `train_step`, optimizer grouping, checkpoint configuration, and provenance have the same executable bodies after type/name changes. `microbatch_loss` differs only in its private nats-to-bits owner, which keeps the same division by `math.log(2.0)`. The module-scope learning-rate function has the old warmup/stable/decay formula; `LearningRateScheduler.state_dict()` retains the format-4 `[None]` lambda record. Return calibration reads the same transformed CPU batch when the staged batch is consumed and saves the same sample/target record.

Focused CPU checks: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 UV_CACHE_DIR=/tmp/hal-uv-cache uv run pytest -q tests/test_checkpoints.py tests/test_action_sequence_checkpoint.py tests/experiments/test_059_muon_action_sequence.py -m 'not integration'` passed with 90 tests, 4 CUDA skips, and 4 sandbox NVML warnings. The relevant tests include `test_module_scope_lr_schedule_preserves_checkpoint_record_and_next_step`, `test_training_validation_change_and_next_update_resume_are_exact` (both CPU variants), `test_drained_prefetch_restores_loader_masks_and_optimizer_update`, `test_resume_lineage_permits_only_the_declared_source_transition`, and `test_checkpoint_validation_preserves_rng_and_accepts_a_resumed_descendant`. These synthetic and proxy checks do not replace the required real update-2048 and update-4096 resume comparisons.
