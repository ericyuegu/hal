# O64 validation

Local validation uses Python 3.14.4 and PyTorch 2.11.0+cu130. Commands use
`UV_CACHE_DIR=/tmp/hal-uv-cache UV_NO_SYNC=1`. CUDA tests run with host access on
the local RTX 3060. Scratch uses `TMPDIR=outputs/o64-validation-tmp` (an absolute
workspace path in the actual environment) and pytest `--basetemp=outputs/t`.

| Command | Final result |
| --- | --- |
| `uv run ruff format --check .` | Passed; 344 files |
| `uv run ruff check .` | Passed |
| `uv run ty check --python-version 3.14 --error-on-warning hal experiments/059_muon_action_sequence.py experiments/063_percentile_conditioning.py experiments/064_production_percentile.py scripts` | Passed; zero diagnostics |
| `uv run pytest -q tests/experiments/test_064_production_percentile.py -k validation --basetemp=outputs/t -x --tb=short` | Seven passed, 12 deselected; CPU and CUDA; 5.40 s |
| `uv run pytest -q -m "not integration" --basetemp=outputs/t -x --tb=short -rs` | 2,190 passed, eight optional skips, 65 deselected, 26 warnings; 176.97 s |
| `HAL_REQUIRE_INTEGRATION=1 uv run pytest -q tests/test_roundtrip.py tests/test_session_cleanup.py -m integration` | Eight passed, six deselected, seven warnings; 60.16 s; no skips |

The eight optional non-integration skips are two production netplay GPU
qualifications and six tests for an unavailable optional local v7 dataset
subset. Required roundtrip/session fixtures were present.

The O64 tests cover config differences from O63, parameter and optimizer
membership, all 26 action offsets, loss coefficients and gradients, unchanged
percentile targets and RNG draws, checkpoint identity, u95 defaults, and exact
CPU, CUDA eager and CUDA compiled next-update resume. Final-prefix validation
is compared with the original full-prefix calculation for all three arms,
with nonzero conditioning, padding and unavailable labels, on CPU and CUDA.
A separate test checks teacher-forced score aggregation across batch sizes.
The requested independent review found no remaining correctness blocker.

The full corpus audit verified all 44 pinned train indexes and split row
identities. All replay lengths exceed the required 293 frames. It streamed
metadata only; no MDS dataset was downloaded. See
[corpus_length_audit.json](corpus_length_audit.json).

## B200 qualification

The initial device-resident training check passed. A real-prefetch followup
passed eight updates but failed validation batch 128 with resident training
CUDA graphs. A single 32-row validation batch passed in the matched timing
benchmark, but the subsequent full-cohort attempt at batch 32 failed with CUDA
OOM in eager history attention. Both failures are retained.

Validation previously decoded all 128 suffix prefixes while scoring only the
last. The final implementation decodes only that prefix and restores the
original batch 128. It preserves full-suffix value/percentile computation,
labeler draw shape/order, the full context memory, cohort and training path.
The final B200 qualification passed ten training updates, all 2,048 validation
samples in 7.623 seconds, and eight subsequent training updates. It used
production learning rates, compilation/CUDA graphs, and real prefetch/dropout.
All losses and gradient norms were finite. Peak allocated/reserved memory was
166.60/175.86 GiB. No dataset was downloaded. See
[full_validation_preflight.json](full_validation_preflight.json).

The matched throughput/profile investigation rejected static caches, explicit
rotary backward, Flash SDPA and max-autotune. No rejected implementation remains
in maintained code. Its raw results and limitations are in
[performance/report.md](performance/report.md).

## Earlier checks and failures

The original O64 implementation passed 12 focused tests and a full CUDA suite
(2,183 passed). The temporary cache candidate passed 16 focused tests and a full
suite (2,187 passed); it was later removed because its timing overlapped the
repeated control. The interim batch-32 implementation passed 2,184 tests before
the final-prefix optimization. These do not replace the final checks above.

Earlier full-suite attempts hit host temporary-file quotas and a Unix socket
path-length limit. Workspace scratch and a short pytest basetemp resolved them.
The first EvalArgs fixture omitted its required checkpoint argument. The first
full-prefix reference fixture used another experiment's typed model boundary
and was rejected by runtime checks; it was replaced with a direct reference
calculation on the same model. Both fixture failures were corrected. Formatting
checks also found a long line and an extra blank line during editing; the final
format check passed. No required fixture was skipped.
