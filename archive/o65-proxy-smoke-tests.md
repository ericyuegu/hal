# Retired o65 proxy smoke tests

The following optional tests belong to the earlier proxy campaign. They required
an old checkpoint or machine-specific paths, skipped in normal validation, and
did not qualify the current production continuation:

- `tests/test_o65_proxy_checkpoint.py`: two dtype cases against the O63
  bootstrap30 checkpoint, SHA-256
  `65e7e43e5cbb18a365a005922ec1cfd026ad807c045eae13b331f487b4ec82e6`.
- `tests/test_o65_reward.py::test_short_gpu_reward_treatment`: two native updates
  from `runs/proxy/final.pt` on an RTX 3060.
- `tests/test_o65_fixed_layout.py::test_short_gpu_run_has_no_compiles_or_captures_after_startup`:
  three native updates from the same proxy with an older layout.

Their last retained source is `7718afdb7e0ce67b232e53679bc777af5ac235a2`.
Use `git show <commit>:<path>` to inspect that source. Historical reports under
`experiments/o65/training-throughput*` retain their original results and skips.

Current reward migration, exact resume, fixed layout, cached inference, replay
gradients and CUDA graph tests remain maintained. Production qualification must
use the actual checkpoint-32 continuation, its production shapes and final
compiler-cache identity; the proxy tests cannot substitute for that run.
