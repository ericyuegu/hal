# Cached batching proxy parity

Control: one 059 `ActionSequenceTransformer` evaluated with independent batch-one
KV caches. Candidate: the same model and tokens evaluated through `KVCachePool`
with gathered rows and one batched trunk/decoder call. The proxy uses width 32,
two trunk layers, context 8, and seed 47. It is a mechanical test, not a
production checkpoint qualification.

- CPU FP32: batched hidden states, projected history attention, and all
  persistent KV rows matched serial execution at `atol=2e-6, rtol=2e-5`
  through row permutations, sparse ready sets, multiple wraps, and a
  single-row reset. Test: `tests/test_action_sequence_kv_batch.py`.
- CPU policy: two arbitrary stream IDs used one trunk call and one decoder
  call for four-frame updates; actions and keyed RNG counters matched serial
  execution, including a three-frame update decomposed as 2+1. Test:
  `tests/test_action_sequence_policy_batch.py`.
- RTX 3060 BF16: 256 forced conditional categorical comparisons over eight
  four-frame updates and multiple wraps were finite. Mean KL = 0 nats;
  p99 KL = 0 nats. The tiny proxy does not establish the production
  checkpoint's proposed `5e-4` mean / `5e-3` p99 limits.
- RTX 3060 CUDA graphs: capacity-one and capacity-two policies captured every
  1/2/4-frame update shape and decoder bucket before stream admission.
  Subsequent B1 and B2 predictions replayed the prepared graphs.
- RTX 3060 graph versus eager: 45 one-frame netplay-profile predictions crossed
  multiple ring wraps and a generation reset at frame 24. Position metadata
  matched exactly; hidden, active trunk KV slots, and active projected-history
  KV slots matched within `atol=0.035, rtol=0.035`. Inactive ring slots are not
  compared because reset invalidates them without zeroing old bytes.
- RTX 3060 shared model: three cached policies prepared H4/prefix0,
  H8/prefix3, and H8/prefix4 on one model object, then ran A/B/C/A under
  `fail_on_recompile`. Their cache storage, target frames, horizons, prefixes,
  and captured call sets remained independent.

The matched direct-policy H8/prefix3/stride4/update4 benchmark used the same
bundle and replay on the RTX 3060 for 2,400 frames. Control p95/p99 were
8.44974/8.95109 ms with 536.33 MiB peak allocation. After binding CUDA
Linear weights to BF16 once at artifact load, candidate p95/p99 were
8.50424/8.86401 ms with 536.32 MiB peak allocation. The first candidate
trial was retained as a failed diagnostic: its FP32 Linear weights caused
repeated autocast conversion, p95 20.24299 ms, and 1002.59 MiB peak.
Raw records: `runs/refactor-059/control-3060-trial-1/results.json`,
`runs/refactor-059/candidate-3060-trial-1/results.json`, and
`runs/refactor-059/candidate-3060-trial-2/results.json`. This measures direct
policy calls; it is not the service transport or long-running hardware gate.

CUDA command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/pytest -q -s
tests/test_action_sequence_kv_batch.py::test_cuda_bf16_batched_conditional_logits_match_serial_before_and_after_eviction`
and `tests/test_action_sequence_policy_batch.py::test_cuda_graphs_are_captured_before_stream_admission`.
The graph/eager test additionally requires
`HAL_REQUIRE_NETPLAY_HARDWARE_QUALIFICATION=1` and is marked integration.
