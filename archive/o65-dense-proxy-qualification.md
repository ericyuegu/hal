# Historical dense-proxy comparison

The dense-proxy comparison was recorded at
`2ff3e77adfa0eb14505dfdd16758886e56c1aa6e`, against trainer control `aacdfee0`.
Its results are described in `experiments/o65/performance.md`. The original
reports remain in Git and in the recorded Modal volumes. They did not establish
statistical equivalence or qualify the later checkpoint-32 continuation.

The maintained test and its two helpers were removed after
`cd48e073203dcbf034171c440a3ec0e22a483051`. That commit preserves their exact
source at `tests/test_065_dense_qualification.py`, `tests/o65_precision.py`, and
`tests/o65_reference.py`.

The helper imported the historical trainer into the current library and replaced
two entries in `sys.modules`. Its replay adapter expected `log_distributions`,
but the recorded trainer's evaluation object contains `distributions` in
probability units. This was no longer a working numerical qualification. Reproduce
the historical comparison from its recorded commit, environment, checkpoint and
data, rather than combining that trainer with current library modules.

Current per-decision/window output and gradient parity remains in
`tests/test_window_replay_model.py`. The actor objective, masks, reference
detachment and graph replay checks remain in `tests/test_window_replay_actor.py`
and `tests/test_actor_diagnostic_targets.py`. Full-model qualification must use
the captured checkpoint-32 actor operation and the exact repaired continuation;
the historical proxy result cannot substitute for it.
