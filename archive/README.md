# Research archive

This directory preserves research programs, notes, and evidence from before the maintained 059 path. It is not a supported runtime. The programs can refer to library modules and commands that were later removed.

To reproduce a historical result, use its recorded Git commit, Python environment, data manifests, and artifacts. The pre-refactor source control is commit `d7454f9d1a136f7c745d2d4478af22e19cd65faf`. Historical CSV rows keep their original source paths; their paths are evidence, not active entrypoints.

The maintained experiment and its current evidence remain in `experiments/059_muon_action_sequence.py` and `experiments/o59/`.

`archive/scripts/probe_sm120.py` preserves the earlier FlexAttention/Blackwell diagnostic from `docker/probe_sm120.py`. The maintained 059 trunk uses a different attention backend, so this probe is not part of current startup.
