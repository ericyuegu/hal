# HAL

Training superhuman AI for *Super Smash Bros. Melee*. 

This project is under active development and is not ready for public use. 

Blog post: https://ericyuegu.com/melee-pt1

# Setup

This project targets Python ≥ 3.14 on Ubuntu 20.04+. Dependencies are managed by [uv](https://docs.astral.sh/uv/).

`peppi-py` (the slp parser used by the data pipeline) is pulled from a fork and built from source via `maturin`, so a Rust toolchain is required:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if you don't have uv
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
. "$HOME/.cargo/env"
uv sync
```

The first `uv sync` will compile `peppi-py` (~35s); subsequent syncs reuse the cached build.

For macOS, `libmelee` requires a system installation of enet:
```bash
brew install enet
CFLAGS="-I$(brew --prefix enet)/include" \
LDFLAGS="-L$(brew --prefix enet)/lib -lenet" \
uv sync
```

## Dolphin emulator

Download the latest Slippi ExiAI AppImage (e.g. `Slippi_Online-x86_64-ExiAI.AppImage`) into `~/data/ssbm/` and extract it once:

```bash
chmod +x ~/data/ssbm/Slippi_Online-x86_64-ExiAI.AppImage
( cd ~/data/ssbm && ./Slippi_Online-x86_64-ExiAI.AppImage --appimage-extract )
```

`libmelee` should be pointed at `~/data/ssbm/squashfs-root/AppRun`. The ExiAI build forces a Null video backend, so it runs headless with no X display required. To build the emulator from source instead, follow the instructions [here](https://github.com/ericyuegu/slippi-Ishiiruka/tree/ubuntu-20.04).

## Downloading data

You can obtain raw `.slp` files from the [Slippi Discord](https://discord.gg/qaHgPwpr) server.

# HOW-TO

Paths to the repo, Dolphin, ISO, and replay directory are resolved by `hal/local_paths.py` from environment variables, with defaults that match the layout above (`~/data/ssbm/...`). To override, copy `.env.example` to `.env` and edit, or `export` the variables in your shell profile.

## Processing replays to MDS format

The data pipeline runs in three stages:

1. **`build_index`** walks loose `.slp` files (or streams from a `.7z` archive) and writes `index.jsonl` — one row of metadata per replay.
2. **`filter_replays`** is a pure-function pass over `index.jsonl` that emits a `paths.txt` for the next stage based on rank / character / version / frame-count predicates.
3. **`process_replays`** consumes `paths.txt` + `index.jsonl`, parses every kept replay's frames, and writes MDS shards (`train`/`val`/`test`) plus a `manifest.jsonl` sidecar.

`paths.txt` is self-describing — each line is either a filesystem path or `archive://<abs-archive>!<member>`. A single `paths.txt` may mix loose files with members from one or more archives, and Stage 3 will bucket and stream them appropriately.

```bash
# Stage 1 — index loose .slp files on disk
uv run python -m hal.data.build_index \
    --root ~/data/ssbm/dev \
    --output ~/data/hal/index.jsonl

# Stage 1 — index .slp members directly from a .7z (no extraction; tmpfs-backed)
uv run python -m hal.data.build_index \
    --archive ~/data/ssbm/melee_public_slp_dataset_v2.7z \
    --output ~/data/hal/index.jsonl

# Stage 1 — fold another archive into the same index
uv run python -m hal.data.build_index \
    --archive ~/data/ssbm/ranked-anonymized-2-151807.7z \
    --output ~/data/hal/index.jsonl \
    --incremental

# Stage 2 — filter to a paths.txt (no .slp opens; runs in seconds)
uv run python -m hal.data.filter_replays \
    --index ~/data/hal/index.jsonl \
    --output ~/data/hal/paths.txt \
    --min-frames 1500 \
    --rank master diamond platinum

# Stage 3 — paths.txt + index.jsonl → MDS shards + manifest.jsonl
# (Archive entries in paths.txt are streamed automatically; no --archive flag.)
uv run python -m hal.data.process_replays \
    --paths-file ~/data/hal/paths.txt \
    --index ~/data/hal/index.jsonl \
    --output ~/data/hal/mds
```

In archive mode `build_index` and `process_replays` materialize each member to a bounded tmpfs ring (default `/dev/shm/...`, ~64 files in flight) just long enough for `peppi-py` to parse it, then unlink — the archive itself is never extracted to persistent storage.

End-to-end smoke tests live in `tests/test_archive_streaming.py` and skip automatically when the `~/data/ssbm/dev.7z` fixture isn't present:

```bash
uv run pytest tests/test_archive_streaming.py -v
```

## Training

```bash
uv run python hal/training/simple_trainer.py --n_gpus 1 --data.data_dir /path/to/mds --arch GPTv5Controller-512-6-8-dropout
```

## Evaluation

```bash
uv run python hal/eval/eval.py --model_dir /path/to/model_dir --n_workers 1
```
