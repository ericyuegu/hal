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

Two builds of [Slippi-Ishiiruka](https://github.com/project-slippi/Ishiiruka) are supported, side by side:

- **`exiai`** — the headless ExiAI fork. Forces a Null video backend, no X display required. This is the default for training/eval rollouts.
- **`slippi`** — the upstream Slippi netplay build with the WX GUI. Use this for interactive debugging, replay playback, or watching rollouts.

Place each build under `~/data/dolphin/<name>/` with the AppImage extracted once. The ISO sits next to them:

```
~/data/
  dolphin/
    ssbm.ciso
    exiai/
      Slippi_Online-x86_64-ExiAI.AppImage
      squashfs-root/   # AppRun
    slippi/
      Slippi_Online-x86_64.AppImage
      squashfs-root/   # AppRun
```

To extract an AppImage:

```bash
chmod +x ~/data/dolphin/exiai/Slippi_Online-x86_64-ExiAI.AppImage
( cd ~/data/dolphin/exiai && ./Slippi_Online-x86_64-ExiAI.AppImage --appimage-extract )
```

`libmelee` defaults to `~/data/dolphin/exiai/squashfs-root/AppRun`. Override via `HAL_EMULATOR_PATH` to point at the slippi build instead. To build either from source, follow the instructions [here](https://github.com/ericyuegu/slippi-Ishiiruka/tree/ubuntu-20.04).

## Downloading data

You can obtain raw `.slp` files from the [Slippi Discord](https://discord.gg/qaHgPwpr) server.

# HOW-TO

Paths to the repo, Dolphin, ISO, and data directories are resolved by `hal/paths.py` from environment variables, with defaults rooted at `~/data/` (override via `HAL_DATA_HOME`). The layout below uses `~/data/raw/` for human-source `.7z` archives + extracted `.slp` trees, `~/data/processed/` for pipeline outputs (index, MDS shards, manifests), `~/data/scratch/` for throwaway recordings, and `~/data/runs/<run_id>/` for eval rollouts. To override individual paths, copy `.env.example` to `.env` and edit, or `export` the variables in your shell profile.

## Processing replays to MDS format

The data pipeline lives in `hal/scripts/` and runs in three stages:

1. **`index`** walks loose `.slp` files (or streams from a `.7z` archive) and writes `index.jsonl` — one row of metadata per replay.
2. **`filter`** is a pure-function pass over `index.jsonl` that emits a `paths.txt` for the next stage based on rank / character / version / frame-count predicates.
3. **`materialize`** consumes `paths.txt` + `index.jsonl`, parses every kept replay's frames, and writes MDS shards (`train`/`val`/`test`) plus a `manifest.jsonl` sidecar.

`paths.txt` is self-describing — each line is either a filesystem path or `archive://<abs-archive>!<member>`. A single `paths.txt` may mix loose files with members from one or more archives, and the materialize stage will bucket and stream them appropriately.

```bash
# Stage 1 — index loose .slp files on disk
uv run python -m hal.scripts.index \
    --root ~/data/raw/dev \
    --output ~/data/processed/index.jsonl

# Stage 1 — index .slp members directly from a .7z (no extraction; tmpfs-backed)
uv run python -m hal.scripts.index \
    --archive ~/data/raw/melee_public_slp_dataset_v2.7z \
    --output ~/data/processed/index.jsonl

# Stage 1 — fold another archive into the same index
uv run python -m hal.scripts.index \
    --archive ~/data/raw/ranked-anonymized-2-151807.7z \
    --output ~/data/processed/index.jsonl \
    --incremental

# Stage 2 — filter to a paths.txt (no .slp opens; runs in seconds)
uv run python -m hal.scripts.filter \
    --index ~/data/processed/index.jsonl \
    --output ~/data/processed/paths.txt \
    --min-frames 1500 \
    --rank master diamond platinum

# Stage 3 — paths.txt + index.jsonl → MDS shards + manifest.jsonl
# (Archive entries in paths.txt are streamed automatically; no --archive flag.)
uv run python -m hal.scripts.materialize \
    --paths-file ~/data/processed/paths.txt \
    --index ~/data/processed/index.jsonl \
    --output ~/data/processed/mds
```

In archive mode `index` and `materialize` materialize each member to a bounded tmpfs ring (default `/dev/shm/...`, ~64 files in flight) just long enough for `peppi-py` to parse it, then unlink — the archive itself is never extracted to persistent storage.

End-to-end smoke tests live in `tests/test_archive_streaming.py` and skip automatically when the `~/data/raw/dev.7z` fixture isn't present:

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
