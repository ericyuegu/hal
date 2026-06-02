# HAL

Training superhuman AI for *Super Smash Bros. Melee* via imitation learning and RL.

Blog: https://ericyuegu.com/melee-pt1.

## Setup

```bash
git clone git@github.com:ericyuegu/hal.git
cd hal
uv sync

# download emulator & datasets
cp .env.example .env && $EDITOR .env  # fill in AWS_* creds — ask Eric
source .env                           # or use direnv
uv run fetch                          # download ~2 GB into <repo>/data/
uv run pytest tests/                  # expect: 39 passed
```

All HAL data lives under `<repo>/data/` (gitignored). After fetch:

```
data/
├── raw/
│   └── dev.7z                        # 37 MB slp archive (+ ranked-anonymized-*.7z if you have them)
├── processed/
│   └── dev/
│       └── mds/
│           ├── train/
│           ├── val/
│           ├── test/
│           ├── manifest.jsonl
│           └── stats.json
├── emulator/
│   ├── ssbm.ciso                     # ISO
│   └── exiai/
│       └── squashfs-root/
│           └── AppRun                # headless Dolphin
├── scratch/                          # throwaway Dolphin recordings, debug dumps
└── runs/                             # eval rollouts (per run_id)
```

Real data on a different drive? Symlink: `ln -s /mnt/big/hal/data data` (or symlink individual files under `data/raw/`).

Fetch a single fixture with `fetch --name {dev.7z | dev-mds | ssbm.ciso | dolphin-exiai}`. Override path defaults via `HAL_*` env vars — see `hal/paths.py`.


## Data pipeline

Three stages in `hal/scripts/` turn `.slp` files (loose or `.7z`-archived) into MDS shards:

1. **`index`** — walk replays; write `index.jsonl` (one row of metadata per file).
2. **`filter`** — query `index.jsonl`; emit `paths.txt` of replays to keep.
3. **`materialize`** — read `paths.txt` + `index.jsonl`; write MDS shards + `manifest.jsonl` + `stats.json`.

`paths.txt` lines are either filesystem paths or `archive://<abs-archive>!<member>` synthetic paths; one file may mix both, and the materialize stage streams archive members from a bounded tmpfs ring (`/dev/shm/...`) without ever extracting to disk.

```bash
# step 1
uv run hal.scripts.build_index --archive data/raw/dev.7z --output /tmp/index.jsonl

# step 2
uv run hal.scripts.filter --index /tmp/index.jsonl --output /tmp/paths.txt

# step 3
uv run hal.scripts.materialize --paths-file /tmp/paths.txt --index /tmp/index.jsonl --output /tmp/mds
```


## Cloud training (vast.ai)

`scripts/launch_vast.py` rents a GPU and runs one training command on it,
fire-and-forget: it pushes the current git SHA, queues for an offer under a price
ceiling, rents it, and injects the SHA + command. The box clones that SHA into the
prebuilt image (`docker/Dockerfile`), `uv sync`s, fetches data from R2, trains, then
tears *itself* down — **destroy** on success (checkpoints in R2, logs in W&B),
**stop** on failure (keeps `/opt/hal/train.log` for inspection). The instance is
stateless; a preempted run resumes from its latest R2 checkpoint with `--resume`.

```bash
uv run scripts/launch_vast.py                          # search-only: print offers, rent nothing
uv run scripts/launch_vast.py --dry-run -- <cmd>       # preflight + search, print what would be sent
uv run scripts/launch_vast.py --max-price 0.80 \
    -- uv run experiments/001_flow_matching_baseline.py --cfg.max-steps 100000
uv run scripts/launch_vast.py \                        # resume a preempted run
    -- uv run experiments/001_flow_matching_baseline.py --resume <run_name>
```

Everything after `--` is the training command. The host needs only a vast API key
(`~/.config/vastai`); R2/W&B secrets live as vast **account** env-vars (`AWS_*`,
`WANDB_API_KEY`) and inject into the box out-of-band — the launcher just verifies
they're present and refuses a dirty tree. Useful flags: `--max-price` (impatience
knob), `--keep-alive` (leave the box up to SSH in). Watch W&B or `vastai logs <id>`.

To smoke-test the image locally instead: `docker compose -f docker/compose.yaml run --rm hal <cmd>`.

## Adding a new fixture (maintainer)

The fixture registry is `hal/fixtures.py`. Two backends:

**R2 (private artifacts: replays, derived bundles, ISO):**

```bash
rclone copy -P --stats-one-line <local-path> r2:hal/fixtures/<key>
sha256sum <local-path>
```

Then add a `Fixture(...)` entry with `r2_key="fixtures/<key>"`, the sha256, and a repo-relative `dest=Path("data/<subdir>/...")`. If the artifact is a directory tree, tar it with `tar --use-compress-program='zstd -19 -T0'` and set `extract="tar_zst"` (the tarball must contain its files at the root, not under a wrapper dir).

**Upstream URLs (public binaries from GitHub Releases / CDNs):**

```bash
curl -sL <url> | sha256sum
```

Then add a `Fixture(..., url="<pinned-tag-url>", sha256=..., extract=...)` entry. Don't re-host upstream binaries to R2 — pin the tag and verify by sha256. If the release is ever yanked, mirror to R2 then.

After editing `hal/fixtures.py`, the new fixture is included in `fetch` and selectable via `--name`. Sha mismatch is a hard error — `ensure()` fails loud.

## Appendix: prerequisites

Tested on Ubuntu 20.04+. Linux is highly recommended — macOS works with one extra dep; Windows is dodgy.

### uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Rust toolchain

`peppi-py` is pulled from a fork and compiled via `maturin` during `uv sync` (one-time, ~35s; cached after). Only required if you're going to parse `.slp` files locally — i.e. run `hal.scripts.{index,filter,materialize}`, `roundtrip`, or anything that imports `hal.data.extract`. The plain `fetch` step doesn't touch peppi-py at runtime, but `uv sync` still tries to compile it, so install Rust once if you don't have it:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
. "$HOME/.cargo/env"
```

### macOS

`libmelee` needs a system `enet`:

```bash
brew install enet
CFLAGS="-I$(brew --prefix enet)/include" LDFLAGS="-L$(brew --prefix enet)/lib -lenet" uv sync
```
