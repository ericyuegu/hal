# %% [markdown]
# Filter exploration: tune the heuristics in `hal/scripts/filter.py` against the
# actual distribution of borderline replays.
#
# How to use:
#  1. In another terminal: `cd ~/src/slippilab && npm run dev` (port 3000).
#  2. SSH-forward `-L 3000:localhost:3000 -L 8000:localhost:8000`.
#  3. Step through the cells. Adjust `cfg` and re-run the bucketing cell to
#     iterate. The last cell emits the equivalent `python -m hal.scripts.filter`
#     invocation for the thresholds you settled on.
#
# Buckets:
#  - `accept_clean`  passes everything, no marginal stat
#  - `accept_near`   passes everything but `frames < min_frames + slack`
#  - `reject_near`   fails exactly one *soft* predicate by a slack-sized margin
#  - `reject_hard`   fails ≥2 predicates, or any hard one

# %%
import random
import urllib.parse
from collections import Counter
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import melee
import py7zr
from loguru import logger

from hal.data.archive import parse_archive_member_path
from hal.data.index import ReplayIndexEntry
from hal.data.index import read_jsonl
from hal.data.replay_stats import PlayerStatsMins
from hal.paths import REPO_DIR
from hal.policy import INCLUDED_STAGES
from hal.scripts.filter import _resolve_ids
from hal.scripts.filter import _resolve_stages
from hal.scripts.filter import build_predicates
from hal.wire import CHARACTERS_BY_NAME
from hal.wire import slp_stage_to_libmelee

INDEX = Path("~/src/hal/data/processed/ranked-anonymized-2/index.jsonl").expanduser()
OUT_DIR = Path(__file__).parent / ".filter_explore"
OUT_DIR.mkdir(exist_ok=True)
EXTRACT_CACHE = OUT_DIR / "extracted"
EXTRACT_CACHE.mkdir(exist_ok=True)

SLIPPILAB_URL = "http://localhost:5173"
SLIPPILAB_PUBLIC = Path("~/src/slippilab/public").expanduser()
# Symlink EXTRACT_CACHE into slippilab's public/ so vite serves the slps at
# `<SLIPPILAB_URL>/<SERVE_MOUNT>/...` — same origin as the app, no second port.
SERVE_MOUNT = "hal-replays"
_serve_dir = SLIPPILAB_PUBLIC / SERVE_MOUNT
if not _serve_dir.exists():
    if not SLIPPILAB_PUBLIC.exists():
        raise FileNotFoundError(f"slippilab public/ not found at {SLIPPILAB_PUBLIC}")
    _serve_dir.symlink_to(EXTRACT_CACHE)
    logger.info(f"symlinked {_serve_dir} -> {EXTRACT_CACHE}")


def materialize(entry_path: str) -> Path:
    """Resolve an entry's path to a file under EXTRACT_CACHE (so it's served via
    slippilab's vite server). Archive members get extracted; plain files get
    symlinked. Cached by basename, so re-runs are instant.
    """
    parsed = parse_archive_member_path(entry_path)
    if parsed is None:
        src = Path(entry_path)
        if not src.is_absolute():
            src = Path(REPO_DIR) / src
        cached = EXTRACT_CACHE / src.name
        if not cached.exists():
            cached.symlink_to(src)
        return cached
    archive, member = parsed
    if not archive.is_absolute():
        archive = Path(REPO_DIR) / archive
    cached = EXTRACT_CACHE / f"{archive.stem}__{Path(member).name}"
    if cached.exists():
        return cached
    with py7zr.SevenZipFile(str(archive), "r") as z:
        z.extract(path=str(EXTRACT_CACHE), targets=[member])
    (EXTRACT_CACHE / member).rename(cached)
    return cached


def _failure_key(failed: list[str]) -> str:
    return failed[0] if len(failed) == 1 else " + ".join(sorted(failed))


def group_rejections(
    buckets: dict,
) -> tuple[dict[str, list[ReplayIndexEntry]], dict[str, list[ReplayIndexEntry]]]:
    """Group all rejects by exact failure set, and by each failed predicate.

    ``by_predicate`` lists an entry once per predicate it failed (sum of group
    sizes can exceed total rejects). ``by_exact`` lists each entry once.
    """
    by_exact: dict[str, list[ReplayIndexEntry]] = defaultdict(list)
    by_predicate: dict[str, list[ReplayIndexEntry]] = defaultdict(list)

    for entry, label in buckets["reject_near"]:
        by_exact[label].append(entry)
        by_predicate[label].append(entry)
    for entry, failed in buckets["reject_hard"]:
        by_exact[_failure_key(failed)].append(entry)
        for label in failed:
            by_predicate[label].append(entry)

    return (
        dict(sorted(by_exact.items(), key=lambda kv: -len(kv[1]))),
        dict(sorted(by_predicate.items(), key=lambda kv: -len(kv[1]))),
    )


def _safe_reason_name(label: str) -> str:
    return label.replace("/", "_").replace("=", "_").replace("@", "_at_").replace(" + ", "__")


@dataclass
class TuneCfg:
    min_frames: int = 1500
    max_frames: int | None = None
    completed_only: bool = True
    stages: list[str] = field(default_factory=lambda: [s.name for s in INCLUDED_STAGES])
    characters: list[str] = field(default_factory=list)
    ranks: list[str] = field(default_factory=list)

    # Stats predicates (require --with-stats index).
    min_damage_dealt: float | None = None
    min_damage_taken: float | None = None
    min_stocks_remaining: int | None = None
    min_inputs: int | None = None

    # Reject replays where any player has >= max_cheap_deaths stocks lost at
    # <= cheap_death_pct percent. Set max_cheap_deaths=None to disable.
    max_cheap_deaths: int | None = 2
    cheap_death_pct: float = 10.0

    # Keep if any player has >= this many stock losses. None = disable.
    min_death_count: int | None = 3

    # Slack windows defining "near" the boundary.
    frame_slack: int = 300


cfg = TuneCfg(min_damage_dealt=100, completed_only=False)


# %% [markdown]
# Load the full index once.

# %%
entries: list[ReplayIndexEntry] = list(read_jsonl(INDEX, verify_schema_version=False))
logger.info(f"loaded {len(entries)} entries from {INDEX}")


# %% [markdown]
# Score every entry. We reuse `filter.build_predicates` so the gate is exactly
# the same as production; the bucketing adds the "near" overlay on top.


# %%
def score(cfg: TuneCfg, entries: list[ReplayIndexEntry]):
    stages = _resolve_stages(cfg.stages) if cfg.stages else None
    chars = _resolve_ids(cfg.characters, CHARACTERS_BY_NAME, "character") if cfg.characters else None
    ranks = {r.strip().lower() for r in cfg.ranks} if cfg.ranks else None
    mins = PlayerStatsMins(
        damage_dealt=cfg.min_damage_dealt,
        damage_taken=cfg.min_damage_taken,
        stocks_remaining=cfg.min_stocks_remaining,
        inputs=cfg.min_inputs,
    )

    preds = build_predicates(
        min_frames=cfg.min_frames if cfg.min_frames > 0 else None,
        max_frames=cfg.max_frames,
        completed_only=cfg.completed_only,
        stages=stages,
        characters=chars,
        ranks=ranks,
        mins=mins if mins.any_set() else None,
    )

    if cfg.max_cheap_deaths is not None:
        max_cheap = cfg.max_cheap_deaths
        pct = cfg.cheap_death_pct

        def _no_cheap_deaths(e: ReplayIndexEntry, max_cheap: int = max_cheap, pct: float = pct) -> bool:
            if e.stats is None:
                return True
            return all(sum(1 for dp in p.death_percents if dp <= pct) < max_cheap for p in e.stats.players)

        preds.append((f"max_cheap_deaths<{max_cheap}@{pct}%", _no_cheap_deaths))

    if cfg.min_death_count is not None:
        floor = cfg.min_death_count

        def _any_player_death_count_ok(e: ReplayIndexEntry, floor: int = floor) -> bool:
            if e.stats is None:
                return True
            return any(len(p.death_percents) >= floor for p in e.stats.players)

        preds.append((f"any_death_count>={floor}", _any_player_death_count_ok))

    # "Soft" predicates are the ones with a continuous knob — fails within a
    # slack window count as near-rejects. Everything else (completed_only,
    # ranks, characters, stages) is hard.
    def is_soft_near(entry: ReplayIndexEntry, label: str) -> bool:
        if label.startswith("min_frames="):
            return entry.frame_count >= cfg.min_frames - cfg.frame_slack
        if label.startswith("max_frames=") and cfg.max_frames is not None:
            return entry.frame_count <= cfg.max_frames + cfg.frame_slack
        return False

    accepted: list[ReplayIndexEntry] = []
    accept_near: list[tuple[ReplayIndexEntry, str]] = []
    reject_near: list[tuple[ReplayIndexEntry, str]] = []
    reject_hard: list[tuple[ReplayIndexEntry, list[str]]] = []
    fail_counts: Counter[str] = Counter()

    for entry in entries:
        failed = [label for label, pred in preds if not pred(entry)]
        for f in failed:
            fail_counts[f] += 1
        if not failed:
            # near-accept by length only (cheapest, most common knob)
            if cfg.min_frames > 0 and entry.frame_count < cfg.min_frames + cfg.frame_slack:
                accept_near.append((entry, f"frames={entry.frame_count}"))
            else:
                accepted.append(entry)
        elif len(failed) == 1 and is_soft_near(entry, failed[0]):
            reject_near.append((entry, failed[0]))
        else:
            reject_hard.append((entry, failed))

    return {
        "accept_clean": accepted,
        "accept_near": accept_near,
        "reject_near": reject_near,
        "reject_hard": reject_hard,
        "fail_counts": fail_counts,
    }


buckets = score(cfg, entries)
reject_by_exact, reject_by_predicate = group_rejections(buckets)
total = len(entries)
for name in ("accept_clean", "accept_near", "reject_near", "reject_hard"):
    n = len(buckets[name])
    logger.info(f"  {name:14s} {n:7d}  ({100 * n / total:5.1f}%)")
logger.info("rejections by predicate (entry may appear in multiple groups):")
for label, items in reject_by_predicate.items():
    logger.info(f"  reject[{label}]: {len(items)}")
logger.info("rejections by exact failure set:")
for label, items in reject_by_exact.items():
    logger.info(f"  reject_exact[{label}]: {len(items)}")


# %% [markdown]
# Persist the four file lists. Archive members (`archive://…`) get a separate
# `.archive-skipped.txt` because slippilab can't fetch them as plain URLs.


# %%
def _bucket_entries(bucket_value) -> list[tuple[ReplayIndexEntry, str]]:
    out: list[tuple[ReplayIndexEntry, str]] = []
    for item in bucket_value:
        entry, note = (item, "") if isinstance(item, ReplayIndexEntry) else (item[0], str(item[1]))
        out.append((entry, note))
    return out


# File lists keep the entry.path (synthetic archive:// paths included) — they're
# the meaningful identifier from the index. Extraction happens lazily in show_bucket.
for name in ("accept_clean", "accept_near", "reject_near", "reject_hard"):
    items = _bucket_entries(buckets[name])
    (OUT_DIR / f"{name}.txt").write_text("\n".join(e.path for e, _ in items) + ("\n" if items else ""))
    logger.info(f"wrote {len(items)} -> {OUT_DIR / (name + '.txt')}")

reject_dir = OUT_DIR / "reject_by_predicate"
reject_dir.mkdir(exist_ok=True)
for label, items in reject_by_predicate.items():
    path = reject_dir / f"{_safe_reason_name(label)}.txt"
    path.write_text("\n".join(e.path for e in items) + ("\n" if items else ""))
    logger.info(f"wrote {len(items)} -> {path}")


# %% [markdown]
# No standalone file server: slps are served by slippilab's vite dev server via
# the `public/hal-replays` symlink created above. Only port 5173 needs forwarding.


# %% [markdown]
# Sample N from a chosen bucket and emit slippilab URLs.


# %%
def _summary(entry: ReplayIndexEntry) -> str:
    try:
        stage_name = slp_stage_to_libmelee(entry.stage).name
    except Exception:
        stage_name = f"stage={entry.stage}"
    chars = "/".join(str(p.character) for p in entry.players)
    completed = entry.outcome.completed if entry.outcome else None
    if entry.stats:
        deaths = " | ".join("[" + ",".join(f"{dp:.0f}" for dp in p.death_percents) + "]" for p in entry.stats.players)
        death_counts = [len(p.death_percents) for p in entry.stats.players]
        count_tag = f"deaths={death_counts} "
    else:
        deaths = "?"
        count_tag = ""
    return (
        f"frames={entry.frame_count:5d} {stage_name:18s} chars={chars} death_pcts={deaths} {count_tag}"
        f"rank={entry.rank_filename} v={'.'.join(map(str, entry.slp_version))} completed={completed}"
    )


def show_bucket(name: str, n: int = 25, offset: int = 0) -> None:
    bucket = buckets[name]
    items = bucket[offset : offset + n]
    print(f"\n=== {name}  showing [{offset}:{offset + len(items)}] of {len(bucket)} ===\n")
    for item in items:
        entry, note = (item, "") if isinstance(item, ReplayIndexEntry) else item
        try:
            local = materialize(entry.path)
        except Exception as e:
            print(f"  [extract failed: {e!r}] {entry.path}")
            continue
        replay_url = f"{SLIPPILAB_URL}/{SERVE_MOUNT}/{local.name}"
        link = f"{SLIPPILAB_URL}/?replayUrl={urllib.parse.quote(replay_url, safe=':/')}"
        print(f"  {_summary(entry)}  why={note}")
        print(f"    {link}")


def _print_entries(header: str, entries: list[ReplayIndexEntry]) -> None:
    print(f"\n=== {header} ===\n")
    for entry in entries:
        try:
            local = materialize(entry.path)
        except Exception as e:
            print(f"  [extract failed: {e!r}] {entry.path}")
            continue
        replay_url = f"{SLIPPILAB_URL}/{SERVE_MOUNT}/{local.name}"
        link = f"{SLIPPILAB_URL}/?replayUrl={urllib.parse.quote(replay_url, safe=':/')}"
        print(f"  {_summary(entry)}")
        print(f"    {link}")


def show_rejections(reason: str, n: int = 25, offset: int = 0) -> None:
    """Sample from ``reject_by_predicate[reason]`` (predicate label from scoring)."""
    items = reject_by_predicate[reason]
    page = items[offset : offset + n]
    _print_entries(f"reject[{reason}]  showing [{offset}:{offset + len(page)}] of {len(items)}", page)


def show_rejections_sample(n: int = 15, *, seed: int | None = None) -> None:
    """Random sample from each predicate rejection group."""
    rng = random.Random(seed)
    for reason, items in reject_by_predicate.items():
        k = min(n, len(items))
        if k == 0:
            continue
        sample = rng.sample(items, k)
        _print_entries(f"reject[{reason}]  sample {k} of {len(items)}", sample)


show_bucket("reject_near", n=15)

# %%
buckets

# %%
show_bucket("accept_near", n=15)

# %%
show_bucket("reject_hard")

# %%
show_rejections_sample(15)

# %%
show_bucket("accept_clean", n=25, offset=25)

# %%

reject_by_predicate.keys()
# %%
rng = random.Random(0)
samples = rng.sample(reject_by_predicate["max_frames=26000"], 15)
_print_entries(f"reject['max_frames=26000']  sample 15 of {len(samples)}", samples)

# %% [markdown]
# Iterate: edit `cfg`, rerun the scoring cell. Examples:
#
# ```python
# cfg.min_frames = 1200
# cfg.frame_slack = 200
# buckets = score(cfg, entries)
# show_bucket("reject_near")
# ```

# %% [markdown]
# Final: print the equivalent CLI invocation.


# %%
def emit_cli(cfg: TuneCfg) -> str:
    parts = ["python -m hal.scripts.filter", f"--index {INDEX}", "--output /tmp/paths.txt"]
    if cfg.min_frames != 1500:
        parts.append(f"--min-frames {cfg.min_frames}")
    if cfg.max_frames is not None:
        parts.append(f"--max-frames {cfg.max_frames}")
    if not cfg.completed_only:
        parts.append("--no-completed-only")
    default_stages = [s.name for s in INCLUDED_STAGES]
    if cfg.stages != default_stages:
        parts.append("--stages " + " ".join(cfg.stages) if cfg.stages else "--stages")
    if cfg.characters:
        parts.append("--characters " + " ".join(cfg.characters))
    if cfg.ranks:
        parts.append("--ranks " + " ".join(cfg.ranks))
    if cfg.min_damage_dealt is not None:
        parts.append(f"--min-damage-dealt {cfg.min_damage_dealt}")
    if cfg.min_damage_taken is not None:
        parts.append(f"--min-damage-taken {cfg.min_damage_taken}")
    if cfg.min_stocks_remaining is not None:
        parts.append(f"--min-stocks-remaining {cfg.min_stocks_remaining}")
    if cfg.min_inputs is not None:
        parts.append(f"--min-inputs {cfg.min_inputs}")
    return " \\\n  ".join(parts)


print(emit_cli(cfg))
