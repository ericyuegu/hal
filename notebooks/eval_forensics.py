# %%
"""Forensics on closed-loop eval .slp recordings (model vs lvl-9 CPU).

Answers two questions the replays can settle offline, no GPU / no emulator:

Q1 (contamination damage) - the instant-restart harness never clears the policy's
    rolling context at match boundaries, so every match after a boot's first opens
    on stale two-match context. Quantify by match ordinal within each boot: stocks
    lost per active minute and stock losses in the opening frames, ordinal 0 (clean)
    vs ordinal >= 1 (contaminated).

Q2 (failure budget) - "stocks lost per minute" hides opposite problems. Classify
    every ego stock loss as SD vs legitimate KO and characterise deaths: percent at
    death (full histogram so the SD threshold is judgeable), death position, frames
    since match start and since the stock began.

Parsing goes through the repo extractor (``hal.data.extract.extract_replay``); the
start block is read separately (``skip_frames=True``, robust even on torn files) to
identify the model port and whether the match reached a real GameEnd. Instant-restart
matches cut at the per-boot frame cap are torn at the final frame - peppi raises a Rust
``PanicException`` (a ``BaseException``, so ``extract_replay``'s ``except Exception``
does not catch it). ``_repair_to_last_frame`` trims the raw event stream back to the
last complete frame bookend so those files parse.

Data is pulled from R2 into the scratch dir (see ``_PROVENANCE``); run top-to-bottom
with ``uv run notebooks/eval_forensics.py``. Tables print to stdout; plots land in
``<scratch>/forensics/plots``.
"""

import struct
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import matplotlib
import numpy as np
import peppi_py

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hal.data.extract import extract_replay  # noqa: E402
from hal.wire import peppi_port_to_libmelee  # noqa: E402
from hal.wire import slp_stage_to_libmelee  # noqa: E402

# %%
# Config + reference constants.

SCRATCH = Path.home() / "data" / "scratch" / "eval_forensics"
DATA_ROOT = SCRATCH / "data"
PLOT_DIR = SCRATCH / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

FRAMES_PER_MINUTE = 3600  # 60 fps
SD_PERCENT_THRESHOLD = 50.0  # percent-at-death below this reads as a self-destruct
OPENING_WINDOWS = (51, 128, 256)  # frames; 256 = the contamination premise, 51 = this run's Lc

# How each data subdir was populated, and what it is. Auto-discovery uses the dirs on
# disk; this only labels them for the printout. rclone remote ``r2`` bucket ``hal``.
_PROVENANCE: dict[str, str] = {
    "172638_final": "012 260616-172638 (Lc256, gpt-16k) replays/final - single-match-per-boot",
    "004736_final": "012 260616-004736 (Lc256, gpt-16k) replays/final - single-match-per-boot",
    "183711_step004096": "013 260618-183711 (Lc51, iso-flop-short-ctx) replays/step_004096 - MULTI-match-per-boot",
}

# libmelee Stage value -> (name, ledge x, blastzone [left, right, top, bottom]).
# Frozen reference geometry for the six legal stages (from libmelee EDGE_POSITION /
# stages.BLASTZONES) so the notebook stays self-contained.
STAGE_GEO: dict[int, tuple[str, float, tuple[float, float, float, float]]] = {
    25: ("FINAL_DESTINATION", 88.47, (-246.0, 246.0, 188.0, -140.0)),
    24: ("BATTLEFIELD", 71.31, (-224.0, 224.0, 200.0, -108.8)),
    18: ("POKEMON_STADIUM", 90.66, (-230.0, 230.0, 180.0, -111.0)),
    26: ("DREAMLAND", 80.18, (-255.0, 255.0, 250.0, -123.0)),
    8: ("FOUNTAIN_OF_DREAMS", 66.26, (-198.75, 198.75, 202.5, -146.25)),
    6: ("YOSHIS_STORY", 58.91, (-175.7, 173.6, 168.0, -91.0)),
}


# %%
# Robust .slp reading. Torn instant-restart files (cut at the per-boot frame cap) have a
# final frame missing one port's Post + the bookend, so peppi's struct build panics. We
# trim to the last complete frame (last 0x3C bookend) and re-point rawLength; this mirrors
# hal.data.slp_finalize but at frame- rather than event-granularity. Only invoked on the
# files that actually fail to parse, so clean GameEnd files keep their end block.

_SLP_HEADER = b"{U\x03raw[$U#l"
_LEN_OFF = len(_SLP_HEADER)
_RAW_START = _LEN_OFF + 4
_EVENT_PAYLOADS = 0x35
_FRAME_BOOKEND = 0x3C
_SLP_FOOTER = b"U\x08metadata{}}"


def _repair_to_last_frame(path: Path) -> bool:
    """Trim a torn .slp back to its last complete frame bookend, in place. Returns
    True if it rewrote the file. No-op (False) for non-raw or bookend-less files."""
    data = path.read_bytes()
    if data[:_LEN_OFF] != _SLP_HEADER or data[_RAW_START] != _EVENT_PAYLOADS:
        return False
    raw_len = struct.unpack(">i", data[_LEN_OFF:_RAW_START])[0]
    declared = data[_RAW_START + 1]
    sizes = {_EVENT_PAYLOADS: declared}
    p = _RAW_START + 2
    for _ in range((declared - 1) // 3):
        sizes[data[p]] = struct.unpack(">H", data[p + 1 : p + 3])[0]
        p += 3
    raw_end = _RAW_START + raw_len if raw_len else len(data)
    cur = _RAW_START + 1 + declared
    last_bookend: int | None = None
    while cur < raw_end and data[cur] in sizes:
        nxt = cur + 1 + sizes[data[cur]]
        if nxt > raw_end:
            break
        if data[cur] == _FRAME_BOOKEND:
            last_bookend = nxt
        cur = nxt
    if last_bookend is None:
        return False
    out = bytearray(data[:last_bookend])
    struct.pack_into(">i", out, _LEN_OFF, last_bookend - _RAW_START)
    out += _SLP_FOOTER
    path.write_bytes(bytes(out))
    return True


def _safe_extract(path: Path) -> dict[str, np.ndarray] | None:
    """extract_replay with a torn-file repair fallback. peppi's Rust panic is a
    BaseException, so guard that explicitly (never swallow Ctrl-C / exit)."""
    try:
        return extract_replay(str(path))
    except KeyboardInterrupt, SystemExit:
        raise
    except BaseException:
        pass
    if not _repair_to_last_frame(path):
        return None
    try:
        return extract_replay(str(path))
    except KeyboardInterrupt, SystemExit:
        raise
    except BaseException:
        return None


@dataclass(frozen=True, slots=True)
class StartInfo:
    """Start/end-block facts read with skip_frames=True (robust on torn files)."""

    model_prefix: str  # "p1" or "p2" - extract_replay maps prefixes to sorted occupied ports
    stage_val: int
    reached_game_end: bool  # True = clean GameEnd, False = cut at the per-boot frame cap


def _read_start_info(path: Path) -> StartInfo | None:
    """Identify the model port (the non-CPU player) and whether the match ended
    cleanly. Returns None unless the game is a 1v1 with exactly one CPU."""
    try:
        g = peppi_py.read_slippi(str(path), skip_frames=True)
    except KeyboardInterrupt, SystemExit:
        raise
    except BaseException:
        return None
    players = list(g.start.players)
    if len(players) != 2:
        return None
    ports = sorted(peppi_port_to_libmelee(pl.port) for pl in players)

    def is_cpu(pl: object) -> bool:
        lvl = getattr(pl, "cpu_level", None)
        return getattr(pl, "type", None) == 1 or (lvl is not None and lvl > 0)

    cpu = [pl for pl in players if is_cpu(pl)]
    model = [pl for pl in players if not is_cpu(pl)]
    if len(cpu) != 1 or len(model) != 1:
        return None
    model_port = peppi_port_to_libmelee(model[0].port)
    model_prefix = "p1" if model_port == ports[0] else "p2"
    # g.start.stage is the slp-native external id; normalize to the libmelee Stage value
    # (same convention extract_replay stores) so STAGE_GEO lookups match.
    stage_val = int(slp_stage_to_libmelee(int(g.start.stage)).value)
    return StartInfo(model_prefix=model_prefix, stage_val=stage_val, reached_game_end=g.end is not None)


# %%
# Per-match record + death extraction.


@dataclass(frozen=True, slots=True)
class Death:
    percent: float  # ego percent at the frame the stock is lost (pre-reset)
    x: float
    y: float
    frame_id: int  # >= 0, i.e. frames since GO
    frames_since_stock: int  # since this stock began (previous death, or match start)


@dataclass(frozen=True, slots=True)
class MatchRecord:
    run: str
    boot: str
    ordinal: int  # position of this match within its boot (0 = clean context)
    stage_val: int
    reached_game_end: bool
    n_active_frames: int
    # Peak ego percent within [0, W) active frames, one per OPENING_WINDOWS - a denser
    # contamination probe than the (spawn-protected, rare) opening stock losses: a match
    # that opens on stale context should eat more damage early.
    opening_peak_percent: dict[int, float] = field(default_factory=dict)
    deaths: list[Death] = field(default_factory=list)

    @property
    def active_minutes(self) -> float:
        return self.n_active_frames / FRAMES_PER_MINUTE

    def opening_deaths(self, window: int) -> int:
        return sum(1 for d in self.deaths if d.frame_id < window)


def _ego_deaths(
    frame_id: np.ndarray, stock: np.ndarray, percent: np.ndarray, x: np.ndarray, y: np.ndarray
) -> list[Death]:
    """A death = a downward step in the ego stock count on an active frame. Read the
    dying-stock state from the frame just before the decrement (percent still holds the
    death value; position is where the body crossed the blastzone)."""
    active = frame_id >= 0
    fid, st, pc, xx, yy = frame_id[active], stock[active], percent[active], x[active], y[active]
    if st.size < 2:
        return []
    drops = np.flatnonzero(np.diff(st) < 0)  # index i: stock lost between i and i+1
    deaths: list[Death] = []
    stock_start_frame = int(fid[0])
    for i in drops:
        f = int(fid[i])
        deaths.append(
            Death(
                percent=float(pc[i]),
                x=float(xx[i]),
                y=float(yy[i]),
                frame_id=f,
                frames_since_stock=f - stock_start_frame,
            )
        )
        stock_start_frame = f
    return deaths


def build_record(run: str, boot: str, ordinal: int, path: Path) -> MatchRecord | None:
    info = _read_start_info(path)
    if info is None:
        return None
    cols = _safe_extract(path)
    if cols is None:
        return None
    p = info.model_prefix
    frame_id = cols["frame"]
    active = frame_id >= 0  # post-GO frames
    n_active = int(np.sum(active))
    fid_a = frame_id[active]
    pct_a = cols[f"{p}_percent"][active]
    opening_peak = {w: float(pct_a[fid_a < w].max()) if np.any(fid_a < w) else 0.0 for w in OPENING_WINDOWS}
    deaths = _ego_deaths(
        frame_id, cols[f"{p}_stock"], cols[f"{p}_percent"], cols[f"{p}_position_x"], cols[f"{p}_position_y"]
    )
    return MatchRecord(
        run=run,
        boot=boot,
        ordinal=ordinal,
        stage_val=info.stage_val,
        reached_game_end=info.reached_game_end,
        n_active_frames=n_active,
        opening_peak_percent=opening_peak,
        deaths=deaths,
    )


# %%
# Discovery: group every .slp by its parent dir (a boot: ``match_NNN`` in the single-match
# 012 layout, ``boot_NNN`` in the multi-match 013 layout). Ordinal = chronological rank of
# the filename within the boot (Slippi names files ``Game_<timestamp>.slp``, so lexical
# sort is chronological).


def discover(run_dir: Path) -> list[tuple[str, int, Path]]:
    by_boot: dict[str, list[Path]] = defaultdict(list)
    for slp in run_dir.rglob("*.slp"):
        by_boot[str(slp.parent.relative_to(run_dir))].append(slp)
    out: list[tuple[str, int, Path]] = []
    for boot, files in by_boot.items():
        for ordinal, path in enumerate(sorted(files, key=lambda q: q.name)):
            out.append((boot, ordinal, path))
    return out


run_dirs = sorted(d for d in DATA_ROOT.iterdir() if d.is_dir()) if DATA_ROOT.exists() else []
print(f"data root: {DATA_ROOT}")
print(f"run dirs discovered: {[d.name for d in run_dirs]}\n")

records: list[MatchRecord] = []
parse_stats: dict[str, dict[str, int]] = {}
for rd in run_dirs:
    triples = discover(rd)
    ok = skipped = 0
    for boot, ordinal, path in triples:
        rec = build_record(rd.name, boot, ordinal, path)
        if rec is None:
            skipped += 1
            continue
        records.append(rec)
        ok += 1
    n_boots = len({b for b, _, _ in triples})
    parse_stats[rd.name] = {"slp": len(triples), "boots": n_boots, "parsed": ok, "skipped": skipped}
    prov = _PROVENANCE.get(rd.name, "(unlabeled)")
    print(f"{rd.name}\n    {prov}\n    slp={len(triples)} boots={n_boots} parsed={ok} skipped={skipped}")

print(f"\ntotal matches parsed: {len(records)}")


# %%
# Q1 - contamination damage by match ordinal within a boot.
#
# ordinal 0 = boot's first match, context freshly cleared. ordinal >= 1 = every later
# match, whose 256/51-frame rolling context still spans the previous match (zero training
# support). Report stocks-lost per active minute and opening-window stock losses per
# ordinal bucket, plus a paired view over the boots that actually played >= 2 matches.

print("\n" + "=" * 96)
print("Q1  CONTAMINATION DAMAGE  (ordinal 0 = clean context, ordinal >=1 = contaminated)")
print("=" * 96)


_Q1_HEADER = (
    ["bucket", "matches", "active_min", "stocks", "st/min"]
    + [f"d<{w}f/m" for w in OPENING_WINDOWS]
    + [f"pk%<{w}f" for w in OPENING_WINDOWS]
)


def _q1_cells(label: str, recs: list[MatchRecord]) -> list[str]:
    """One formatted table row for an ordinal bucket. ``st/min`` and ``d<Wf/m`` are the
    requested stock metrics; ``pk%<Wf`` (mean peak ego percent in the opening W frames) is
    the denser damage-taken probe."""
    if not recs:
        return [label, "(none)"]
    stocks = sum(len(r.deaths) for r in recs)
    minutes = sum(r.active_minutes for r in recs)
    cells = [label, f"{len(recs)}", f"{minutes:.2f}", f"{stocks}", f"{stocks / minutes:.3f}" if minutes else "nan"]
    cells += [f"{sum(r.opening_deaths(w) for r in recs) / len(recs):.3f}" for w in OPENING_WINDOWS]
    cells += [f"{np.mean([r.opening_peak_percent[w] for r in recs]):.1f}" for w in OPENING_WINDOWS]
    return cells


def _print_q1(rows: list[tuple[str, list[MatchRecord]]]) -> None:
    print("    " + "".join(f"{h:>12}" for h in _Q1_HEADER))
    for label, recs in rows:
        print("    " + "".join(f"{c:>12}" for c in _q1_cells(label, recs)))


for rd in run_dirs:
    run = rd.name
    run_recs = [r for r in records if r.run == run]
    if not run_recs:
        continue
    max_ord = max(r.ordinal for r in run_recs)
    multimatch_boots = {b for b in {r.boot for r in run_recs} if sum(1 for r in run_recs if r.boot == b) >= 2}
    print(
        f"\n--- {run}  ({len(run_recs)} matches, {len({r.boot for r in run_recs})} boots, "
        f"max ordinal={max_ord}, boots with >=2 matches={len(multimatch_boots)})"
    )
    if max_ord == 0:
        print("    every boot played exactly one match -> no ordinal>=1 exists; contamination")
        print("    cannot arise here (single-match-per-boot layout). Q1 is vacuous for this run.")
    ord0 = [r for r in run_recs if r.ordinal == 0]
    ordn = [r for r in run_recs if r.ordinal >= 1]
    _print_q1([("ordinal 0", ord0), ("ordinal >=1", ordn)])
    # Paired view: within the boots that played >= 2 matches, ordinal 0 vs its continuations.
    if multimatch_boots:
        print(f"    paired (only the {len(multimatch_boots)} multi-match boots):")
        _print_q1(
            [
                ("  ord 0", [r for r in ord0 if r.boot in multimatch_boots]),
                ("  ord>=1", [r for r in ordn if r.boot in multimatch_boots]),
            ]
        )


# %%
# Q2 - failure budget: SD vs legitimate KO, and death characterisation.
#
# Every ego death across all runs is pooled. SD heuristic: percent < 50 at death. We also
# flag spatially-obvious SDs (died past the ledge while below KO percent, or below the
# bottom blastzone) so the percent threshold can be sanity-checked against geometry.

print("\n" + "=" * 96)
print("Q2  FAILURE BUDGET  (every ego stock loss classified)")
print("=" * 96)

all_deaths: list[tuple[Death, int]] = [(d, r.stage_val) for r in records for d in r.deaths]
print(f"\ntotal ego deaths pooled: {len(all_deaths)}  (over {len(records)} matches)")


def _pct_quartiles(vals: np.ndarray) -> str:
    q = np.percentile(vals, [0, 25, 50, 75, 100])
    return f"min={q[0]:.0f} q25={q[1]:.0f} median={q[2]:.0f} q75={q[3]:.0f} max={q[4]:.0f} mean={vals.mean():.1f}"


def _death_direction(d: Death, stage_val: int) -> str:
    """Which blastzone the body was crossing at death: bottom / side / top. Pick the
    nearest by |position| as a fraction of that blastzone (position is captured a frame
    or two shy of the actual crossing, so fractions run a touch below 1.0)."""
    if stage_val not in STAGE_GEO:
        return "unknown"
    _, _edge, (bl, br, top, bot) = STAGE_GEO[stage_val]
    frac_bottom = d.y / bot if bot else 0.0  # both negative -> positive fraction
    frac_top = d.y / top if top else 0.0
    frac_side = abs(d.x) / (br if d.x >= 0 else -bl)
    return max((frac_bottom, "bottom"), (frac_side, "side"), (frac_top, "top"))[1]


if all_deaths:
    pct = np.array([d.percent for d, _ in all_deaths])
    ys = np.array([d.y for d, _ in all_deaths])
    xs = np.array([d.x for d, _ in all_deaths])
    since_stock = np.array([d.frames_since_stock for d, _ in all_deaths])
    since_start = np.array([d.frame_id for d, _ in all_deaths])

    sd_by_pct = pct < SD_PERCENT_THRESHOLD
    # Spatial SD flags relative to each death's stage geometry.
    offstage = np.zeros(len(all_deaths), dtype=bool)
    below_stage = np.zeros(len(all_deaths), dtype=bool)
    for i, (d, sv) in enumerate(all_deaths):
        if sv not in STAGE_GEO:
            continue
        _, edge, (bl, br, _top, bot) = STAGE_GEO[sv]
        offstage[i] = d.x < bl * 0.85 or d.x > br * 0.85 or abs(d.x) > edge + 40
        below_stage[i] = d.y < bot * 0.7

    print(
        f"\nSD fraction (percent < {SD_PERCENT_THRESHOLD:.0f}): "
        f"{sd_by_pct.mean():.3f}  ({int(sd_by_pct.sum())}/{len(pct)})"
    )
    print(f"percent-at-death quartiles: {_pct_quartiles(pct)}")
    print(f"deaths below bottom blastzone-ish (y < 0.7*bottom): {int(below_stage.sum())} ({below_stage.mean():.3f})")
    print(f"deaths far past the ledge (|x| beyond stage): {int(offstage.sum())} ({offstage.mean():.3f})")
    print(
        f"low-percent (<{SD_PERCENT_THRESHOLD:.0f}) AND (offstage or below stage): "
        f"{int((sd_by_pct & (offstage | below_stage)).sum())}"
    )

    # Death direction (which blastzone) x SD/KO cross-tab - the striking spatial pattern.
    directions = [_death_direction(d, sv) for d, sv in all_deaths]
    print("\ndeath direction (blastzone crossed)   |  all    SD(<50)   KO(>=50)")
    for dirn in ("bottom", "side", "top", "unknown"):
        mask = np.array([x == dirn for x in directions])
        if not mask.any():
            continue
        n = int(mask.sum())
        n_sd = int((mask & sd_by_pct).sum())
        print(f"    {dirn:<34} {n:4d} ({n / len(directions):.2f})  {n_sd:4d}    {n - n_sd:4d}")

    print(
        f"\nmedian frames since stock began at death: {np.median(since_stock):.0f} "
        f"({np.median(since_stock) / 60:.1f}s)"
    )
    print(
        f"median frames since match start at death:  {np.median(since_start):.0f} ({np.median(since_start) / 60:.1f}s)"
    )

    print("\npercent-at-death histogram (10-pt bins):")
    edges = np.arange(0, max(210, pct.max() + 10), 10)
    counts, _ = np.histogram(pct, bins=edges)
    for lo, c in zip(edges[:-1], counts, strict=False):
        if c:
            print(f"    [{int(lo):3d},{int(lo) + 10:3d})  {c:4d}  {'#' * int(round(40 * c / counts.max()))}")

    # Per-run SD breakdown so the two eval eras are comparable.
    print("\nper-run SD fraction and death count:")
    for rd in run_dirs:
        rp = np.array([d.percent for r in records if r.run == rd.name for d in r.deaths])
        if rp.size:
            print(
                f"    {rd.name:<26} deaths={rp.size:4d}  SD<{SD_PERCENT_THRESHOLD:.0f}={np.mean(rp < SD_PERCENT_THRESHOLD):.3f}  "
                f"median_pct={np.median(rp):.0f}"
            )

    # X-position-at-death histogram (ledge clustering).
    print("\ndeath x-position histogram (20-unit bins) - ledge x for reference below:")
    xe = np.arange(-280, 300, 20)
    xc, _ = np.histogram(xs, bins=xe)
    for lo, c in zip(xe[:-1], xc, strict=False):
        if c:
            print(f"    x[{int(lo):4d},{int(lo) + 20:4d})  {c:4d}  {'#' * int(round(40 * c / xc.max()))}")
    print(
        "    (legal-stage ledge x: "
        + ", ".join(f"{n[:4]}=+-{e:.0f}" for _, (n, e, _b) in sorted(STAGE_GEO.items()))
        + ")"
    )


# %%
# Plots -> scratch/plots.

if all_deaths:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(pct, bins=np.arange(0, max(210, pct.max() + 10), 10), color="#4C78A8", edgecolor="white")
    ax.axvline(SD_PERCENT_THRESHOLD, color="#E45756", ls="--", label=f"SD threshold {SD_PERCENT_THRESHOLD:.0f}%")
    ax.set_xlabel("ego percent at death")
    ax.set_ylabel("stock losses")
    ax.set_title(f"Q2 percent-at-death  (n={len(pct)}, SD fraction={sd_by_pct.mean():.2f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "q2_percent_at_death.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    sc = ax.scatter(xs, ys, c=pct, cmap="viridis", s=14, alpha=0.7)
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    for _, (_, edge, (_bl, _br, _top, bot)) in STAGE_GEO.items():
        ax.axvline(edge, color="grey", lw=0.3, ls=":")
        ax.axvline(-edge, color="grey", lw=0.3, ls=":")
        ax.axhline(bot, color="grey", lw=0.3, ls=":")
    fig.colorbar(sc, ax=ax, label="percent at death")
    ax.set_xlabel("x at death")
    ax.set_ylabel("y at death")
    ax.set_title("Q2 death positions (colour = percent; dotted = stage ledges/floors)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "q2_death_positions.png", dpi=120)
    plt.close(fig)

    # Q1 opening-window damage taken (mean peak ego percent) by ordinal, for any
    # multi-match run. Opening deaths are ~0 (spawn-protected), so peak percent is the
    # informative probe: contamination would push ordinal>=1 above ordinal 0.
    for rd in run_dirs:
        run_recs = [r for r in records if r.run == rd.name]
        if not run_recs or max(r.ordinal for r in run_recs) == 0:
            continue
        ord0 = [r for r in run_recs if r.ordinal == 0]
        ordn = [r for r in run_recs if r.ordinal >= 1]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        width = 0.38
        idx = np.arange(len(OPENING_WINDOWS))
        for off, label, recs, col in (
            (-width / 2, "ordinal 0 (clean)", ord0, "#4C78A8"),
            (width / 2, "ordinal >=1 (contaminated)", ordn, "#E45756"),
        ):
            vals = [float(np.mean([r.opening_peak_percent[w] for r in recs])) for w in OPENING_WINDOWS]
            ax.bar(idx + off, vals, width, label=label, color=col)
        ax.set_xticks(idx)
        ax.set_xticklabels([f"first {w}f" for w in OPENING_WINDOWS])
        ax.set_ylabel("mean peak ego percent in opening window")
        ax.set_title(f"Q1 opening-window damage taken by ordinal\n{rd.name}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"q1_opening_damage_{rd.name}.png", dpi=120)
        plt.close(fig)

print(f"\nplots written to {PLOT_DIR}")
print("done.")
