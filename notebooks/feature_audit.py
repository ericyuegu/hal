# %%
"""Every model input and target feature, end to end: raw MDS value -> transform -> model input.

One window of the LOCAL dev MDS (schema v5) is carried down both construction
paths and printed side by side:

* train path  — ``StreamingDataset`` -> ``WindowDataset`` -> ``collate_train_batch``
  (which calls ``hal.training.features.preprocess`` in the DataLoader worker);
* eval path   — canonical libmelee frame dict -> ``flatten_canonical_frame`` ->
  ``closed_loop._live_batch_from_rolling`` -> the SAME ``preprocess``.

The two paths must agree feature for feature at every context position. Cell 4
measures that; cells 3 and 5 show what each column is, which model block reads it,
and the three places where the two paths are deliberately not symmetric.

The dev MDS is v5, so the loader guard is told so explicitly (``schema_version=5``).
Do NOT ``python -m hal.scripts.fetch --name dev-mds``: the upstream bundle is stale
(pre-v5) and a fetch overwrites the local rebuild.

CPU only, no network, one window. Run from the repo root:
``uv run notebooks/feature_audit.py``.
"""

# %%
from pathlib import Path

import numpy as np
import torch
from melee import Character
from melee import Stage
from melee.stages import BLASTZONES
from melee.stages import EDGE_POSITION
from streaming import StreamingDataset
from streaming.base.util import clean_stale_shared_memory

from hal.data.feature_stats import FeatureStatsSufficient
from hal.data.feature_stats import load_sufficient_stats
from hal.data.feature_stats import merge_sufficient
from hal.data.schema import SCHEMA_VERSION
from hal.training.canonical import flatten_canonical_frame
from hal.training.closed_loop import _live_batch_from_rolling
from hal.training.dataloader import WindowDataset
from hal.training.dataloader import collate_train_batch
from hal.training.dataloader import collate_windows
from hal.training.ego_stats import consolidate_key
from hal.training.ego_stats import load_consolidated_stats
from hal.training.features import _SPATIAL_SCALES
from hal.training.features import ACTION_CHANNELS
from hal.training.features import CAT_FEATURES
from hal.training.features import FLOAT_FEATURES
from hal.training.features import SPATIAL_COLUMNS
from hal.training.features import SPATIAL_MASKS
from hal.training.features import _classify
from hal.training.features import _is_masked
from hal.training.features import derive_spatial
from hal.training.features import preprocess
from hal.wire import MASK_FLOAT
from hal.wire import MASK_INT32
from hal.wire import POST_FIELD_SUFFIXES
from hal.wire import post_field_path

MDS_ROOT = Path("data/processed/dev/mds")
SPLIT = "train"
MDS_SCHEMA_VERSION = 5  # the local dev bundle; code SCHEMA_VERSION is ahead of it
L_CTX = 256
L_CHUNK = 1
SEED = 0

# The four per-player blocks the model builds a float+mask+categorical token from,
# and the two embedding tables it indexes with raw libmelee ids. Both live in the
# experiment file (016/019 ``_PLAYER_PREFIXES``, ``char_vocab``/``stage_vocab``),
# not in ``features.py`` — repeated here so the audit can say who reads what.
MODEL_PLAYER_PREFIXES = ("ego", "ego_nana", "opp_nana", "opp")
EXPERIMENT_EMBEDS = {"character": (32, 12), "stage": (32, 4)}  # 016-base TrainConfig defaults


def table(header: tuple[str, ...], rows: list[tuple], widths: tuple[int, ...]) -> None:
    """Plain fixed-width table. No pandas: the repo has it, but a printed audit
    reads better without a DataFrame's truncation rules."""
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths, strict=True))
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(str(cell).ljust(w) for cell, w in zip(row, widths, strict=True)))


def fmt(value) -> str:
    """One raw or model-input scalar. Masked entries print as MASK, not as their
    dtype sentinel, because the sentinel is what the two paths disagree about."""
    arr = np.asarray(value)
    if bool(_is_masked(arr)):
        return "MASK"
    if arr.dtype.kind in "iu":
        return str(int(arr))
    return f"{float(arr):.4f}"


# %%
# --- Cell 1: consolidated dataset stats ---------------------------------------
# ``load_consolidated_stats`` Welford-merges the per-port p1_*/p2_* blocks into one
# distribution per bare feature name, because every window is relabelled ego/opp and
# both perspectives are fed to the same weights. ``preprocess`` looks these up by
# ``consolidate_key(name)``. Only FLOAT_FEATURES (and their nana twins) use them;
# sticks, triggers, buttons and categoricals never touch stats.

stats_path = MDS_ROOT / "stats.json"
stats = load_consolidated_stats(stats_path)

# Sample counts are not part of FeatureStats but explain the placeholder rows, so
# merge them the same way straight from the sufficient file.
merged: dict[str, FeatureStatsSufficient] = {}
for name, block in load_sufficient_stats(stats_path).items():
    key = consolidate_key(name)
    merged[key] = merge_sufficient(merged[key], block) if key in merged else block
counts: dict[str, int] = {k: v.count for k, v in merged.items()}


def float_transform(name: str) -> str:
    """The exact branch ``preprocess`` takes for a float column, with the numbers."""
    s = stats[consolidate_key(name)]
    if "position" in name or "percent" in name:
        return f"standardize (x-{s.mean:.4f})/{s.std:.4f}"
    return f"min-max [{s.min:.4f}, {s.max:.4f}] -> [-1,1]"


print(f"stats: {stats_path}  ({len(stats)} consolidated float features)\n")
rows = []
for name in sorted(stats):
    s = stats[name]
    rows.append(
        (
            name,
            f"{counts[name]:,}",
            f"{s.mean:.4f}",
            f"{s.std:.4f}",
            f"{s.min:.4f}",
            f"{s.max:.4f}",
            # Stats exist for every float MDS column, but only the kind 'float' ones are
            # normalized; sticks and triggers keep their native range and ignore stats.
            float_transform(name) if _classify(f"ego_{name}") == "float" else "-",
        )
    )
table(
    ("feature", "count", "mean", "std", "min", "max", "preprocess transform"),
    rows,
    (24, 12, 11, 11, 11, 11, 46),
)
print(
    "\nnote: count == 0 rows (the nana block on a dataset with no Ice Climbers) get the"
    "\nunit-Gaussian placeholder from FeatureStatsSufficient.finalize; the column is fully"
    "\nmasked anyway, so preprocess zeroes it and only the mask sidecar carries signal."
    "\nStick/trigger/button and categorical columns are absent from this table by design."
)

# %%
# --- Cell 2: one train window, raw and tensorized -----------------------------
# StreamingDataset -> WindowDataset (v5 guard) -> one [ctx | chunk] window. ``raw`` is
# the pre-preprocess dict of MDS columns (already ego/opp relabelled); ``stacked`` is
# the same with a leading batch axis; ``train_feats`` is what the model receives.

clean_stale_shared_memory()
mds = StreamingDataset(local=str(MDS_ROOT / SPLIT), batch_size=1, shuffle=False)
windows = WindowDataset(mds, L_CTX, L_CHUNK, seed=SEED, schema_version=MDS_SCHEMA_VERSION)
raw = next(iter(windows))

CTX_PAD = int(raw["ctx_pad"])
FRAME_T = L_CTX - 1  # audited position: newest context frame, always real whatever the pad is
stacked = collate_windows([raw])
train_batch = collate_train_batch([raw], stats=stats, L_ctx=L_CTX)
train_feats = train_batch.context.features

# Columns are the window's own keys minus the per-window scalar; ``frame`` stays in
# the list because the audit must show that it is classified as drop.
raw_columns = [k for k in raw if raw[k].ndim > 0]

stage_id = int(raw["stage"][FRAME_T])
print(f"split={SPLIT}  replays={mds.num_samples}  window=[{L_CTX} ctx | {L_CHUNK} chunk]  ctx_pad={CTX_PAD}")
print(f"audited frame t = {FRAME_T} (context index); episode frame id = {int(raw['frame'][FRAME_T])}")
print(
    f"stage={Stage(stage_id).name}  "
    f"ego={Character(int(raw['ego_character'][FRAME_T])).name}  "
    f"opp={Character(int(raw['opp_character'][FRAME_T])).name}"
)
print(f"raw columns: {len(raw_columns)}   model-input tensors after preprocess: {len(train_feats)}")
print(f"target: {tuple(train_batch.target.shape)} = [B, L_chunk, d_action]  ({train_batch.target.dtype})")

# %%
# --- Cell 3: the audit table --------------------------------------------------
# One row per raw column: what ``_classify`` calls it, the exact transform
# ``preprocess`` applies, the raw value at t, the model-input value at t, and which
# block of the model's input token reads it (016/019 assembly). "unread" means the
# column survives preprocess but no model consumes it.

TRIGGERS = ("trigger_l", "trigger_r")


def transform_of(name: str, kind: str) -> str:
    if kind == "drop":
        return "drop (never reaches the model)"
    if kind == "button":
        return "native {0,1} (masked -> 0.0)"
    if kind == "stick_trigger":
        return "native [0,1], 1/140 grid" if name.endswith(TRIGGERS) else "native [-1,1], 1/80 grid"
    if kind == "cat":
        for feature, (vocab, dim) in CAT_FEATURES.items():
            if name.endswith(f"_{feature}"):
                return f"embed id (vocab {vocab} -> dim {dim})"
        base = "stage" if name == "stage" else "character"
        vocab, dim = EXPERIMENT_EMBEDS[base]
        return f"embed id (vocab {vocab} -> dim {dim}, cfg)"
    return float_transform(name)


def consumer_of(name: str) -> str:
    for prefix in MODEL_PLAYER_PREFIXES:
        for feature in FLOAT_FEATURES:
            if name == f"{prefix}_{feature}":
                return f"{prefix} float block"
        for feature in CAT_FEATURES:
            if name == f"{prefix}_{feature}":
                return f"{prefix} cat embed"
    if name in tuple(f"ego_{channel}" for channel in ACTION_CHANNELS):
        return "ego action history"
    if name in ("ego_character", "opp_character"):
        return "char embed"
    if name == "stage":
        return "stage embed (+ spatial)"
    return "unread"


def blocks(columns: list[str]) -> list[tuple[str, list[str]]]:
    """Group the raw columns the way the model's input token is assembled."""
    seen: set[str] = set()
    out: list[tuple[str, list[str]]] = []

    def take(title: str, names: list[str]) -> None:
        picked = [n for n in names if n in columns and n not in seen]
        seen.update(picked)
        if picked:
            out.append((title, picked))

    take("global / per-replay constants", ["frame", "stage", "ego_character", "opp_character"])
    for prefix in MODEL_PLAYER_PREFIXES:
        take(
            f"{prefix} gamestate",
            [f"{prefix}_{f}" for f in FLOAT_FEATURES] + [f"{prefix}_{c}" for c in CAT_FEATURES],
        )
    take(
        "ego controller history",
        [f"ego_{channel}" for channel in ACTION_CHANNELS] + ["ego_button_start"],
    )
    take("opp controller", sorted(n for n in columns if _classify(n) in ("button", "stick_trigger")))
    take("other", sorted(set(columns) - seen))
    return out


header = ("column", "kind", "transform", f"raw @ t={FRAME_T}", "model input", "read by")
widths = (26, 14, 40, 12, 12, 24)
for title, names in blocks(raw_columns):
    print(f"\n### {title}")
    rows = []
    for name in names:
        kind = _classify(name)
        model = fmt(train_feats[name][0, FRAME_T]) if name in train_feats else "-"
        rows.append((name, kind, transform_of(name, kind), fmt(raw[name][FRAME_T]), model, consumer_of(name)))
    table(header, rows, widths)

# %%
# Targets. ``stack_actions`` reads the ego controller columns a second time, at the
# chunk positions, so the same 14 columns are both an input (the history slice) and
# the supervised target (one frame later). START is in neither: it is excluded from
# ACTION_CHANNELS so the policy can never pause the match.
print(f"\n### target action chunk (window positions {L_CTX}..{L_CTX + L_CHUNK - 1})")
table(
    ("channel", "raw @ L_ctx", "target[0,0]"),
    [
        (channel, fmt(raw[f"ego_{channel}"][L_CTX]), fmt(train_batch.target[0, 0, i]))
        for i, channel in enumerate(ACTION_CHANNELS)
    ],
    (18, 14, 14),
)

# %%
# Derived spatial block. ``preprocess`` calls ``derive_spatial`` on every batch that
# carries ``stage``, so train and closed loop compute it from one code path. Values
# are raw game units times a FIXED scale (never dataset stats: a distance has a known
# physical scale). Geometry comes from libmelee's own stage tables.
# Run over the whole window and clip to the context, since collate_train_batch keeps
# only the first L_ctx positions of every feature.
spatial = derive_spatial(stacked)
for name in SPATIAL_COLUMNS:
    torch.testing.assert_close(torch.from_numpy(spatial[name][:, :L_CTX]), train_feats[name], rtol=0, atol=0)

SHARED_FORMULAS = {
    "rel_dx": "opp_x - ego_x",
    "rel_dy": "opp_y - ego_y",
    "rel_dist": "hypot(rel_dx, rel_dy)",
    "rel_dx_ego_facing": "rel_dx * sign(ego_direction)",
    "rel_dx_opp_facing": "-rel_dx * sign(opp_direction)",
}
PLAYER_FORMULAS = {
    "ledge_dx": "|{p}_x| - edge_x",
    "ledge_dy": "{p}_y - 0.0 (ledge lip y)",
    "offstage": "1.0 if |{p}_x| > edge_x else 0.0",
    "blast_left": "{p}_x - blast_left",
    "blast_right": "blast_right - {p}_x",
    "blast_top": "blast_top - {p}_y",
    "blast_bottom": "{p}_y - blast_bottom",
    "dpos_x": "{p}_x[t] - {p}_x[t-1]",
    "dpos_y": "{p}_y[t] - {p}_y[t-1]",
}
MASK_FORMULAS = {
    "spatial_mask": "1 - observed; observed = stage known AND ego/opp pos+dir unmasked",
    "spatial_dpos_mask": "1 - (observed[t] AND observed[t-1])",
}

edge_x = EDGE_POSITION[Stage(stage_id)]
blast = BLASTZONES[Stage(stage_id)]
print(f"\n### derived spatial block (25 columns), stage {Stage(stage_id).name}")
print(f"geometry: edge_x={edge_x}  blastzones (l, r, t, b)={blast}  ledge y=0.0")
rows = []
for name in SPATIAL_COLUMNS:
    if name in SPATIAL_MASKS:
        formula, scale = MASK_FORMULAS[name], 1.0
    elif name in SHARED_FORMULAS:
        formula, scale = SHARED_FORMULAS[name], _SPATIAL_SCALES[name]
    else:
        player, suffix = name.split("_", 1)
        formula, scale = PLAYER_FORMULAS[suffix].format(p=player), _SPATIAL_SCALES[name]
    rows.append((name, formula, f"x {scale:g}", fmt(train_feats[name][0, FRAME_T])))
table(("column", "formula (raw game units)", "scale", f"value @ t={FRAME_T}"), rows, (26, 62, 10, 12))

# %%
# Mask sidecars. ``preprocess`` emits ``{name}_mask`` for a float column ONLY when
# that batch has at least one masked entry (``mask.any()``), so the model's input
# keys are batch-dependent; 016/019 substitute zeros for an absent sidecar. The
# spatial masks are the exception — both are emitted unconditionally, so the block's
# width never depends on the data.
print("\n### float columns and their mask sidecars (this batch)")
rows = []
for name in raw_columns:
    if _classify(name) != "float":
        continue
    mask = _is_masked(stacked[name])
    rows.append(
        (
            name,
            "yes" if f"{name}_mask" in train_feats else "no",
            f"{float(mask.mean()):.3f}",
            "all frames masked" if mask.all() else ("some frames masked" if mask.any() else "no masked frame"),
        )
    )
table(("float column", "mask emitted", "masked frac", "why"), rows, (26, 14, 13, 22))
print(f"unconditional: {', '.join(SPATIAL_MASKS)}")

# %%
# --- Cell 4: eval-path rebuild + parity diff ----------------------------------
# Rebuild the SAME frames the way the closed-loop driver does: fabricate the nested
# canonical gamestate dict libmelee hands ``Session.step`` (per port, per
# ``wire.POST_FIELD_SUFFIXES``), flatten it, feed the rolling-buffer window builder,
# and run the identical ``preprocess``. Only the real context frames are fed, so the
# builder re-creates the window's left pad itself.


def value_at(column: str, i: int) -> float:
    """One raw cell as the online reader would deliver it: a float, with any mask
    sentinel (NaN or MASK_INT32) collapsed to MASK_FLOAT — libmelee reports an
    unavailable field as absent, never as an int sentinel."""
    x = raw[column][i]
    return MASK_FLOAT if bool(_is_masked(np.asarray(x))) else float(x)


def post_at(prefix: str, i: int) -> dict:
    """Canonical post-frame dict for one player. Suffixes with no v5 column (the whole
    v6 addition) are simply absent, which is exactly what ``canonical_post_field``
    turns into MASK_FLOAT against a build that never recorded them."""
    post: dict = {}
    for suffix in POST_FIELD_SUFFIXES:
        column = f"{prefix}_{suffix}"
        if column not in raw:
            continue
        path = post_field_path(suffix)
        if len(path) == 1:
            post[path[0]] = value_at(column, i)
        else:
            post.setdefault(path[0], {})[path[1]] = value_at(column, i)
    return post


def follower_at(prefix: str, i: int) -> dict | None:
    """Nana. Fully-masked in the MDS <=> no follower on the wire, which is what makes
    the two paths land on the same masked columns."""
    nana = [f"{prefix}_nana_{s}" for s in POST_FIELD_SUFFIXES if f"{prefix}_nana_{s}" in raw]
    if all(bool(_is_masked(np.asarray(raw[c][i]))) for c in nana):
        return None
    return {"post": post_at(f"{prefix}_nana", i)}


def canonical_frame(i: int) -> dict:
    """Ego is libmelee port 1, opp port 2; ``_matchup`` is what ``drive_vec`` injects."""
    return {
        "id": int(raw["frame"][i]),
        "ports": {
            1: {"leader": {"post": post_at("ego", i)}, "follower": follower_at("ego", i)},
            2: {"leader": {"post": post_at("opp", i)}, "follower": follower_at("opp", i)},
        },
        "_matchup": {
            "stage": int(raw["stage"][i]),
            "character": {1: int(raw["ego_character"][i]), 2: int(raw["opp_character"][i])},
        },
    }


real = range(CTX_PAD, L_CTX)
flat_history = [flatten_canonical_frame(canonical_frame(i)) for i in real]
# Online, this buffer holds the policy's own intended actions. Here it replays the
# recorded ego inputs so the two paths are comparable column for column.
ego_inputs_hist = [
    np.array([value_at(f"ego_{channel}", i) for channel in ACTION_CHANNELS], dtype=np.float32) for i in real
]
assert not np.isnan(np.stack(ego_inputs_hist)).any(), (
    "recorded ego controller input is masked; history is not faithful"
)
eval_batch = _live_batch_from_rolling(flat_history, ego_inputs_hist, ego_prefix="p1", L_ctx=L_CTX)
eval_feats = preprocess(eval_batch, stats)

print(f"eval-path raw columns: {len(eval_batch)}   model-input tensors: {len(eval_feats)}")
print(f"train-path raw columns: {len(raw_columns)}   model-input tensors: {len(train_feats)}\n")

shared = sorted(set(train_feats) & set(eval_feats))
rows = []
worst = 0.0
for name in shared:
    a = train_feats[name].to(torch.float64)
    b = eval_feats[name].to(torch.float64)
    diff = float((a - b).abs().max())
    worst = max(worst, diff)
    rows.append(
        (
            name,
            str(train_feats[name].dtype).removeprefix("torch."),
            str(eval_feats[name].dtype).removeprefix("torch."),
            fmt(train_feats[name][0, FRAME_T]),
            fmt(eval_feats[name][0, FRAME_T]),
            f"{diff:.3e}" if diff else "0",
        )
    )
print(f"### per-feature train vs eval, max|diff| over all {L_CTX} context positions")
table(("feature", "train dtype", "eval dtype", "train @ t", "eval @ t", "max|diff|"), rows, (28, 12, 12, 12, 12, 11))
print(f"\nshared features: {len(shared)}   worst max|diff|: {worst:.3e}")
nonzero = [(name, row[-1]) for name, row in zip(shared, rows, strict=True) if row[-1] != "0"]
print(f"nonzero diffs: {len(nonzero)}" + (f" -> {nonzero}" if nonzero else " (paths agree bit for bit)"))
print(f"train-only features: {sorted(set(train_feats) - set(eval_feats))}")
print(f"eval-only features: {sorted(set(eval_feats) - set(train_feats))}")

# %%
# --- Cell 5: asymmetry callouts -----------------------------------------------
# (a) The opponent's controller columns exist on the train path only.
train_only_raw = sorted(set(raw_columns) - set(eval_batch))
opp_controller = [n for n in raw_columns if n.startswith("opp_") and _classify(n) in ("button", "stick_trigger")]
print("### (a) opponent controller columns: train path only, and no model reads them")
print(f"in the MDS window : {len(opp_controller)} columns -> {opp_controller}")
print(f"in the eval batch : {[n for n in opp_controller if n in eval_batch]}")
print(f"read by the model : {sorted({consumer_of(n) for n in opp_controller})}")
print(
    "They pass preprocess (kind 'button' / 'stick_trigger') but no model gathers them\n"
    "into the input token. This is deliberate: netplay does not show the opponent's\n"
    "inputs, so this conditioning would break human parity.\n"
    "ego_button_start / opp_button_start are unread for a different reason: START is\n"
    "not in ACTION_CHANNELS, because a pause ends the rollout.\n"
    f"raw columns present at train but not at eval: {train_only_raw}"
)

# %%
# (b) Categoricals arrive by two different routes and converge on one tensor.
print("\n### (b) categoricals: int32 + MASK_INT32 at train, float + NaN at eval")
rows = []
for prefix in MODEL_PLAYER_PREFIXES:
    for feature in CAT_FEATURES:
        name = f"{prefix}_{feature}"
        rows.append(
            (
                name,
                f"{stacked[name].dtype} / {MASK_INT32}",
                f"{eval_batch[name].dtype} / NaN",
                fmt(raw[name][FRAME_T]),
                str(train_feats[name].dtype).removeprefix("torch."),
                str(eval_feats[name].dtype).removeprefix("torch."),
            )
        )
table(
    ("categorical", "train raw / sentinel", "eval raw / sentinel", f"raw @ t={FRAME_T}", "train out", "eval out"),
    rows,
    (24, 24, 24, 12, 10, 10),
)
print(
    "The MDS writes ints with wire.mask_value sentinels; the online reader goes through\n"
    "wire.canonical_post_field, which is float-typed and reports an absent field as\n"
    "MASK_FLOAT (NaN). preprocess detects each with the matching rule (_is_masked) and\n"
    "casts both to int64, so the embedding lookup is identical. Two routes, one tensor."
)

# %%
# (c) Mask sidecars are batch-conditional on BOTH paths.
train_sidecars = sorted(k for k in train_feats if k.endswith("_mask") and k not in SPATIAL_MASKS)
eval_sidecars = sorted(k for k in eval_feats if k.endswith("_mask") and k not in SPATIAL_MASKS)
print("\n### (c) mask sidecars are conditional on mask.any(), on both paths")
print(f"train sidecars ({len(train_sidecars)}): {train_sidecars}")
print(f"eval  sidecars ({len(eval_sidecars)}): {eval_sidecars}")
print(f"identical set: {train_sidecars == eval_sidecars}")
missing = [f"{p}_{f}" for p in MODEL_PLAYER_PREFIXES for f in FLOAT_FEATURES if f"{p}_{f}_mask" not in train_feats]
print(f"float columns with NO sidecar this batch (model substitutes zeros): {missing}")
print(f"always emitted, data-independent: {list(SPATIAL_MASKS)}")

# %%
# (d) Schema v6, not yet materialized. The extractor/wire/canonical side already
# carries the new post fields, so the EVAL path already builds them every frame —
# they arrive as NaN here because the v5 window has no column to fill them from, and
# `_classify` routes every one of them to 'drop'. After the v6 re-materialization the
# same names appear as real train columns and will still drop until a feature spec
# claims them (WP5 / experiment 021). Nothing to fix here; this is the inventory.
v6_suffixes = [s for s in POST_FIELD_SUFFIXES if f"ego_{s}" not in raw]
print(f"\n### (d) schema v6 columns (code SCHEMA_VERSION={SCHEMA_VERSION}, this MDS={MDS_SCHEMA_VERSION})")
rows = []
for suffix in v6_suffixes:
    name = f"ego_{suffix}"
    rows.append(
        (
            name,
            "absent" if name not in raw else "present",
            "present" if name in eval_batch else "absent",
            fmt(eval_batch[name][0, FRAME_T]) if name in eval_batch else "-",
            _classify(name),
        )
    )
items = sorted(k for k in eval_batch if k.startswith("item"))
rows.append((f"item0..3_* ({len(items)} cols)", "absent", "present", "MASK", _classify(items[0])))
table(("v6 column (ego)", "train raw", "eval raw", f"eval @ t={FRAME_T}", "_classify"), rows, (28, 12, 12, 14, 10))
print(
    "The global item slots come from wire.canonical_item_columns and are all MASK here\n"
    "(the fabricated frames carry no items). Velocities are the interesting entry: the\n"
    "spatial block's dpos_x/dpos_y are finite-difference proxies for them today."
)
