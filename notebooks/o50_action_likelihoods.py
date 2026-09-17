"""Inspect the likelihood of sampled O50 actions at each executed frame.

First run the H2H evaluator with ``--trace-actions``. Point this notebook at the
downloaded evaluation's ``action-traces`` directory:

    HAL_ACTION_TRACE_DIR=/path/to/eval/action-traces jupyter lab

The trace stores the exact masked logits and uniform draw used by live sampling.
It does not run the model again.
"""

# %%
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from IPython.display import display

from hal.eval.action_trace import ACTION_TRACE_SCHEMA_VERSION
from hal.training import scoring
from hal.wire import ACTION_CHANNELS

TRACE_ROOT = Path(os.environ.get("HAL_ACTION_TRACE_DIR", "action-traces")).expanduser()
SAVE_FIGURES = True

# %%
tables = []
for manifest_path in sorted(TRACE_ROOT.rglob("manifest.json")):
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != ACTION_TRACE_SCHEMA_VERSION:
        raise ValueError(f"unsupported trace schema in {manifest_path}: {manifest.get('schema_version')}")
    parts = sorted(manifest_path.parent.glob("part-*.parquet"))
    if not parts:
        continue
    table = pa.concat_tables([pq.read_table(path) for path in parts])
    tables.append(table)
if not tables:
    raise FileNotFoundError(f"no action trace parts below {TRACE_ROOT}")

trace = pa.concat_tables(tables).to_pandas()
trace[["model", "decode_seed", "slot_id", "generation"]].drop_duplicates().sort_values(
    ["model", "decode_seed", "slot_id", "generation"]
)

# %%
# Select one match-side stream. Change these values to inspect another game.
MODEL = sorted(trace.model.unique())[0]
model_trace = trace[trace.model == MODEL]
DECODE_SEED = int(model_trace.decode_seed.min())
seed_trace = model_trace[model_trace.decode_seed == DECODE_SEED]
SLOT_ID = int(seed_trace.slot_id.min())
slot_trace = seed_trace[seed_trace.slot_id == SLOT_ID]
GENERATION = int(slot_trace.generation.min())

selected = slot_trace[slot_trace.generation == GENERATION].sort_values(["execution_frame", "group"])
if selected.empty:
    raise ValueError("the selected model, seed, slot, and generation has no actions")
print(
    f"{MODEL}: seed={DECODE_SEED}, slot={SLOT_ID}, generation={GENERATION}, "
    f"temperature={selected.temperature.iloc[0]:g}, frames={selected.execution_frame.min()}–"
    f"{selected.execution_frame.max()}"
)

# %%
group_colors = {
    "buttons": "#4c78a8",
    "main_stick": "#f58518",
    "c_stick": "#54a24b",
    "triggers": "#e45756",
}
actions = selected.drop_duplicates("execution_frame").copy()
actions["joint_probability_percent"] = 100.0 * np.exp(actions.action_log_probability.astype(float))
actions["surprise_bits"] = -actions.action_log_probability.astype(float) / math.log(2.0)

fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True, constrained_layout=True)
for group, rows in selected.groupby("group", sort=False):
    axes[0].plot(
        rows.execution_frame,
        100.0 * rows.sampled_probability,
        marker=".",
        linewidth=1,
        label=group,
        color=group_colors[group],
    )
axes[0].set_yscale("log")
axes[0].set_ylabel("sampled group likelihood (%)")
axes[0].legend(ncol=4)
axes[0].grid(alpha=0.2)

axes[1].plot(actions.execution_frame, actions.joint_probability_percent, linewidth=1, color="#b279a2")
axes[1].set_yscale("log")
axes[1].set_ylabel("joint action likelihood (%)")
axes[1].grid(alpha=0.2)

for group, rows in selected.groupby("group", sort=False):
    axes[2].scatter(
        rows.execution_frame,
        rows["rank"],
        s=9,
        alpha=0.7,
        label=group,
        color=group_colors[group],
    )
axes[2].set_yscale("log", base=2)
axes[2].invert_yaxis()
axes[2].set_ylabel("sampled rank (1 is best)")
axes[2].set_xlabel("executed game frame")
axes[2].grid(alpha=0.2)
fig.suptitle(f"{MODEL} sampled-action likelihood")

if SAVE_FIGURES:
    plot_dir = TRACE_ROOT / "plots"
    plot_dir.mkdir(exist_ok=True)
    fig.savefig(plot_dir / f"{MODEL}-seed{DECODE_SEED}-slot{SLOT_ID}-generation{GENERATION}.png", dpi=160)
plt.show()


# %%
def action_label(group: str, index: int) -> str:
    """Return a readable controller value for one categorical index."""
    if group == "buttons":
        bits = scoring.combo_to_buttons(torch.tensor([index]))[0].bool().tolist()
        names = [name.removeprefix("button_") for name, active in zip(ACTION_CHANNELS[6:], bits) if active]
        return "+".join(names) if names else "neutral"
    if group == "main_stick":
        x, y = scoring.STICK_CLUSTER_CENTERS_MAIN[index].tolist()
        return f"({x:+.2f}, {y:+.2f})"
    if group == "c_stick":
        x, y = scoring.STICK_CLUSTER_CENTERS_C[index].tolist()
        return f"({x:+.2f}, {y:+.2f})"
    if group == "triggers":
        count = len(scoring.TRIGGER_CENTERS)
        left = float(scoring.TRIGGER_CENTERS[index // count])
        right = float(scoring.TRIGGER_CENTERS[index % count])
        return f"L={left:.2f}, R={right:.2f}"
    raise ValueError(f"unknown controller group {group!r}")


def probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Match the live sampler's float32 softmax."""
    scaled = np.asarray(logits, dtype=np.float32) / np.float32(temperature)
    scaled -= np.max(scaled)
    values = np.exp(scaled)
    return values / values.sum()


# %%
# Inspect all candidate actions at one executed frame.
FRAME = int(actions.execution_frame.iloc[0])
frame_rows = selected[selected.execution_frame == FRAME]
TOP_K = 12

fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
candidate_tables = []
for axis, (_, row) in zip(axes.flat, frame_rows.iterrows(), strict=True):
    probs = probabilities(np.asarray(row.logits), float(row.temperature))
    sampled = int(row.sampled_index)
    top = np.argsort(probs)[::-1][:TOP_K]
    shown = top if sampled in top else np.append(top[:-1], sampled)
    shown = shown[np.argsort(probs[shown])]
    labels = [f"{index}: {action_label(row.group, int(index))}" for index in shown]
    colors = ["#e45756" if index == sampled else "#4c78a8" for index in shown]
    axis.barh(labels, 100.0 * probs[shown], color=colors)
    axis.set_xscale("log")
    axis.set_xlabel("likelihood (%)")
    axis.set_title(f"{row.group}: sampled {sampled} at rank {row['rank']}")
    candidate_tables.append(
        pd.DataFrame(
            {
                "group": row.group,
                "index": shown[::-1],
                "action": [action_label(row.group, int(index)) for index in shown[::-1]],
                "likelihood_percent": 100.0 * probs[shown[::-1]],
                "sampled": shown[::-1] == sampled,
            }
        )
    )
fig.suptitle(
    f"Frame {FRAME}: joint likelihood={100 * math.exp(float(frame_rows.action_log_probability.iloc[0])):.3e}%"
)
display(pd.concat(candidate_tables, ignore_index=True))
plt.show()

# %%
# The compact per-frame table is useful for sorting or export.
frame_summary = selected.pivot(
    index="execution_frame", columns="group", values=["sampled_index", "sampled_probability", "rank"]
)
frame_summary.join(actions.set_index("execution_frame")[["joint_probability_percent", "surprise_bits"]]).head(30)
