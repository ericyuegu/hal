"""Audit Experiment 040's reward scale and AWR temperature.

Run from the repository root after exporting the R2 variables from ``.env``::

    uv run notebooks/040_awr_constants.py

The sample follows the production loader's natural replay mixture. Within each
source it reads seeded contiguous spans spread across the index, balancing
representativeness against shard-download locality. Window and ego selection
use the same deterministic functions and geometry as the production loader.
"""

# %% Imports and frozen treatment
import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from streaming import StreamingDataset

from hal import streams
from hal.data.policy_world_schema import decode_policy_world_replay
from hal.training import returns
from hal.training.dataloader import _choose_chunk_starts
from hal.training.replay_reservoir import _stable_replay_rng

SEED = 0
SAMPLE_REPLAYS = 50_000
L_CTX = 128
L_CHUNK = 24
WINDOWS_PER_REPLAY = 2
STOCK_VALUE = 120.0
WIN_REWARD = 50.0
DAMAGE_SHAPING = 1.0
GAMMA = 0.99618
BETA = 199.5
WEIGHT_MAX = 3.5
SAMPLE_SPANS_PER_SOURCE = 8


@dataclass(frozen=True, slots=True)
class SourceAudit:
    name: str
    requested_replays: int
    terminal_replays: int
    eligible_positions: int


def replay_quotas(total: int) -> dict[str, int]:
    """Largest-remainder allocation under the production replay mixture."""
    counts = streams.POLICY_WORLD_V7_TRAIN_REPLAYS
    corpus_total = sum(counts.values())
    exact = {name: total * count / corpus_total for name, count in counts.items()}
    quotas = {name: math.floor(value) for name, value in exact.items()}
    remainder = total - sum(quotas.values())
    order = sorted(counts, key=lambda name: (exact[name] - quotas[name], name), reverse=True)
    for name in order[:remainder]:
        quotas[name] += 1
    if sum(quotas.values()) != total:
        raise RuntimeError("largest-remainder allocation did not preserve the requested total")
    return quotas


def stratified_spans(
    population: int,
    sample_size: int,
    rng: np.random.Generator,
    *,
    max_spans: int = SAMPLE_SPANS_PER_SOURCE,
) -> tuple[tuple[int, int], ...]:
    """Allocate exact, non-overlapping contiguous blocks across index strata."""
    if not 0 <= sample_size <= population:
        raise ValueError(f"sample_size must be in [0, {population}], got {sample_size}")
    if max_spans < 1:
        raise ValueError(f"max_spans must be positive, got {max_spans}")
    if sample_size == 0:
        return ()
    span_count = min(max_spans, sample_size)
    population_base, population_remainder = divmod(population, span_count)
    sample_base, sample_remainder = divmod(sample_size, span_count)
    spans = []
    lane_start = 0
    for index in range(span_count):
        lane_size = population_base + (index < population_remainder)
        block_size = sample_base + (index < sample_remainder)
        latest_start = lane_start + lane_size - block_size
        block_start = int(rng.integers(lane_start, latest_start + 1))
        spans.append((block_start, block_start + block_size))
        lane_start += lane_size
    return tuple(spans)


def loader_weighted_returns(sample: dict[str, object], *, epoch: int = 0) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return production rows and matching 256-frame variance windows."""
    replay_id = str(sample["replay_id"])
    decoded = decode_policy_world_replay(sample)
    labeled = returns.replay_returns(
        decoded,
        gamma=GAMMA,
        damage_shaping=DAMAGE_SHAPING,
        win_reward=WIN_REWARD,
        stock_value=STOCK_VALUE,
        suffix="awr_return",
    )
    if not bool(np.asarray(labeled["p1_awr_return_valid"]).any()):
        return [], []
    frames = int(sample["num_frames"])
    rng = _stable_replay_rng(SEED, epoch, replay_id)
    chunk_starts = _choose_chunk_starts(frames, L_CTX, L_CHUNK, WINDOWS_PER_REPLAY, rng)
    ports = ["p1" if rng.random() < 0.5 else "p2" for _ in chunk_starts]
    selected = []
    diagnostic_256 = []
    for chunk_start, port in zip(chunk_starts, ports, strict=True):
        # collate_awr_batch takes window[1:L_ctx+1], and the model drops left
        # padding. The surviving episode indices are therefore these bounds.
        stop = int(chunk_start) + 1
        start = max(1, stop - L_CTX)
        replay_return = np.asarray(labeled[f"{port}_awr_return"])
        selected.append(replay_return[start:stop])
        diagnostic_256.append(replay_return[max(1, stop - 256) : stop])
    return selected, diagnostic_256


def source_rows(source: streams.StreamSource, quota: int, rng: np.random.Generator):
    """Yield seeded stratified spans while verifying the immutable index."""
    dataset = StreamingDataset(
        remote=f"{source.remote}/train",
        local=str(source.local_root / "train"),
        batch_size=1,
        shuffle=False,
        cache_limit="1024gb",
        predownload=64,
    )
    actual = int(dataset.samples_per_stream[0])
    expected = streams.POLICY_WORLD_V7_TRAIN_REPLAYS[source.name]
    if actual != expected:
        raise RuntimeError(f"{source.name}: index has {actual:,} replays; manifest says {expected:,}")
    if quota > actual:
        raise ValueError(f"{source.name}: requested {quota:,} of only {actual:,} replays")
    for start, stop in stratified_spans(actual, quota, rng):
        for index in range(start, stop):
            yield dataset[index]


# %% Stream the stratified replay sample
def calibrate(sample_replays: int = SAMPLE_REPLAYS) -> dict[str, object]:
    rng = np.random.default_rng(SEED)
    quotas = replay_quotas(sample_replays)
    selected_returns: list[np.ndarray] = []
    diagnostic_256_returns: list[np.ndarray] = []
    audits: list[SourceAudit] = []
    for source in streams.POLICY_WORLD_V7_SOURCES:
        quota = quotas[source.name]
        if quota == 0:
            audits.append(SourceAudit(source.name, 0, 0, 0))
            continue
        terminal = positions = 0
        for sample in source_rows(source, quota, rng):
            windows, diagnostic_windows = loader_weighted_returns(sample)
            terminal += bool(windows)
            positions += sum(len(window) for window in windows)
            selected_returns.extend(windows)
            diagnostic_256_returns.extend(diagnostic_windows)
        audit = SourceAudit(source.name, quota, terminal, positions)
        audits.append(audit)
        print(
            f"{source.name}: {terminal:,}/{quota:,} terminal replays, {positions:,} eligible positions",
            flush=True,
        )

    if not selected_returns:
        raise RuntimeError("the calibration sample contained no eligible return positions")
    pooled = np.concatenate(selected_returns).astype(np.float64, copy=False)
    if not np.isfinite(pooled).all():
        raise FloatingPointError("eligible return sample contains a non-finite value")

    return_mean = float(pooled.mean())

    # Law-of-total-variance diagnostics across production and 256-frame windows.
    window_means = np.array([window.mean() for window in selected_returns], dtype=np.float64)
    window_sizes = np.array([len(window) for window in selected_returns], dtype=np.float64)
    between_variance = float(np.average(np.square(window_means - return_mean), weights=window_sizes))
    total_variance = float(pooled.var())
    between_fraction = between_variance / total_variance if total_variance else 0.0
    pooled_256 = np.concatenate(diagnostic_256_returns).astype(np.float64, copy=False)
    means_256 = np.array([window.mean() for window in diagnostic_256_returns], dtype=np.float64)
    sizes_256 = np.array([len(window) for window in diagnostic_256_returns], dtype=np.float64)
    baseline_256 = float(pooled_256.mean())
    between_256 = float(np.average(np.square(means_256 - baseline_256), weights=sizes_256))
    variance_256 = float(pooled_256.var())

    return {
        "sample_seed": SEED,
        "sample_replays": sample_replays,
        "sample_spans_per_source": SAMPLE_SPANS_PER_SOURCE,
        "terminal_replays": sum(audit.terminal_replays for audit in audits),
        "eligible_positions": int(pooled.size),
        "loader_geometry": {
            "L_ctx": L_CTX,
            "L_chunk": L_CHUNK,
            "windows_per_replay": WINDOWS_PER_REPLAY,
        },
        "reward": {
            "stock_value": STOCK_VALUE,
            "win_reward": WIN_REWARD,
            "damage_shaping": DAMAGE_SHAPING,
            "gamma": GAMMA,
        },
        "awr_beta": BETA,
        "awr_weight_max": WEIGHT_MAX,
        "return_mean": return_mean,
        "return_std": float(pooled.std()),
        "return_quantiles": {
            str(percentile): float(value)
            for percentile, value in zip(
                (1, 5, 25, 50, 75, 95, 99), np.percentile(pooled, (1, 5, 25, 50, 75, 95, 99)), strict=True
            )
        },
        "between_loader_window_variance_fraction": between_fraction,
        "between_256_frame_window_variance_fraction": between_256 / variance_256 if variance_256 else 0.0,
        "sources": [
            {
                "name": audit.name,
                "requested_replays": audit.requested_replays,
                "terminal_replays": audit.terminal_replays,
                "eligible_positions": audit.eligible_positions,
            }
            for audit in audits
        ],
    }


# %% Print the immutable config inputs and diagnostics
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-replays", type=int, default=SAMPLE_REPLAYS)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".json"))
    args = parser.parse_args()
    if args.sample_replays < 1:
        raise SystemExit("--sample-replays must be positive")
    payload = json.dumps(calibrate(args.sample_replays), indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(payload)
    temporary.replace(args.output)
    print(payload, end="")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
