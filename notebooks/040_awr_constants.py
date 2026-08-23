"""Calibrate Experiment 040's fixed AWR baseline and weight normalizer.

Run from the repository root after exporting the R2 variables from ``.env``::

    uv run notebooks/040_awr_constants.py

The sample follows the production loader's natural replay mixture. Within each
source it reads one seeded contiguous span to avoid turning a 50k-replay audit
into a near-full-corpus random shard download. Window and ego selection use the
same deterministic functions and geometry as the production reservoir loader.
"""

# %% Imports and frozen treatment
import argparse
import json
import math
from dataclasses import dataclass

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
    assert sum(quotas.values()) == total
    return quotas


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
    """Yield a seeded contiguous sample while verifying the immutable index."""
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
    start = int(rng.integers(0, actual - quota + 1))
    for index in range(start, start + quota):
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

    baseline = float(pooled.mean())
    advantage = pooled - baseline
    raw_weight = np.minimum(np.exp(advantage / BETA), WEIGHT_MAX)
    weight_norm = float(raw_weight.mean())
    weight = raw_weight / weight_norm
    ess = float(weight.sum() ** 2 / (len(weight) * np.square(weight).sum()))
    clip_fraction = float(np.mean(raw_weight == WEIGHT_MAX))

    # Law-of-total-variance diagnostics across production and 256-frame windows.
    window_means = np.array([window.mean() for window in selected_returns], dtype=np.float64)
    window_sizes = np.array([len(window) for window in selected_returns], dtype=np.float64)
    between_variance = float(np.average(np.square(window_means - baseline), weights=window_sizes))
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
        "return_baseline": baseline,
        "weight_norm": weight_norm,
        "weight_ess": ess,
        "weight_clip_fraction": clip_fraction,
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
    args = parser.parse_args()
    if args.sample_replays < 1:
        raise SystemExit("--sample-replays must be positive")
    print(json.dumps(calibrate(args.sample_replays), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
