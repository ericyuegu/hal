"""Capture matched conditional logits from immutable 059 and batched candidate."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.controller import POLICY_BUTTON_MASK
from hal.controller import ControllerAction
from hal.data.extract import extract_replay
from hal.inference.api import PolicyInput
from hal.inference.api import PredictionRequest
from hal.inference.api import RuntimeConfig
from hal.wire import ACTION_CHANNELS
from hal.wire import BUTTON_BITS

BASE_FRAMES = (0, 512)
SOURCES = frozenset((3, 7, 11, 15, 19, 23, 247, 251, 255, 259, 263, 267, 271, 275, 279))
HORIZON = 8
PREFIX = 3


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def replay_action(rows: dict[str, np.ndarray], frame: int) -> ControllerAction:
    buttons = (
        sum(
            BUTTON_BITS[name.removeprefix("button_")]
            for name in ACTION_CHANNELS[6:]
            if rows["p1_" + name][frame] > 0.5
        )
        & POLICY_BUTTON_MASK
    )
    return ControllerAction(*(float(rows["p1_" + name][frame]) for name in ACTION_CHANNELS[:6]), buttons)


def replay_vector(rows: dict[str, np.ndarray], frame: int) -> np.ndarray:
    return np.asarray([rows["p1_" + name][frame] for name in ACTION_CHANNELS], dtype=np.float32)


def capture_logits(policy: object, mode: str, rows: dict[str, np.ndarray], source: int) -> tuple[torch.Tensor, ...]:
    model = policy.model
    observed_vectors = np.stack([replay_vector(rows, base + source) for base in BASE_FRAMES])
    future_vectors = np.stack(
        [
            np.stack([replay_vector(rows, base + source + depth) for depth in range(1, HORIZON + 1)])
            for base in BASE_FRAMES
        ]
    )
    observed = model.codec.quantize(torch.from_numpy(observed_vectors).cuda())
    forced = model.codec.quantize(torch.from_numpy(future_vectors).cuda())
    values = torch.full((len(BASE_FRAMES),), 20.0, device="cuda")
    present = torch.ones(len(BASE_FRAMES), dtype=torch.bool, device="cuda")
    if mode == "candidate":
        cache_pool = policy._cache_pool
        if cache_pool is None:
            raise RuntimeError("candidate cache was not prepared")
        cache_rows = tuple(policy._prediction_streams[index].row for index in range(len(BASE_FRAMES)))
        cache = cache_pool.gather(cache_rows, len(BASE_FRAMES))
        _, logits = model.temporal.sample_indices_with_logits(
            cache.hidden,
            observed,
            tuple(range(1, HORIZON + 1)),
            values,
            present,
            argmax=True,
            forced_prefix=forced,
            history=cache.memory(),
        )
        return tuple(group.float().cpu() for group in logits)
    per_stream = []
    for index in range(len(BASE_FRAMES)):
        cache = policy._prediction_streams[index].cache
        if cache is None:
            raise RuntimeError("control cache was not admitted")
        _, logits = model.temporal.sample_indices_with_logits(
            cache.hidden,
            observed[index : index + 1],
            tuple(range(1, HORIZON + 1)),
            values[index : index + 1],
            present[index : index + 1],
            argmax=True,
            forced_prefix=forced[index : index + 1],
            history=cache.memory(),
        )
        per_stream.append(tuple(group.float().cpu() for group in logits))
    return tuple(torch.cat([groups[index] for groups in per_stream]) for index in range(len(per_stream[0])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("control", "candidate"))
    parser.add_argument("bundle", type=Path)
    parser.add_argument("replay", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    if args.mode == "control":
        from hal.inference.backends.history_decoder.policy import load_o59_policy

        policy = load_o59_policy(
            args.bundle,
            device="cuda",
            seed=1001,
            compiled=False,
            history_mode="kv_cache",
            kv_update_frames=4,
            kv_cuda_graphs=False,
        )
    else:
        from hal.inference.action_sequence_artifact import load_action_sequence_policy

        policy = load_action_sequence_policy(
            args.bundle,
            device="cuda",
            seed=1001,
            compiled=False,
            history_mode="kv_cache",
            kv_update_frames=4,
            kv_cuda_graphs=False,
        )
    policy.prepare_prediction(RuntimeConfig(2, (2,), replan_interval_frames=4), HORIZON, PREFIX)
    extracted = extract_replay(str(args.replay))
    if extracted is None or len(extracted["stage"]) <= BASE_FRAMES[-1] + max(SOURCES) + HORIZON:
        raise ValueError("replay does not cover the matched cache-wrap cohort")
    rows = {name: np.asarray(value) for name, value in extracted.items()}
    saved: dict[int, tuple[torch.Tensor, ...]] = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, max(SOURCES) + 1, 4):
            requests = []
            for stream_id, base in enumerate(BASE_FRAMES):
                observations = []
                for frame in range(base + start, base + start + 4):
                    observation = {name: rows[name][frame].item() for name in policy.spec.required_observation_fields}
                    action = replay_action(rows, frame)
                    if args.mode == "control":
                        item = PolicyInput(
                            stream_id,
                            frame,
                            1,
                            observation,
                            action,
                            (NEUTRAL_CONTROLLER_ACTION,) * 2,
                            player_identity="IBDW#0",
                            reset=frame == base,
                        )
                    else:
                        item = PolicyInput(
                            stream_id,
                            frame,
                            1,
                            observation,
                            action,
                            player_identity="IBDW#0",
                            reset=frame == base,
                        )
                    observations.append(item)
                requests.append(
                    PredictionRequest(
                        stream_id,
                        1,
                        start // 4,
                        base + start + 3,
                        tuple(observations),
                        (NEUTRAL_CONTROLLER_ACTION,) * PREFIX,
                    )
                )
            policy.predict(tuple(requests))
            source = start + 3
            if source in SOURCES:
                saved[source] = capture_logits(policy, args.mode, rows, source)
    result = {
        "mode": args.mode,
        "bundle_sha256": sha256(args.bundle),
        "replay_sha256": sha256(args.replay),
        "source_sha256": sha256(Path(__file__)),
        "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(),
        "frames_per_stream": max(SOURCES) + 1,
        "base_frames": BASE_FRAMES,
        "sources": sorted(SOURCES),
        "logits": saved,
    }
    torch.save(result, args.output)
    print(json.dumps({name: value for name, value in result.items() if name != "logits"}, indent=2))


if __name__ == "__main__":
    main()
