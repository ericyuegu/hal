"""Replay a schema-2 dense inference fault capsule with a validated policy artifact."""

import argparse
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import Tensor

from hal.inference.action_sequence_artifact import build_action_sequence_model
from hal.inference.action_sequence_artifact import read_action_sequence_artifact
from hal.inference.checkpoints import resolve_checkpoint
from hal.inference.engine import configure_inference_process
from hal.inference.sampling import StreamGroupRng
from hal.inference.window_policy import WindowPolicy
from hal.inference.window_policy import condition_ego_players
from hal.models.controller_codec import CONTROLLER_GROUP_NAMES
from hal.representation.features import Context
from hal.wire import ACTION_DIM


@dataclass(frozen=True, slots=True)
class FaultInputs:
    context: Context
    fixed_actions: Tensor
    horizon: int
    stream_ids: tuple[int, ...]
    generations: tuple[int, ...]
    player_ids: tuple[int, ...]
    desired_returns: tuple[float | None, ...]
    temperatures: tuple[float, ...]
    sampling_seed: int
    sampling_generations: tuple[tuple[int, int], ...]
    sampling_counters: tuple[tuple[int, int, str, int], ...]
    checkpoint_sha256: str | None

    def sampling(self) -> StreamGroupRng:
        rng = StreamGroupRng(self.sampling_seed, CONTROLLER_GROUP_NAMES)
        rng.restore(generations=self.sampling_generations, counters=self.sampling_counters)
        return rng


def _typed_values[T](values: object, value_type: type[T]) -> tuple[T, ...]:
    if not isinstance(values, list) or any(type(value) is not value_type for value in values):
        raise ValueError(f"fault metadata requires an array of {value_type.__name__}")
    return cast(tuple[T, ...], tuple(values))


def _number_values(values: object, *, allow_none: bool) -> tuple[float | None, ...]:
    if not isinstance(values, list):
        raise ValueError("fault metadata requires an array of numbers")
    converted = []
    for value in values:
        if allow_none and value is None:
            converted.append(None)
        elif type(value) in (int, float):
            converted.append(float(cast(int | float, value)))
        else:
            raise ValueError("fault metadata requires numeric conditioning values")
    return tuple(converted)


def load_fault_inputs(path: Path) -> FaultInputs:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported process fault capsule version")
    raw = payload.get("policy")
    if not isinstance(raw, dict) or raw.get("schema_version") != 2:
        raise ValueError("replay requires a current schema-2 dense policy snapshot")
    metadata = cast(Mapping[str, object], raw)
    ids = _typed_values(metadata["stream_ids"], int)
    resets = _typed_values(metadata["reset"], bool)
    request_generations = _typed_values(metadata["generations"], int)
    if any(generation < 1 for generation in request_generations):
        raise ValueError("fault request generations must be positive")
    players = _typed_values(metadata["player_id"], int)
    pads = _typed_values(metadata["ctx_pad"], int)
    returns = _number_values(metadata["desired_return"], allow_none=True)
    temperatures = cast(tuple[float, ...], _number_values(metadata["temperature"], allow_none=False))
    values = _typed_values(metadata["value_names"], str)
    masks = _typed_values(metadata["mask_names"], str)
    categories = _typed_values(metadata["cat_names"], str)
    emitted = _typed_values(metadata["emitted_masks"], bool)
    horizon, prefix, seed = (metadata[key] for key in ("horizon", "fixed_prefix_frames", "sampling_seed"))
    if type(horizon) is not int or type(prefix) is not int or type(seed) is not int:
        raise ValueError("fault horizon, prefix, and seed must be integers")
    if not 0 <= prefix < horizon or seed < 0:
        raise ValueError("invalid fault horizon, prefix, or seed")
    rows = len(ids)
    if (
        rows < 1
        or len(set(ids)) != rows
        or any(len(item) != rows for item in (resets, request_generations, players, pads, returns, temperatures))
    ):
        raise ValueError("fault stream metadata has inconsistent rows")
    if len(masks) != len(emitted) or len(set((*values, *masks, *categories))) != len(values) + len(masks) + len(
        categories
    ):
        raise ValueError("fault feature names or masks are inconsistent")
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
        if set(arrays.files) != {"floats", "cats", "fixed_actions"}:
            raise ValueError("fault arrays must contain floats, cats, and fixed_actions")
        floats = arrays["floats"]
        cats = arrays["cats"]
        fixed = arrays["fixed_actions"]
    if floats.ndim != 3 or floats.shape[:2] != (len(values) + len(masks), rows) or floats.dtype != np.float32:
        raise ValueError("fault float arrays have the wrong shape or dtype")
    length = floats.shape[2]
    if cats.shape != (len(categories), rows, length) or cats.dtype != np.int64:
        raise ValueError("fault categorical arrays have the wrong shape or dtype")
    if fixed.shape != (rows, prefix, ACTION_DIM) or fixed.dtype != np.float32:
        raise ValueError("fault fixed actions have the wrong shape or dtype")
    if any(pad < 0 or pad >= length for pad in pads):
        raise ValueError("fault context pads are outside the context window")
    features = {name: torch.from_numpy(floats[index]) for index, name in enumerate(values)}
    features.update(
        {name: torch.from_numpy(floats[len(values) + index]) for index, name in enumerate(masks) if emitted[index]}
    )
    features.update({name: torch.from_numpy(cats[index]) for index, name in enumerate(categories)})
    raw_generations, raw_counters = metadata["sampling_generations"], metadata["sampling_counters"]
    if not isinstance(raw_generations, list) or not isinstance(raw_counters, list):
        raise ValueError("fault sampling metadata must be arrays")
    generations = []
    for entry in raw_generations:
        row = _typed_values(entry, int)
        if len(row) != 2:
            raise ValueError("fault sampling generation needs a stream and generation")
        generations.append((row[0], row[1]))
    counters = []
    for entry in raw_counters:
        if not isinstance(entry, list) or len(entry) != 4:
            raise ValueError("fault sampling counter needs stream, generation, group, count")
        stream, generation, group, count = entry
        if any(type(value) is not int for value in (stream, generation, count)) or not isinstance(group, str):
            raise ValueError("fault sampling counter has invalid fields")
        counters.append((stream, generation, group, count))
    checkpoint = metadata["checkpoint_sha256"]
    if checkpoint is not None and not isinstance(checkpoint, str):
        raise ValueError("fault checkpoint identity must be a hash or absent")
    result = FaultInputs(
        Context(features, torch.tensor(pads, dtype=torch.long)),
        torch.from_numpy(fixed),
        horizon,
        ids,
        request_generations,
        players,
        returns,
        temperatures,
        seed,
        tuple(generations),
        tuple(counters),
        checkpoint,
    )
    result.sampling()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capsule", type=Path)
    parser.add_argument("artifact")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuda-sync-debug", action="store_true")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    if args.cuda_sync_debug and os.environ.get("CUDA_LAUNCH_BLOCKING") != "1":
        os.execvpe(sys.executable, [sys.executable, *sys.argv], {**os.environ, "CUDA_LAUNCH_BLOCKING": "1"})
    configure_inference_process()
    captured = load_fault_inputs(args.capsule)
    artifact = read_action_sequence_artifact(resolve_checkpoint(args.artifact))
    if captured.checkpoint_sha256 is not None and captured.checkpoint_sha256 != artifact.checkpoint_sha256:
        raise ValueError("fault capsule belongs to a different checkpoint")
    model = build_action_sequence_model(artifact, device=args.device)
    rows = len(captured.stream_ids)
    bucket = 1 << (rows - 1).bit_length()
    executor = WindowPolicy(
        model,
        context_frames=artifact.model_config.L_ctx,
        prediction_frames=captured.horizon,
        prepared_buckets=(bucket,),
        compiled=not args.eager,
        compile_mode=args.compile_mode,
        amp_dtype="bfloat16" if args.device.startswith("cuda") else "float32",
        return_conditioning=artifact.model_config.return_conditioning,
    )
    context = condition_ego_players(captured.context.to(args.device), captured.player_ids)
    if (
        context.ctx_pad.shape[0] != rows
        or next(iter(context.features.values())).shape[1] != artifact.model_config.L_ctx
    ):
        raise ValueError("fault context differs from the artifact architecture")
    fixed = captured.fixed_actions.to(args.device)
    for repeat in range(args.repeats):
        actions = executor.decode(
            context,
            captured.horizon,
            streams=captured.sampling(),
            stream_ids=captured.stream_ids,
            sampling_generations=tuple(generation - 1 for generation in captured.generations),
            desired_returns=captured.desired_returns,
            temperatures=captured.temperatures,
            committed=fixed,
        )
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        if not bool(torch.isfinite(actions).all()):
            raise RuntimeError(f"fault replay {repeat + 1} returned non-finite controller actions")
    print(f"replayed {args.repeats} times: {rows} streams, horizon {captured.horizon}")


if __name__ == "__main__":
    main()
