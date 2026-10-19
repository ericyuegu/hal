"""Batch conditioned evaluation streams through the rollout inference kernels."""

import math
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from hal.data.feature_stats import FeatureStats
from hal.data.schema import MDS_PER_FRAME_DTYPES
from hal.eval.self_play_policy import PlayerCondition
from hal.eval.self_play_policy import evaluation_uniforms
from hal.inference.column_context import PackedColumnContextBuilder
from hal.inference.cuda_graph import CapturedCall
from hal.inference.decision_decoder import DecisionDecoder
from hal.inference.kv_cache import KVCache
from hal.inference.kv_cache import forward_tokens_with_block_handoff
from hal.models.attention import KVMemory
from hal.models.self_play import SelfPlayTransformer
from hal.representation.observations import project_observation_columns
from hal.sim.inputs import native_controller_readback
from hal.sim.native import OBSERVATION_FIELDS
from hal.sim.native import NativeFrameBatch
from hal.sim.rollout import ObservationRow
from hal.sim.rollout import PolicyRuntimeSpec
from hal.sim.rollout import Slot
from hal.wire import ACTION_DIM


@dataclass(frozen=True, slots=True)
class EvaluationStream:
    slot: Slot
    condition: PlayerCondition
    seed: int
    logical_player: int


class BatchedSelfPlayModel:
    """Retain queues and block caches for a fixed set of evaluation streams.

    A missing stream can return only with an explicit reset. This permits a
    completed Dolphin worker to leave the cohort without changing its batch size.
    """

    def __init__(
        self,
        model: SelfPlayTransformer,
        statistics: dict[str, FeatureStats],
        streams: tuple[EvaluationStream, ...],
        *,
        compiled: bool,
    ) -> None:
        if not streams or len({stream.slot for stream in streams}) != len(streams):
            raise ValueError("evaluation streams must be nonempty and have unique slots")
        for stream in streams:
            if stream.slot.match < 0 or stream.slot.port not in (1, 2):
                raise ValueError("evaluation stream needs a valid match and port")
            if not 0 <= stream.condition.player_id < model.cfg.player_vocab_size:
                raise ValueError("player identity is outside the checkpoint vocabulary")
            if any(type(value) is not int or value < 0 for value in (stream.seed, stream.logical_player)):
                raise ValueError("sampling seed and logical player must be nonnegative integers")
        self.model = model
        self.streams = streams
        self.device = next(model.parameters()).device
        self.compiled = compiled
        self.frames = np.zeros(len(streams), dtype=np.int64)
        self.last_frame: list[int | None] = [None] * len(streams)
        self.missing = np.zeros(len(streams), dtype=np.bool_)
        self.sides = torch.tensor([stream.slot.port - 1 for stream in streams], device=self.device)
        self.identities = torch.tensor([stream.condition.player_id for stream in streams], device=self.device)
        self.delays = torch.tensor([stream.condition.delay for stream in streams], device=self.device)
        self.replans = torch.tensor([stream.condition.replan for stream in streams], device=self.device)
        self.log_lambda = torch.tensor(
            [math.log(stream.condition.lambda_value) for stream in streams], device=self.device
        )
        model.ranges.normalize(self.log_lambda, self.delays, self.replans)
        configurations = tuple(dict.fromkeys((stream.condition.delay, stream.condition.replan) for stream in streams))
        self.decoder = DecisionDecoder(model, configurations, compiled=compiled)
        self.builder = PackedColumnContextBuilder(statistics, OBSERVATION_FIELDS, self.device)
        self.active = KVCache(model, 4, self.device, batch_size=len(streams), dtype=torch.float32)
        self.secondary = KVCache(model, 4, self.device, batch_size=len(streams), dtype=torch.float32)
        neutral = model.codec.quantize(torch.zeros(1, ACTION_DIM, device=self.device))[0]
        self.queue = neutral.expand(len(streams), self.decoder.max_delay + 4, 4).clone()
        self.neutral = neutral
        self.append_graph: CapturedCall[tuple[Tensor, Tensor]] | None = None
        self.last_sampled: dict[Slot, Tensor] = {}
        self.last_log_probs: dict[Slot, Tensor] = {}

    def _append(self, records: Tensor, actions: Tensor, resets: Tensor) -> tuple[Tensor, Tensor]:
        actions = torch.where(resets[:, None, None, None], 0, actions)
        features = self.builder(records, actions, self.sides, self.identities, torch.ones_like(self.sides))
        observed = self.model.codec.quantize(
            actions[torch.arange(len(self.streams), device=self.device), 0, self.sides]
        )
        tokens = self.model.context_tokens(features, observed[:, None])
        hidden = forward_tokens_with_block_handoff(self.model, tokens, self.active, self.secondary)
        return hidden, observed

    @torch.inference_mode()
    def step(
        self,
        records: NDArray[np.uint8],
        actions: NDArray[np.float32],
        frame_ids: Sequence[int | None],
        reset_flags: Sequence[bool],
    ) -> NDArray[np.float32]:
        batch = len(self.streams)
        if records.shape != (batch, 1, 4 * len(OBSERVATION_FIELDS)) or records.dtype != np.uint8:
            raise ValueError("evaluation records must contain packed schema words")
        if actions.shape != (batch, 1, 2, ACTION_DIM) or actions.dtype != np.float32 or not np.isfinite(actions).all():
            raise ValueError("observed controllers must be finite float32 HAL action vectors")
        if len(frame_ids) != batch or len(reset_flags) != batch:
            raise ValueError("evaluation frame metadata must match the stream count")
        resets = []
        for row, (frame_id, explicit_reset) in enumerate(zip(frame_ids, reset_flags, strict=True)):
            if frame_id is None:
                continue
            previous = self.last_frame[row]
            if self.missing[row] and not explicit_reset:
                raise ValueError("a missing evaluation stream must reset before returning")
            if explicit_reset or previous is None:
                resets.append(row)
            elif frame_id != previous + 1:
                raise ValueError("O65 observations must be consecutive; mark episode resets explicitly")
        for row, frame_id in enumerate(frame_ids):
            self.missing[row] = frame_id is None
            if frame_id is not None:
                self.last_frame[row] = frame_id
        self.frames[resets] = 0
        reset_mask = torch.zeros(batch, dtype=torch.bool, device=self.device)
        if resets:
            reset_at = torch.tensor(resets, device=self.device)
            reset_mask[reset_at] = True
            self.queue[reset_at] = self.neutral
            for row in resets:
                self.active.reset(row)
                self.secondary.reset(row)
        inputs = (
            torch.from_numpy(records).to(self.device),
            torch.from_numpy(actions).to(self.device),
            reset_mask,
        )
        if self.compiled and self.device.type == "cuda":
            if self.append_graph is None:
                static = tuple(value.clone() for value in inputs)
                self.append_graph = CapturedCall(
                    lambda: self._append(*static),
                    static,
                    device=self.device,
                )
                # The first capture has no ingested history to preserve.
                self.active.reset()
                self.secondary.reset()
            hidden, observed = self.append_graph(inputs)
        else:
            hidden, observed = self._append(*inputs)
        ready = [
            row
            for row, stream in enumerate(self.streams)
            if frame_ids[row] is not None and self.frames[row] % stream.condition.replan == 0
        ]
        if ready:
            at = torch.tensor(ready, device=self.device)
            count = max(self.streams[row].condition.replan for row in ready)
            uniforms = np.concatenate(
                [
                    evaluation_uniforms(
                        self.streams[row].seed, self.streams[row].logical_player, int(self.frames[row]), count
                    )
                    for row in ready
                ],
                axis=2,
            )
            memory = self.active.memory()
            output = self.decoder(
                hidden[at],
                observed[at],
                self.queue[at, : self.decoder.max_delay],
                self.log_lambda[at],
                self.delays[at],
                self.replans[at],
                torch.from_numpy(uniforms).to(self.device),
                sampled_offsets=count,
                history=KVMemory(memory.kv[:, at], memory.positions[at], memory.query_position[at], memory.window),
                validate=False,
            )
            offsets = self.delays[at, None] + torch.arange(count, device=self.device)[None]
            self.queue[at[:, None], offsets] = torch.where(
                output.valid[..., None], output.actions, self.queue[at[:, None], offsets]
            )
            for index, row in enumerate(ready):
                condition = self.streams[row].condition
                self.last_sampled[self.streams[row].slot] = output.actions[index, : condition.replan]
                self.last_log_probs[self.streams[row].slot] = output.log_probs[index, : condition.replan]
        due = self.model.codec.dequantize(self.queue[:, 0]).float().cpu().numpy()
        self.queue[:, :-1] = self.queue[:, 1:].clone()
        self.queue[:, -1] = self.neutral
        self.frames += 1
        return due


class BatchedSelfPlayPolicy:
    def __init__(self, models: tuple[BatchedSelfPlayModel, ...]) -> None:
        slots = [stream.slot for model in models for stream in model.streams]
        if not slots or len(set(slots)) != len(slots):
            raise ValueError("batched policy requires distinct stream slots")
        self.models = models
        self.slots = frozenset(slots)

    @property
    def runtime_spec(self) -> PolicyRuntimeSpec:
        return PolicyRuntimeSpec(256, 1, 1, 0, ACTION_DIM, observed_actions=True)

    def plan_rows(self, rows: Mapping[Slot, Sequence[ObservationRow]]) -> Mapping[Slot, np.ndarray]:
        if not rows or not self.slots.issuperset(rows) or any(len(values) != 1 for values in rows.values()):
            raise ValueError("batched evaluation requires known slots and one observation per player")
        result = {}
        for model in self.models:
            words = np.zeros((len(model.streams), 1, len(OBSERVATION_FIELDS)), dtype=np.uint32)
            actions = np.zeros((len(model.streams), 1, 2, ACTION_DIM), dtype=np.float32)
            frames: list[int | None] = []
            resets = []
            for index, stream in enumerate(model.streams):
                values = rows.get(stream.slot)
                if values is None:
                    frames.append(None)
                    resets.append(False)
                    continue
                row = values[0]
                flat = project_observation_columns(row.flat, OBSERVATION_FIELDS)
                for column, name in enumerate(OBSERVATION_FIELDS):
                    words.view(MDS_PER_FRAME_DTYPES[name])[index, 0, column] = flat[name]
                actions[index, 0, stream.slot.port - 1] = row.action
                frames.append(row.frame_id)
                resets.append(row.reset)
            due = model.step(words.view(np.uint8), actions, frames, resets)
            result.update(
                (stream.slot, due[index : index + 1])
                for index, stream in enumerate(model.streams)
                if stream.slot in rows
            )
        return result

    def plan_native(self, frame: NativeFrameBatch, mask: NDArray[np.bool_] | None = None) -> NDArray[np.float32]:
        if frame.records is None:
            raise ValueError("batched MSL evaluation requires native frame records")
        if mask is None:
            mask = np.ones(len(frame.frame_id), dtype=np.bool_)
        if mask.shape != frame.frame_id.shape or mask.dtype != np.bool_:
            raise ValueError("native evaluation mask must be bool [matches]")
        result = np.zeros_like(frame.applied_action)
        observed = native_controller_readback(frame.applied_action)
        for model in self.models:
            lanes = [stream.slot.match for stream in model.streams]
            records = np.ascontiguousarray(frame.records["columns"][lanes, None]).view(np.uint8)
            due = model.step(
                records,
                observed[lanes, None],
                [int(frame.frame_id[lane]) if mask[lane] else None for lane in lanes],
                [bool(frame.reset[lane]) for lane in lanes],
            )
            for index, stream in enumerate(model.streams):
                result[stream.slot.match, stream.slot.port - 1] = due[index]
        return result
