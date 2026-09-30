"""Execute prepared action-sequence predictions over independent observation streams."""

import secrets
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace
from functools import partial
from typing import Literal
from typing import cast

import numpy as np
import torch
from torch import Tensor

from hal.controller import ControllerAction
from hal.controller import action_vec_to_controller
from hal.controller import controller_to_action_vec
from hal.data.feature_stats import FeatureStats
from hal.data.schema import Rank
from hal.inference.action_sequence_artifact import REQUIRED_OBSERVATION_FIELDS
from hal.inference.api import ActionPlan
from hal.inference.api import PolicyInput
from hal.inference.api import PolicySpec
from hal.inference.api import PredictionRequest
from hal.inference.api import PreparedInferenceProfile
from hal.inference.api import RuntimeConfig
from hal.inference.api import action_plan
from hal.inference.api import contiguous_horizons
from hal.inference.api import validate_prediction_request
from hal.inference.cuda_graph import CaptureCounter
from hal.inference.cuda_graph import CapturedCall
from hal.inference.cuda_graph import prepared_shape_compilation
from hal.inference.gpu_observations import GpuObservationBatch
from hal.inference.gpu_observations import GpuObservationUpdates
from hal.inference.kv_cache import KVCache
from hal.inference.kv_cache import KVCachePool
from hal.inference.kv_cache import forward_with_kv_cache
from hal.inference.observation_history import ObservationHistory
from hal.inference.sampling import StreamGroupRng
from hal.inference.window_policy import DenseWindowPredictionPolicy
from hal.inference.window_policy import WindowPolicy
from hal.models.action_sequence import ActionSequenceTransformer
from hal.models.attention import KVMemory
from hal.models.controller_codec import CONTROLLER_GROUP_NAMES
from hal.representation.features import BASE_ITEMS_PROJECTION
from hal.representation.features import ITEM_COLUMNS
from hal.representation.features import feature_kind
from hal.representation.player_identity import FIRST_CONNECT_CODE_ID
from hal.representation.player_identity import MASKED_PLAYER_ID
from hal.wire import ACTION_CHANNELS

_ROUTES = {
    port: tuple(
        (name, f"{'ego' if (name.startswith('p1_') == (port == 1)) else 'opp'}_{name[3:]}")
        if name.startswith(("p1_", "p2_"))
        else (name, name)
        for name in REQUIRED_OBSERVATION_FIELDS
    )
    for port in (1, 2)
}


def _action_vector(action: ControllerAction) -> np.ndarray:
    return controller_to_action_vec(action, dtype=np.float32)


def _validate_prepared_observation(item: PolicyInput) -> None:
    for field, relative in _ROUTES[item.controlled_port]:
        value = item.observation[field]
        is_int = isinstance(value, (int, np.integer)) and not isinstance(value, bool)
        expects_int = feature_kind(relative, ITEM_COLUMNS) in ("cat", "button")
        if is_int != expects_int:
            raise ValueError(f"observation field {relative!r} has a noncanonical scalar type")


def _decode_cached(policy: ActionSequencePolicy, cache: KVCache, inputs: tuple[Tensor, ...]) -> tuple[Tensor, Tensor]:
    return policy._kv_decoder(cache.hidden, cache.memory(), *inputs)


@dataclass(slots=True)
class _Stream:
    history: ObservationHistory
    gpu: GpuObservationUpdates
    port: int
    player_id: int
    last_frame: int
    generation: int = -1
    sequence: int = -1
    row: int = -1
    cache: KVCache | None = None
    pending_frames: int = 0


class ActionSequencePolicy:
    """Stateful action sequence with explicit prediction requests."""

    _prediction_frames = 4
    _prefix_frames = 2
    history_mode: Literal["window", "kv_cache"] = "window"

    def __init__(
        self,
        model: ActionSequenceTransformer,
        stats: dict[str, FeatureStats],
        codes: tuple[str, ...],
        *,
        spec: PolicySpec,
        checkpoint_sha256: str,
        return_p90: float,
        capability_version: int,
        device: torch.device,
        seed: int | None,
        compiled: bool,
        history_mode: Literal["window", "kv_cache"] = "window",
        kv_update_frames: int = 2,
        kv_cuda_graphs: bool = True,
    ) -> None:
        self.model = model
        if history_mode not in ("window", "kv_cache") or kv_update_frames not in (1, 2, 4):
            raise ValueError("history must be window or kv_cache, with one, two, or four frames per update")
        self.history_mode = history_mode
        self.kv_update_frames = kv_update_frames
        self.kv_cuda_graphs = kv_cuda_graphs and compiled and device.type == "cuda"
        self.capture_counter = CaptureCounter()
        self.cfg = model.cfg
        self.checkpoint_sha256 = checkpoint_sha256
        self.return_p90 = return_p90
        self.capability_version = capability_version
        self.stats = stats
        self.device = device
        self.code_to_id = {code: FIRST_CONNECT_CODE_ID + i for i, code in enumerate(codes)}
        self.spec = spec
        self._compiled = compiled and device.type == "cuda"
        self._seed = secrets.randbits(64) if seed is None else seed
        self._rng = StreamGroupRng(self._seed, CONTROLLER_GROUP_NAMES)
        self._runtime: RuntimeConfig | None = None
        self._prepared_ready = False
        self._dense_policy: DenseWindowPredictionPolicy | None = None
        self._kv_trunk = self._advance
        self._cache_pool: KVCachePool | None = None
        self._free_rows: set[int] = set()
        self._observation_batches: dict[tuple[int, int], GpuObservationBatch] = {}
        self._update_calls: dict[tuple[int, int], CapturedCall[Tensor]] = {}
        self._decoder_calls: dict[int, CapturedCall[tuple[Tensor, Tensor]]] = {}
        self._decode_inputs: dict[int, tuple[Tensor, ...]] = {}
        self._kv_decoder = self._sample_with_kv_cache
        self._prediction_streams: dict[int, _Stream] = {}
        self._row_histories: tuple[ObservationHistory, ...] = ()
        self._row_updates: tuple[GpuObservationUpdates, ...] = ()

    def _player_id(self, identity: str | None) -> int:
        if identity is None:
            return MASKED_PLAYER_ID
        rank = Rank.__members__.get(identity)
        if rank is not None:
            return int(rank)
        try:
            return self.code_to_id[identity]
        except KeyError as error:
            raise ValueError(f"059 identity {identity!r} is absent from the checkpoint") from error

    @property
    def prepared_update_shapes(self) -> tuple[int, ...]:
        if not self._prepared_ready:
            raise RuntimeError("action-sequence policy has not been prepared")
        if self.history_mode == "kv_cache":
            return tuple(count for count in (1, 2, 4) if count <= self.kv_update_frames)
        return tuple(count for count in (1, 2, 4) if count <= self.cfg.L_ctx)

    def prepare_prediction(self, runtime: RuntimeConfig, horizon: int, prefix_frames: int) -> None:
        self._prepared_ready = False
        if horizon not in self.supported_horizons or not max(runtime.transport_delays) <= prefix_frames < horizon:
            raise ValueError("059 prediction shape requires contiguous trained heads and a fixed input prefix")
        if set(runtime.transport_delays) - set(self.spec.supported_transport_delays):
            raise ValueError("059 prediction input delay is unsupported")
        self.model.temporal.configure_live_horizons(tuple(sorted(set(self.model.temporal.live_horizons) | {horizon})))
        self._prediction_frames = horizon
        self._prefix_frames = prefix_frames
        self._runtime = runtime
        self.reset_prediction()
        if self.history_mode == "kv_cache":
            # One additional profile covers the two local fixed-prefix shapes
            # that can share a declared transport delay.
            profile_count = len(self.spec.supported_transport_delays) + 1
            shape_count = runtime.max_batch_size.bit_length() * 3 * profile_count
            with prepared_shape_compilation(shape_count):
                self._prepare_kv_cache()
            self._prepared_ready = True
            return
        buckets = []
        bucket = 1
        while bucket // 2 < runtime.max_batch_size:
            buckets.append(bucket)
            bucket *= 2
        executor = WindowPolicy(
            self.model,
            context_frames=self.cfg.L_ctx,
            prediction_frames=horizon,
            prepared_buckets=tuple(buckets),
            compiled=self._compiled,
            amp_dtype="bfloat16",
            return_conditioning=self.cfg.return_conditioning,
        )
        self._dense_policy = DenseWindowPredictionPolicy(
            executor,
            self.stats,
            seed=self._seed,
            player_codes=tuple(self.code_to_id),
            spec=self.spec,
            checkpoint_sha256=self.checkpoint_sha256,
        )
        self._dense_policy.prepare_prediction(runtime, horizon, prefix_frames)
        self._prepared_ready = True

    def validate_prepared_profile(self, profile: PreparedInferenceProfile) -> None:
        runtime = self._runtime
        if (
            not self._prepared_ready
            or runtime is None
            or (self.history_mode == "kv_cache" and self._cache_pool is None)
        ):
            raise ValueError("action-sequence policy has not been prepared")
        if self.history_mode == "window" and self._dense_policy is None:
            raise ValueError("action-sequence window policy has not been prepared")
        if (
            profile.checkpoint_sha256 != self.checkpoint_sha256
            or profile.execution_mode != self.history_mode
            or profile.prediction_horizon_frames != self._prediction_frames
            or profile.fixed_prefix_frames != self._prefix_frames
            or profile.capacity != runtime.max_batch_size
        ):
            raise ValueError("prepared inference profile differs from the action-sequence policy")
        supported = (
            self.prepared_update_shapes if self.history_mode == "kv_cache" else tuple(range(1, self.cfg.L_ctx + 1))
        )
        if not set(profile.update_shapes) <= set(supported):
            raise ValueError("prepared inference profile declares an unprepared observation update shape")

    def _advance(self, features: dict[str, Tensor], observed: Tensor, cache: KVCache) -> Tensor:
        return forward_with_kv_cache(self.model, features, observed, cache)

    def _sample_with_kv_cache(
        self,
        hidden: Tensor,
        memory: KVMemory,
        observed: Tensor,
        uniforms: Tensor,
        forced: Tensor,
        return_value: Tensor,
        condition_present: Tensor,
        temperature: Tensor,
    ) -> tuple[Tensor, Tensor]:
        indices = self.model.temporal.sample_indices(
            hidden,
            observed,
            self.model.head_offsets[: self._prediction_frames],
            return_value,
            condition_present,
            argmax=False,
            uniforms=uniforms,
            temperature=temperature,
            forced_prefix=forced,
            history=memory,
        )
        return indices, self.model.estimate_value(hidden)

    def _prepare_kv_cache(self) -> None:
        if self._compiled:
            self._kv_trunk = torch.compile(self._advance, dynamic=False, fullgraph=True)
            self._kv_decoder = torch.compile(self._sample_with_kv_cache, dynamic=False, fullgraph=True)
        assert self._runtime is not None
        capacity = self._runtime.max_batch_size
        pool = KVCachePool(self.model, capacity, self.device, max_update_frames=4)
        self._cache_pool = pool
        self._free_rows = set(range(capacity))
        self._observation_batches.clear()
        self._update_calls.clear()
        self._decoder_calls.clear()
        self._decode_inputs.clear()
        dummy = {
            name: (0 if feature_kind(relative, ITEM_COLUMNS) in ("cat", "button") else 0.0)
            for name, relative in _ROUTES[1]
        }
        relative = {name: dummy[field] for field, name in _ROUTES[1]}
        template = ObservationHistory.from_frame(
            relative, "p1", self.stats, self.cfg.L_ctx, ITEM_COLUMNS, BASE_ITEMS_PROJECTION
        )
        histories = [template]
        histories.extend(ObservationHistory(template.layout, self.cfg.L_ctx) for _ in range(capacity - 1))
        self._row_histories = tuple(histories)
        self._row_updates = tuple(
            GpuObservationUpdates(history, self.model.codec, self.device, self.kv_update_frames)
            for history in self._row_histories
        )
        neutral = np.zeros(len(ACTION_CHANNELS), dtype=np.float32)
        for history, stage in zip(self._row_histories, self._row_updates, strict=True):
            history.gather(relative, neutral)
            history.push()
            stage.push(neutral)
            stage.upload(MASKED_PLAYER_ID)
            history.reset()
            stage.reset()
        history = self._row_histories[0]
        stage = self._row_updates[0]
        for _ in range(self.kv_update_frames):
            history.gather(relative, neutral)
            history.push()
            stage.push(neutral)
        buckets = (1, *(bucket for bucket in pool.scratch if bucket > 1))
        with (
            torch.inference_mode(),
            torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"),
        ):
            for bucket in buckets:
                cache = pool.row_view(0) if capacity == 1 else pool.scratch[bucket]
                for count in (1, 2, 4):
                    if count > self.kv_update_frames:
                        continue
                    batch = GpuObservationBatch(stage, bucket, count)
                    batch.gather((stage,) * bucket, (MASKED_PLAYER_ID,) * bucket)
                    self._observation_batches[(bucket, count)] = batch
                    operation = partial(self._kv_trunk, batch.features(), batch.actions, cache)
                    if self.kv_cuda_graphs:
                        self._update_calls[(bucket, count)] = CapturedCall(
                            operation, (), cache.buffers(), device=self.device, counter=self.capture_counter
                        )
                    else:
                        operation()
                    cache.reset()
                first = self._observation_batches[(bucket, 1)]
                self._kv_trunk(first.features(), first.actions, cache)
                observed = first.actions[:, -1]
                uniforms = torch.full(
                    (self._prediction_frames, len(CONTROLLER_GROUP_NAMES), bucket), 0.5, device=self.device
                )
                forced = torch.zeros(
                    (bucket, self._prefix_frames, len(CONTROLLER_GROUP_NAMES)), dtype=torch.long, device=self.device
                )
                values = torch.full((bucket,), 20.0, device=self.device)
                present = torch.ones(bucket, dtype=torch.bool, device=self.device)
                temperature = torch.ones(bucket, 1, device=self.device)
                inputs = (observed, uniforms, forced, values, present, temperature)
                self._decode_inputs[bucket] = inputs
                operation = partial(_decode_cached, self, cache, inputs)
                if self.kv_cuda_graphs:
                    self._decoder_calls[bucket] = CapturedCall(
                        operation, (), cache.buffers(), device=self.device, counter=self.capture_counter
                    )
                    sampled, state_values = self._decoder_calls[bucket](())
                else:
                    sampled, state_values = operation()
                # Torch 2.11 lazily initializes controller dequantization kernels
                # on their first shape. Prepare the full delivered-action path.
                decoded = self.model.codec.dequantize(sampled)[
                    :, self._prefix_frames : self._prediction_frames
                ].float()
                delivered = torch.cat((decoded.flatten(1), state_values[:, None]), dim=1).cpu().numpy()
                for row in delivered:
                    for action in row[:-1].reshape(-1, len(ACTION_CHANNELS)):
                        action_vec_to_controller(action)
                cache.reset()
            pool.storage.reset()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.reset_prediction()

    def _advance_batch(self, streams: tuple[_Stream, ...], count: int, columns: slice | None = None) -> None:
        pool = self._cache_pool
        if pool is None:
            raise RuntimeError("KV cache was not prepared")
        bucket = 1 << (len(streams) - 1).bit_length()
        batch = self._observation_batches[(bucket, count)]
        stages = tuple(stream.gpu for stream in streams)
        if not all(isinstance(stage, GpuObservationUpdates) for stage in stages):
            raise RuntimeError("cached stream lacks prepared observation updates")
        batch.gather(
            stages,
            tuple(stream.player_id for stream in streams),
            columns,
        )
        cache = pool.gather(tuple(stream.row for stream in streams), bucket, direct=pool.capacity == 1)
        with (
            torch.inference_mode(),
            torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"),
        ):
            if self.kv_cuda_graphs:
                self._update_calls[(bucket, count)](())
            else:
                self._kv_trunk(batch.features(), batch.actions, cache)
        pool.scatter(tuple(stream.row for stream in streams), cache)
        for stream in streams:
            stream.pending_frames -= count

    def _ingest(self, item: PolicyInput, stream: _Stream | None, row: int) -> _Stream:
        player_id = self._player_id(item.player_identity)
        if item.reset or (stream is not None and item.frame_id != stream.last_frame + 1):
            stream = None
        if stream is not None and (stream.port != item.controlled_port or stream.player_id != player_id):
            raise ValueError("059 port or identity changed without reset")
        flat = {
            relative: int(item.observation[name])
            if isinstance(item.observation[name], (int, np.integer))
            else float(item.observation[name])
            for name, relative in _ROUTES[item.controlled_port]
        }
        if stream is None:
            history = self._row_histories[row]
            stage = self._row_updates[row]
            history.reset()
            stage.reset()
            stream = _Stream(
                history,
                stage,
                item.controlled_port,
                player_id,
                item.frame_id,
                row=row,
            )
            if self._cache_pool is None:
                raise RuntimeError("KV cache was not prepared")
            self._cache_pool.reset(row)
            stream.cache = self._cache_pool.row_view(row)
        action = _action_vector(item.applied_action)
        stream.history.gather(flat, action)
        stream.history.push()
        stream.gpu.push(action)
        if stream.cache is not None:
            stream.pending_frames += 1
        stream.last_frame = item.frame_id
        return stream

    @torch.inference_mode()
    def _plan_cached_batch(
        self, requests: tuple[PredictionRequest, ...], streams: tuple[_Stream, ...]
    ) -> tuple[tuple[tuple[ControllerAction, ...], float], ...]:
        pool = self._cache_pool
        if pool is None:
            raise RuntimeError("KV cache was not prepared")
        bucket = 1 << (len(streams) - 1).bit_length()
        stages = tuple(stream.gpu for stream in streams)
        if not all(isinstance(stage, GpuObservationUpdates) for stage in stages):
            raise RuntimeError("cached stream lacks prepared observation updates")
        batch = self._observation_batches[(bucket, 1)]
        batch.gather(
            stages,
            tuple(stream.player_id for stream in streams),
        )
        cache = pool.gather(tuple(stream.row for stream in streams), bucket, direct=pool.capacity == 1)
        if len(streams) < bucket:
            cache.positions[len(streams) :, 0] = 0
            cache.next_position[len(streams) :] = 1
        committed = np.array(
            [[_action_vector(action) for action in request.fixed_actions] for request in requests], dtype=np.float32
        ).reshape(len(requests), self._prefix_frames, len(ACTION_CHANNELS))
        if len(streams) < bucket:
            committed = np.pad(committed, ((0, bucket - len(streams)), (0, 0), (0, 0)))
        forced = self.model.codec.quantize(torch.from_numpy(committed).to(self.device))
        self._rng.begin(
            tuple(request.stream_id for request in requests),
            tuple(request.generation - 1 for request in requests),
            device="cpu",
        )
        draws = torch.stack(
            [
                torch.stack(
                    [
                        self._rng.uniforms(name, [depth >= self._prefix_frames] * len(streams))
                        for name in CONTROLLER_GROUP_NAMES
                    ]
                )
                for depth in range(self._prediction_frames)
            ]
        )
        uniforms = torch.full((self._prediction_frames, len(CONTROLLER_GROUP_NAMES), bucket), 0.5, device=self.device)
        uniforms[:, :, : len(streams)] = draws.to(self.device)
        values = torch.zeros(bucket, device=self.device)
        present = torch.zeros(bucket, dtype=torch.bool, device=self.device)
        temperature = torch.ones(bucket, 1, device=self.device)
        for index, request in enumerate(requests):
            item = request.observations[-1]
            if item.desired_return is not None:
                values[index] = item.desired_return
                present[index] = True
            temperature[index, 0] = item.temperature
        inputs = (batch.actions[:, -1], uniforms, forced, values, present, temperature)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            if self.kv_cuda_graphs:
                for target, source in zip(self._decode_inputs[bucket], inputs, strict=True):
                    target.copy_(source)
                indices, state_values = self._decoder_calls[bucket](())
            else:
                indices, state_values = self._kv_decoder(cache.hidden, cache.memory(), *inputs)
            actions = self.model.codec.dequantize(indices[: len(streams)])[
                :, self._prefix_frames : self._prediction_frames
            ].float()
            delivered = torch.cat((actions.flatten(1), state_values[: len(streams), None]), dim=1).cpu().numpy()
        return tuple(
            (
                tuple(action_vec_to_controller(row) for row in plan[:-1].reshape(-1, len(ACTION_CHANNELS))),
                float(plan[-1]),
            )
            for plan in delivered
        )

    @property
    def sampling_seed(self) -> int:
        return self._seed

    @property
    def context_frames(self) -> int:
        return self.cfg.L_ctx

    @property
    def supported_horizons(self) -> tuple[int, ...]:
        return contiguous_horizons(self.model.head_offsets)

    @property
    def prediction_horizon(self) -> int:
        return self._prediction_frames

    def reset_prediction(self, *, seed: int | None = None) -> None:
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("sampling seed must be a nonnegative integer")
            self._seed = seed
        if self._dense_policy is not None:
            self._dense_policy.reset_prediction(seed=self._seed)
        self._prediction_streams.clear()
        if self._cache_pool is not None:
            self._cache_pool.storage.reset()
        for history, stage in zip(self._row_histories, self._row_updates, strict=True):
            history.reset()
            stage.reset()
        if self._runtime is not None:
            self._free_rows = set(range(self._runtime.max_batch_size))
        self._rng = StreamGroupRng(self._seed, CONTROLLER_GROUP_NAMES)

    def release_stream(self, stream_id: int) -> None:
        """Return one admitted row to the prepared pool after its match ends."""
        if self._dense_policy is not None and self.history_mode == "window":
            self._dense_policy.release_stream(stream_id)
            return
        stream = self._prediction_streams.pop(stream_id, None)
        if stream is None:
            return
        if self._cache_pool is not None:
            self._cache_pool.reset(stream.row)
        self._row_histories[stream.row].reset()
        self._row_updates[stream.row].reset()
        self._free_rows.add(stream.row)
        self._rng.release(stream_id)

    @torch.inference_mode()
    def predict(self, requests: Sequence[PredictionRequest]) -> Sequence[ActionPlan]:
        if not self._prepared_ready:
            raise RuntimeError("action sequence is not prepared for prediction")
        if self.history_mode == "window":
            if self._dense_policy is None:
                raise RuntimeError("dense action sequence is not prepared")
            with torch.inference_mode(False):
                return self._dense_policy.predict(requests)
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("action sequence is not prepared for prediction")
        if not requests or len(requests) > runtime.max_batch_size:
            raise ValueError("invalid action sequence prediction batch size")
        if len({request.stream_id for request in requests}) != len(requests):
            raise ValueError("duplicate action sequence prediction stream")
        ordered = tuple(requests)
        for request in ordered:
            if request.generation < 1:
                raise ValueError("action sequence prediction generation must start at one")
            validate_prediction_request(
                self.spec,
                runtime,
                request,
                context_frames=self.context_frames,
                prefix_frames=self._prefix_frames,
            )
            stream = self._prediction_streams.get(request.stream_id)
            generation = None if stream is None else stream.generation
            for observation in request.observations:
                _validate_prepared_observation(observation)
            if generation is not None and request.generation < generation:
                raise ValueError("obsolete action sequence prediction generation")
            if stream is not None and generation == request.generation and request.sequence <= stream.sequence:
                raise ValueError("obsolete action sequence prediction sequence")
            if (
                stream is not None
                and generation == request.generation
                and request.observations[0].frame_id != stream.last_frame + 1
            ):
                raise ValueError("prediction observations must start at the next frame")
            if (
                stream is not None
                and generation == request.generation
                and any(item.reset for item in request.observations)
            ):
                raise ValueError("prediction reset requires a new generation")
        new_ids = {request.stream_id for request in ordered if request.stream_id not in self._prediction_streams}
        if len(new_ids) > len(self._free_rows):
            raise ValueError("prepared action sequence stream capacity is exhausted")
        groups: dict[int, list[tuple[int, PredictionRequest, int, _Stream | None]]] = {}
        for index, request in enumerate(ordered):
            previous = self._prediction_streams.get(request.stream_id)
            if previous is None:
                row = min(self._free_rows)
                self._free_rows.remove(row)
            else:
                row = previous.row
            current = previous if previous is not None and previous.generation == request.generation else None
            groups.setdefault(len(request.observations), []).append((index, request, row, current))
        plans: dict[int, ActionPlan] = {}
        for observation_count, group in groups.items():
            states = [entry[3] for entry in group]
            for at in range(observation_count):
                for index, (_, request, row, _) in enumerate(group):
                    item = request.observations[at]
                    states[index] = self._ingest(replace(item, reset=states[index] is None), states[index], row)
                if self.history_mode == "kv_cache" and (at + 1) % self.kv_update_frames == 0:
                    self._advance_batch(tuple(cast(_Stream, state) for state in states), self.kv_update_frames)
            streams = tuple(cast(_Stream, state) for state in states)
            remainder = observation_count % self.kv_update_frames
            if remainder == 3:
                self._advance_batch(streams, 2, slice(-3, -1))
                self._advance_batch(streams, 1, slice(-1, None))
            elif remainder:
                self._advance_batch(streams, remainder)
            tails = self._plan_cached_batch(tuple(entry[1] for entry in group), streams)
            for (index, request, _, _), stream, (tail, state_value) in zip(group, streams, tails, strict=True):
                if stream.last_frame != request.source_frame:
                    raise ValueError("obsolete action sequence prediction request")
                stream.generation = request.generation
                stream.sequence = request.sequence
                self._prediction_streams[request.stream_id] = stream
                plans[index] = action_plan(request, tail, state_value=state_value)
        return tuple(plans[index] for index in range(len(ordered)))
