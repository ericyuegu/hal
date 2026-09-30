"""Dense action-sequence execution for the official 059 evaluation profile."""

import contextlib
import functools
import math
import time
from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from hal.controller import action_vec_to_controller
from hal.controller import controller_to_action_vec
from hal.data.feature_stats import FeatureStats
from hal.data.schema import Rank
from hal.inference.action_sequence_artifact import REQUIRED_OBSERVATION_FIELDS
from hal.inference.api import ActionPlan
from hal.inference.api import PolicySpec
from hal.inference.api import PredictionRequest
from hal.inference.api import PreparedInferenceProfile
from hal.inference.api import RuntimeConfig
from hal.inference.api import action_plan
from hal.inference.api import contiguous_horizons
from hal.inference.api import validate_prediction_request
from hal.inference.benchmark import DecodeTelemetry
from hal.inference.observation_history import ObservationHistory
from hal.inference.observation_history import ObservationWindows
from hal.inference.observation_history import stack_observation_windows
from hal.inference.sampling import StreamGroupRng
from hal.inference.warmup import canonical_context
from hal.inference.warmup import synthetic_context
from hal.models.action_sequence import ActionSequenceTransformer
from hal.models.controller_codec import CONTROLLER_GROUP_COUNT
from hal.models.controller_codec import CONTROLLER_GROUP_NAMES
from hal.models.sampling import validate_sampling_temperature
from hal.representation.features import ACTION_CHANNELS
from hal.representation.features import BASE_ITEMS_PROJECTION
from hal.representation.features import ITEM_COLUMNS
from hal.representation.features import NEUTRAL_ACTION
from hal.representation.features import Context
from hal.representation.features import stack_actions
from hal.representation.player_identity import FIRST_CONNECT_CODE_ID
from hal.representation.player_identity import MASKED_PLAYER_ID
from hal.sim.rollout import covering_power_of_two


def _sample_decoder(
    model: ActionSequenceTransformer,
    offsets: tuple[int, ...],
    hidden: Tensor,
    ctx_pad: Tensor,
    observed: Tensor,
    uniforms: Tensor,
    forced_prefix: Tensor,
    return_value: Tensor,
    condition_present: Tensor,
    temperature: Tensor,
) -> tuple[Tensor, Tensor]:
    indices = model.temporal.sample_indices(
        hidden,
        observed,
        offsets,
        return_value,
        condition_present,
        argmax=False,
        uniforms=uniforms,
        temperature=temperature,
        ctx_pad=ctx_pad,
        forced_prefix=forced_prefix,
    )
    return indices, model.estimate_value(hidden)


def _sample_decoder_with_logits(
    model: ActionSequenceTransformer,
    offsets: tuple[int, ...],
    hidden: Tensor,
    ctx_pad: Tensor,
    observed: Tensor,
    uniforms: Tensor,
    forced_prefix: Tensor,
    return_value: Tensor,
    condition_present: Tensor,
    temperature: Tensor,
) -> tuple[Tensor, tuple[Tensor, ...], Tensor]:
    indices, logits = model.temporal.sample_indices_with_logits(
        hidden,
        observed,
        offsets,
        return_value,
        condition_present,
        argmax=False,
        uniforms=uniforms,
        temperature=temperature,
        ctx_pad=ctx_pad,
        forced_prefix=forced_prefix,
    )
    return indices, logits, model.estimate_value(hidden)


def _pad_context(ctx: Context, bucket: int) -> Context:
    rows = ctx.ctx_pad.shape[0]
    if rows == bucket:
        return ctx
    if rows > bucket:
        raise ValueError("cannot pad a context to a smaller bucket")
    extra = bucket - rows
    features = {
        name: torch.cat((value, torch.zeros((extra, *value.shape[1:]), dtype=value.dtype, device=value.device)))
        for name, value in ctx.features.items()
    }
    ctx_pad = torch.cat(
        (
            ctx.ctx_pad,
            torch.full(
                (extra,),
                ctx.features[next(iter(ctx.features))].shape[1] - 1,
                dtype=ctx.ctx_pad.dtype,
                device=ctx.ctx_pad.device,
            ),
        )
    )
    return Context(features=features, ctx_pad=ctx_pad)


def condition_ego_player(ctx: Context, player_id: int) -> Context:
    """Attach one runtime ego identity without specializing the compiled graph."""
    return condition_ego_players(ctx, (player_id,) * ctx.ctx_pad.shape[0])


def condition_ego_players(ctx: Context, player_ids: Sequence[int]) -> Context:
    """Attach independently selected ego identities for one inference batch."""
    if "opp_player_id" in ctx.features:
        raise ValueError("opponent identity must never enter inference")
    if len(player_ids) != ctx.ctx_pad.shape[0] or any(
        not isinstance(player_id, int) or isinstance(player_id, bool) or player_id < 0 for player_id in player_ids
    ):
        raise ValueError("one non-negative player ID is required per inference row")
    reference = ctx.features[next(iter(ctx.features))]
    features = dict(ctx.features)
    features["ego_player_id"] = (
        torch.tensor(player_ids, dtype=torch.long, device=reference.device)[:, None]
        .expand(reference.shape[:2])
        .contiguous()
    )
    return replace(ctx, features=features)


@dataclass(frozen=True, slots=True)
class DecodedPlan:
    state_values: Tensor
    actions: Tensor
    indices: Tensor
    logits: tuple[Tensor, ...]
    uniforms: Tensor


@dataclass(frozen=True, slots=True)
class _PreparedDecode:
    rows: int
    bucket: int
    ctx_pad: Tensor
    hidden: Tensor
    observed: Tensor
    uniforms: Tensor
    forced_prefix: Tensor
    return_value: Tensor
    condition_present: Tensor
    temperature: Tensor


class WindowPolicy:
    """Hardware-bucketed compiled trunk and unrolled dense-prefix decoders.

    Evaluation compiles each required program synchronously on first use. Runtime
    calls use the smallest compiled bucket that fits. Padding and slot-keyed random
    streams leave real rows unchanged.
    """

    def __init__(
        self,
        model: ActionSequenceTransformer,
        *,
        context_frames: int,
        prediction_frames: int,
        prepared_buckets: tuple[int, ...],
        compiled: bool,
        amp_dtype: str = "bfloat16",
        return_conditioning: bool = True,
        compile_mode: str = "default",
        temperature: float = 1.0,
        desired_return: float | None = None,
    ) -> None:
        if context_frames < 1 or prediction_frames < 1:
            raise ValueError("context and prediction lengths must be positive")
        if amp_dtype not in {"bfloat16", "float32"}:
            raise ValueError(f"unsupported inference amp dtype {amp_dtype!r}")
        if desired_return is not None and (not math.isfinite(desired_return) or not return_conditioning):
            raise ValueError("desired return requires finite input and enabled conditioning")
        self.desired_return = desired_return
        self.model = model
        self.context_frames = context_frames
        self.prediction_frames = prediction_frames
        self.amp_dtype = amp_dtype
        self.prepared_buckets = tuple(sorted(set(prepared_buckets)))
        if not self.prepared_buckets or any(bucket < 1 or bucket & (bucket - 1) for bucket in self.prepared_buckets):
            raise ValueError(f"prepared_buckets must be positive powers of two, got {self.prepared_buckets}")
        self.compiled = bool(compiled and next(model.parameters()).device.type == "cuda")
        self.compile_mode = compile_mode
        self.temperature = validate_sampling_temperature(temperature)
        self.attention_backend = "dense_sdpa"
        self.compile_seconds = 0.0
        self._warmed: set[tuple[int, int, int]] = set()
        self._trunks: dict[int, Callable] = {}
        self._decoders: dict[tuple[int, int, int], Callable] = {}
        self._trace_decoders: dict[tuple[int, int, int], Callable] = {}

    @property
    def uses_cuda_graphs(self) -> bool:
        return self.compiled and self.compile_mode == "reduce-overhead"

    def _bucket(self, rows: int) -> int:
        try:
            return next(bucket for bucket in self.prepared_buckets if bucket >= rows)
        except StopIteration as exc:
            if self.compiled:
                raise ValueError(
                    f"inference batch {rows} exceeds largest prepared bucket {self.prepared_buckets[-1]}"
                ) from exc
            return covering_power_of_two(rows)

    def _amp_context(self, device: torch.device | str):
        if self.amp_dtype == "bfloat16" and torch.device(device).type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _trunk(self, bucket: int) -> Callable:
        if bucket not in self._trunks:
            forward = self.model.forward_dense
            self._trunks[bucket] = (
                torch.compile(forward, dynamic=False, fullgraph=True, mode=self.compile_mode)
                if self.compiled
                else forward
            )
        return self._trunks[bucket]

    def _decoder(self, bucket: int, horizon: int, committed_frames: int) -> Callable:
        key = (bucket, horizon, committed_frames)
        if key not in self._decoders:
            offsets = self.model.head_offsets[:horizon]

            call = functools.partial(_sample_decoder, self.model, offsets)
            self._decoders[key] = torch.compile(call, dynamic=False, mode=self.compile_mode) if self.compiled else call
        return self._decoders[key]

    def _trace_decoder(self, bucket: int, horizon: int, committed_frames: int) -> Callable:
        key = (bucket, horizon, committed_frames)
        if key not in self._trace_decoders:
            offsets = self.model.head_offsets[:horizon]

            call = functools.partial(_sample_decoder_with_logits, self.model, offsets)
            self._trace_decoders[key] = (
                torch.compile(call, dynamic=False, mode=self.compile_mode) if self.compiled else call
            )
        return self._trace_decoders[key]

    def _prepare_decode(
        self,
        ctx: Context,
        horizon: int,
        *,
        streams: StreamGroupRng | None,
        stream_ids: Sequence[int] | None,
        sampling_generations: Sequence[int] | None,
        desired_returns: Sequence[float | None] | None,
        temperatures: Sequence[float] | None,
        gen: torch.Generator | None,
        committed: Tensor | None,
    ) -> _PreparedDecode:
        if horizon != self.prediction_frames:
            raise ValueError(f"horizon must be {self.prediction_frames}")
        rows = ctx.ctx_pad.shape[0]
        if committed is not None and (
            committed.ndim != 3
            or committed.shape[0] != rows
            or committed.shape[2] != len(ACTION_CHANNELS)
            or committed.shape[1] > horizon
        ):
            raise ValueError("committed actions have the wrong shape")
        if streams is not None and (
            stream_ids is None
            or sampling_generations is None
            or len(stream_ids) != rows
            or len(sampling_generations) != rows
        ):
            raise ValueError("stream sampling requires one stream ID and generation per row")
        if desired_returns is not None and len(desired_returns) != rows:
            raise ValueError("return conditioning needs one value per row")
        if temperatures is not None and len(temperatures) != rows:
            raise ValueError("sampling temperature needs one value per row")
        returns = (self.desired_return,) * rows if desired_returns is None else tuple(desired_returns)
        if any(value is not None and not math.isfinite(value) for value in returns):
            raise ValueError("desired returns must be finite or absent")
        values = (self.temperature,) * rows if temperatures is None else tuple(temperatures)
        values = tuple(validate_sampling_temperature(value) for value in values)
        committed_frames = 0 if committed is None else committed.shape[1]
        bucket = self._bucket(rows)
        padded = _pad_context(ctx, bucket)
        if "ego_player_id" not in padded.features:
            padded = condition_ego_player(padded, MASKED_PLAYER_ID)
        padded = canonical_context(padded, items=True)
        observed = self.model.codec.quantize(stack_actions(padded.features))
        uniform_parts: list[Tensor] = []
        if streams is not None:
            assert stream_ids is not None and sampling_generations is not None
            streams.begin(stream_ids, sampling_generations, device=ctx.ctx_pad.device)
        for frame in range(horizon):
            groups = []
            for name in CONTROLLER_GROUP_NAMES:
                if frame < committed_frames:
                    real = torch.full((rows,), 0.5, device=ctx.ctx_pad.device)
                elif streams is None:
                    real = torch.rand(rows, device=ctx.ctx_pad.device, generator=gen)
                else:
                    real = streams.uniforms(name)
                groups.append(F.pad(real, (0, bucket - rows), value=0.5))
            uniform_parts.append(torch.stack(groups))
        uniforms = torch.stack(uniform_parts)
        if self.uses_cuda_graphs:
            # The trunk and decoder share one graph step so the next trunk replay
            # can reuse its output storage only after the decoder consumes it.
            torch.compiler.cudagraph_mark_step_begin()
        with self._amp_context(ctx.ctx_pad.device):
            hidden = self._trunk(bucket)(padded.features, padded.ctx_pad, observed)
            if committed is None:
                forced_prefix = torch.empty(bucket, 0, CONTROLLER_GROUP_COUNT, dtype=torch.long, device=hidden.device)
            else:
                padded_committed = F.pad(committed, (0, 0, 0, 0, 0, bucket - rows))
                forced_prefix = self.model.codec.quantize(padded_committed)
        return_value = torch.tensor([0.0 if value is None else value for value in returns], device=hidden.device)
        return_value = F.pad(return_value, (0, bucket - rows))
        condition_present = torch.tensor([value is not None for value in returns], device=hidden.device)
        condition_present = F.pad(condition_present, (0, bucket - rows))
        temperature = torch.tensor(values, device=hidden.device).unsqueeze(-1)
        temperature = F.pad(temperature, (0, 0, 0, bucket - rows), value=1.0)
        return _PreparedDecode(
            rows,
            bucket,
            padded.ctx_pad,
            hidden,
            observed[:, -1],
            uniforms,
            forced_prefix,
            return_value,
            condition_present,
            temperature,
        )

    @torch.no_grad()
    def prewarm(self, rows: int, horizon: int, *, committed_frames: int) -> float:
        """Compile and replay the exact evaluation program before Dolphin starts."""
        if horizon != self.prediction_frames or not 0 <= committed_frames <= horizon:
            raise ValueError("invalid prewarm horizon or committed-prefix length")
        bucket = self._bucket(rows)
        key = (bucket, horizon, committed_frames)
        if key in self._warmed or not self.compiled:
            self._warmed.add(key)
            return 0.0
        device = next(self.model.parameters()).device
        started = time.perf_counter()
        context = condition_ego_player(
            synthetic_context(self.context_frames, rows, device, items=True), MASKED_PLAYER_ID
        )
        neutral = torch.as_tensor(NEUTRAL_ACTION, device=device).expand(rows, committed_frames, -1)
        self.decode(context, horizon, committed=neutral)
        self.decode(context, horizon, committed=neutral)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        self.compile_seconds += elapsed
        self._warmed.add(key)
        print(
            f"[inference] synchronously compiled batch {bucket}, horizon {horizon} in {elapsed:.1f}s",
            flush=True,
        )
        return elapsed

    @torch.no_grad()
    def decode(
        self,
        ctx: Context,
        horizon: int,
        *,
        streams: StreamGroupRng | None = None,
        stream_ids: Sequence[int] | None = None,
        sampling_generations: Sequence[int] | None = None,
        desired_returns: Sequence[float | None] | None = None,
        temperatures: Sequence[float] | None = None,
        argmax: bool = False,
        gen: torch.Generator | None = None,
        committed: Tensor | None = None,
    ) -> Tensor:
        actions, _ = self.decode_prediction(
            ctx,
            horizon,
            streams=streams,
            stream_ids=stream_ids,
            sampling_generations=sampling_generations,
            desired_returns=desired_returns,
            temperatures=temperatures,
            argmax=argmax,
            gen=gen,
            committed=committed,
        )
        return actions

    @torch.no_grad()
    def decode_prediction(
        self,
        ctx: Context,
        horizon: int,
        *,
        streams: StreamGroupRng | None = None,
        stream_ids: Sequence[int] | None = None,
        sampling_generations: Sequence[int] | None = None,
        desired_returns: Sequence[float | None] | None = None,
        temperatures: Sequence[float] | None = None,
        argmax: bool = False,
        gen: torch.Generator | None = None,
        committed: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        prepared = self._prepare_decode(
            ctx,
            horizon,
            streams=streams,
            stream_ids=stream_ids,
            sampling_generations=sampling_generations,
            desired_returns=desired_returns,
            temperatures=temperatures,
            gen=gen,
            committed=committed,
        )
        with self._amp_context(ctx.ctx_pad.device):
            if argmax:
                indices = self.model.temporal.sample_indices(
                    prepared.hidden,
                    prepared.observed,
                    self.model.head_offsets[:horizon],
                    prepared.return_value,
                    prepared.condition_present,
                    argmax=True,
                    temperature=self.temperature,
                    ctx_pad=prepared.ctx_pad,
                    forced_prefix=prepared.forced_prefix,
                )
                state_values = self.model.estimate_value(prepared.hidden)
            else:
                indices, state_values = self._decoder(prepared.bucket, horizon, prepared.forced_prefix.shape[1])(
                    prepared.hidden,
                    prepared.ctx_pad,
                    prepared.observed,
                    prepared.uniforms,
                    prepared.forced_prefix,
                    prepared.return_value,
                    prepared.condition_present,
                    prepared.temperature,
                )
        return self.model.codec.dequantize(indices[: prepared.rows]), state_values[: prepared.rows]

    @torch.no_grad()
    def decode_with_trace(
        self,
        ctx: Context,
        horizon: int,
        *,
        streams: StreamGroupRng,
        stream_ids: Sequence[int],
        sampling_generations: Sequence[int],
        desired_returns: Sequence[float | None] | None = None,
        temperatures: Sequence[float] | None = None,
        committed: Tensor | None = None,
    ) -> DecodedPlan:
        """Decode once and retain the exact logits and uniforms used to sample."""
        prepared = self._prepare_decode(
            ctx,
            horizon,
            streams=streams,
            stream_ids=stream_ids,
            sampling_generations=sampling_generations,
            desired_returns=desired_returns,
            temperatures=temperatures,
            gen=None,
            committed=committed,
        )
        with self._amp_context(ctx.ctx_pad.device):
            indices, logits, state_values = self._trace_decoder(
                prepared.bucket, horizon, prepared.forced_prefix.shape[1]
            )(
                prepared.hidden,
                prepared.ctx_pad,
                prepared.observed,
                prepared.uniforms,
                prepared.forced_prefix,
                prepared.return_value,
                prepared.condition_present,
                prepared.temperature,
            )
        real_indices = indices[: prepared.rows]
        return DecodedPlan(
            state_values=state_values[: prepared.rows],
            actions=self.model.codec.dequantize(real_indices),
            indices=real_indices,
            logits=tuple(values[: prepared.rows] for values in logits),
            uniforms=prepared.uniforms[:, :, : prepared.rows],
        )


@dataclass(slots=True)
class _WindowStream:
    history: ObservationHistory
    generation: int
    sequence: int
    last_frame: int
    port: int
    player_id: int
    reset_pending: bool


@dataclass(frozen=True, slots=True)
class _FaultInput:
    windows: ObservationWindows
    fixed_actions: np.ndarray
    metadata: dict[str, object]


def _relative_observation(item: PredictionRequest, port: int, frame_index: int) -> dict[str, float | int]:
    observation = item.observations[frame_index].observation
    relative: dict[str, float | int] = {}
    for name in REQUIRED_OBSERVATION_FIELDS:
        if name.startswith(("p1_", "p2_")):
            prefix = "ego" if int(name[1]) == port else "opp"
            relative[f"{prefix}_{name[3:]}"] = observation[name]
        else:
            relative[name] = observation[name]
    return relative


class DenseWindowPredictionPolicy:
    """Batch the official dense 059 profile over independent replay histories."""

    history_mode = "window"

    def __init__(
        self,
        executor: WindowPolicy,
        stats: dict[str, FeatureStats],
        *,
        seed: int,
        ego_player_id: int = MASKED_PLAYER_ID,
        player_codes: tuple[str, ...] = (),
        spec: PolicySpec | None = None,
        checkpoint_sha256: str | None = None,
        telemetry: DecodeTelemetry | None = None,
        trace_sink: Callable[[Sequence[PredictionRequest], DecodedPlan], None] | None = None,
    ) -> None:
        self.executor = executor
        self.stats = stats
        self.ego_player_id = ego_player_id
        self.code_to_id = {code: FIRST_CONNECT_CODE_ID + index for index, code in enumerate(player_codes)}
        self.telemetry = telemetry
        self.trace_sink = trace_sink
        self.checkpoint_sha256 = checkpoint_sha256
        self._seed = seed
        self._rng = StreamGroupRng(seed, CONTROLLER_GROUP_NAMES)
        self._streams: dict[int, _WindowStream] = {}
        self._runtime: RuntimeConfig | None = None
        self._preparation_completed = False
        self._fully_prepared = False
        self._horizon = executor.prediction_frames
        self._prefix = 0
        self._last_fault_input: _FaultInput | None = None
        self.spec = spec or PolicySpec("059 official dense", "o59-history-decoder", REQUIRED_OBSERVATION_FIELDS, (0,))

    @property
    def sampling_seed(self) -> int:
        return self._seed

    @property
    def context_frames(self) -> int:
        return self.executor.context_frames

    @property
    def supported_horizons(self) -> tuple[int, ...]:
        return contiguous_horizons(self.executor.model.head_offsets)

    @property
    def prediction_horizon(self) -> int:
        return self._horizon

    @property
    def prepared_update_shapes(self) -> tuple[int, ...]:
        if not self._preparation_completed:
            raise RuntimeError("dense policy has not been prepared")
        return tuple(count for count in (1, 2, 4) if count <= self.context_frames)

    def prepare_prediction(
        self, runtime: RuntimeConfig, horizon: int, prefix_frames: int, *, prewarm_executor: bool = True
    ) -> None:
        self._preparation_completed = False
        self._fully_prepared = False
        delay = runtime.require_single_delay()
        if delay not in self.spec.supported_transport_delays or horizon != self.executor.prediction_frames:
            raise ValueError("dense 059 prediction delay or horizon is unsupported")
        if prefix_frames < 0 or prefix_frames >= horizon:
            raise ValueError("dense 059 fixed prefix is invalid")
        self._runtime = runtime
        self._horizon = horizon
        self._prefix = prefix_frames
        self.reset_prediction()
        if prewarm_executor:
            buckets = {self.executor._bucket(rows) for rows in range(1, runtime.max_batch_size + 1)}
            for bucket in sorted(buckets):
                self.executor.prewarm(bucket, horizon, committed_frames=prefix_frames)
        self._fully_prepared = prewarm_executor or not self.executor.compiled
        self._preparation_completed = True

    def validate_prepared_profile(self, profile: PreparedInferenceProfile) -> None:
        runtime = self._runtime
        if not self._fully_prepared or runtime is None or self.checkpoint_sha256 is None:
            raise ValueError("dense policy lacks prepared artifact identity")
        if (
            profile.checkpoint_sha256 != self.checkpoint_sha256
            or profile.execution_mode != "window"
            or profile.prediction_horizon_frames != self._horizon
            or profile.fixed_prefix_frames != self._prefix
            or profile.capacity != runtime.max_batch_size
        ):
            raise ValueError("prepared inference profile differs from the dense policy")
        if any(shape > self.context_frames for shape in profile.update_shapes):
            raise ValueError("prepared inference profile declares an unsupported observation update shape")

    def reset_prediction(self, *, seed: int | None = None) -> None:
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("sampling seed must be a nonnegative integer")
            self._seed = seed
        self._streams.clear()
        self._rng = StreamGroupRng(self._seed, CONTROLLER_GROUP_NAMES)
        self._last_fault_input = None

    def release_stream(self, stream_id: int) -> None:
        self._streams.pop(stream_id, None)
        self._rng.release(stream_id)

    def fault_snapshot(self) -> tuple[dict[str, object], dict[str, np.ndarray]]:
        """Return the last packed CPU input without touching the model or CUDA."""
        captured = self._last_fault_input
        if captured is None:
            return {}, {}
        return dict(captured.metadata), {
            "floats": captured.windows.floats,
            "cats": captured.windows.cats,
            "fixed_actions": captured.fixed_actions,
        }

    def _player_id(self, identity: str | None) -> int:
        if identity is None:
            return self.ego_player_id
        rank = Rank.__members__.get(identity)
        if rank is not None:
            return int(rank)
        try:
            return self.code_to_id[identity]
        except KeyError as error:
            raise ValueError(f"059 identity {identity!r} is absent from the checkpoint") from error

    @torch.no_grad()
    def predict(self, requests: Sequence[PredictionRequest]) -> Sequence[ActionPlan]:
        runtime = self._runtime
        if not self._preparation_completed or runtime is None:
            raise RuntimeError("dense 059 evaluation policy is not prepared")
        if not requests or len(requests) > runtime.max_batch_size:
            raise ValueError("invalid dense 059 prediction batch")
        if len({request.stream_id for request in requests}) != len(requests):
            raise ValueError("duplicate dense 059 stream")
        histories: list[ObservationHistory] = []
        resets: list[bool] = []
        for request in requests:
            validate_prediction_request(
                self.spec, runtime, request, context_frames=self.context_frames, prefix_frames=self._prefix
            )
            stream = self._streams.get(request.stream_id)
            if stream is not None and request.generation < stream.generation:
                raise ValueError("obsolete dense 059 stream generation")
            if stream is not None and request.generation == stream.generation:
                if request.sequence <= stream.sequence or request.observations[0].frame_id != stream.last_frame + 1:
                    raise ValueError("obsolete or noncontiguous dense 059 request")
            else:
                stream = None
            port = request.observations[0].controlled_port
            player_id = self._player_id(request.observations[-1].player_identity)
            if stream is not None and stream.port != port:
                raise ValueError("dense 059 controlled port changed without reset")
            if stream is not None and stream.player_id != player_id:
                raise ValueError("dense 059 player identity changed without reset")
            for index, item in enumerate(request.observations):
                if stream is not None and item.reset:
                    raise ValueError("dense 059 reset requires a new generation")
                relative = _relative_observation(request, port, index)
                if stream is None:
                    history = ObservationHistory.from_frame(
                        relative, "p1", self.stats, self.context_frames, ITEM_COLUMNS, BASE_ITEMS_PROJECTION
                    )
                    stream = _WindowStream(history, request.generation, -1, item.frame_id, port, player_id, True)
                history = stream.history
                history.gather(relative, controller_to_action_vec(item.applied_action, dtype=np.float32))
                history.push()
                stream.last_frame = item.frame_id
            assert stream is not None
            stream.sequence = request.sequence
            self._streams[request.stream_id] = stream
            histories.append(stream.history)
            resets.append(stream.reset_pending)
        device = next(self.executor.model.parameters()).device
        windows = stack_observation_windows(histories, self.context_frames)
        pad_values = [self.context_frames - history.count for history in histories]
        player_ids = [self._streams[request.stream_id].player_id for request in requests]
        committed = np.array(
            [
                [controller_to_action_vec(action, dtype=np.float32) for action in request.fixed_actions]
                for request in requests
            ],
            dtype=np.float32,
        ).reshape(len(requests), self._prefix, len(ACTION_CHANNELS))
        stream_ids = [request.stream_id for request in requests]
        sampling_generations = [request.generation - 1 for request in requests]
        desired_returns = [request.observations[-1].desired_return for request in requests]
        temperatures = [request.observations[-1].temperature for request in requests]
        layout = windows.layout
        self._last_fault_input = _FaultInput(
            windows,
            committed,
            {
                "schema_version": 2,
                "stream_ids": stream_ids,
                "generations": [request.generation for request in requests],
                "sequences": [request.sequence for request in requests],
                "source_frames": [request.source_frame for request in requests],
                "reset": resets,
                "controlled_ports": [request.observations[-1].controlled_port for request in requests],
                "player_id": player_ids,
                "player_identity": [request.observations[-1].player_identity for request in requests],
                "desired_return": desired_returns,
                "temperature": temperatures,
                "ctx_pad": pad_values,
                "fixed_prefix_frames": self._prefix,
                "horizon": self._horizon,
                "sampling_seed": self._seed,
                "sampling_generations": [
                    [stream_id, generation] for stream_id, generation in self._rng.generations.items()
                ],
                "sampling_counters": [list(item) for item in self._rng.state()],
                "checkpoint_sha256": self.checkpoint_sha256,
                "value_names": list(layout.value_names),
                "mask_names": list(layout.mask_names),
                "cat_names": list(layout.cat_names),
                "emitted_masks": windows.emitted.tolist(),
            },
        )
        features = windows.features(device, next(self.executor.model.parameters()).dtype)
        pads = torch.tensor(pad_values, device=device)
        context = condition_ego_players(Context(features=features, ctx_pad=pads), player_ids)
        started = time.perf_counter()
        committed_tensor = torch.from_numpy(committed).to(device)
        if self.trace_sink is None:
            actions, state_values = self.executor.decode_prediction(
                context,
                self._horizon,
                streams=self._rng,
                stream_ids=stream_ids,
                sampling_generations=sampling_generations,
                desired_returns=desired_returns,
                temperatures=temperatures,
                committed=committed_tensor,
            )
        else:
            decoded = self.executor.decode_with_trace(
                context,
                self._horizon,
                streams=self._rng,
                stream_ids=stream_ids,
                sampling_generations=sampling_generations,
                desired_returns=desired_returns,
                temperatures=temperatures,
                committed=committed_tensor,
            )
            self.trace_sink(requests, decoded)
            actions, state_values = decoded.actions, decoded.state_values
        if self.telemetry is not None:
            self.telemetry.record(rows=len(requests), horizon=self._horizon, seconds=time.perf_counter() - started)
        plans: list[ActionPlan] = []
        delivered = (
            torch.cat((actions[:, self._prefix :].float().flatten(1), state_values[:, None]), dim=1).cpu().numpy()
        )
        for request, row in zip(requests, delivered, strict=True):
            tail = tuple(action_vec_to_controller(action) for action in row[:-1].reshape(-1, len(ACTION_CHANNELS)))
            plans.append(action_plan(request, tail, state_value=float(row[-1])))
            self._streams[request.stream_id].reset_pending = False
        return plans
