"""Portable O59 checkpoint export and one-slot live inference."""

from __future__ import annotations

import hashlib
import json
import math
import secrets
import tempfile
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import fields
from dataclasses import replace
from pathlib import Path
from typing import Any
from typing import Final
from typing import Literal
from typing import cast

import numpy as np
import torch
from torch import Tensor

from hal import streams
from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.controller import POLICY_BUTTON_MASK
from hal.controller import ControllerAction
from hal.data.feature_stats import FeatureStats
from hal.data.schema import Rank
from hal.eval.policy_sampling import SlotGroupRng
from hal.inference.api import PolicyInput
from hal.inference.api import PolicyOutput
from hal.inference.api import PolicySpec
from hal.inference.api import RuntimeConfig
from hal.inference.api import validate_policy_inputs
from hal.inference.bundle import BundleDescription
from hal.inference.bundle import PolicyBundleManifest
from hal.inference.bundle import extract_policy_bundle
from hal.inference.bundle import write_policy_bundle
from hal.inference.checkpoints import resolve_checkpoint
from hal.inference.chunks import ChunkRequest
from hal.inference.chunks import ChunkResponse
from hal.inference.chunks import chunk_response
from hal.inference.chunks import contiguous_horizons
from hal.inference.chunks import validate_chunk_request
from hal.inference.cuda_graph import CapturedCall
from hal.inference.gpu_history import GpuContextHistory
from hal.inference.gpu_history import GpuTokenHistory
from hal.inference.kv_cache import KVCache
from hal.inference.kv_cache import KVMemory
from hal.inference.kv_cache import forward_with_kv_cache
from hal.inference.o59_model import GPT
from hal.inference.o59_model import Architecture
from hal.inference.o59_model import AWRCalibration
from hal.inference.o59_model import ReturnCalibration
from hal.inference.o59_model import TrainConfig
from hal.training.context_history import ContextHistory
from hal.training.controller_codec import CONTROLLER_GROUP_NAMES
from hal.training.ego_stats import load_consolidated_mixture_stats
from hal.training.features import ACTION_CHANNELS
from hal.training.features import BASE_ITEMS_PROJECTION
from hal.training.features import ITEM_COLUMNS
from hal.training.features import feature_kind
from hal.training.player_identity import FIRST_CONNECT_CODE_ID
from hal.training.player_identity import MASKED_PLAYER_ID
from hal.training.player_identity import PlayerVocabulary
from hal.training.player_identity import decode_player_codes
from hal.training.player_identity import encode_player_codes
from hal.wire import BUTTON_BITS

O59_BACKEND: Final[str] = "o59-history-decoder"
O59_BACKEND_VERSION: Final[int] = 1
_CONFIG_MEMBER: Final[str] = "backend.json"
_CHECKPOINT_MEMBER: Final[str] = "checkpoint.pt"
_STATS_MEMBER: Final[str] = "stats.json"
_ACTION_FIELDS: Final[frozenset[str]] = frozenset(f"ego_{name}" for name in ACTION_CHANNELS)
_MODEL_FIELDS: Final[frozenset[str]] = BASE_ITEMS_PROJECTION.columns - _ACTION_FIELDS


def _canonical_field(name: str) -> str:
    if name.startswith("ego_"):
        return f"p1_{name[4:]}"
    if name.startswith("opp_"):
        return f"p2_{name[4:]}"
    return name


O59_REQUIRED_OBSERVATION_FIELDS: Final[tuple[str, ...]] = tuple(sorted(map(_canonical_field, _MODEL_FIELDS)))
_ROUTES: Final[dict[int, tuple[tuple[str, str], ...]]] = {
    port: tuple(
        (name, f"{'ego' if (name.startswith('p1_') == (port == 1)) else 'opp'}_{name[3:]}")
        if name.startswith(("p1_", "p2_"))
        else (name, name)
        for name in O59_REQUIRED_OBSERVATION_FIELDS
    )
    for port in (1, 2)
}
for _routes in _ROUTES.values():
    if {relative for _, relative in _routes} != _MODEL_FIELDS:
        raise RuntimeError("O59 observation routing differs from its projection")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_from_state(values: object) -> TrainConfig:
    if not isinstance(values, dict):
        raise ValueError("O59 checkpoint config must be an object")
    values = cast(dict[str, object], values)
    expected = {item.name for item in fields(TrainConfig)} - {"arch", "awr"}
    if set(values) != {
        "experiment_id",
        "checkpoint_format_version",
        "architecture",
        "awr_calibration",
        "max_steps",
        "warmup_steps",
        *expected,
    }:
        raise ValueError("O59 checkpoint config fields changed")
    if (values["experiment_id"], values["checkpoint_format_version"]) != ("059_muon_history_decoder_v5", 4):
        raise ValueError("O59 checkpoint identity changed")
    architecture = values["architecture"]
    calibration = values["awr_calibration"]
    if not isinstance(architecture, dict) or set(architecture) != {item.name for item in fields(Architecture)}:
        raise ValueError("O59 architecture fields changed")
    if not isinstance(calibration, dict) or set(calibration) != {item.name for item in fields(AWRCalibration)}:
        raise ValueError("O59 AWR fields changed")
    cfg = TrainConfig(
        arch=Architecture(**cast(Any, architecture)),
        awr=AWRCalibration(**cast(Any, calibration)),
        **cast(Any, {name: values[name] for name in expected}),
    )
    if values["max_steps"] != cfg.max_steps or values["warmup_steps"] != cfg.warmup_steps:
        raise ValueError("O59 checkpoint schedule changed")
    if cfg.arch != Architecture() or cfg.awr != AWRCalibration():
        raise ValueError("O59 checkpoint architecture or calibration is unsupported")
    if (cfg.prediction_frames, cfg.delay_frames, cfg.replan_interval_frames) != (4, 2, 2):
        raise ValueError("O59 checkpoint timing is unsupported")
    if cfg.source_names != tuple(source.name for source in streams.POLICY_WORLD_V8_SOURCES):
        raise ValueError("O59 checkpoint source list changed")
    if cfg.mds_schema_version != 7 or not cfg.return_conditioning:
        raise ValueError("O59 checkpoint data or return conditioning changed")
    return cfg


def _checkpoint(path: Path) -> tuple[dict[str, object], TrainConfig, tuple[str, ...], float]:
    raw = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if not isinstance(raw, dict):
        raise ValueError("O59 checkpoint root must be an object")
    cfg = _config_from_state(raw.get("cfg"))
    model_state = raw.get("model")
    if not isinstance(model_state, dict):
        raise ValueError("O59 checkpoint has no model state")
    encoded = model_state.get("player_code_bytes")
    if not isinstance(encoded, Tensor) or encoded.dtype != torch.uint8 or encoded.ndim != 1:
        raise ValueError("O59 checkpoint player codes are invalid")
    codes = decode_player_codes(encoded.numpy().tobytes())
    if encode_player_codes(codes) != encoded.numpy().tobytes():
        raise ValueError("O59 player codes are not canonical")
    vocabulary = PlayerVocabulary(codes)
    if vocabulary.size != cfg.player_vocab_size or vocabulary.sha256 != cfg.player_vocab_sha256:
        raise ValueError("O59 player vocabulary does not match its checkpoint")
    protocol = {
        "version": 1,
        "horizon": 60,
        "gamma": 0.99855,
        "reward": "damage_opp-damage_ego+120*(stock_loss_opp-stock_loss_ego)+50*(last_stock_opp-last_stock_ego)",
        "scale": 120.0,
        "alignment": "sum(k=1..60, gamma**(k-1)*r[t+k])",
        "availability": "full observed horizon or known terminal with zero rewards thereafter",
        "evaluation": "positive p90 at every replan; separate unconditioned comparison",
        "modulation": "per-block RMSNorm affine, 128-wide SiLU, biased zero-initialized projections",
        "dropout": "independent CPU generator, per context position",
        "calibration_windows": 65_536,
    }
    if raw.get("conditioning_protocol") != protocol:
        raise ValueError("O59 checkpoint conditioning protocol changed")
    calibration = raw.get("return_calibration")
    if not isinstance(calibration, dict):
        raise ValueError("O59 checkpoint return calibration is missing")
    calibrator = ReturnCalibration()
    calibrator.load_state_dict(cast(dict[str, object], calibration))
    targets = calibrator.targets()
    if not isinstance(targets, tuple) or len(targets) != 3:
        raise ValueError("O59 checkpoint return targets are missing")
    p90 = targets[2]
    if not isinstance(p90, float) or not math.isfinite(p90):
        raise ValueError("O59 checkpoint return p90 is invalid")
    masker = raw.get("return_masker")
    if not isinstance(masker, dict) or set(masker) != {"version", "probability", "enabled", "generator"}:
        raise ValueError("O59 checkpoint return masker is missing")
    if (masker["version"], masker["probability"], masker["enabled"]) != (1, cfg.return_dropout, True):
        raise ValueError("O59 checkpoint return masker changed")
    generator_state = masker["generator"]
    if not isinstance(generator_state, Tensor):
        raise ValueError("O59 checkpoint return masker RNG is invalid")
    torch.Generator(device="cpu").set_state(generator_state.cpu())
    with torch.device("meta"):
        model = GPT(cfg, vocabulary)
    model.load_state_dict(cast(dict[str, Tensor], model_state), strict=True, assign=True)
    if raw.get("wandb_id") != "vywk3cih":
        raise ValueError("O59 checkpoint is not W&B run vywk3cih")
    return raw, cfg, codes, p90


def export_o59_policy(checkpoint_source: str | Path, destination: str | Path) -> PolicyBundleManifest:
    checkpoint = resolve_checkpoint(str(checkpoint_source))
    raw, cfg, _codes, p90 = _checkpoint(checkpoint)
    source_names = cfg.source_names
    sources = tuple(streams.BY_NAME[name] for name in source_names)
    paths = tuple(streams.ensure_stats(source.local_root / "stats.json") for source in sources)
    stats = load_consolidated_mixture_stats(
        paths,
        tuple(float(streams.POLICY_WORLD_V8_TRAIN_REPLAYS[name]) for name in source_names),
        expected_mds_schema_version=cfg.mds_schema_version,
    )
    stats_payload = {
        name: {"mean": item.mean, "std": item.std, "min": item.min, "max": item.max}
        for name, item in sorted(stats.items())
    }
    stats_bytes = json.dumps(stats_payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    config = {
        "schema_version": 1,
        "checkpoint_sha256": _sha256(checkpoint),
        "wandb_id": raw["wandb_id"],
        "step": raw["step"],
        "return_p90": p90,
        "stats_sha256": hashlib.sha256(stats_bytes).hexdigest(),
        "source_stats_sha256": {name: _sha256(path) for name, path in zip(source_names, paths, strict=True)},
    }
    with tempfile.TemporaryDirectory(prefix="hal-o59-export-") as temporary:
        root = Path(temporary)
        config_path = root / _CONFIG_MEMBER
        stats_path = root / _STATS_MEMBER
        config_path.write_text(json.dumps(config, sort_keys=True, separators=(",", ":")))
        stats_path.write_bytes(stats_bytes)
        return write_policy_bundle(
            destination,
            BundleDescription(
                policy_name="O59 cooldown32768",
                backend=O59_BACKEND,
                backend_version=O59_BACKEND_VERSION,
                required_observation_fields=O59_REQUIRED_OBSERVATION_FIELDS,
                supported_transport_delays=(2,),
                requires_player_code=False,
                source_sha256=config["checkpoint_sha256"],
            ),
            {_CONFIG_MEMBER: config_path, _STATS_MEMBER: stats_path, _CHECKPOINT_MEMBER: checkpoint},
        )


def _action_vector(action: ControllerAction) -> np.ndarray:
    if action.buttons & ~POLICY_BUTTON_MASK:
        raise ValueError("O59 controller action has unsupported buttons")
    buttons = [float(bool(action.buttons & BUTTON_BITS[name.removeprefix("button_")])) for name in ACTION_CHANNELS[6:]]
    return np.asarray(
        (action.main_x, action.main_y, action.c_x, action.c_y, action.trigger_l, action.trigger_r, *buttons),
        dtype=np.float32,
    )


def _controller_action(values: np.ndarray) -> ControllerAction:
    buttons = 0
    for value, name in zip(values[6:], ACTION_CHANNELS[6:], strict=True):
        if value > 0.5:
            buttons |= BUTTON_BITS[name.removeprefix("button_")]
    return ControllerAction(
        main_x=float(np.clip(values[0], -1, 1)),
        main_y=float(np.clip(values[1], -1, 1)),
        c_x=float(np.clip(values[2], -1, 1)),
        c_y=float(np.clip(values[3], -1, 1)),
        trigger_l=float(np.clip(values[4], 0, 1)),
        trigger_r=float(np.clip(values[5], 0, 1)),
        buttons=buttons,
    )


@dataclass(slots=True)
class _Stream:
    history: ContextHistory
    gpu: GpuContextHistory | GpuTokenHistory
    port: int
    player_id: int
    last_frame: int
    reset_pending: bool
    queued: deque[ControllerAction]
    cache: KVCache | None = None
    pending_frames: int = 0


class O59Policy:
    """Saved 4/2/2 synchronous behavior and configurable real-time chunks."""

    _prediction_frames = 4
    _prefix_frames = 2
    history_mode: Literal["window", "kv_cache"] = "window"

    def __init__(
        self,
        model: GPT,
        cfg: TrainConfig,
        stats: dict[str, FeatureStats],
        codes: tuple[str, ...],
        *,
        device: torch.device,
        seed: int | None,
        compiled: bool,
        history_mode: Literal["window", "kv_cache"] = "window",
        kv_update_frames: int = 2,
        kv_cuda_graphs: bool = True,
    ) -> None:
        self.model = model
        if history_mode not in ("window", "kv_cache") or kv_update_frames not in (1, 2):
            raise ValueError("history must be window or kv_cache, with one or two frames per update")
        self.history_mode = history_mode
        self.kv_update_frames = kv_update_frames
        self.kv_cuda_graphs = kv_cuda_graphs and compiled and device.type == "cuda"
        self.cfg = cfg
        self.stats = stats
        self.device = device
        self.code_to_id = {code: FIRST_CONNECT_CODE_ID + i for i, code in enumerate(codes)}
        self.spec = PolicySpec("O59 cooldown32768", O59_BACKEND, O59_REQUIRED_OBSERVATION_FIELDS, (2,))
        self._compiled = compiled and device.type == "cuda"
        self._seed = secrets.randbits(64) if seed is None else seed
        self._rng = SlotGroupRng(self._seed, CONTROLLER_GROUP_NAMES)
        self._runtime: RuntimeConfig | None = None
        self._stream: _Stream | None = None
        self._trunk = model.forward_dense
        self._decoder = self._sample
        self._kv_trunk = self._advance
        self._caches: dict[int, KVCache] = {}
        self._kv_decoder = self._sample_with_kv_cache
        self.decode_seconds: list[float] = []
        self._chunk_streams: dict[int, _Stream] = {}
        self._chunk_generations: dict[int, int] = {}
        self._chunk_mode = False

    def _player_id(self, identity: str | None) -> int:
        if identity is None:
            return MASKED_PLAYER_ID
        rank = Rank.__members__.get(identity)
        if rank is not None:
            return int(rank)
        try:
            return self.code_to_id[identity]
        except KeyError as error:
            raise ValueError(f"O59 identity {identity!r} is absent from the checkpoint") from error

    def _sample(
        self,
        hidden: Tensor,
        pad: Tensor,
        observed: Tensor,
        uniforms: Tensor,
        forced: Tensor,
        return_value: Tensor,
        condition_present: Tensor,
        temperature: Tensor,
    ) -> Tensor:
        return self.model.temporal.sample_indices(
            hidden,
            observed,
            self.model.head_offsets[: self._prediction_frames],
            return_value,
            condition_present,
            argmax=False,
            uniforms=uniforms,
            temperature=temperature,
            ctx_pad=pad,
            forced_prefix=forced,
        )

    def prepare(self, config: RuntimeConfig) -> None:
        if self._runtime is not None:
            raise RuntimeError("O59 policy is already prepared")
        if not self._chunk_mode and (
            config.max_batch_size != 1
            or config.transport_delays != (2,)
            or config.replan_interval_frames not in (None, 2)
        ):
            raise ValueError("O59 serving requires one slot, delay 2, replan 2")
        self._runtime = config
        if self.history_mode == "kv_cache":
            self._prepare_kv_cache()
            return
        if self._compiled:
            self._trunk = torch.compile(self.model.forward_dense, dynamic=False, fullgraph=True, mode="default")
            self._decoder = torch.compile(self._sample, dynamic=False, mode="default")
        dummy = {
            name: (0 if feature_kind(relative, ITEM_COLUMNS) in ("cat", "button") else 0.0)
            for name, relative in _ROUTES[1]
        }
        relative = {relative: dummy[name] for name, relative in _ROUTES[1]}
        history = ContextHistory.from_frame(
            relative, "p1", self.stats, self.cfg.arch.L_ctx, ITEM_COLUMNS, BASE_ITEMS_PROJECTION
        )
        gpu = GpuContextHistory(history, self.model.codec, self.device)
        neutral_action = np.zeros(len(ACTION_CHANNELS), dtype=np.float32)
        for _ in range(self.cfg.arch.L_ctx):
            history.gather(relative, neutral_action)
            history.push(None)
            gpu.push(neutral_action)
        with torch.inference_mode():
            context = gpu.context(MASKED_PLAYER_ID, 0, True)
            observed = gpu.action_indices()
            forced = self.model.codec.quantize(
                torch.from_numpy(np.zeros((self._prefix_frames, len(ACTION_CHANNELS)), dtype=np.float32))
                .to(self.device)
                .unsqueeze(0)
            )
            uniforms = torch.full((self._prediction_frames, len(CONTROLLER_GROUP_NAMES), 1), 0.5, device=self.device)
            value = torch.tensor([20.0], device=self.device)
            present = torch.tensor([True], device=self.device)
            temperature = torch.tensor(1.0, device=self.device)
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                for _ in range(2):
                    hidden = self._trunk(context.features, context.ctx_pad, observed)
                    self._decoder(
                        hidden, context.ctx_pad, observed[:, -1], uniforms, forced, value, present, temperature
                    )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        if self._compiled and not self._chunk_mode:
            warm = PolicyInput(
                stream_id=0,
                frame_id=0,
                controlled_port=1,
                observation=dummy,
                applied_action=NEUTRAL_CONTROLLER_ACTION,
                pending_actions=(NEUTRAL_CONTROLLER_ACTION,) * 2,
                reset=True,
            )
            self.step((warm,))
            self._stream = None
            self._rng = SlotGroupRng(self._seed, CONTROLLER_GROUP_NAMES)
            self.decode_seconds.clear()

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
    ) -> Tensor:
        return self.model.temporal.sample_indices(
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

    def _prepare_kv_cache(self) -> None:
        self._caches.clear()
        if self.device.type == "cuda":
            # Autocast uses these same rounded weights; retain them between calls.
            for module in self.model.modules():
                if isinstance(module, torch.nn.Linear):
                    module.to(dtype=torch.bfloat16)
        if self._compiled:
            self._kv_trunk = torch.compile(self._advance, dynamic=False, fullgraph=True)
            self._kv_decoder = torch.compile(self._sample_with_kv_cache, dynamic=False, fullgraph=True)
        with torch.inference_mode():
            assert self._runtime is not None
            for slot in range(self._runtime.max_batch_size):
                self._stream = None
                for item in self.warmup_context(slot, self.context_frames - 1, 2):
                    stream = self._ingest(item)
                    if not stream.queued:
                        self._plan(
                            replace(item, pending_actions=(NEUTRAL_CONTROLLER_ACTION,) * self._prefix_frames), stream
                        )
                    stream.queued.popleft()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.reset_chunks()

    def _advance_stream(self, stream: _Stream) -> None:
        if stream.cache is None or not stream.pending_frames:
            return
        context = stream.gpu.context(stream.player_id, 0, False)
        count = stream.pending_frames
        features = {name: value[:, -count:] for name, value in context.features.items()}
        observed = stream.gpu.action_indices()[:, -count:]
        with (
            torch.inference_mode(),
            torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"),
        ):
            if self.kv_cuda_graphs:
                assert isinstance(stream.gpu, GpuTokenHistory)
                inputs = (stream.gpu.floats, stream.gpu.cats, observed, stream.gpu.player)
                call = stream.cache.updates.get(count)
                if call is None:
                    static = tuple(value.clone() for value in inputs)
                    static_features = {
                        name: value[:, -count:]
                        for name, value in stream.gpu.features_from(static[0], static[1], static[3]).items()
                    }
                    cache = stream.cache
                    call = CapturedCall(
                        lambda: self._kv_trunk(static_features, static[2], cache), static, cache.buffers()
                    )
                    cache.updates[count] = call
                call(inputs)
            else:
                self._kv_trunk(features, observed, stream.cache)
        stream.pending_frames = 0

    def _ingest(self, item: PolicyInput) -> _Stream:
        player_id = self._player_id(item.player_identity)
        stream = self._stream
        if item.reset or stream is None or item.frame_id != stream.last_frame + 1:
            stream = None
        if stream is not None and (stream.port != item.controlled_port or stream.player_id != player_id):
            raise ValueError("O59 port or identity changed without reset")
        flat = {
            relative: int(item.observation[name])
            if isinstance(item.observation[name], (int, np.integer))
            else float(item.observation[name])
            for name, relative in _ROUTES[item.controlled_port]
        }
        if stream is None:
            history = ContextHistory.from_frame(
                flat, "p1", self.stats, self.cfg.arch.L_ctx, ITEM_COLUMNS, BASE_ITEMS_PROJECTION
            )
            stream = _Stream(
                history,
                GpuTokenHistory(history, self.model.codec, self.device)
                if self.history_mode == "kv_cache"
                else GpuContextHistory(history, self.model.codec, self.device),
                item.controlled_port,
                player_id,
                item.frame_id,
                True,
                deque(),
            )
            self._stream = stream
            if self.history_mode == "kv_cache":
                cache = self._caches.get(item.stream_id)
                if cache is None:
                    cache = KVCache(self.model, self.kv_update_frames, self.device)
                    self._caches[item.stream_id] = cache
                cache.reset()
                stream.cache = cache
        action = _action_vector(item.applied_action)
        stream.history.gather(flat, action)
        stream.history.push(None)
        stream.gpu.push(action)
        if stream.cache is not None:
            stream.pending_frames += 1
            if stream.pending_frames == self.kv_update_frames:
                self._advance_stream(stream)
        stream.last_frame = item.frame_id
        return stream

    @torch.inference_mode()
    def _plan(self, item: PolicyInput, stream: _Stream) -> None:
        context = stream.gpu.context(stream.player_id, item.stream_id, stream.reset_pending)
        observed = stream.gpu.action_indices()
        committed = np.stack([_action_vector(action) for action in item.pending_actions])
        forced = self.model.codec.quantize(torch.from_numpy(committed).to(self.device).unsqueeze(0))
        self._rng.begin(context)
        draws = []
        for depth in range(self._prediction_frames):
            draws.append(
                torch.stack(
                    [self._rng.uniforms(name, [depth >= self._prefix_frames]) for name in CONTROLLER_GROUP_NAMES]
                )
            )
        uniforms = torch.stack(draws).to(self.device)
        target = item.desired_return
        return_value = torch.tensor([0.0 if target is None else target], device=self.device)
        condition_present = torch.tensor([target is not None], device=self.device)
        temperature = torch.tensor(item.temperature, device=self.device)
        started = time.perf_counter()
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            if self.history_mode == "kv_cache":
                self._advance_stream(stream)
                assert stream.cache is not None
                decode_inputs = (observed[:, -1], uniforms, forced, return_value, condition_present, temperature)
                cache = stream.cache
                if self.kv_cuda_graphs:
                    if cache.decoder is None:
                        static = tuple(value.clone() for value in decode_inputs)
                        cache.decoder = CapturedCall(
                            lambda: self._kv_decoder(cache.hidden, cache.memory(), *static), static
                        )
                    indices = cache.decoder(decode_inputs)
                else:
                    indices = self._kv_decoder(cache.hidden, cache.memory(), *decode_inputs)
            else:
                hidden = self._trunk(context.features, context.ctx_pad, observed)
                indices = self._decoder(
                    hidden,
                    context.ctx_pad,
                    observed[:, -1],
                    uniforms,
                    forced,
                    return_value,
                    condition_present,
                    temperature,
                )
            planned = (
                self.model.codec.dequantize(indices)[0, self._prefix_frames : self._prediction_frames]
                .float()
                .cpu()
                .numpy()
            )
        self.decode_seconds.append(time.perf_counter() - started)
        stream.queued.extend(_controller_action(row) for row in planned)
        stream.reset_pending = False

    @property
    def sampling_seed(self) -> int:
        return self._seed

    @property
    def context_frames(self) -> int:
        return self.cfg.arch.L_ctx

    @property
    def supported_horizons(self) -> tuple[int, ...]:
        return contiguous_horizons(self.model.head_offsets)

    def reset_chunks(self, *, seed: int | None = None) -> None:
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("sampling seed must be a nonnegative integer")
            self._seed = seed
        self._stream = None
        self._chunk_streams.clear()
        self._chunk_generations.clear()
        self._rng = SlotGroupRng(self._seed, CONTROLLER_GROUP_NAMES)
        self.decode_seconds.clear()

    def prepare_chunks(self, runtime: RuntimeConfig, horizon: int, prefix_frames: int) -> None:
        if horizon not in self.supported_horizons or not 0 <= prefix_frames < horizon:
            raise ValueError("O59 chunk shape requires contiguous trained heads and an unforced tail")
        if set(runtime.transport_delays) - set(self.spec.supported_transport_delays):
            raise ValueError("O59 chunk transport delay is unsupported")
        self.model.temporal.configure_live_horizons((horizon,))
        self._chunk_mode = True
        self._prediction_frames = horizon
        self._prefix_frames = prefix_frames
        self._runtime = None
        self.reset_chunks()
        self.prepare(runtime)

    def warmup_context(self, stream_id: int, source_frame: int, transport: int) -> tuple[PolicyInput, ...]:
        observation = {
            name: (0 if feature_kind(relative, ITEM_COLUMNS) in ("cat", "button") else 0.0)
            for name, relative in _ROUTES[1]
        }
        return tuple(
            PolicyInput(
                stream_id,
                frame,
                1,
                observation,
                NEUTRAL_CONTROLLER_ACTION,
                (NEUTRAL_CONTROLLER_ACTION,) * transport,
                reset=frame == source_frame - self.context_frames + 1,
            )
            for frame in range(source_frame - self.context_frames + 1, source_frame + 1)
        )

    @torch.inference_mode()
    def plan_chunks(self, requests: Sequence[ChunkRequest]) -> Sequence[ChunkResponse]:
        if self._runtime is None or not self._chunk_mode:
            raise RuntimeError("O59 chunk policy is not prepared")
        if not requests or len(requests) > self._runtime.max_batch_size:
            raise ValueError("invalid O59 chunk batch size")
        if len({request.stream_id for request in requests}) != len(requests):
            raise ValueError("duplicate O59 chunk stream")
        responses = []
        for request in requests:
            validate_chunk_request(
                self.spec,
                self._runtime,
                request,
                context_frames=self.context_frames,
                prefix_frames=self._prefix_frames,
            )
            stream = self._chunk_streams.get(request.stream_id)
            if self._chunk_generations.get(request.stream_id) != request.generation:
                stream = None
            self._stream = stream
            for item in request.context:
                if stream is None or item.frame_id > stream.last_frame:
                    stream = self._ingest(replace(item, reset=stream is None))
            if stream is None or stream.last_frame != request.source_frame:
                raise ValueError("obsolete O59 chunk request")
            stream.queued.clear()
            self._plan(replace(request.context[-1], pending_actions=request.forced_prefix), stream)
            responses.append(chunk_response(request, tuple(stream.queued)))
            stream.queued.clear()
            self._chunk_streams[request.stream_id] = stream
            self._chunk_generations[request.stream_id] = request.generation
        return tuple(responses)

    def step(self, inputs: Sequence[PolicyInput]) -> Sequence[PolicyOutput]:
        if self._runtime is None:
            raise RuntimeError("O59 policy is not prepared")
        validate_policy_inputs(self.spec, self._runtime, inputs)
        return self.step_prevalidated(inputs)

    @torch.inference_mode()
    def step_prevalidated(self, inputs: Sequence[PolicyInput]) -> Sequence[PolicyOutput]:
        if len(inputs) != 1:
            raise ValueError("O59 policy requires one input")
        item = inputs[0]
        stream = self._ingest(item)
        if not stream.queued:
            self._plan(item, stream)
        return (PolicyOutput(item.stream_id, stream.queued.popleft()),)


def load_o59_policy(
    path: str | Path,
    *,
    device: str,
    seed: int | None,
    compiled: bool,
    history_mode: Literal["window", "kv_cache"] = "window",
    kv_update_frames: int = 2,
    kv_cuda_graphs: bool = True,
) -> O59Policy:
    with extract_policy_bundle(path) as (manifest, root):
        if (
            manifest.backend != O59_BACKEND
            or manifest.backend_version != O59_BACKEND_VERSION
            or manifest.required_observation_fields != O59_REQUIRED_OBSERVATION_FIELDS
            or manifest.supported_transport_delays != (2,)
            or set(member.name for member in manifest.members) != {_CONFIG_MEMBER, _STATS_MEMBER, _CHECKPOINT_MEMBER}
        ):
            raise ValueError("O59 bundle manifest is incompatible")
        config = json.loads((root / _CONFIG_MEMBER).read_text())
        if set(config) != {
            "schema_version",
            "checkpoint_sha256",
            "wandb_id",
            "step",
            "return_p90",
            "stats_sha256",
            "source_stats_sha256",
        }:
            raise ValueError("O59 bundle config fields changed")
        if config["schema_version"] != 1 or config["checkpoint_sha256"] != manifest.source_sha256:
            raise ValueError("O59 bundle config identity changed")
        checkpoint = root / _CHECKPOINT_MEMBER
        if _sha256(checkpoint) != config["checkpoint_sha256"]:
            raise ValueError("O59 checkpoint hash differs from bundle config")
        stats_bytes = (root / _STATS_MEMBER).read_bytes()
        if hashlib.sha256(stats_bytes).hexdigest() != config["stats_sha256"]:
            raise ValueError("O59 stats hash differs from bundle config")
        raw_stats = json.loads(stats_bytes)
        if not isinstance(raw_stats, dict):
            raise ValueError("O59 stats must be an object")
        stats = {name: FeatureStats(**values) for name, values in raw_stats.items()}
        raw, cfg, codes, p90 = _checkpoint(checkpoint)
        if raw["step"] != config["step"] or p90 != config["return_p90"]:
            raise ValueError("O59 checkpoint state differs from bundle config")
        vocabulary = PlayerVocabulary(codes)
        with torch.device("meta"):
            model = GPT(cfg, vocabulary)
        model.load_state_dict(cast(dict[str, Tensor], raw["model"]), strict=True, assign=True)
        model = model.to(device).eval()
    return O59Policy(
        model,
        cfg,
        stats,
        codes,
        device=torch.device(device),
        seed=seed,
        compiled=compiled,
        history_mode=history_mode,
        kv_update_frames=kv_update_frames,
        kv_cuda_graphs=kv_cuda_graphs,
    )
