"""Sample and replay controller chunks with FP32 probability arithmetic."""

from collections.abc import Callable
from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor

from hal.inference.cuda_graph import CapturedCall
from hal.models.action_sequence import DecisionOutput
from hal.models.attention import KVMemory
from hal.models.controller_codec import BUTTONS_GROUP
from hal.models.controller_codec import TRIGGERS_GROUP
from hal.models.self_play import SelfPlayTransformer
from hal.models.self_play import validate_executed_offsets


class DecisionDecoder:
    def __init__(
        self,
        model: SelfPlayTransformer,
        configurations: tuple[tuple[int, int], ...],
        *,
        compiled: bool,
    ) -> None:
        validate_executed_offsets(model.head_offsets, configurations)
        for delay, replan in configurations:
            if (
                not model.ranges.delay[0] <= delay <= model.ranges.delay[1]
                or not model.ranges.replan[0] <= replan <= model.ranges.replan[1]
            ):
                raise ValueError("decision configuration is outside the conditioner ranges")
        self.model = model
        self.compiled = compiled
        if any(parameter.dtype != torch.float32 for parameter in model.temporal.parameters()):
            raise ValueError("O65 action decoding requires FP32 parameters")
        self.graphs: dict[tuple[int, int, bool, int], CapturedCall[DecisionOutput]] = {}
        self.group_sizes = torch.tensor(model.codec.group_vocabs, device=next(model.parameters()).device)
        self.configurations = configurations
        self.max_delay = max(delay for delay, _ in configurations)
        if self.max_delay + 4 > model.cfg.sample_chunk_length:
            raise ValueError("padded decode exceeds the checkpoint's offset embeddings")
        self.variants: dict[int, Callable[..., DecisionOutput]] = {}
        self.replay_variants: dict[int, Callable[..., DecisionOutput]] = {}
        for count in (2, 4):
            decode = partial(
                model.temporal.decode_decisions,
                max_delay=self.max_delay,
                sampled_offsets=count,
                forced_prefix_length=min(delay for delay, _ in configurations),
            )
            self.variants[count] = (
                torch.compile(decode, fullgraph=True, dynamic=False)
                if compiled and next(model.parameters()).device.type == "cpu"
                else decode
            )
            replay = partial(model.temporal.replay_decisions, max_delay=self.max_delay, sampled_offsets=count)
            self.replay_variants[count] = replay

    def __call__(
        self,
        hidden: Tensor,
        observed: Tensor,
        forced_queue: Tensor,
        log_lambda: Tensor,
        delay: Tensor,
        replan: Tensor,
        uniforms: Tensor,
        *,
        sampled_offsets: int,
        ctx_pad: Tensor | None = None,
        history: KVMemory | None = None,
        sampled_actions: Tensor | None = None,
        validate: bool = True,
    ) -> DecisionOutput:
        if hidden.ndim != 3 or min(hidden.shape[:2]) < 1:
            raise ValueError("decision history must be nonempty [B, L, d]")
        batch = hidden.shape[0]
        if observed.shape != (batch, 4) or forced_queue.shape != (batch, self.max_delay, 4):
            raise ValueError("observed actions or forced queue have the wrong shape")
        if delay.shape != (batch,) or replan.shape != (batch,):
            raise ValueError("decision delay and replan must be [B]")
        if sampled_offsets not in self.variants:
            raise ValueError("sampled offsets must be 2 or 4")
        if uniforms.shape != (sampled_offsets, 4, batch):
            raise ValueError("decision uniforms must be [sampled_offsets, groups, B]")
        if sampled_actions is not None and sampled_actions.shape != (batch, sampled_offsets, 4):
            raise ValueError("sampled actions must match the padded decision shape")
        if delay.dtype not in (torch.int32, torch.int64) or replan.dtype not in (torch.int32, torch.int64):
            raise ValueError("decision delay and replan must be integers")
        if validate and not bool((torch.isfinite(uniforms) & (uniforms >= 0) & (uniforms < 1)).all()):
            raise ValueError("decision uniforms must be finite and in [0, 1)")
        if validate:
            valid = torch.zeros_like(delay, dtype=torch.bool)
            for d, r in self.configurations:
                valid |= (delay == d) & (replan == r)
            if not bool(valid.all()) or bool((replan > sampled_offsets).any()):
                raise ValueError("decision batch contains an unsupported configuration")
        for name, actions in (
            ("observed", observed),
            ("forced queue", forced_queue),
            ("sampled actions", sampled_actions),
        ):
            if actions is None:
                continue
            if actions.dtype not in (torch.int32, torch.int64) or actions.shape[-1] != 4:
                raise ValueError(f"{name} must contain integer controller factors")
            if not validate:
                continue
            if not bool(((actions >= 0) & (actions < self.group_sizes)).all()):
                raise ValueError(f"{name} contains an invalid controller factor")
            if bool(
                self.model.codec.button_mask(actions[..., TRIGGERS_GROUP])
                .gather(-1, actions[..., BUTTONS_GROUP, None])
                .any()
            ):
                raise ValueError(f"{name} contains buttons incompatible with its triggers")
        condition = self.model.ranges.normalize(log_lambda, delay, replan, validate=validate)
        bf16 = self.model.decoder_precision == "bf16"
        dtype = torch.bfloat16 if bf16 else torch.float32
        # Own the precision boundary even when evaluation has no outer autocast.
        # Disable weight caching: rollout graphs must read updated actor weights.
        with torch.autocast(hidden.device.type, dtype=torch.bfloat16, enabled=bf16, cache_enabled=False):
            hidden = hidden.to(dtype)
            if history is not None:
                history = KVMemory(history.kv.to(dtype), history.positions, history.query_position, history.window)
            if self.compiled and hidden.is_cuda and sampled_actions is None:
                return self._sample_graph(
                    hidden,
                    observed,
                    forced_queue,
                    delay,
                    replan,
                    condition,
                    uniforms,
                    sampled_offsets,
                    ctx_pad,
                    history,
                )
            variant = (
                self.variants[sampled_offsets] if sampled_actions is None else self.replay_variants[sampled_offsets]
            )
            return variant(
                hidden,
                observed,
                forced_queue,
                delay,
                replan,
                condition,
                uniforms,
                ctx_pad=ctx_pad,
                history=history,
                sampled_actions=sampled_actions,
            )

    def _sample_graph(
        self,
        hidden: Tensor,
        observed: Tensor,
        forced_queue: Tensor,
        delay: Tensor,
        replan: Tensor,
        condition: Tensor,
        uniforms: Tensor,
        sampled_offsets: int,
        ctx_pad: Tensor | None,
        history: KVMemory | None,
    ) -> DecisionOutput:
        if torch.is_grad_enabled():
            raise ValueError("CUDA graph sampling requires no_grad or inference_mode")
        batch = hidden.shape[0]
        bucket = 1 << (batch - 1).bit_length()
        capacity = self.model.cfg.L_ctx + 3
        if history is None:
            key, value = self.model.temporal.history_attention.project_memory(hidden)
            positions = torch.arange(hidden.shape[1], device=hidden.device)[None].expand(batch, -1)
            if ctx_pad is not None:
                positions = torch.where(positions >= ctx_pad[:, None], positions, -1)
            history = KVMemory(
                torch.stack((key, value)), positions, torch.full_like(delay, hidden.shape[1] - 1), self.model.cfg.L_ctx
            )
        if history.kv.shape[3] > capacity:
            raise ValueError("decision history exceeds the prepared graph capacity")
        missing = bucket - batch
        inputs = (
            F.pad(hidden[:, -1:], (0, 0, 0, 0, 0, missing)),
            F.pad(observed, (0, 0, 0, missing)),
            F.pad(forced_queue, (0, 0, 0, 0, 0, missing)),
            F.pad(delay, (0, missing), value=int(self.configurations[0][0])),
            F.pad(replan, (0, missing), value=2),
            F.pad(condition, (0, 0, 0, missing)),
            F.pad(uniforms, (0, missing)),
            F.pad(history.kv, (0, 0, 0, capacity - history.kv.shape[3], 0, 0, 0, missing)),
            F.pad(history.positions, (0, capacity - history.positions.shape[1], 0, missing), value=-1),
            F.pad(history.query_position, (0, missing)),
        )
        graph_key = (
            sampled_offsets,
            bucket,
            torch.is_inference_mode_enabled(),
            history.window,
        )
        captured = self.graphs.get(graph_key)
        if captured is None:
            static = tuple(value.clone() for value in inputs)
            window = history.window

            def sample() -> DecisionOutput:
                return self.variants[sampled_offsets](
                    *static[:7], history=KVMemory(static[7], static[8], static[9], window)
                )

            captured = CapturedCall(sample, static, device=hidden.device)
            self.graphs[graph_key] = captured
        output = captured(inputs)
        # Graph output storage is overwritten on replay. Rollout records outlive it.
        return DecisionOutput(
            output.actions[:batch].clone(),
            output.valid[:batch].clone(),
            output.log_probs[:batch].clone(),
            tuple(value[:batch].clone() for value in output.distributions),
            output.decoder_state[:batch].clone(),
        )
