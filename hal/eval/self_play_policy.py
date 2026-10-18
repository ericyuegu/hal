"""Run conditioned actors frame by frame in synchronous Dolphin and MSL matches."""

import hashlib
import math
from collections import deque
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from hal.data.feature_stats import FeatureStats
from hal.data.schema import MDS_PER_FRAME_DTYPES
from hal.inference.column_context import ColumnContextBuilder
from hal.inference.decision_decoder import DecisionDecoder
from hal.inference.kv_cache import DualBlockRestartCache
from hal.models.self_play import SelfPlayTransformer
from hal.representation.observations import project_observation_columns
from hal.sim.native import OBSERVATION_FIELDS
from hal.sim.rollout import ObservationRow
from hal.sim.rollout import PolicyRuntimeSpec
from hal.sim.rollout import Slot
from hal.wire import ACTION_DIM


@dataclass(frozen=True, slots=True)
class PlayerCondition:
    lambda_value: float = 1.0
    delay: int = 8
    replan: int = 4
    player_id: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(self.lambda_value) or self.lambda_value <= 0:
            raise ValueError("lambda must be finite and positive")
        if type(self.delay) is not int or self.delay < 0 or type(self.replan) is not int or self.replan not in (2, 4):
            raise ValueError("delay must be nonnegative and replan must be 2 or 4")
        if type(self.player_id) is not int or self.player_id < 0:
            raise ValueError("player identity must be a nonnegative integer")


def evaluation_uniforms(seed: int, logical_player: int, frame: int, count: int) -> np.ndarray:
    """Counter-keyed draws are independent of ports, batching, and simulator choice."""
    key = f"o65-evaluation-actions-v1/{seed}/{logical_player}/{frame}".encode()
    counter_seed = int.from_bytes(hashlib.sha256(key).digest()[:16], "little")
    return np.random.Generator(np.random.Philox(counter_seed)).random((count, 4, 1), dtype=np.float32)


class SelfPlayPlayer:
    """Read consecutive observations, update caches, and commit delayed action chunks.

    At frame t, sample offsets delay+1 through delay+replan. Return the queued action
    at offset 1 for execution at t+1. Include transport and compute in delay.
    """

    def __init__(
        self,
        model: SelfPlayTransformer,
        stats: dict[str, FeatureStats],
        condition: PlayerCondition,
        *,
        port: int,
        seed: int,
        logical_player: int,
        compiled: bool = False,
        history_mode: Literal["block", "sliding"] = "block",
    ) -> None:
        if history_mode not in ("block", "sliding"):
            raise ValueError("unknown history mode")
        if condition.player_id >= model.cfg.player_vocab_size:
            raise ValueError("player identity is outside the checkpoint vocabulary")
        if type(seed) is not int or seed < 0 or type(logical_player) is not int or logical_player < 0:
            raise ValueError("sampling seed and logical player must be nonnegative integers")
        self.model = model
        self.condition = condition
        self.device = next(model.parameters()).device
        self.builder = ColumnContextBuilder(stats, port)
        self.decoder = DecisionDecoder(model, ((condition.delay, condition.replan),), compiled=compiled)
        self.model.ranges.normalize(
            torch.tensor([math.log(condition.lambda_value)], device=self.device),
            torch.tensor([condition.delay], device=self.device),
            torch.tensor([condition.replan], device=self.device),
        )
        self.cache = DualBlockRestartCache(model, 1, self.device, dtype=next(model.parameters()).dtype)
        self.history_mode = history_mode
        self.tokens: deque[Tensor] = deque(maxlen=256)
        self.queue: deque[Tensor] = deque()
        self.frame = 0
        self.last_frame: int | None = None
        self.seed = seed
        self.logical_player = logical_player
        self.neutral = model.codec.quantize(torch.zeros(1, ACTION_DIM, device=self.device))[0]
        self.last_sampled: Tensor | None = None
        self.last_log_probs: Tensor | None = None

    def reset(self) -> None:
        self.cache.reset()
        self.tokens.clear()
        self.queue = deque(self.neutral.clone() for _ in range(self.condition.delay))
        self.frame = 0
        self.last_frame = None
        self.last_sampled = None
        self.last_log_probs = None

    @torch.inference_mode()
    def step(self, row: ObservationRow) -> np.ndarray:
        reset = row.reset or self.last_frame is None
        if reset:
            self.reset()
        elif row.frame_id != self.last_frame + 1:
            raise ValueError("O65 observations must be consecutive; mark episode resets explicitly")
        self.last_frame = row.frame_id
        values = project_observation_columns(row.flat, OBSERVATION_FIELDS)
        columns = {
            name: torch.tensor(
                [[values[name]]],
                device=self.device,
                dtype=torch.int64 if np.dtype(MDS_PER_FRAME_DTYPES[name]).kind == "i" else torch.float32,
            )
            for name in OBSERVATION_FIELDS
        }
        actions = torch.zeros(1, 1, 2, ACTION_DIM, device=self.device)
        action = torch.as_tensor(np.array(row.action, copy=True), device=self.device, dtype=torch.float32)
        if action.shape != (ACTION_DIM,) or not bool(torch.isfinite(action).all()):
            raise ValueError("observed controller must be a finite HAL action vector")
        actions[0, 0, self.builder.port - 1] = action
        context = self.builder(
            columns,
            actions,
            torch.zeros(1, 1, device=self.device, dtype=torch.long),
            torch.tensor([[reset]], device=self.device),
            torch.tensor([self.condition.player_id], device=self.device),
        )
        token = self.model.context_tokens(context.features)
        if self.history_mode == "block":
            hidden = self.cache.append(token)
            memory = self.cache.memory()
        else:
            self.tokens.append(token)
            hidden = self.model.trunk.forward_dense(
                torch.cat(tuple(self.tokens), dim=1), context.ctx_pad, force_dense=True
            )
            memory = None
        if self.frame % self.condition.replan == 0:
            if len(self.queue) != self.condition.delay:
                raise RuntimeError("O65 queue length disagrees with D at a decision")
            queue = torch.stack(tuple(self.queue))[None] if self.queue else self.neutral.new_empty(1, 0, 4)
            # Decoder variants reserve four slots even when only two execute.
            uniforms = torch.as_tensor(
                evaluation_uniforms(self.seed, self.logical_player, self.frame, self.condition.replan),
                device=self.device,
            )
            output = self.decoder(
                hidden,
                self.model.codec.quantize(action[None] if not reset else torch.zeros_like(action[None])),
                queue,
                torch.tensor([math.log(self.condition.lambda_value)], device=self.device),
                torch.tensor([self.condition.delay], device=self.device),
                torch.tensor([self.condition.replan], device=self.device),
                uniforms,
                sampled_offsets=self.condition.replan,
                history=memory,
            )
            self.last_sampled = output.actions[0].clone()
            self.last_log_probs = output.log_probs[0].clone()
            self.queue.extend(output.actions[0, : self.condition.replan].unbind())
        due = self.queue.popleft()
        self.frame += 1
        return self.model.codec.dequantize(due).float().cpu().numpy()


class SelfPlayPolicy:
    """Use the same per-frame path on both simulators, including mixed cadence."""

    def __init__(self, players: Mapping[int, SelfPlayPlayer]) -> None:
        if set(players) != {1, 2}:
            raise ValueError("O65 evaluation requires players on ports 1 and 2")
        self.players = dict(players)

    @property
    def runtime_spec(self) -> PolicyRuntimeSpec:
        # Synchronous stepping makes the D-frame queue the complete delay.
        return PolicyRuntimeSpec(256, 1, 1, 0, ACTION_DIM, observed_actions=True)

    def plan_rows(self, rows: Mapping[Slot, Sequence[ObservationRow]]) -> Mapping[Slot, np.ndarray]:
        result = {}
        for slot, observations in rows.items():
            if slot.match != 0 or slot.port not in self.players or len(observations) != 1:
                raise ValueError("O65 evaluation accepts one match and one observation per player")
            result[slot] = self.players[slot.port].step(observations[0])[None]
        return result
