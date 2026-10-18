"""Condition a percentile policy on episode settings and fit a detached residual critic."""

from collections.abc import Mapping

import torch
from torch import Tensor
from torch import nn

from hal.models.action_sequence import ActionSequenceConfig
from hal.models.action_sequence import ActionSequenceTransformer
from hal.models.action_sequence import ReturnConditioner
from hal.models.action_sequence import SwiGLU
from hal.models.action_sequence import decoder_rmsnorm
from hal.models.conditioning import ConditioningRanges
from hal.models.controller_codec import CONTROLLER_GROUP_COUNT
from hal.representation.player_identity import PlayerVocabulary


class QueueResidualHead(nn.Module):
    """A baseline on detached policy features, before any ego action is drawn."""

    def __init__(self, cfg: ActionSequenceConfig, max_queue: int, hidden_dim: int) -> None:
        super().__init__()
        if max_queue < 1 or hidden_dim < 1:
            raise ValueError("critic queue capacity and hidden width must be positive")
        self.max_queue = max_queue
        self.offset_embedding = nn.Embedding(max_queue, cfg.offset_embed_dim)
        self.queue_encoder = SwiGLU(4 * cfg.action_embed_dim + cfg.offset_embed_dim, hidden_dim, hidden_dim)
        self.residual = SwiGLU(cfg.temporal_d_model + 6 + hidden_dim, hidden_dim, 1, output_bias=True)
        nn.init.zeros_(self.residual.down.weight)
        nn.init.zeros_(self.residual.down.bias)

    def forward(
        self,
        decoder_state: Tensor,
        condition: Tensor,
        opponent_condition: Tensor,
        queue_embeddings: Tensor,
        queue_lengths: Tensor,
    ) -> Tensor:
        batch = decoder_state.shape[0]
        if condition.shape != (batch, 3) or opponent_condition.shape != condition.shape:
            raise ValueError("critic conditions must be [B, 3]")
        if queue_embeddings.ndim != 3 or queue_embeddings.shape[:2] != (batch, self.max_queue):
            raise ValueError("opponent queue must match the critic capacity")
        if queue_lengths.shape != (batch,) or queue_lengths.dtype not in (torch.int32, torch.int64):
            raise ValueError("opponent queue lengths must be integer [B]")
        if not torch.compiler.is_compiling() and bool(((queue_lengths < 0) | (queue_lengths > self.max_queue)).any()):
            raise ValueError("opponent queue length is outside its capacity")
        offsets = torch.arange(self.max_queue, device=decoder_state.device)
        embedded_offsets = self.offset_embedding(offsets).expand(batch, -1, -1)
        frames = torch.cat((queue_embeddings.detach(), embedded_offsets), dim=-1)
        valid = offsets[None] < queue_lengths[:, None]
        pooled = torch.where(valid[..., None], frames, 0).sum(dim=1) / queue_lengths.clamp_min(1)[:, None]
        queue = self.queue_encoder(pooled)
        inputs = torch.cat((decoder_state.detach(), condition, opponent_condition, queue), dim=-1)
        return self.residual(inputs).squeeze(-1).float()


class SelfPlayTransformer(ActionSequenceTransformer):
    """O63/O64 architecture with explicit conditioning and independent critic heads."""

    def __init__(
        self,
        cfg: ActionSequenceConfig,
        ranges: ConditioningRanges,
        vocabulary: PlayerVocabulary | None = None,
        *,
        max_opponent_queue: int,
        critic_hidden_dim: int,
    ) -> None:
        super().__init__(cfg, vocabulary, live_horizons=())
        self.ranges = ranges
        with torch.random.fork_rng(devices=[]):
            self.temporal.return_conditioner = ReturnConditioner(cfg, ranges)
            self.percentile_head = SwiGLU(cfg.d_model, cfg.value_hidden_dim, 65, output_bias=True)
            self.critic_residual = QueueResidualHead(cfg, max_opponent_queue, critic_hidden_dim)
        self.percentile_head.requires_grad_(False)

    def load_bc_state_dict(self, state: Mapping[str, Tensor]) -> None:
        """Load BC tensors and widen the conditioner with zero-valued input columns.

        Validate the checkpoint identity and architecture before calling. Check every
        BC key and tensor size before applying the converted state.
        """
        expected = {name for name in self.state_dict() if not name.startswith("critic_residual.")}
        if state.keys() != expected:
            raise ValueError(
                f"BC tensor keys mismatch: missing={sorted(expected - state.keys())}, extra={sorted(state.keys() - expected)}"
            )
        name = "temporal.return_conditioner.embedding.weight"
        old = state[name]
        if old.shape != (self.cfg.return_embed_dim, 1):
            raise ValueError("BC conditioner must have exactly one input column")
        converted = dict(state)
        converted[name] = torch.cat((old, old.new_zeros(old.shape[0], 3)), dim=1)
        converted.update(
            {key: value for key, value in self.state_dict().items() if key.startswith("critic_residual.")}
        )
        self.load_state_dict(converted, strict=True)

    def critic_parameters(self) -> tuple[nn.Parameter, ...]:
        return (*self.value_head.parameters(), *self.critic_residual.parameters())

    def actor_parameters(self) -> tuple[nn.Parameter, ...]:
        excluded = {id(parameter) for parameter in self.critic_parameters()}
        return tuple(
            parameter for parameter in self.parameters() if parameter.requires_grad and id(parameter) not in excluded
        )

    def self_play_value(
        self,
        trunk_state: Tensor,
        decoder_state: Tensor,
        condition: Tensor,
        opponent_condition: Tensor,
        opponent_queue: Tensor,
        opponent_queue_lengths: Tensor,
    ) -> Tensor:
        if trunk_state.ndim != 2 or trunk_state.shape[1] != self.cfg.d_model:
            raise ValueError("critic trunk state must be [B, d_model]")
        if decoder_state.shape != (trunk_state.shape[0], self.cfg.temporal_d_model):
            raise ValueError("critic decoder state must be [B, temporal_d_model]")
        if opponent_queue.shape != (trunk_state.shape[0], self.critic_residual.max_queue, CONTROLLER_GROUP_COUNT):
            raise ValueError("opponent queue must be [B, max_queue, controller_groups]")
        if opponent_queue.dtype not in (torch.int32, torch.int64):
            raise ValueError("opponent queue must contain integer controller factors")
        with torch.autocast(device_type=trunk_state.device.type, enabled=False):
            features = decoder_rmsnorm(trunk_state.detach()).to(self.value_head.up.weight.dtype)
            base = self.value_head(features).squeeze(-1).float()
            queue = self.codec.embed_frame(opponent_queue).detach().to(features.dtype)
            residual = self.critic_residual(
                decoder_state.detach().to(features.dtype),
                condition.to(features.dtype),
                opponent_condition.to(features.dtype),
                queue,
                opponent_queue_lengths,
            )
        return base + residual


def validate_executed_offsets(
    supervised_offsets: tuple[int, ...],
    configurations: tuple[tuple[int, int], ...],
) -> None:
    if (
        not supervised_offsets
        or any(isinstance(offset, bool) or not isinstance(offset, int) or offset < 1 for offset in supervised_offsets)
        or tuple(sorted(set(supervised_offsets))) != supervised_offsets
    ):
        raise ValueError("supervised offsets must be positive, sorted, and unique")
    if not configurations or len(set(configurations)) != len(configurations):
        raise ValueError("decision configurations must be nonempty and unique")
    for delay, replan in configurations:
        if (
            isinstance(delay, bool)
            or not isinstance(delay, int)
            or delay < 0
            or isinstance(replan, bool)
            or not isinstance(replan, int)
            or replan not in (2, 4)
        ):
            raise ValueError("decision delay must be nonnegative and replan must be 2 or 4")
        missing = set(range(delay + 1, delay + replan + 1)) - set(supervised_offsets)
        if missing:
            raise ValueError(f"D={delay}, R={replan} executes unsupervised offsets {sorted(missing)}")
