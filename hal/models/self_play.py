"""Condition a percentile policy on episode settings and fit a distributional critic."""

import math
from collections.abc import Mapping
from typing import Literal

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

type DecoderPrecision = Literal["bf16", "fp32"]


def validate_decoder_precision(value: str) -> DecoderPrecision:
    if value == "bf16":
        return "bf16"
    if value == "fp32":
        return "fp32"
    raise ValueError("decoder precision must be bf16 or fp32")


# The critic predicts a categorical distribution over evenly spaced returns and
# trains on HL-Gauss targets (Farebrother et al., 2024). Stock-only returns with
# a 30-second half-life, including KL costs, stay well inside this range;
# targets outside it are clipped to the edge bins.
VALUE_LIMIT = 600.0
VALUE_BINS = 120
VALUE_SIGMA = 0.75 * (2 * VALUE_LIMIT / VALUE_BINS)


def value_edges(device: torch.device) -> Tensor:
    # Built on the target device, because CUDA graph capture forbids host copies.
    return torch.linspace(-VALUE_LIMIT, VALUE_LIMIT, VALUE_BINS + 1, device=device)


def value_centers(device: torch.device) -> Tensor:
    edges = value_edges(device)
    return (edges[1:] + edges[:-1]) / 2


def value_mean(logits: Tensor) -> Tensor:
    """Return the expected value of each categorical value distribution."""
    if logits.shape[-1] != VALUE_BINS:
        raise ValueError("value logits must end in the value bins")
    return (logits.float().softmax(-1) * value_centers(logits.device)).sum(-1)


def value_targets(values: Tensor) -> Tensor:
    """Spread each clipped target as a Gaussian over the bins, then normalize."""
    edges = value_edges(values.device)
    target = values.detach().float().clamp(-VALUE_LIMIT, VALUE_LIMIT)
    cdf = torch.erf((edges - target[..., None]) / (math.sqrt(2) * VALUE_SIGMA))
    mass = (cdf[..., 1:] - cdf[..., :-1]).clamp_min(0)
    return mass / mass.sum(-1, keepdim=True)


def value_cross_entropy(logits: Tensor, values: Tensor) -> Tensor:
    """Return the per-row cross-entropy between logits and HL-Gauss targets."""
    if logits.shape[:-1] != values.shape or logits.shape[-1] != VALUE_BINS:
        raise ValueError("value logits must match targets plus the value bins")
    return -(value_targets(values) * logits.float().log_softmax(-1)).sum(-1)


class QueueCritic(nn.Module):
    """Predict a value distribution from trunk and decoder state before the ego draw."""

    def __init__(self, cfg: ActionSequenceConfig, max_queue: int, hidden_dim: int) -> None:
        super().__init__()
        if max_queue < 1 or hidden_dim < 1:
            raise ValueError("critic queue capacity and hidden width must be positive")
        self.max_queue = max_queue
        self.hidden_dim = hidden_dim
        # The encoder reads the whole queue in order, so each action keeps its
        # firing offset; a pooled mean would keep only the set of actions.
        self.queue_encoder = SwiGLU(max_queue * (4 * cfg.action_embed_dim + 1), hidden_dim, hidden_dim)
        self.head = SwiGLU(
            cfg.d_model + cfg.temporal_d_model + 8 + hidden_dim, hidden_dim, VALUE_BINS, output_bias=True
        )
        # Uniform initial distributions predict a value of zero.
        nn.init.zeros_(self.head.down.weight)
        nn.init.zeros_(self.head.down.bias)

    def forward(
        self,
        trunk_state: Tensor,
        decoder_state: Tensor,
        condition: Tensor,
        opponent_condition: Tensor,
        queue_embeddings: Tensor,
        queue_lengths: Tensor,
        *,
        validate: bool = True,
    ) -> Tensor:
        batch = decoder_state.shape[0]
        if condition.shape != (batch, 4) or opponent_condition.shape != condition.shape:
            raise ValueError("critic conditions must be [B, 4]")
        if queue_embeddings.ndim != 3 or queue_embeddings.shape[:2] != (batch, self.max_queue):
            raise ValueError("opponent queue must match the critic capacity")
        if queue_lengths.shape != (batch,) or queue_lengths.dtype not in (torch.int32, torch.int64):
            raise ValueError("opponent queue lengths must be integer [B]")
        if (
            validate
            and not torch.compiler.is_compiling()
            and bool(((queue_lengths < 0) | (queue_lengths > self.max_queue)).any())
        ):
            raise ValueError("opponent queue length is outside its capacity")
        offsets = torch.arange(self.max_queue, device=decoder_state.device)
        valid = (offsets[None] < queue_lengths[:, None])[..., None]
        frames = torch.cat((torch.where(valid, queue_embeddings, 0), valid.to(queue_embeddings.dtype)), -1)
        queue = self.queue_encoder(frames.flatten(1))
        inputs = torch.cat((trunk_state, decoder_state, condition, opponent_condition, queue), dim=-1)
        return self.head(inputs).float()


class SelfPlayTransformer(ActionSequenceTransformer):
    def __init__(
        self,
        cfg: ActionSequenceConfig,
        ranges: ConditioningRanges,
        vocabulary: PlayerVocabulary | None = None,
        *,
        max_opponent_queue: int,
        critic_hidden_dim: int,
        decoder_precision: DecoderPrecision = "bf16",
    ) -> None:
        precision = validate_decoder_precision(decoder_precision)
        super().__init__(cfg, vocabulary, live_horizons=())
        self._decoder_precision = precision
        self.ranges = ranges
        with torch.random.fork_rng(devices=[]):
            self.temporal.return_conditioner = ReturnConditioner(cfg, ranges)
            self.percentile_head = SwiGLU(cfg.d_model, cfg.value_hidden_dim, 65, output_bias=True)
            self.critic = QueueCritic(cfg, max_opponent_queue, critic_hidden_dim)
        self.percentile_head.requires_grad_(False)
        # The BC value head predicts the BC reward; the self-play critic replaces it.
        self.value_head.requires_grad_(False)

    @property
    def decoder_precision(self) -> DecoderPrecision:
        return self._decoder_precision

    def load_bc_state_dict(self, state: Mapping[str, Tensor]) -> None:
        """Add zero conditioner columns while loading BC weights.

        Validate BC keys and tensor sizes before applying the converted state.
        """
        expected = {name for name in self.state_dict() if not name.startswith("critic.")}
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
        converted.update({key: value for key, value in self.state_dict().items() if key.startswith("critic.")})
        self.load_state_dict(converted, strict=True)

    def critic_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.critic.parameters())

    def actor_parameters(self) -> tuple[nn.Parameter, ...]:
        excluded = {id(parameter) for parameter in self.critic_parameters()}
        return tuple(
            parameter for parameter in self.parameters() if parameter.requires_grad and id(parameter) not in excluded
        )

    def critic_logits(
        self,
        trunk_state: Tensor,
        decoder_state: Tensor,
        condition: Tensor,
        opponent_condition: Tensor,
        opponent_queue: Tensor,
        opponent_queue_lengths: Tensor,
        *,
        fixed_critic: bool = False,
        validate: bool = True,
    ) -> Tensor:
        """Return value logits; gradients reach the policy network through the given states.

        Callers detach both states for critic fitting and the decoder state for the
        actor's value loss. With fixed_critic, the critic's own parameters receive no
        gradient, so that value loss trains only the trunk.
        """
        if trunk_state.ndim != 2 or trunk_state.shape[1] != self.cfg.d_model:
            raise ValueError("critic trunk state must be [B, d_model]")
        if decoder_state.shape != (trunk_state.shape[0], self.cfg.temporal_d_model):
            raise ValueError("critic decoder state must be [B, temporal_d_model]")
        if opponent_queue.shape != (trunk_state.shape[0], self.critic.max_queue, CONTROLLER_GROUP_COUNT):
            raise ValueError("opponent queue must be [B, max_queue, controller_groups]")
        if opponent_queue.dtype not in (torch.int32, torch.int64):
            raise ValueError("opponent queue must contain integer controller factors")
        with torch.autocast(device_type=trunk_state.device.type, enabled=False):
            dtype = self.critic.head.up.weight.dtype
            inputs = (
                decoder_rmsnorm(trunk_state.float()).to(dtype),
                decoder_state.to(dtype),
                condition.to(dtype),
                opponent_condition.to(dtype),
                self.codec.embed_frame(opponent_queue).detach().to(dtype),
                opponent_queue_lengths,
            )
            if not fixed_critic:
                return self.critic(*inputs, validate=validate)
            parameters = {name: value.detach() for name, value in self.critic.named_parameters()}
            return torch.func.functional_call(self.critic, parameters, inputs, {"validate": validate})


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
