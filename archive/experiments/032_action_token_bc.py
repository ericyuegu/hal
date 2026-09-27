"""PPO-ready unified action-token BC decoder on the experiment-026 trunk.

The observation trunk, corpus, sparse offsets, objective, optimizer, and H4/H6
rolling-context execution contract are inherited unchanged from experiment 026.
Only the decoder changes: each ``(future offset, controller group)`` pair is a
causal token, trained in parallel and decoded with a plan-local KV cache.

Run:
    uv run experiments/032_action_token_bc.py
    uv run experiments/032_action_token_bc.py --eval runs/<run>/final.pt
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from dataclasses import fields
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from torch import Tensor

_BASE_PATH = Path(__file__).with_name("026_temporal_mtp.py")
_SPEC = importlib.util.spec_from_file_location("hal_exp026_for_032", _BASE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
base = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = base
_SPEC.loader.exec_module(base)

# Re-export the unchanged experiment contract. Private helpers intentionally
# remain available because the inherited train/eval functions resolve them in
# ``base`` rather than through this module.
for _name in dir(base):
    if not _name.startswith("__"):
        globals()[_name] = getattr(base, _name)


@dataclass
class TrainConfig(base.TrainConfig):
    decoder_arch_version: int = 4
    # The flattened decoder removes 026's auxiliary trunk heads and FiLM
    # conditioners. Widen only its output MLPs to retain the decoder budget.
    group_head_dim: int = 416
    policy_version: int = 0


def validate_config(cfg: TrainConfig) -> None:
    if cfg.decoder_arch_version != 4:
        raise ValueError(f"unsupported decoder_arch_version={cfg.decoder_arch_version}; expected action-token v4")
    # Exercise every unchanged invariant through the reference validator.
    base.validate_config_original(dataclass_replace(cfg, decoder_arch_version=3))
    if cfg.policy_version < 0:
        raise ValueError("policy_version must be non-negative")


@dataclass(frozen=True, slots=True)
class DecoderPlan:
    tokens: Tensor  # [batch, horizon, canonical groups]
    per_factor_logp: Tensor  # same shape
    entropy: Tensor  # same shape


@dataclass(frozen=True, slots=True)
class PlanSample:
    actions: Tensor
    tokens: Tensor
    per_factor_logp: Tensor
    entropy: Tensor
    value: Tensor
    executed_prefix_mask: Tensor
    policy_version: Tensor


@dataclass(frozen=True, slots=True)
class PlanScore:
    per_factor_logp: Tensor
    entropy: Tensor
    value: Tensor
    executed_prefix_mask: Tensor
    policy_version: Tensor


class ActionTokenDecoder(nn.Module):
    """One causal token for every offset/group pair.

    Teacher forcing shifts the complete flattened action-token sequence right,
    so a prediction can read exactly the earlier factors in ``GROUP_ORDER`` and
    earlier offsets. The one-token path retains K/V only for this plan and the
    caller discards it when sampling returns.
    """

    def __init__(self, cfg: TrainConfig, codec: StructuredControllerCodec) -> None:
        super().__init__()
        self.codec = codec
        self.head_offsets = tuple(cfg.head_offsets)
        self.group_order = tuple(cfg.group_order)
        self.d_model = cfg.temporal_d_model
        self.offset_embedding = nn.Embedding(cfg.sample_chunk_length + 1, cfg.offset_embed_dim)
        self.role_embedding = nn.Embedding(N_GROUPS, cfg.offset_embed_dim)
        self.bos = nn.Parameter(torch.zeros(cfg.action_embed_dim))
        self.token_projection = nn.Linear(
            cfg.d_model + cfg.action_embed_dim + 2 * cfg.offset_embed_dim,
            self.d_model,
        )
        self.blocks = nn.ModuleList([TemporalBlock(cfg) for _ in range(cfg.temporal_layers)])
        self.outputs = nn.ModuleDict(
            {
                name: NonlinearActionHead(self.d_model, cfg.group_head_dim, GROUP_VOCABS[GROUP_INDEX[name]])
                for name in GROUP_NAMES
            }
        )

        offsets = torch.tensor(self.head_offsets).repeat_interleave(N_GROUPS)
        roles = torch.tensor([GROUP_INDEX[name] for name in self.group_order]).repeat(len(self.head_offsets))
        self.register_buffer("token_offsets", offsets, persistent=False)
        self.register_buffer("token_roles", roles, persistent=False)

    def _flatten(self, targets: Tensor) -> Tensor:
        order = torch.tensor([GROUP_INDEX[name] for name in self.group_order], device=targets.device)
        return targets.index_select(-1, order).flatten(-2)

    def _unflatten(self, flat: Tensor, horizon: int) -> Tensor:
        ordered = flat.view(*flat.shape[:-1], horizon, N_GROUPS)
        canonical = torch.empty_like(ordered)
        for position, name in enumerate(self.group_order):
            canonical[..., GROUP_INDEX[name]] = ordered[..., position]
        return canonical

    def _embed_flat(self, flat: Tensor) -> Tensor:
        pieces = []
        for position in range(flat.shape[-1]):
            name = self.group_order[position % N_GROUPS]
            pieces.append(self.codec.group_embedding(name, flat[..., position]))
        return torch.stack(pieces, dim=-2)

    def _inputs(self, hidden: Tensor, targets: Tensor) -> Tensor:
        flat = self._flatten(targets)
        n_tokens = targets.shape[-2] * N_GROUPS
        roles = self.token_roles[:n_tokens].to(hidden.device)
        token_offsets = self.token_offsets[:n_tokens]
        embedded = self._embed_flat(flat)
        bos = self.bos.expand(*embedded.shape[:-2], 1, -1)
        previous = torch.cat((bos, embedded[..., :-1, :]), dim=-2)
        prefix = hidden.shape[:-1]
        trunk = decoder_rmsnorm(hidden)[..., None, :].expand(*prefix, len(roles), -1)
        offsets = self.offset_embedding(token_offsets).view(*((1,) * len(prefix)), len(roles), -1)
        role_emb = self.role_embedding(roles).view(*((1,) * len(prefix)), len(roles), -1)
        return self.token_projection(
            torch.cat((trunk, previous, offsets.expand(*prefix, -1, -1), role_emb.expand(*prefix, -1, -1)), dim=-1)
        )

    def teacher_forced_states(self, hidden: Tensor, observed: Tensor, targets: Tensor) -> Tensor:
        del observed  # The final trunk state already contains the observed action.
        if targets.shape[:2] != hidden.shape[:2] or targets.shape[-1] != N_GROUPS:
            raise ValueError("targets must match hidden's batch/context axes and end in controller groups")
        horizon = targets.shape[-2]
        if horizon < 1 or horizon > len(self.head_offsets):
            raise ValueError("target horizon exceeds configured offsets")
        x = self._inputs(hidden, targets)
        n_tokens = horizon * N_GROUPS
        x = x.reshape(-1, n_tokens, self.d_model)
        for block in self.blocks:
            x = block(x)
        return decoder_rmsnorm(x).view(*hidden.shape[:2], n_tokens, self.d_model)

    def _logits_from_states(self, states: Tensor, targets: Tensor) -> dict[str, Tensor]:
        batch_shape = states.shape[:-2]
        logits: dict[str, Tensor] = {}
        for position, name in enumerate(self.group_order):
            selected = states[..., position::N_GROUPS, :]
            logits[name] = self.outputs[name](selected)
        logits["buttons"] = logits["buttons"].masked_fill(self.codec.button_mask(targets[..., TRIG_G]), float("-inf"))
        return logits

    def teacher_forced_logits_by_group(self, hidden: Tensor, observed: Tensor, targets: Tensor) -> dict[str, Tensor]:
        return self._logits_from_states(self.teacher_forced_states(hidden, observed, targets), targets)

    def teacher_forced_nll(self, hidden: Tensor, observed: Tensor, targets: Tensor) -> Tensor:
        logits = self.teacher_forced_logits_by_group(hidden, observed, targets)
        losses = [
            F.cross_entropy(
                logits[name].float().reshape(-1, GROUP_VOCABS[group]),
                targets[..., group].reshape(-1),
                reduction="none",
            ).view(*targets.shape[:-1])
            for group, name in enumerate(GROUP_NAMES)
        ]
        return torch.stack(losses, dim=-1)

    def teacher_forced_logits(self, hidden: Tensor, observed: Tensor, targets: Tensor) -> list[dict[str, Tensor]]:
        values = self.teacher_forced_logits_by_group(hidden, observed, targets)
        return [
            {name: logits[..., depth, :] for name, logits in values.items()} for depth in range(len(self.head_offsets))
        ]

    def _step(self, trunk: Tensor, previous: Tensor, position: int, caches):
        offset = self.token_offsets[position].expand(trunk.shape[0])
        roles = self.token_roles[position].expand(trunk.shape[0])
        state = self.token_projection(
            torch.cat((trunk, previous, self.offset_embedding(offset), self.role_embedding(roles)), dim=-1)
        )
        next_caches = []
        for block, past in zip(self.blocks, caches, strict=True):
            state, present = block.forward_step(state, past)
            next_caches.append(present)
        return decoder_rmsnorm(state), next_caches, self.group_order[position % N_GROUPS]

    def forced_stepwise_logits(self, hidden: Tensor, observed: Tensor, targets: Tensor) -> list[dict[str, Tensor]]:
        del observed
        if targets.shape != (hidden.shape[0], len(self.head_offsets), N_GROUPS):
            raise ValueError("stepwise targets have the wrong shape")
        flat = self._flatten(targets)
        trunk = decoder_rmsnorm(hidden[:, -1])
        previous = self.bos.expand(hidden.shape[0], -1)
        caches = [None] * len(self.blocks)
        frames = [dict() for _ in self.head_offsets]
        for position in range(len(self.token_roles)):
            state, caches, name = self._step(trunk, previous, position, caches)
            logits = self.outputs[name](state)
            depth = position // N_GROUPS
            if name == "buttons":
                logits = logits.masked_fill(self.codec.button_mask(targets[:, depth, TRIG_G]), float("-inf"))
            frames[depth][name] = logits
            previous = self.codec.group_embedding(name, flat[:, position])
        return frames

    def score_indices(self, hidden: Tensor, tokens: Tensor) -> DecoderPlan:
        horizon = tokens.shape[-2]
        if tokens.shape != (hidden.shape[0], horizon, N_GROUPS) or tuple(self.head_offsets[:horizon]) != tuple(
            range(1, horizon + 1)
        ):
            raise ValueError("tokens must be [batch, dense-prefix horizon, groups]")
        # PPO must score through the exact cached one-token program used by
        # sampling. In reduced precision, parallel causal SDPA is numerically
        # different enough to create a non-unit ratio with unchanged weights.
        flat = self._flatten(tokens)
        trunk = decoder_rmsnorm(hidden[:, -1])
        previous = self.bos.expand(hidden.shape[0], -1)
        caches = [None] * len(self.blocks)
        flat_logp = []
        flat_entropy = []
        for position in range(horizon * N_GROUPS):
            state, caches, name = self._step(trunk, previous, position, caches)
            values = self.outputs[name](state)
            depth = position // N_GROUPS
            group = GROUP_INDEX[name]
            if name == "buttons":
                values = values.masked_fill(self.codec.button_mask(tokens[:, depth, TRIG_G]), float("-inf"))
            dist = torch.distributions.Categorical(logits=values.float())
            expected = tokens[:, depth, group]
            flat_logp.append(dist.log_prob(expected))
            flat_entropy.append(dist.entropy())
            previous = self.codec.group_embedding(name, flat[:, position])
        logp = self._unflatten(torch.stack(flat_logp, dim=-1), horizon)
        entropy = self._unflatten(torch.stack(flat_entropy, dim=-1), horizon)
        return DecoderPlan(tokens=tokens, per_factor_logp=logp, entropy=entropy)

    def sample_plan_indices(
        self,
        hidden: Tensor,
        horizon: int,
        *,
        argmax: bool = False,
        uniforms: Tensor | None = None,
        gen: torch.Generator | None = None,
    ) -> DecoderPlan:
        if horizon not in (4, 6):
            raise ValueError("live decode supports only dense four- or six-frame plans")
        if uniforms is not None and uniforms.shape != (horizon, N_GROUPS, hidden.shape[0]):
            raise ValueError("uniform table must be [frames, groups, batch]")
        n_tokens = horizon * N_GROUPS
        trunk = decoder_rmsnorm(hidden[:, -1])
        previous = self.bos.expand(hidden.shape[0], -1)
        caches = [None] * len(self.blocks)
        flat_picks, flat_logp, flat_entropy = [], [], []
        trigger_pick = None
        for position in range(n_tokens):
            state, caches, name = self._step(trunk, previous, position, caches)
            logits = self.outputs[name](state)
            depth = position // N_GROUPS
            group = GROUP_INDEX[name]
            if name == "buttons":
                assert trigger_pick is not None
                logits = logits.masked_fill(self.codec.button_mask(trigger_pick), float("-inf"))
            uniform = None if uniforms is None else uniforms[depth, group]
            pick = sample_categorical(logits, argmax=argmax, uniform=uniform, gen=gen)
            dist = torch.distributions.Categorical(logits=logits.float())
            flat_picks.append(pick)
            flat_logp.append(dist.log_prob(pick))
            flat_entropy.append(dist.entropy())
            if name == "triggers":
                trigger_pick = pick
            previous = self.codec.group_embedding(name, pick)
        tokens = self._unflatten(torch.stack(flat_picks, dim=-1), horizon)
        logp = self._unflatten(torch.stack(flat_logp, dim=-1), horizon)
        entropy = self._unflatten(torch.stack(flat_entropy, dim=-1), horizon)
        return DecoderPlan(tokens=tokens, per_factor_logp=logp, entropy=entropy)

    def sample_indices(
        self,
        hidden: Tensor,
        observed: Tensor,
        offsets: tuple[int, ...],
        *,
        argmax: bool,
        uniforms: Tensor | None = None,
        gen: torch.Generator | None = None,
    ) -> Tensor:
        del observed
        if offsets not in (self.head_offsets[:4], self.head_offsets[:6]):
            raise ValueError("live decode may compute only the dense four- or six-offset prefix")
        return self.sample_plan_indices(hidden, len(offsets), argmax=argmax, uniforms=uniforms, gen=gen).tokens

    def rollout_conditioned_logits(self, hidden: Tensor, observed: Tensor):
        # Greedy ancestral diagnostic, including the sparse auxiliary tail.
        del observed
        trunk = decoder_rmsnorm(hidden[:, -1])
        previous = self.bos.expand(hidden.shape[0], -1)
        caches = [None] * len(self.blocks)
        frames = [dict() for _ in self.head_offsets]
        flat = []
        trigger_pick = None
        for position in range(len(self.token_roles)):
            state, caches, name = self._step(trunk, previous, position, caches)
            logits = self.outputs[name](state)
            depth = position // N_GROUPS
            if name == "buttons":
                logits = logits.masked_fill(self.codec.button_mask(trigger_pick), float("-inf"))
            pick = logits.argmax(-1)
            frames[depth][name] = logits
            flat.append(pick)
            if name == "triggers":
                trigger_pick = pick
            previous = self.codec.group_embedding(name, pick)
        return frames, self._unflatten(torch.stack(flat, dim=-1), len(self.head_offsets))


class GPT(base.GPT):
    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__(cfg)
        self.temporal = ActionTokenDecoder(cfg, self.codec)
        self.value_head = nn.Linear(cfg.d_model, 1)
        # Pure BC supplies no value target. PPO (or an explicit value warmup)
        # must train this neutral head before its estimates are consumed.
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)
        self.register_buffer("policy_version", torch.tensor(cfg.policy_version, dtype=torch.long), persistent=True)

    def _plan_context(self, context: Context) -> tuple[Tensor, Tensor]:
        observed = self.codec.quantize(stack_actions(context.features))
        return self(context.features, context.ctx_pad, observed), observed[:, -1]

    @torch.no_grad()
    def sample_plan(
        self,
        context: Context,
        horizon: int,
        *,
        argmax: bool = False,
        uniforms: Tensor | None = None,
        gen: torch.Generator | None = None,
    ) -> PlanSample:
        hidden, _ = self._plan_context(context)
        plan = self.temporal.sample_plan_indices(hidden, horizon, argmax=argmax, uniforms=uniforms, gen=gen)
        batch = plan.tokens.shape[0]
        return PlanSample(
            actions=self.codec.dequantize(plan.tokens),
            tokens=plan.tokens,
            per_factor_logp=plan.per_factor_logp,
            entropy=plan.entropy,
            value=self.value_head(decoder_rmsnorm(hidden[:, -1])).squeeze(-1),
            executed_prefix_mask=torch.ones(batch, horizon, N_GROUPS, dtype=torch.bool, device=hidden.device),
            policy_version=self.policy_version.expand(batch),
        )

    def score_plan(self, context: Context, tokens: Tensor, executed_prefix_mask: Tensor | None = None) -> PlanScore:
        hidden, _ = self._plan_context(context)
        plan = self.temporal.score_indices(hidden, tokens)
        if executed_prefix_mask is None:
            mask = torch.ones_like(tokens, dtype=torch.bool)
        else:
            if executed_prefix_mask.shape == tokens.shape[:-1]:
                executed_prefix_mask = executed_prefix_mask[..., None].expand_as(tokens)
            if executed_prefix_mask.shape != tokens.shape:
                raise ValueError("executed_prefix_mask must be [batch,horizon] or [batch,horizon,groups]")
            mask = executed_prefix_mask.bool()
            flat_mask = mask.index_select(
                -1, torch.tensor([GROUP_INDEX[name] for name in self.group_order], device=mask.device)
            ).flatten(1)
            if ((~flat_mask[:, :-1]) & flat_mask[:, 1:]).any():
                raise ValueError("executed_prefix_mask must describe a contiguous action-token prefix")
        return PlanScore(
            per_factor_logp=plan.per_factor_logp,
            entropy=plan.entropy,
            value=self.value_head(decoder_rmsnorm(hidden[:, -1])).squeeze(-1),
            executed_prefix_mask=mask,
            policy_version=self.policy_version.expand(tokens.shape[0]),
        )


_CHECKPOINT_ARCH_FIELDS = base._CHECKPOINT_ARCH_FIELDS | {"policy_version"}


def config_from_state(values: dict) -> TrainConfig:
    missing = _CHECKPOINT_ARCH_FIELDS - values.keys()
    if missing:
        raise ValueError(f"checkpoint is not an experiment-032 architecture; missing {sorted(missing)}")
    known = {item.name for item in fields(TrainConfig)}
    return TrainConfig(**{name: value for name, value in values.items() if name in known})


def model_tag(cfg: TrainConfig) -> str:
    offsets = "-".join(map(str, cfg.head_offsets))
    return (
        f"atbc032-d{cfg.d_model}-L{cfg.n_layers}-h{cfg.n_heads}-Lc{cfg.L_ctx}-"
        f"t{cfg.temporal_d_model}x{cfg.temporal_layers}-o{offsets}-s{cfg.exec_horizon}-{cfg.observation_bundle}"
    )


def subsystem_parameter_counts(model: GPT) -> dict[str, int]:
    groups = {
        "trunk": model.trunk,
        "observation": model.ctx_proj,
        "codec": model.codec,
        "action_token_decoder": model.temporal,
        "ppo_value": model.value_head,
    }
    return {name: sum(parameter.numel() for parameter in module.parameters()) for name, module in groups.items()}


_base_train = base.train


def train(cfg: TrainConfig, stats, **kwargs) -> None:
    """Run the inherited 026 loop with unmistakable treatment metadata."""
    original_init = base.wandb.init

    def init_032(*args, **init_kwargs):
        init_kwargs["tags"] = ["gpt", "action-token", "autoregressive", "bc", "032"]
        return original_init(*args, **init_kwargs)

    base.wandb.init = init_032
    try:
        _base_train(cfg, stats, **kwargs)
    finally:
        base.wandb.init = original_init


@dataclass
class Args(base.Args):
    cfg: TrainConfig = dataclass_field(default_factory=TrainConfig)


# Install treatment classes into inherited functions. This preserves one tested
# implementation of the full training/evaluation pipeline while checkpoints
# instantiate only the action-token architecture.
dataclass_replace = base.replace
base.validate_config_original = base.validate_config
base.TrainConfig = TrainConfig
base.GPT = GPT
base.validate_config = validate_config
base.config_from_state = config_from_state
base.model_tag = model_tag
base.subsystem_parameter_counts = subsystem_parameter_counts
base.train = train
base.__file__ = __file__


if __name__ == "__main__":
    base.main(tyro.cli(Args))
