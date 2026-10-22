"""Architecture-based dense model FLOP estimates for O65 telemetry.

Count two FLOPs per multiply-add, causal attention products, and three times
forward work for a backward pass. Exclude nonlinearities, sampling, optimizer
work and the simulator. Padding counts as work. This is model MFU, not a kernel
instruction count; BF16 peak normalization also includes the small FP32 heads.
"""

from dataclasses import dataclass

import torch
from torch import nn

from hal.models.self_play import SelfPlayTransformer
from hal.training import mfu
from hal.training import self_play_distributed as distributed
from hal.wire import ITEM_SLOTS


def phase_metrics(
    work: tuple[float, float, float, float],
    frames: int,
    values: int,
    metrics: dict[str, float],
    *,
    device: torch.device,
    peak: float | None,
) -> dict[str, float]:
    rollout, actor, preparation, critic, frames_total, values_total = distributed.sum_tensor(
        torch.tensor((*work, frames, values), device=device, dtype=torch.float64)
    ).tolist()
    result: dict[str, float] = {"mfu/bf16_peak_known": float(peak is not None)}
    for name, flops, units in (
        ("rollout", rollout, frames_total),
        ("actor", actor, metrics["data/trained_decisions"]),
        ("preparation", preparation, values_total),
        ("critic", critic, values_total),
    ):
        seconds = metrics.get(f"systems/{name}_seconds", 0)
        result[f"flops/{name}_total"] = flops
        if units:
            result[f"flops/{name}_per_{'match_frame' if name == 'rollout' else 'decision'}"] = flops / units
        if flops and seconds > 0:
            result[f"throughput/{name}_tflops"] = flops / seconds / 1e12
            if peak is not None:
                result[f"mfu/{name}"] = mfu.model_flops_utilization(flops, seconds, distributed.world_size() * peak)
    return result


def linear_weights(module: nn.Module) -> int:
    return sum(layer.weight.numel() for layer in module.modules() if isinstance(layer, nn.Linear))


@dataclass(frozen=True, slots=True)
class SelfPlayFlops:
    encode: int
    trunk: int
    trunk_layers: int
    trunk_width: int
    decoder: int
    decoder_layers: int
    decoder_width: int
    cross_attention: int
    decision: int
    heads: int
    critic: int

    @classmethod
    def from_model(cls, model: SelfPlayTransformer) -> SelfPlayFlops:
        temporal = model.temporal
        cfg = model.cfg
        return cls(
            encode=2
            * (
                linear_weights(model.observation_encoder)
                + linear_weights(model.player_projection)
                + ITEM_SLOTS * linear_weights(model.item_encoder)
                + linear_weights(temporal.history_attention.key_value)
            ),
            trunk=2 * linear_weights(model.trunk),
            trunk_layers=cfg.n_layers,
            trunk_width=cfg.d_model,
            decoder=2
            * (
                linear_weights(temporal.blocks)
                + (4 * cfg.action_embed_dim + cfg.offset_embed_dim) * cfg.temporal_d_model
                + linear_weights(temporal.history_attention.query)
                + linear_weights(temporal.history_attention.output)
            ),
            decoder_layers=cfg.temporal_layers,
            decoder_width=cfg.temporal_d_model,
            cross_attention=cfg.temporal_d_model,
            decision=2
            * (
                cfg.d_model * cfg.temporal_d_model
                + linear_weights(temporal.return_conditioner)
                + linear_weights(temporal.trunk_outputs)
            ),
            heads=2 * (linear_weights(temporal.outputs) + linear_weights(temporal.group_condition)),
            critic=2 * (linear_weights(model.value_head) + linear_weights(model.critic_residual)),
        )

    def history(self, length: int, *, cached: bool = False) -> int:
        pairs = length * 256 if cached else length * (length + 1) // 2
        return length * (self.encode + self.trunk) + 4 * self.trunk_layers * self.trunk_width * pairs

    def decode(self, offsets: int, samples: int, *, context: int = 256) -> int:
        pairs = offsets * (offsets + 1) // 2
        return (
            self.decision
            + offsets * self.decoder
            + samples * self.heads
            + 4 * self.decoder_layers * self.decoder_width * pairs
            + 4 * self.cross_attention * offsets * context
        )

    def rollout(self, players: int, appended: int, offsets: int, samples: int, *, restart: bool) -> int:
        return players * (
            self.history(appended, cached=True)
            + (self.history(128) if restart else 0)
            + self.decode(offsets, samples)
            + self.critic
        )

    def replay(self, windows: int, slots: int, offsets: int, samples: int) -> int:
        return windows * (self.history(256) + slots * self.decode(offsets, samples))
