"""Exponential-moving-average shadow weights for a torch module.

Self-play RL keeps a slowly-trailing *behavior* copy of the live policy: the
collector acts under EMA weights while the learner updates the fast weights,
which stabilizes the opponent distribution. This holds one shadow tensor per
parameter and per buffer.

Parameters are averaged (``shadow = decay*shadow + (1-decay)*param``); buffers
are copied verbatim from the live module, never averaged. A buffer is running
state that must stay internally consistent with the parameters it annotates
(BatchNorm running mean/var, RoPE ``inv_freq``, causal masks) — a blended value
would describe no coherent model and could silently corrupt inference under the
EMA weights. Verbatim copy keeps the shadow a faithful, runnable module.
"""

import torch


class EMAWeights:
    def __init__(self, module: torch.nn.Module, decay: float) -> None:
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"decay must be in [0, 1], got {decay}")
        self.decay = decay
        self._params: dict[str, torch.Tensor] = {name: p.detach().clone() for name, p in module.named_parameters()}
        self._buffers: dict[str, torch.Tensor] = {name: b.detach().clone() for name, b in module.named_buffers()}

    @torch.no_grad()
    def update(self, module: torch.nn.Module) -> None:
        for name, p in module.named_parameters():
            self._params[name].mul_(self.decay).add_(p, alpha=1.0 - self.decay)
        for name, b in module.named_buffers():
            self._buffers[name].copy_(b)

    @torch.no_grad()
    def copy_to(self, module: torch.nn.Module) -> None:
        for name, p in module.named_parameters():
            p.copy_(self._params[name])
        for name, b in module.named_buffers():
            b.copy_(self._buffers[name])

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {**self._params, **self._buffers}

    @torch.no_grad()
    def load_state_dict(self, sd: dict[str, torch.Tensor]) -> None:
        expected = set(self._params) | set(self._buffers)
        got = set(sd)
        if got != expected:
            raise ValueError(f"EMA state_dict key mismatch: missing {expected - got}, unexpected {got - expected}")
        for name, dst in self._params.items():
            dst.copy_(sd[name])
        for name, dst in self._buffers.items():
            dst.copy_(sd[name])
