"""CartPole actor/critic MLPs for the gym gate (G1).

Plain ``nn.Module``s that emit logits (actor) and a scalar value (critic). They
carry no tianshou coupling; ``gym_train`` wraps the actor in a
``ProbabilisticActorPolicy`` and hands the critic to ``BehaviorLogpPPO``. Both
follow the CleanRL orthogonal-init recipe (hidden gain ``sqrt(2)``, policy head
gain 0.01 so the initial policy is near-uniform, value head gain 1.0, zero
biases), which is what lets PPO reach 475 on CartPole within the frame budget.

The forward signatures accept numpy or torch observations (the buffer stores
numpy, envpool hands numpy) and convert once on entry.
"""

import numpy as np
import torch
from beartype import beartype
from jaxtyping import Float
from jaxtyping import jaxtyped
from torch import nn

CARTPOLE_OBS_DIM = 4
CARTPOLE_N_ACT = 2
HIDDEN = 64

Obs = Float[np.ndarray, "batch obs_dim"] | Float[torch.Tensor, "batch obs_dim"]


def _orthogonal(layer: nn.Linear, gain: float) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class MLPActor(nn.Module):
    """obs -> logits over the discrete action space."""

    def __init__(self, obs_dim: int = CARTPOLE_OBS_DIM, n_act: int = CARTPOLE_N_ACT) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            _orthogonal(nn.Linear(obs_dim, HIDDEN), np.sqrt(2)),
            nn.Tanh(),
            _orthogonal(nn.Linear(HIDDEN, HIDDEN), np.sqrt(2)),
            nn.Tanh(),
        )
        self.head = _orthogonal(nn.Linear(HIDDEN, n_act), 0.01)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        obs: Obs,
        state: object = None,
        info: object = None,
    ) -> tuple[Float[torch.Tensor, "batch n_act"], None]:
        device = self.head.weight.device
        x = torch.as_tensor(obs, dtype=torch.float32, device=device)
        return self.head(self.trunk(x)), None


class MLPCritic(nn.Module):
    """obs -> scalar state value V(s)."""

    def __init__(self, obs_dim: int = CARTPOLE_OBS_DIM) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            _orthogonal(nn.Linear(obs_dim, HIDDEN), np.sqrt(2)),
            nn.Tanh(),
            _orthogonal(nn.Linear(HIDDEN, HIDDEN), np.sqrt(2)),
            nn.Tanh(),
        )
        self.head = _orthogonal(nn.Linear(HIDDEN, 1), 1.0)

    @jaxtyped(typechecker=beartype)
    def forward(self, obs: Obs) -> Float[torch.Tensor, "batch 1"]:
        device = self.head.weight.device
        x = torch.as_tensor(obs, dtype=torch.float32, device=device)
        return self.head(self.trunk(x))
