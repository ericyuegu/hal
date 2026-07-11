"""CartPole MLP and Atari Nature-CNN actor/critic nets for the gym gates.

Plain ``nn.Module``s that emit logits (actor) and a scalar value (critic). They
carry no tianshou coupling; ``gym_train`` wraps the actor in a
``ProbabilisticActorPolicy`` and hands the critic to ``BehaviorLogpPPO``. Both
families follow the CleanRL orthogonal-init recipe (hidden gain ``sqrt(2)``,
policy head gain 0.01 so the initial policy is near-uniform, value head gain 1.0,
zero biases), which is what makes their learning curves comparable to CleanRL's.

The forward signatures accept numpy or torch observations (the buffer stores
numpy, envpool hands numpy) and convert once on entry.

CartPole (G1) uses two independent MLPs (separate trunks). Atari (G2) mirrors
CleanRL ``ppo_atari.py``: a single Nature-CNN trunk feeds a policy head and a
value head. ``NatureCNNActor``/``NatureCNNCritic`` are thin adapters over one
shared ``NatureCNN`` instance so the conv weights are learned jointly and the
parameter count matches CleanRL; ``ActorCritic``'s parameter dedup keeps the
shared trunk in the optimizer exactly once. Because tianshou invokes the actor
(for the action dist) and the critic (for V) separately, the shared trunk runs
twice per minibatch at update time — a deliberate, correctness-first tradeoff
(CleanRL's fused ``get_action_and_value`` runs it once) rather than contorting
tianshou's split actor/critic API.
"""

import numpy as np
import torch
from beartype import beartype
from jaxtyping import Float
from jaxtyping import UInt8
from jaxtyping import jaxtyped
from torch import nn

CARTPOLE_OBS_DIM = 4
CARTPOLE_N_ACT = 2
HIDDEN = 64

ATARI_STACK = 4
ATARI_HW = 84
ATARI_FEATURES = 512

Obs = Float[np.ndarray, "batch obs_dim"] | Float[torch.Tensor, "batch obs_dim"]
ImgObs = UInt8[np.ndarray, "batch stack h w"] | UInt8[torch.Tensor, "batch stack h w"]


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


class NatureCNN(nn.Module):
    """Shared Nature-CNN trunk + policy/value heads (CleanRL ``ppo_atari`` Agent).

    conv(32,8,s4) -> conv(64,4,s2) -> conv(64,3,s1) -> flatten -> fc(512), all
    ReLU; then a policy head (logits over ``n_act``) and a scalar value head.
    ``features`` scales the uint8 [B,4,84,84] observation by 1/255 on entry, once.
    Not used directly by tianshou; the ``NatureCNNActor``/``NatureCNNCritic``
    adapters expose it as the (actor, critic) pair the algorithm expects.
    """

    def __init__(self, n_act: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            _orthogonal(nn.Conv2d(ATARI_STACK, 32, kernel_size=8, stride=4), np.sqrt(2)),
            nn.ReLU(),
            _orthogonal(nn.Conv2d(32, 64, kernel_size=4, stride=2), np.sqrt(2)),
            nn.ReLU(),
            _orthogonal(nn.Conv2d(64, 64, kernel_size=3, stride=1), np.sqrt(2)),
            nn.ReLU(),
            nn.Flatten(),
            _orthogonal(nn.Linear(64 * 7 * 7, ATARI_FEATURES), np.sqrt(2)),
            nn.ReLU(),
        )
        self.policy_head = _orthogonal(nn.Linear(ATARI_FEATURES, n_act), 0.01)
        self.value_head = _orthogonal(nn.Linear(ATARI_FEATURES, 1), 1.0)

    @jaxtyped(typechecker=beartype)
    def features(self, obs: ImgObs) -> Float[torch.Tensor, "batch 512"]:
        device = self.policy_head.weight.device
        x = torch.as_tensor(obs, device=device).float().div_(255.0)
        return self.trunk(x)


class NatureCNNActor(nn.Module):
    """obs -> logits; a policy-head adapter over a (possibly shared) ``NatureCNN``."""

    def __init__(self, cnn: NatureCNN) -> None:
        super().__init__()
        self.cnn = cnn

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        obs: ImgObs,
        state: object = None,
        info: object = None,
    ) -> tuple[Float[torch.Tensor, "batch n_act"], None]:
        return self.cnn.policy_head(self.cnn.features(obs)), None


class NatureCNNCritic(nn.Module):
    """obs -> scalar V(s); a value-head adapter over a (possibly shared) ``NatureCNN``."""

    def __init__(self, cnn: NatureCNN) -> None:
        super().__init__()
        self.cnn = cnn

    @jaxtyped(typechecker=beartype)
    def forward(self, obs: ImgObs) -> Float[torch.Tensor, "batch 1"]:
        return self.cnn.value_head(self.cnn.features(obs))
