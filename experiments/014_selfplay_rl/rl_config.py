"""Frozen config objects for the 014 self-play PPO experiment.

The config is split by concern so each axis varies independently: ``PPOConfig``
is pure PPO/GAE math (shared by the gym and Melee entry points); ``PipelineConfig``
and ``EMAConfig`` govern the collector/learner plumbing (double-buffering, the
trailing behavior policy); ``RewardConfig`` is the Melee reward shaping; and the
phase configs (``GymConfig``, ``MeleeRLConfig``) carry per-environment run knobs.
Entry scripts compose these with tyro; nothing here imports tyro. Melee PPO
overrides a few ``PPOConfig`` defaults at the entry-script level (see
``MeleeRLConfig``).
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PPOConfig:
    lr: float = 3e-4
    clip: float = 0.2
    epochs: int = 4
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    target_kl: float = 0.015
    max_grad_norm: float = 0.5


@dataclass(frozen=True, slots=True)
class EMAConfig:
    decay: float = 0.995


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    overlap: bool = True


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Per-step shaped reward weights; win_bonus is terminal only (0 on truncation)."""

    damage_dealt: float = 0.01
    damage_taken: float = 0.01
    stock_take: float = 1.0
    stock_loss: float = 1.0
    win_bonus: float = 2.0


@dataclass(frozen=True, slots=True)
class GymConfig:
    task: str = "CartPole-v1"
    num_envs: int = 8
    horizon: int = 128
    total_frames: int = 500_000


@dataclass(frozen=True, slots=True)
class MeleeRLConfig:
    """Melee self-play run knobs; later milestones extend this.

    Melee PPO overrides a few ``PPOConfig`` defaults at the entry-script level:
    gamma=0.997, ent_coef=0.003, lr=3e-5.
    """

    n_boots: int = 4
    rollout_frames: int = 4096
    refresh_every: int = 64
    value_warmup_iters: int = 20
    kl_il_coef: float = 0.05
    warm_start: str = "260616-004736_012_multi_token_gpt-d256-L8-h4-Lc256-o1.5.9.13_ranked-anon-1_gpt-16k-b1024"
    # Reboot the whole self-play wave every N learned iterations so the character matchups
    # rotate through the full training prior (each reboot advances to the next prior slice),
    # not just the first n_boots. In-progress streams flush truncated via the orphan path.
    reboot_every_iters: int = 150
