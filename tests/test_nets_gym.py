"""Nature-CNN net + pinned Pong env-factory tests (fast, no training).

Guards the two things a curve-comparable Atari run depends on: the CNN handles
uint8 [B,4,84,84] input and emits the right-shaped logits/value, its heads carry
the CleanRL orthogonal-init scales (near-zero policy head, unit value head), and
the envpool Pong factory resolves to the pinned wrapper config with the expected
observation contract.
"""

import numpy as np
import torch
from nets_gym import ATARI_FEATURES
from nets_gym import NatureCNN
from nets_gym import NatureCNNActor
from nets_gym import NatureCNNCritic

N_ACT = 6
BATCH = 5


def _obs(batch: int = BATCH) -> np.ndarray:
    return np.random.randint(0, 256, size=(batch, 4, 84, 84), dtype=np.uint8)


def test_nature_cnn_shapes_and_dtype() -> None:
    cnn = NatureCNN(N_ACT)
    actor, critic = NatureCNNActor(cnn), NatureCNNCritic(cnn)

    logits, state = actor(_obs())
    value = critic(_obs())

    assert logits.shape == (BATCH, N_ACT)
    assert value.shape == (BATCH, 1)
    assert logits.dtype == torch.float32
    assert value.dtype == torch.float32
    assert state is None
    # trunk feature width matches CleanRL fc512
    assert cnn.features(_obs()).shape == (BATCH, ATARI_FEATURES)


def test_nature_cnn_accepts_torch_uint8() -> None:
    cnn = NatureCNN(N_ACT)
    obs = torch.from_numpy(_obs())
    assert obs.dtype == torch.uint8
    logits, _ = NatureCNNActor(cnn)(obs)
    assert logits.shape == (BATCH, N_ACT)


def test_shared_trunk_is_one_instance() -> None:
    # Actor and critic must reference the SAME conv trunk (CleanRL shares the net);
    # otherwise the optimizer would learn two independent feature extractors.
    cnn = NatureCNN(N_ACT)
    actor, critic = NatureCNNActor(cnn), NatureCNNCritic(cnn)
    assert actor.cnn is critic.cnn
    assert actor.cnn.trunk[0].weight is critic.cnn.trunk[0].weight


def test_orthogonal_head_init_scales() -> None:
    # Orthogonal init with gain g on a [out, in] weight gives per-entry std ~ g/sqrt(in).
    # Policy head gain 0.01 -> near-zero (near-uniform initial policy); value head gain 1.0.
    cnn = NatureCNN(N_ACT)
    scale = np.sqrt(ATARI_FEATURES)
    policy_std = cnn.policy_head.weight.std().item() * scale
    value_std = cnn.value_head.weight.std().item() * scale

    assert 0.005 < policy_std < 0.02  # ~ 0.01
    assert 0.5 < value_std < 1.5  # ~ 1.0
    assert torch.count_nonzero(cnn.policy_head.bias) == 0
    assert torch.count_nonzero(cnn.value_head.bias) == 0


def test_pong_env_factory_pinned_config() -> None:
    import envpool
    from gym_train import PONG_ENV_KWARGS

    num_envs = 8
    env = envpool.make("Pong-v5", env_type="gymnasium", num_envs=num_envs, seed=0, **PONG_ENV_KWARGS)
    obs, _ = env.reset()

    assert obs.shape == (num_envs, 4, 84, 84)
    assert obs.dtype == np.uint8
    # the pinned config we assert against verbatim
    assert PONG_ENV_KWARGS == {
        "img_height": 84,
        "img_width": 84,
        "gray_scale": True,
        "stack_num": 4,
        "frame_skip": 4,
        "noop_max": 30,
        "episodic_life": True,
        "reward_clip": True,
        "repeat_action_probability": 0.0,
        "full_action_space": False,
    }
