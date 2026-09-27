"""O59 uses contiguous trained heads without changing checkpoint configuration."""

from dataclasses import replace
from typing import Literal

import pytest
import torch

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.data.feature_stats import FeatureStats
from hal.data.feature_stats import consolidate_key
from hal.eval.scheduling import ActionScheduler
from hal.eval.scheduling import FrameTiming
from hal.inference.action_sequence_artifact import REQUIRED_OBSERVATION_FIELDS
from hal.inference.action_sequence_policy import ActionSequencePolicy
from hal.inference.api import PolicySpec
from hal.inference.api import PredictionRequest
from hal.inference.api import RuntimeConfig
from hal.inference.warmup import make_warmup_observations
from hal.models.action_sequence import ActionSequenceConfig
from hal.models.action_sequence import ActionSequenceTransformer
from hal.representation.features import ITEM_COLUMNS
from hal.representation.features import feature_kind


def policy(*, history_mode: Literal["window", "kv_cache"] = "window") -> ActionSequencePolicy:
    config = ActionSequenceConfig(
        d_model=32,
        n_layers=1,
        n_heads=4,
        L_ctx=8,
        temporal_d_model=32,
        temporal_layers=1,
        temporal_heads=4,
        temporal_ff_dim=64,
        group_head_dim=16,
        value_hidden_dim=16,
        item_hidden_dim=8,
        item_dim=5,
    )
    model = ActionSequenceTransformer(config).eval()
    stats: dict[str, FeatureStats] = {}
    for name in REQUIRED_OBSERVATION_FIELDS:
        relative = (
            f"ego_{name[3:]}" if name.startswith("p1_") else f"opp_{name[3:]}" if name.startswith("p2_") else name
        )
        if feature_kind(relative, ITEM_COLUMNS) not in ("cat", "button", "stick_trigger"):
            stats[consolidate_key(relative)] = FeatureStats(0.0, 1.0, -10.0, 10.0)
    spec = PolicySpec("small 059", "hal.action_sequence.test", REQUIRED_OBSERVATION_FIELDS, (0, 2))
    return ActionSequencePolicy(
        model,
        stats,
        (),
        spec=spec,
        checkpoint_sha256="test",
        return_p90=20.0,
        capability_version=1,
        device=torch.device("cpu"),
        seed=5,
        compiled=False,
        history_mode=history_mode,
    )


def test_o59_can_decode_twelve_contiguous_heads_and_reset_calibration_state() -> None:
    chunks = policy()
    assert chunks.supported_horizons == tuple(range(1, 13))
    chunks.prepare_prediction(RuntimeConfig(1, (2,)), 12, 4)
    request = PredictionRequest(
        0,
        1,
        0,
        7,
        make_warmup_observations(chunks.spec, chunks.context_frames, 0, 7, 2),
        (NEUTRAL_CONTROLLER_ACTION,) * 4,
    )
    first = chunks.predict((request,))[0]
    assert len(first.actions) == 8 and first.actions[0].target_frame == 12
    assert chunks.cfg.head_offsets[:4] == (1, 2, 3, 4)
    assert chunks.prediction_horizon == 12
    chunks.reset_prediction()
    assert chunks.predict((request,))[0] == first


def test_prediction_requires_contiguous_new_observations() -> None:
    chunks = policy()
    chunks.prepare_prediction(RuntimeConfig(1, (2,)), 4, 2)
    context = make_warmup_observations(chunks.spec, chunks.context_frames, 0, 7, 2)
    chunks.predict((PredictionRequest(0, 1, 0, 7, context, (NEUTRAL_CONTROLLER_ACTION,) * 2),))
    next_context = make_warmup_observations(chunks.spec, chunks.context_frames, 0, 10, 2)
    with pytest.raises(ValueError, match="noncontiguous"):
        chunks.predict((PredictionRequest(0, 1, 1, 10, next_context[-2:], (NEUTRAL_CONTROLLER_ACTION,) * 2),))


def test_prediction_rejects_overlapping_observations() -> None:
    chunks = policy()
    chunks.prepare_prediction(RuntimeConfig(1, (2,)), 4, 2)
    context = make_warmup_observations(chunks.spec, chunks.context_frames, 0, 7, 2)
    chunks.predict((PredictionRequest(0, 1, 0, 7, context, (NEUTRAL_CONTROLLER_ACTION,) * 2),))
    overlap = make_warmup_observations(chunks.spec, 3, 0, 9, 2)
    with pytest.raises(ValueError, match="noncontiguous"):
        chunks.predict((PredictionRequest(0, 1, 1, 9, overlap, (NEUTRAL_CONTROLLER_ACTION,) * 2),))


def test_prediction_accepts_delta_and_rejects_unannounced_reset() -> None:
    chunks = policy()
    chunks.prepare_prediction(RuntimeConfig(1, (2,)), 4, 2)
    context = make_warmup_observations(chunks.spec, chunks.context_frames, 0, 7, 2)
    chunks.predict((PredictionRequest(0, 1, 0, 7, context, (NEUTRAL_CONTROLLER_ACTION,) * 2),))
    next_context = make_warmup_observations(chunks.spec, chunks.context_frames, 0, 9, 2)
    delta = tuple(replace(item, reset=False) for item in next_context[-2:])
    plan = chunks.predict((PredictionRequest(0, 1, 1, 9, delta, (NEUTRAL_CONTROLLER_ACTION,) * 2),))[0]
    assert [item.target_frame for item in plan.actions] == [12, 13]
    with pytest.raises(ValueError, match="reset requires a new generation"):
        chunks.predict(
            (
                PredictionRequest(
                    0,
                    1,
                    2,
                    10,
                    (replace(next_context[-1], frame_id=10, reset=True),),
                    (NEUTRAL_CONTROLLER_ACTION,) * 2,
                ),
            )
        )


def test_prediction_rejects_replayed_sequence_and_older_generation() -> None:
    chunks = policy()
    chunks.prepare_prediction(RuntimeConfig(1, (2,)), 4, 2)
    request = PredictionRequest(
        0,
        2,
        3,
        7,
        make_warmup_observations(chunks.spec, chunks.context_frames, 0, 7, 2),
        (NEUTRAL_CONTROLLER_ACTION,) * 2,
    )
    chunks.predict((request,))
    with pytest.raises(ValueError, match="obsolete"):
        chunks.predict((request,))
    with pytest.raises(ValueError, match="generation"):
        chunks.predict((replace(request, generation=1),))


@pytest.mark.parametrize("history_mode", ["window", "kv_cache"])
def test_incremental_prediction_is_deterministic_for_twenty_frames(
    history_mode: Literal["window", "kv_cache"],
) -> None:
    torch.manual_seed(29)
    first_policy = policy(history_mode=history_mode)
    torch.manual_seed(29)
    second_policy = policy(history_mode=history_mode)
    runtime = RuntimeConfig(1, (0,))
    first_policy.prepare_prediction(runtime, 4, 2)
    second_policy.prepare_prediction(runtime, 4, 2)
    scheduler = ActionScheduler(FrameTiming(0, 0, 2, 2, 4), first_policy.context_frames, generation=1)

    for frame in range(20):
        item = make_warmup_observations(first_policy.spec, 1, 0, frame, 0)[0]
        item = replace(
            item,
            reset=frame == 0,
            applied_action=scheduler.submitted.get(frame, NEUTRAL_CONTROLLER_ACTION),
        )
        assert scheduler.observe(item)
        request = scheduler.request_plan()
        if request is not None:
            first_plan = first_policy.predict((request,))[0]
            second_plan = second_policy.predict((request,))[0]
            assert first_plan == second_plan
            assert scheduler.accept_plan(first_plan, frame)
        scheduler.action_to_submit(frame)
