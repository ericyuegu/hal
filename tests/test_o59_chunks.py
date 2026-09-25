"""O59 uses contiguous trained heads without changing checkpoint configuration."""

from dataclasses import replace

import torch

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.data.feature_stats import FeatureStats
from hal.inference.api import RuntimeConfig
from hal.inference.chunks import ChunkRequest
from hal.inference.o59 import _MODEL_FIELDS
from hal.inference.o59 import O59Policy
from hal.inference.o59_model import GPT
from hal.inference.o59_model import Architecture
from hal.inference.o59_model import TrainConfig
from hal.training.ego_stats import consolidate_key
from hal.training.features import ITEM_COLUMNS
from hal.training.features import feature_kind


def policy() -> O59Policy:
    arch = replace(
        Architecture(),
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
    cfg = TrainConfig(arch=arch)
    model = GPT(cfg).eval()
    stats = {
        consolidate_key(name): FeatureStats(0, 1, -10, 10)
        for name in _MODEL_FIELDS
        if feature_kind(name, ITEM_COLUMNS) not in ("cat", "button", "stick_trigger")
    }
    return O59Policy(model, cfg, stats, (), device=torch.device("cpu"), seed=5, compiled=False)


def test_o59_chunk_and_existing_plan_decode_identically() -> None:
    torch.manual_seed(5)
    old = policy()
    torch.manual_seed(5)
    chunks = policy()
    runtime = RuntimeConfig(1, (2,))
    old.prepare(runtime)
    chunks.prepare_chunks(runtime, 4, 2)
    context = old.warmup_context(0, 7, 2)
    for item in context:
        stream = old._ingest(item)
    old._plan(context[-1], stream)
    request = ChunkRequest(0, 1, 0, 7, context, (NEUTRAL_CONTROLLER_ACTION,) * 2)
    response = chunks.plan_chunks((request,))[0]
    assert tuple(item.action for item in response.actions[2:]) == tuple(stream.queued)


def test_o59_can_decode_twelve_contiguous_heads_and_reset_calibration_state() -> None:
    chunks = policy()
    assert chunks.supported_horizons == tuple(range(1, 13))
    chunks.prepare_chunks(RuntimeConfig(1, (2,)), 12, 4)
    request = ChunkRequest(0, 1, 0, 7, chunks.warmup_context(0, 7, 2), (NEUTRAL_CONTROLLER_ACTION,) * 4)
    first = chunks.plan_chunks((request,))[0]
    assert len(first.actions) == 12 and first.actions[4].target_frame == 12
    assert (chunks.cfg.prediction_frames, chunks.cfg.delay_frames, chunks.cfg.replan_interval_frames) == (4, 2, 2)
    chunks.reset_chunks()
    assert chunks.plan_chunks((request,))[0] == first
