"""Checkpoint/source adapters in the torch-carrying H2H CLI."""

from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hal.scripts import h2h


class _CapturedPolicy:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


def _fake_historical_module() -> ModuleType:
    module = ModuleType("fake_historical_013")
    module.torch = torch
    module.RecedingHorizon = _CapturedPolicy
    module.__hal_source__ = {"commit": "a" * 40, "blob": "b" * 40, "path": "experiments/013.py"}

    def decode(model, ctx, *, temp, gen):
        del model, ctx, temp
        sample = torch.randint(0, 2**20, (1,), generator=gen, dtype=torch.int64)
        return sample.to(torch.float32).reshape(1, 1, 1)

    module.decode = decode
    model = torch.nn.Linear(1, 1, bias=False)
    cfg = SimpleNamespace(decode_temp=1.0, L_ctx=51)
    module._load_ckpt = lambda _path: (model, cfg, {"stat": object()}, {"step": 16_384})
    return module


def test_historical_builder_records_source_and_uses_private_seeded_rng(monkeypatch) -> None:
    module = _fake_historical_module()
    monkeypatch.setattr(h2h, "import_experiment", lambda _spec: module)
    build, protocol = h2h.load_policy_builder(
        h2h.ModelArgs(
            name="013",
            checkpoint="checkpoint.pt",
            experiment="git:wandb-sha:experiments/013_interleaved_groups.py",
            family="historical_interleaved_groups",
        )
    )

    assert protocol["source"] == module.__hal_source__
    assert protocol["trunk_passes_per_action"] == 4
    assert protocol["exec_horizon"] == 1
    assert protocol["decode_settings"] == {"temp": 1.0}

    reference = build(19)
    expected = [reference.predict_chunk(None, None) for _ in range(2)]
    policy = build(19)
    opponent = build(19)
    actual = [policy.predict_chunk(None, None)]
    opponent.predict_chunk(None, None)
    opponent.predict_chunk(None, None)
    actual.append(policy.predict_chunk(None, None))
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(got, want)


def test_git_experiment_import_resolves_commit_and_blob_without_checkout() -> None:
    module = h2h.import_experiment("git:e8a1b5b:experiments/013_interleaved_groups.py")
    source = module.__hal_source__
    assert source["commit"] == "e8a1b5bf371e845582a143524b97eaf42c94d658"
    assert len(source["blob"]) == 40
    assert source["path"] == "experiments/013_interleaved_groups.py"
    assert module.N_GROUPS == 4
    assert module.TrainConfig().L_ctx == 256


def test_real_026_checkpoint_loads_through_temporal_mtp_builder() -> None:
    checkpoint = Path(
        "runs/260810-071709_026_temporal_mtp_mtp026-d384-L8-h6-Lc128-t128x2-"
        "o1-2-3-4-5-6-9-12-16-20-s4-base_ranked-anon-1_production-seed0-d384-b512/final.pt"
    )
    if not checkpoint.is_file():
        pytest.skip("production 026 checkpoint is not present")
    build, protocol = h2h.load_policy_builder(
        h2h.ModelArgs(
            name="026",
            checkpoint=str(checkpoint),
            experiment="experiments/026_temporal_mtp.py",
            family="temporal_mtp",
        )
    )
    policy = build(17)
    assert protocol["decode_settings"] == {"temp": 1.0}
    assert protocol["exec_horizon"] == 4
    assert policy.runtime_spec.context_frames == 128
    assert policy.runtime_spec.prediction_frames == 4
