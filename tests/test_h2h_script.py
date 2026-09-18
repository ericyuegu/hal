"""Checkpoint/source adapters in the torch-carrying H2H CLI."""

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hal.scripts import h2h
from hal.training.player_identity import encode_player_codes


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
    monkeypatch.setattr(h2h, "checkpoint_sha256", lambda _path: "a" * 64)
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


@dataclass(frozen=True)
class _O50Architecture:
    L_ctx: int = 256
    head_offsets: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 9, 12)


@dataclass(frozen=True)
class _O50Config:
    arch: _O50Architecture = _O50Architecture()
    prediction_frames: int = 4
    delay_frames: int = 2
    replan_interval_frames: int = 2


class _FakeO50(torch.nn.Module):
    def __init__(self, cfg: _O50Config) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.register_buffer(
            "player_code_bytes",
            torch.from_numpy(np.frombuffer(encode_player_codes(("IBDW#0",)), dtype=np.uint8).copy()),
        )
        self.cfg = cfg
        self.temporal = SimpleNamespace(live_horizons=(cfg.prediction_frames,))


def test_o50_builder_applies_identity_and_dense_deployment_timing(monkeypatch) -> None:
    module = ModuleType("fake_o50")
    cfg = _O50Config()
    model = _FakeO50(cfg)
    module.load_checkpoint = lambda _path: (model, cfg, {}, {"step": 99})

    def make_policy(_model, _stats, _cfg, *, decode_temperature=1.0, action_trace=None, **kwargs):
        return _CapturedPolicy(decode_temperature=decode_temperature, action_trace=action_trace, **kwargs)

    module.make_policy = make_policy
    monkeypatch.setattr(h2h, "import_experiment", lambda _spec: module)
    monkeypatch.setattr(h2h, "checkpoint_sha256", lambda _path: "a" * 64)

    trace = object()
    build, protocol = h2h.load_policy_builder(
        h2h.ModelArgs(
            name="IBDW#0",
            checkpoint="checkpoint.pt",
            experiment="experiments/050_scaled_temporal_awr.py",
            family="temporal_mtp",
            player_identity="IBDW#0",
            prediction_frames=6,
            delay_frames=2,
            replan_interval_frames=3,
            temperature=0.1,
        ),
        action_trace=trace,
    )
    policy = build(17)

    assert protocol["player_identity"] == "IBDW#0"
    assert protocol["prediction_frames"] == 6
    assert protocol["delay_frames"] == 2
    assert protocol["replan_interval_frames"] == 3
    assert protocol["decode_settings"] == {"temp": 0.1}
    assert protocol["action_trace_schema_version"] == 1
    assert model.cfg.prediction_frames == 6
    assert model.temporal.live_horizons == (6,)
    assert policy.ego_player_id == 4
    assert policy.delay_frames == 2
    assert policy.replan_interval_frames == 3
    assert policy.decode_temperature == 0.1
    assert policy.action_trace is trace


def test_nested_upload_prefix_must_be_empty_and_safe(monkeypatch) -> None:
    calls = []
    client = SimpleNamespace(
        list_objects_v2=lambda **kwargs: calls.append(kwargs) or {"KeyCount": 0},
    )
    monkeypatch.setattr(h2h.r2, "client", lambda: client)
    monkeypatch.setattr(h2h.r2, "bucket", lambda: "hal")

    h2h._require_empty_upload_prefix("eval-temp01", "runs/training-run/h2h-evals")

    assert calls == [
        {
            "Bucket": "hal",
            "Prefix": "runs/training-run/h2h-evals/eval-temp01/",
            "MaxKeys": 1,
        }
    ]
    with pytest.raises(ValueError, match="safe path components"):
        h2h._require_empty_upload_prefix("eval-temp01", "runs/../h2h-evals")


def test_o50_builder_rejects_sparse_heads_as_dense_horizon(monkeypatch) -> None:
    module = ModuleType("fake_o50")
    cfg = _O50Config()
    model = _FakeO50(cfg)
    module.load_checkpoint = lambda _path: (model, cfg, {}, {"step": 99})
    module.make_policy = lambda *_args, **_kwargs: None
    monkeypatch.setattr(h2h, "import_experiment", lambda _spec: module)
    monkeypatch.setattr(h2h, "checkpoint_sha256", lambda _path: "a" * 64)

    with pytest.raises(ValueError, match="requires dense heads"):
        h2h.load_policy_builder(
            h2h.ModelArgs(
                name="IBDW#0",
                checkpoint="checkpoint.pt",
                experiment="experiments/050_scaled_temporal_awr.py",
                family="temporal_mtp",
                prediction_frames=8,
            )
        )


def test_conditioned_o52_builder_uses_portable_transport_path(monkeypatch) -> None:
    prepared = []
    built = []

    class FakePolicy:
        def prepare(self, runtime) -> None:
            prepared.append(runtime)

    source = SimpleNamespace(
        config=SimpleNamespace(
            experiment_id="052_adamw_temporal_awr_v1",
            prediction_frames=4,
            architecture=SimpleNamespace(L_ctx=256, head_offsets=(1, 2, 3, 4, 5, 6, 9, 12)),
        ),
        transport_delay=2,
        replan_interval_frames=2,
        source_sha256="a" * 64,
        step=16_383,
        model=torch.nn.Linear(1, 1),
        player_id=lambda identity: 0 if identity is None else 4,
    )

    def new_policy(**kwargs):
        built.append(kwargs)
        return FakePolicy()

    source.new_policy = new_policy
    monkeypatch.setattr(h2h, "load_o50_checkpoint", lambda *_args, **_kwargs: source)
    monkeypatch.setattr(h2h, "import_experiment", lambda _spec: pytest.fail("legacy wrapper was imported"))
    monkeypatch.setattr(h2h.torch.cuda, "is_available", lambda: False)

    build, protocol = h2h.load_policy_builder(
        h2h.ModelArgs(
            name="052",
            checkpoint="control.pt",
            experiment="experiments/052_adamw_temporal_awr.py",
            family="conditioned_temporal_mtp",
        ),
        max_batch_size=8,
    )
    adapter = build(19)

    assert isinstance(adapter, h2h.PolicyBatchAdapter)
    assert prepared[0].max_batch_size == 8
    assert prepared[0].transport_delays == (2,)
    assert prepared[0].replan_interval_frames == 2
    assert built == [
        {
            "seed": 19,
            "compiled": False,
            "allow_masked_player_identity": True,
            "name": "052",
        }
    ]
    assert protocol["pending_prefix_conditioned"] is True
    assert protocol["transport_semantics"] == "conditioned_pending_actions_v1"
    assert protocol["forced_prefix_consumes_sampling_draws"] is False


def test_conditioned_builder_rejects_horizon_shorter_than_delay_and_cadence(monkeypatch) -> None:
    source = SimpleNamespace(
        config=SimpleNamespace(
            experiment_id="052_adamw_temporal_awr_v1",
            prediction_frames=4,
            architecture=SimpleNamespace(L_ctx=256, head_offsets=(1, 2, 3, 4)),
        ),
        transport_delay=2,
        replan_interval_frames=2,
        source_sha256="a" * 64,
        step=16_383,
        model=torch.nn.Linear(1, 1),
    )
    monkeypatch.setattr(h2h, "load_o50_checkpoint", lambda *_args, **_kwargs: source)
    monkeypatch.setattr(h2h.torch.cuda, "is_available", lambda: False)

    with pytest.raises(ValueError, match=r"3 \+ 2 > 4"):
        h2h.load_policy_builder(
            h2h.ModelArgs(
                name="052",
                checkpoint="control.pt",
                experiment="experiments/052_adamw_temporal_awr.py",
                family="conditioned_temporal_mtp",
                delay_frames=3,
                replan_interval_frames=2,
            )
        )
