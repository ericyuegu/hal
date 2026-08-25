import hashlib
import json
from dataclasses import dataclass
from dataclasses import replace
from types import SimpleNamespace

import melee
import pytest
import torch

from hal.eval import self_play
from hal.training.features import ITEM_COLUMNS
from hal.wire import ITEM_SLOTS
from hal.wire import item_column


@dataclass(frozen=True)
class _Config:
    exec_horizon: int = 6
    inference_mode: str = "compiled"
    eval_seed: int = 17
    observation_bundle: str = "base"
    L_ctx: int = 4


@dataclass(frozen=True)
class _ItemConfig(_Config):
    item_conditioning: bool = True


class _Inference:
    instances = []

    def __init__(self, model, cfg, *, bucket, compile_mode):
        self.model = model
        self.cfg = cfg
        self.bucket = bucket
        self.compile_mode = compile_mode
        self.decoded = []
        self.instances.append(self)

    def decode(self, context, horizon):
        self.decoded.append((context, horizon))


class _NativePrewarmInference(_Inference):
    def __init__(self, model, cfg, *, bucket, compile_mode):
        super().__init__(model, cfg, bucket=bucket, compile_mode=compile_mode)
        self.prewarmed = []

    def prewarm(self, horizon):
        self.prewarmed.append(horizon)
        return 12.5


@pytest.fixture
def benchmark_fakes(monkeypatch):
    configs = [
        SimpleNamespace(
            stage=melee.Stage.BATTLEFIELD,
            character_port_1=melee.Character.FOX,
            character_port_2=melee.Character.FALCO,
        )
        for _ in range(3)
    ]
    calls = {}

    monkeypatch.setattr(self_play, "mirrored_configs", lambda n: configs[:n])
    monkeypatch.setattr(
        self_play,
        "default_session_cfg",
        lambda replay_dir, *, instant_match_restart: (replay_dir, instant_match_restart),
    )

    def run_matches(session_cfg, matches, policy_factory, **kwargs):
        calls["session_cfg"] = session_cfg
        calls["matches"] = matches
        calls["kwargs"] = kwargs
        calls["policies"] = [policy_factory(), policy_factory()]
        return [[[0, 1, 2]], [[0, 1]], []]

    monkeypatch.setattr(self_play, "run_matches_vec", run_matches)
    return calls


def _benchmark(tmp_path, benchmark_fakes, inference_class, **kwargs):
    checkpoint = tmp_path / "final.pt"
    checkpoint.write_bytes(b"checkpoint")
    model = torch.nn.Linear(1, 1)
    loaded = []
    policies = []

    def load_checkpoint(path):
        loaded.append(path)
        return model, _Config(), {"stats": object()}, {"step": 123}

    def make_policy(model_arg, stats, cfg, **policy_kwargs):
        policies.append((model_arg, stats, cfg, policy_kwargs))
        return object()

    metrics = self_play.benchmark_checkpoint(
        str(checkpoint),
        load_checkpoint=load_checkpoint,
        make_inference=inference_class,
        make_policy=make_policy,
        **kwargs,
    )
    return checkpoint, loaded, policies, metrics


def test_benchmark_checkpoint_owns_the_self_play_flow(tmp_path, benchmark_fakes) -> None:
    _Inference.instances.clear()
    checkpoint, loaded, policies, metrics = _benchmark(
        tmp_path,
        benchmark_fakes,
        _Inference,
        n_matches=3,
        max_frames=99,
        eager=True,
        instant_match_restart=True,
        process_cohorts=2,
    )

    inference = _Inference.instances[-1]
    assert loaded == [str(checkpoint)]
    assert inference.bucket == 8
    assert inference.compile_mode == "default"
    assert inference.cfg.inference_mode == "eager"
    assert [horizon for _, horizon in inference.decoded] == [6, 6]
    assert all(context.ctx_pad.shape == (8,) for context, _ in inference.decoded)
    assert [policy[3]["decode_seed"] for policy in policies] == [17, 18]
    assert all(policy[3]["inference"] is inference for policy in policies)
    assert all(isinstance(policy[3]["telemetry"], self_play.DecodeTelemetry) for policy in policies)

    assert len(benchmark_fakes["matches"]) == 3
    assert all(match.model_ports == (1, 2) for match in benchmark_fakes["matches"])
    assert benchmark_fakes["kwargs"]["max_frames"] == 99
    assert benchmark_fakes["kwargs"]["max_parallel"] == 4
    assert benchmark_fakes["kwargs"]["start_retries"] == 0
    assert benchmark_fakes["kwargs"]["process_cohorts"] == 2
    assert benchmark_fakes["session_cfg"][1] is True

    output = tmp_path / "self_play_benchmark_3x99_instant_restart_c2" / "metrics.json"
    payload = json.loads(output.read_text())
    assert payload["checkpoint"] == str(checkpoint.resolve())
    assert payload["checkpoint_sha256"] == hashlib.sha256(b"checkpoint").hexdigest()
    assert payload["process_cohorts"] == 2
    assert metrics["checkpoint_step"] == 123.0
    assert metrics["completed_workers"] == 2.0
    assert metrics["captured_frames"] == 5.0


def test_benchmark_uses_native_prewarm_and_numbered_output_directories(tmp_path, benchmark_fakes) -> None:
    _NativePrewarmInference.instances.clear()
    options = {"n_matches": 1, "max_frames": 2}
    _benchmark(tmp_path, benchmark_fakes, _NativePrewarmInference, **options)
    _, _, _, metrics = _benchmark(tmp_path, benchmark_fakes, _NativePrewarmInference, **options)

    inference = _NativePrewarmInference.instances[-1]
    assert inference.prewarmed == [6]
    assert inference.decoded == []
    assert metrics["compile_seconds"] == 12.5
    base = "self_play_benchmark_1x2_single_match"
    assert (tmp_path / base / "metrics.json").exists()
    output = tmp_path / f"{base}_run02" / "metrics.json"
    assert output.exists()
    assert "process_cohorts" not in json.loads(output.read_text())


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"n_matches": 0}, "n_matches must be >= 1"),
        ({"max_frames": 1}, "max_frames must be >= 2"),
    ],
)
def test_benchmark_rejects_invalid_workloads(options, message) -> None:
    def unused(*args, **kwargs):
        raise AssertionError("validation should happen before loading")

    with pytest.raises(ValueError, match=message):
        self_play.benchmark_checkpoint(
            "unused.pt",
            load_checkpoint=unused,
            make_inference=unused,
            make_policy=unused,
            **options,
        )


def test_decode_telemetry_metrics() -> None:
    telemetry = self_play.DecodeTelemetry()
    telemetry.record(rows=2, horizon=4, seconds=0.05)
    telemetry.record(rows=6, horizon=4, seconds=0.15)

    metrics = telemetry.metrics()
    assert metrics["decode_calls"] == 2.0
    assert metrics["decode_rows"] == 8.0
    assert metrics["decode_executed_frames_per_s"] == pytest.approx(160.0)
    assert metrics["decode_calls_over_100ms"] == 1.0


def test_canonical_context_matches_prewarm_feature_keys() -> None:
    cfg = _Config()
    synthetic = self_play.synthetic_context(cfg, 2, torch.device("cpu"))
    expected = self_play.canonical_context(synthetic, cfg.observation_bundle)
    dropped = [name for name in synthetic.features if name.endswith("_mask")][:6]
    live = replace(
        synthetic,
        features={name: synthetic.features[name] for name in reversed(synthetic.features) if name not in dropped},
    )

    actual = self_play.canonical_context(live, cfg.observation_bundle)

    assert list(actual.features) == list(expected.features)
    for name in dropped:
        assert torch.equal(actual.features[name], torch.zeros_like(expected.features[name]))


def test_item_conditioning_carries_the_projectile_block() -> None:
    """A config that opts in gets every item slot, with the float mask sidecars the
    live decode can omit. A config that does not stays byte-identical."""
    cfg = _ItemConfig()
    context = self_play.synthetic_context(cfg, 2, torch.device("cpu"))

    base = self_play.synthetic_context(_Config(), 2, torch.device("cpu"))
    assert [name for name in base.features if name.startswith("item")] == []
    for slot in range(ITEM_SLOTS):
        for name in ITEM_COLUMNS.cats:
            assert context.features[item_column(slot, name)].dtype == torch.long
        for name in ITEM_COLUMNS.floats:
            column = item_column(slot, name)
            assert context.features[column].shape == (2, cfg.L_ctx)
            assert context.features[f"{column}_mask"].shape == (2, cfg.L_ctx)

    absent = f"{item_column(0, next(iter(ITEM_COLUMNS.floats)))}_mask"
    live = replace(context, features={k: v for k, v in context.features.items() if k != absent})
    assert absent not in self_play.canonical_context(live, cfg.observation_bundle).features
    filled = self_play.canonical_context(live, cfg.observation_bundle, items=True)
    assert list(filled.features) == sorted(context.features)
    assert torch.equal(filled.features[absent], torch.zeros_like(context.features[absent]))
