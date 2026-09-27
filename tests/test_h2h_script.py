"""Explicit artifact loading and shared-checkpoint H2H batching."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from hal.eval import h2h as evaluation
from hal.scripts import h2h


def test_nested_upload_prefix_must_be_empty_and_safe(monkeypatch) -> None:
    calls = []
    client = SimpleNamespace(list_objects_v2=lambda **values: calls.append(values) or {"KeyCount": 0})
    monkeypatch.setattr(h2h.r2, "client", lambda: client)
    monkeypatch.setattr(h2h.r2, "bucket", lambda: "hal")
    h2h._validate_empty_upload_prefix("eval-temp01", "runs/training-run/h2h-evals")
    assert calls == [{"Bucket": "hal", "Prefix": "runs/training-run/h2h-evals/eval-temp01/", "MaxKeys": 1}]
    with pytest.raises(ValueError, match="safe path components"):
        h2h._validate_empty_upload_prefix("eval-temp01", "runs/../h2h-evals")


@pytest.mark.parametrize("same_checkpoint", (True, False))
def test_h2h_prepares_one_policy_per_checkpoint_and_keeps_port_settings(monkeypatch, same_checkpoint) -> None:
    allocations = []
    prepared = []
    models = (
        evaluation.H2HModel("alpha", "a.hal", "IBDW#0", 17.0, 0.8),
        evaluation.H2HModel("beta", "b.hal", "ZAIN#0", 22.0, 1.1),
    )

    def read_artifact(path):
        return SimpleNamespace(
            checkpoint_sha256="a" * 64 if path.name == "a.hal" or same_checkpoint else "b" * 64,
            statistics={},
            vocabulary=SimpleNamespace(codes=()),
            return_p90=19.75,
            spec=SimpleNamespace(supported_transport_delays=(0, 2, 3)),
            capability_version=2,
        )

    class Registry:
        def register_artifact(self, artifact, **options):
            allocations.append((artifact.checkpoint_sha256, options))
            return object()

    class Policy:
        def __init__(self, _model, _statistics, _codes, **options):
            self.checkpoint_sha256 = options["checkpoint_sha256"]
            self.seeds = []

        def prepare_prediction(self, runtime, horizon, prefix):
            prepared.append((runtime.max_batch_size, horizon, prefix))

        def reset_prediction(self, *, seed=None):
            self.seeds.append(seed)

    class Adapter:
        def __init__(self, policy, runtime, timing, *, settings_by_port, observed_actions):
            self.policy = policy
            self.settings_by_port = settings_by_port
            self.runtime_spec = (4, 2, 0)

    monkeypatch.setattr(evaluation, "resolve_checkpoint", Path)
    monkeypatch.setattr(evaluation, "read_action_sequence_artifact", read_artifact)
    monkeypatch.setattr(evaluation, "ModelRegistry", Registry)
    monkeypatch.setattr(evaluation, "ActionSequencePolicy", Policy)
    monkeypatch.setattr(evaluation, "PolicyBatchAdapter", Adapter)
    factory = evaluation.prepare_h2h_policies(models, max_parallel=3, profile="local", device="cpu", compiled=False)
    policy_count = 1 if same_checkpoint else 2
    assert len(allocations) == policy_count
    assert prepared == [(6 if same_checkpoint else 3, 4, 0)] * policy_count
    router = factory({1: "alpha", 2: "beta"}, 123)
    assert (router.by_port[1] is router.by_port[2]) is same_checkpoint
    assert router.by_port[1].settings_by_port[1] == evaluation.PolicySettings("IBDW#0", 17.0, 0.8)
    assert router.by_port[2].settings_by_port[2] == evaluation.PolicySettings("ZAIN#0", 22.0, 1.1)
    mirror = factory({1: "beta", 2: "alpha"}, 456)
    assert mirror.by_port[1].settings_by_port[1].player_identity == "ZAIN#0"
    assert mirror.by_port[2].settings_by_port[2].player_identity == "IBDW#0"
    assert len(allocations) == policy_count
