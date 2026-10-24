"""Fixed-weight gradient controls must use complete, disjoint recorded batches."""

import dataclasses
import json
from pathlib import Path

import pytest
import torch
from test_065_loop import PROVENANCE
from test_065_loop import exp
from test_065_loop import transformer_trainer

from hal.models.replay_actor import ActorTargets
from hal.models.replay_actor import actor_loss
from hal.models.window_replay import ReplaySlots
from hal.models.window_replay import replay_windows
from scripts.o65_gradient_noise import accumulate
from scripts.o65_gradient_noise import comparison
from scripts.o65_gradient_noise import load_chunk
from scripts.o65_gradient_noise import load_manifest


def test_capture_covers_every_batch_and_preserves_loss_and_gradient(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = transformer_trainer(
        "cpu", development=True, decoder_precision="fp32", matches=(0, 1, 2, 3), episode_length=10000
    )
    run.config = dataclasses.replace(run.config, windows_per_step=4, decisions_per_window="all")
    batch = run.prepare(run.engine.collect(0))
    run.update = 1
    output = tmp_path / "inputs"
    before = exp._state_hash(run.policy.model.state_dict())
    exp.save_update_gradient_inputs(run, batch, output, PROVENANCE)
    manifest = load_manifest(output, PROVENANCE["checkpoint_sha256"])
    assert manifest["decoder_precision"] == "fp32"
    windows = [window for step in manifest["steps"] for window in step]
    assert len(windows) == len(set(windows)) == int((batch.placement.selected.any(1)).sum())
    assert all(len(step) == 4 for step in manifest["steps"][:-1])
    count = 0
    first = next(iter(manifest["chunks"]))
    saved = load_chunk(output / first, len(manifest["steps"]))
    slots = ReplaySlots(*saved["slots"])
    targets = ActorTargets(*saved["targets"])
    context, observed = saved["context"], saved["observed"]
    tokens = run.policy.model.context_tokens(context, observed)
    hidden = run.policy.model.trunk(tokens, torch.zeros(len(tokens), dtype=torch.long))
    actual = replay_windows(run.policy.model, hidden, observed, slots, grad=True)
    loss = actor_loss(actual, targets).loss
    parameters = run.policy.actor_parameters()
    gradient = torch.zeros(sum(parameter.numel() for parameter in parameters))
    expected_values = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    accumulate(loss, parameters, gradient, retain_graph=False)
    expected = torch.cat(
        [
            (torch.zeros_like(parameter) if value is None else value).flatten()
            for parameter, value in zip(parameters, expected_values, strict=True)
        ]
    )
    torch.testing.assert_close(gradient, expected, atol=0, rtol=0)
    assert comparison(gradient, expected)["cosine"] == pytest.approx(1)
    for name in manifest["chunks"]:
        chunk = load_chunk(output / name, len(manifest["steps"]))
        count += int(ActorTargets(*chunk["targets"]).selected_WK.sum())
    assert count == manifest["denominator"] == int(batch.selected.sum())
    assert exp._state_hash(run.policy.model.state_dict()) == before
    recorded: list[tuple[tuple[torch.Tensor, ...], exp.ActorTargets]] = []
    original = run.policy.actor_step

    def record(
        windows: exp.ReplayBatch, targets: exp.ActorTargets, *, clip_epsilon: float
    ) -> tuple[exp.ActorLoss, exp.ReplayOutput]:
        recorded.append((windows.inputs(), targets))
        return original(windows, targets, clip_epsilon=clip_epsilon)

    monkeypatch.setattr(run.policy, "actor_step", record)
    run.fit(batch)
    assert len(recorded) == len(manifest["chunks"])
    for name, (inputs, expected_targets) in zip(manifest["chunks"], recorded, strict=True):
        chunk = load_chunk(output / name, len(manifest["steps"]))
        for saved_value, current_value in zip(chunk["slots"], inputs[5:], strict=True):
            torch.testing.assert_close(saved_value, current_value, atol=0, rtol=0)
        for field in dataclasses.fields(ActorTargets):
            saved_value = getattr(ActorTargets(*chunk["targets"]), field.name)
            current_value = getattr(expected_targets, field.name)
            torch.testing.assert_close(saved_value, current_value, atol=0, rtol=0)
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        load_manifest(output, "different")
    for precision in ("fp32", "bf16"):
        manifest["decoder_precision"] = precision
        (output / "manifest.json").write_text(json.dumps(manifest))
        assert load_manifest(output, PROVENANCE["checkpoint_sha256"])["decoder_precision"] == precision
    manifest["decoder_precision"] = "invalid"
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="schema or checkpoint mismatch"):
        load_manifest(output, PROVENANCE["checkpoint_sha256"])
    manifest["decoder_precision"] = "fp32"
    manifest["steps"][1][0] = manifest["steps"][0][0]
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="disjoint"):
        load_manifest(output, PROVENANCE["checkpoint_sha256"])


def test_gradient_comparison_has_a_known_scale() -> None:
    first = torch.tensor([1.0, 0.0])
    second = torch.tensor([0.0, 1.0])
    assert comparison(first, second)["cosine"] == 0
    assert comparison(first, second)["relative_l2"] == pytest.approx(2**0.5)
    assert comparison(first, torch.zeros(2))["cosine"] is None
    with pytest.raises(ValueError, match="NaN"):
        comparison(first, torch.tensor([float("nan"), 0.0]))


@pytest.mark.parametrize("rollout,nsys,warm", [(True, False, 1), (False, True, 1)])
def test_update_capture_requires_one_full_first_update(rollout: bool, nsys: bool, warm: int) -> None:
    with pytest.raises(ValueError, match="training input capture"):
        exp.ProfileConfig(
            warm_updates=warm, rollout_only=rollout, nsys=nsys, update_gradient_inputs=Path("/cache/gradients")
        )


def test_update_capture_allows_warming_every_group() -> None:
    exp.ProfileConfig(warm_updates=4, update_gradient_inputs=Path("/cache/gradients"))


def test_profile_counts_only_finished_games() -> None:
    run = transformer_trainer("cpu", decoder_precision="fp32")
    rollout = run.engine.collect(0)
    episode = run.engine.matches[0].episode
    results = tuple(
        exp.EpisodeResult(0, episode, duration, (0.0, 0.0), timeout, None)
        for duration, timeout in ((100, False), (200, False), (300, True))
    )
    metrics = exp.profile_rollout_metrics(dataclasses.replace(rollout, results=results))
    assert metrics["profile/finished_games"] == 2
    assert metrics["profile/finished_game_frames"] == 300
    metrics = exp.profile_rollout_metrics(dataclasses.replace(rollout, results=()))
    assert metrics["profile/finished_games"] == metrics["profile/finished_game_frames"] == 0
