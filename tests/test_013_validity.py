"""Validity and default-parity guards for experiment 013."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from hal.training.features import A_DIM
from hal.training.features import ACTION_CHANNELS
from hal.training.features import CAT_FEATURES
from hal.training.features import FLOAT_FEATURES
from hal.training.features import Context
from hal.training.features import TrainBatch

_EXP_DIR = Path(__file__).resolve().parent.parent / "experiments"


def _load_experiment(filename: str):
    spec = importlib.util.spec_from_file_location(filename.split(".")[0], _EXP_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


exp012 = _load_experiment("012_multi_token.py")
exp013 = _load_experiment("013_multi_token.py")


def _zeros_features(exp, batch_size: int, length: int) -> dict[str, torch.Tensor]:
    features: dict[str, torch.Tensor] = {}
    for prefix in exp._PLAYER_PREFIXES:
        for feat in FLOAT_FEATURES:
            features[f"{prefix}_{feat}"] = torch.zeros(batch_size, length)
        for name in CAT_FEATURES:
            features[f"{prefix}_{name}"] = torch.zeros(batch_size, length, dtype=torch.long)
    for ch in ACTION_CHANNELS:
        features[f"ego_{ch}"] = torch.zeros(batch_size, length)
    for key in ("ego_character", "opp_character", "stage"):
        features[key] = torch.zeros(batch_size, length, dtype=torch.long)
    return features


def _batch(exp, *, batch_size: int = 2, length: int = 4, target_length: int = 2) -> TrainBatch:
    features = _zeros_features(exp, batch_size, length)
    target = torch.zeros(batch_size, target_length, A_DIM)
    target[:, :, 0] = 0.25
    target[:, :, ACTION_CHANNELS.index("button_a")] = 1.0
    return TrainBatch(
        context=Context(features=features, ctx_pad=torch.tensor([0, 1][:batch_size], dtype=torch.long)),
        target=target,
    )


def _tiny_cfg(exp, **kwargs):
    defaults = dict(
        d_model=64,
        n_layers=1,
        n_heads=2,
        L_ctx=4,
        head_offsets=(1, 2),
        max_steps=8,
        warmup_steps=2,
        batch_size=2,
        val_n_batches=1,
    )
    return exp.TrainConfig(**{**defaults, **kwargs})


def test_default_model_loss_and_decode_match_012_exactly():
    """013's validity fixes must not alter the default model or learning objective."""
    cfg012 = _tiny_cfg(exp012)
    cfg013 = _tiny_cfg(exp013)
    torch.manual_seed(7)
    model012 = exp012.GPT(cfg012).eval()
    torch.manual_seed(7)
    model013 = exp013.GPT(cfg013).eval()
    exp013._load_model_state(model013, model012.state_dict())
    batch = _batch(exp013)

    nll012, trans012 = exp012.action_loss(model012, batch)
    nll013, trans013 = exp013.action_loss(model013, batch)
    assert nll012.keys() == nll013.keys()
    for key in nll012:
        assert torch.equal(nll012[key], nll013[key])
        assert torch.equal(trans012[key], trans013[key])
    assert torch.equal(
        exp012.objective(nll012, trans012, cfg012.aux_loss_weight, cfg012.transition_loss_weight),
        exp013.objective(nll013, trans013, cfg013.aux_loss_weight, cfg013.transition_loss_weight),
    )
    assert torch.equal(
        exp012.decode(model012, batch.context, argmax=True),
        exp013.decode(model013, batch.context, argmax=True),
    )


def test_validate_config_rejects_invalid_rotary_and_execution_geometry():
    with pytest.raises(ValueError, match="head_dim"):
        exp013.validate_config(
            _tiny_cfg(exp013, d_model=30, n_heads=2),
            has_button_combo_counts=False,
        )
    with pytest.raises(ValueError, match="contiguous prefix"):
        exp013.validate_config(
            _tiny_cfg(exp013, head_offsets=(1, 5), exec_horizon=2),
            has_button_combo_counts=False,
        )


def test_action_loss_rejects_a_target_shorter_than_the_farthest_head():
    model = exp013.GPT(_tiny_cfg(exp013)).eval()
    with pytest.raises(ValueError, match="target horizon"):
        exp013.action_loss(model, _batch(exp013, target_length=1))


def test_button_count_artifact_is_dataset_scoped_and_checkpointed(tmp_path: Path):
    data_root = tmp_path / "dataset"
    data_root.mkdir()
    counts = [0] * exp013.scoring.N_BUTTON_COMBOS
    counts[0] = 7
    artifact = tmp_path / "button_counts.json"
    artifact.write_text(json.dumps({"version": 1, "data_root": str(data_root), "total_frames": 7, "counts": counts}))
    cfg = _tiny_cfg(exp013, data_root=str(data_root), button_combo_counts_path=str(artifact))
    loaded = exp013._load_button_combo_counts(cfg)
    assert loaded is not None and loaded.tolist() == counts

    model = exp013.GPT(cfg)
    model.button_combo_counts.copy_(loaded)
    assert torch.equal(model.state_dict()["button_combo_counts"], loaded)
    dead = exp013._btn_support_dead(model, min_count=1, device=torch.device("cpu"))
    assert not bool(dead[0]) and bool(dead[1:].all())

    bad = json.loads(artifact.read_text())
    bad["data_root"] = str(tmp_path / "different-dataset")
    artifact.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="does not match"):
        exp013._load_button_combo_counts(cfg)


def test_support_masking_fails_without_dataset_counts():
    cfg = _tiny_cfg(exp013, decode_btn_support_min=1)
    with pytest.raises(ValueError, match="requires button_combo_counts_path"):
        exp013.validate_config(cfg, has_button_combo_counts=False)


@pytest.mark.parametrize("n_matchups", [16, 96])
def test_eval_matchup_count_is_independent_of_cpu_concurrency(monkeypatch, n_matchups: int):
    cfg = _tiny_cfg(exp013, eval_parallel_per_cpu=0.5)
    model = exp013.GPT(cfg).eval()
    captured: dict[str, int] = {}

    def fake_sweep(_policy_factory, **kwargs):
        captured["n_matchups"] = kwargs["n_matchups"]
        captured["max_parallel"] = kwargs["max_parallel"]
        return []

    monkeypatch.setattr(exp013.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(exp013, "default_session_cfg", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(exp013, "sweep_vs_cpu_prior", fake_sweep)
    monkeypatch.setattr(exp013, "vs_cpu_metrics", lambda _results, *, seed: {"seed": float(seed)})

    result = exp013.eval_vs_cpu(model, {}, cfg, max_frames=10, n_matchups=n_matchups)
    assert captured == {"n_matchups": n_matchups, "max_parallel": 4}
    assert result == {"seed": float(cfg.eval_seed)}


def test_policy_sampling_is_repeatable_for_an_explicit_seed():
    cfg = _tiny_cfg(exp013)
    model = exp013.GPT(cfg).eval()
    context = _batch(exp013).context
    policy_a = exp013.make_policy(model, {}, cfg, device="cpu", decode_seed=19)
    policy_b = exp013.make_policy(model, {}, cfg, device="cpu", decode_seed=19)
    assert (policy_a.predict_chunk(context, None) == policy_b.predict_chunk(context, None)).all()


def test_reconstruction_uses_every_configured_decode_knob(monkeypatch):
    cfg = _tiny_cfg(exp013)
    model = exp013.GPT(cfg).eval()
    batch = _batch(exp013)
    seen: list[dict] = []

    def fake_decode(_model, ctx, **kwargs):
        seen.append(kwargs)
        return torch.zeros(ctx.ctx_pad.shape[0], 1, A_DIM)

    monkeypatch.setattr(exp013, "decode", fake_decode)
    gen = torch.Generator().manual_seed(3)
    exp013.recon_metrics(
        model,
        [batch],
        argmax=False,
        temp=0.7,
        temps=(0.8, 0.9, 1.0, 1.1),
        btn_support_min=12,
        min_p=0.05,
        click_trigger_fix=True,
        gen=gen,
    )
    assert seen == [
        {
            "temp": 0.7,
            "temps": (0.8, 0.9, 1.0, 1.1),
            "btn_support_min": 12,
            "min_p": 0.05,
            "click_trigger_fix": True,
            "argmax": False,
            "gen": gen,
        }
    ]
