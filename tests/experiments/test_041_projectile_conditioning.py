"""Contracts for the projectile-conditioned experiment.

The flag-off architecture must stay bit-identical to experiment 026, the pooled set
encoder must ignore which slots the live items occupy, the two observation paths must
deliver the same item tensors, and the configured replay format must be one that
carries the projectile block at all.
"""

import importlib.util
import math
import sys
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from types import ModuleType

import melee
import numpy as np
import pytest
import torch
from streaming import MDSWriter
from streaming import StreamingDataset
from torch import Tensor

from hal.data.feature_stats import FeatureStats
from hal.data.policy_world_schema import POLICY_WORLD_MDS_COLUMNS
from hal.data.policy_world_schema import encode_policy_world_replay
from hal.eval.self_play import synthetic_context
from hal.training.canonical import flatten_canonical_frame
from hal.training.closed_loop import _build_layout
from hal.training.closed_loop import _Rings
from hal.training.dataloader import make_loader
from hal.training.dataloader import relabel_ego
from hal.training.features import A_DIM
from hal.training.features import ACTION_CHANNELS
from hal.training.features import BASE_ACTION_PROJECTION
from hal.training.features import BASE_ITEMS_PROJECTION
from hal.training.features import ITEM_COLUMNS
from hal.training.features import ITEM_INPUT_COLUMNS
from hal.training.features import V6_PLAYER_COLUMNS
from hal.training.features import Context
from hal.training.features import preprocess
from hal.wire import ITEM_SLOTS
from hal.wire import MASK_INT32
from hal.wire import item_column


def _load(name: str, filename: str) -> ModuleType:
    """Experiments load by path: their filenames start with a digit."""
    path = Path(__file__).resolve().parents[2] / "experiments" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


exp = _load("test_exp041", "041_projectile_conditioning.py")
exp026 = _load("test_exp026_for_041", "026_temporal_mtp.py")

# A CPU-sized model. Every item width differs from every other so a mixed-up
# concatenation cannot pass by coincidence.
_TINY: dict[str, object] = {
    "d_model": 32,
    "n_layers": 1,
    "n_heads": 4,
    "L_ctx": 4,
    "temporal_d_model": 32,
    "temporal_layers": 1,
    "temporal_heads": 4,
    "temporal_ff_dim": 64,
    "group_head_dim": 64,
    "batch_size": 2,
    "reservoir_capacity": 4,
    "compile_trunk": False,
    "compile_temporal": False,
    "num_workers": 0,
    "push_to_r2": False,
    "inference_mode": "eager",
}
_TINY_ITEMS: dict[str, object] = {
    "item_type_dim": 6,
    "item_state_dim": 3,
    "item_hidden_dim": 8,
    "item_dim": 5,
}

_ITEM_CATS: tuple[str, ...] = tuple(ITEM_COLUMNS.cats)
_ITEM_FLOATS: tuple[str, ...] = tuple(ITEM_COLUMNS.floats)
_ITEM_CAT_COLUMNS = {item_column(slot, name) for slot in range(ITEM_SLOTS) for name in _ITEM_CATS}

EGO_PORT, OPP_PORT = 1, 2
EGO_PREFIX = "p1"
STAGE = int(melee.Stage.FINAL_DESTINATION.value)

# The local v7 subset the schema tests also read. Absent on a fresh checkout.
_V7_TRAIN = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "ranked-anonymized-1" / "mds-v7-sub4" / "train"
)


def _cfg(**overrides) -> exp.TrainConfig:
    return exp.TrainConfig(**{**_TINY, **_TINY_ITEMS, **overrides})


# --- flag-off parity ----------------------------------------------------------


def test_flag_off_architecture_equals_experiment_026() -> None:
    """With projectiles off, 041 builds nothing new: same parameters, same widths."""
    model = exp.GPT(_cfg(item_conditioning=False))
    baseline = exp026.GPT(exp026.TrainConfig(**_TINY))

    assert [name for name, _ in model.named_parameters() if "item" in name] == []
    got = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    expected = {name: tuple(value.shape) for name, value in baseline.state_dict().items()}
    assert got == expected
    assert model.ctx_proj.weight.shape == baseline.ctx_proj.weight.shape


def test_flag_on_adds_exactly_the_item_modules() -> None:
    cfg = _cfg()
    off = exp.GPT(_cfg(item_conditioning=False))
    on = exp.GPT(cfg)

    added = set(on.state_dict()) - set(off.state_dict())
    assert set(off.state_dict()) <= set(on.state_dict())
    assert added == {"item_type_emb.weight", "item_state_emb.weight", "item_up.weight", "item_down.weight"}
    assert on.item_type_emb.weight.shape == (256, cfg.item_type_dim)
    assert on.item_state_emb.weight.shape == (256, cfg.item_state_dim)
    # type + state embeddings, four floats, four sidecars, one presence flag.
    slot_width = cfg.item_type_dim + cfg.item_state_dim + 2 * len(_ITEM_FLOATS) + 1
    assert on.item_up.weight.shape == (cfg.item_hidden_dim, slot_width)
    assert on.item_down.weight.shape == (cfg.item_dim, cfg.item_hidden_dim)
    assert on.ctx_proj.weight.shape[1] == off.ctx_proj.weight.shape[1] + cfg.item_dim


def test_optimizer_and_counts_place_the_item_modules() -> None:
    cfg = _cfg()
    model = exp.GPT(cfg)
    optimizer = exp.make_optimizer(model, cfg)

    groups = {id(parameter): group for group in optimizer.param_groups for parameter in group["params"]}
    tables = (model.item_type_emb.weight, model.item_state_emb.weight)
    linears = (model.item_up.weight, model.item_down.weight)
    assert all(groups[id(w)]["weight_decay"] == 0.0 for w in tables)
    assert all(groups[id(w)]["weight_decay"] == cfg.adam_weight_decay for w in linears)
    assert all(not groups[id(w)]["use_muon"] for w in tables + linears)

    counts = exp.subsystem_parameter_counts(model)
    off_counts = exp.subsystem_parameter_counts(exp.GPT(_cfg(item_conditioning=False)))
    item_parameters = sum(w.numel() for w in tables + linears)
    assert counts["observation"] - off_counts["observation"] == item_parameters + cfg.item_dim * cfg.d_model


# --- the pooled set encoder ---------------------------------------------------


def _item_features(items: Mapping[int, Mapping[str, float]], *, masks: bool = True) -> dict[str, Tensor]:
    """Model-ready item columns for a ``[2, 3]`` batch: ``items`` maps a slot to its
    live projectile, and every other slot is the preprocessed empty form (id 0, zeroed
    floats, sidecar 1.0)."""
    shape = (2, 3)
    features: dict[str, Tensor] = {}
    for slot in range(ITEM_SLOTS):
        item = items.get(slot)
        for name in _ITEM_CATS:
            value = 0 if item is None else int(item[name])
            features[item_column(slot, name)] = torch.full(shape, value, dtype=torch.long)
        for name in _ITEM_FLOATS:
            column = item_column(slot, name)
            features[column] = torch.full(shape, 0.0 if item is None else float(item[name]))
            if masks:
                features[f"{column}_mask"] = torch.full(shape, 1.0 if item is None else 0.0)
    return features


_LASER = {"type": 6, "state": 2, "pos_x": 12.5, "pos_y": -3.25, "vel_x": 1.5, "vel_y": 0.0}
_TURNIP = {"type": 210, "state": 4, "pos_x": -40.0, "pos_y": 18.75, "vel_x": -0.5, "vel_y": 2.25}


def _pooled(model: torch.nn.Module, items: Mapping[int, Mapping[str, float]], *, masks: bool = True) -> Tensor:
    with torch.no_grad():
        return model._item_features(_item_features(items, masks=masks))


def test_pooling_is_permutation_invariant_over_the_slots() -> None:
    """A slot holds its item until an OLDER item despawns, so live items shift slots
    mid-match. The pooled value must not move with them."""
    torch.manual_seed(0)
    model = exp.GPT(_cfg()).eval()

    first = _pooled(model, {0: _LASER, 1: _TURNIP})
    permuted = _pooled(model, {3: _TURNIP, 1: _LASER})
    one_item = _pooled(model, {2: _LASER})

    assert first.shape == (2, 3, model.cfg.item_dim)
    assert torch.allclose(first, permuted, atol=1e-6)
    # Non-vacuity: the pooled value does depend on the set, just not on the slots.
    assert not torch.allclose(first, one_item, atol=1e-4)


def test_an_empty_frame_pools_to_exactly_zero() -> None:
    torch.manual_seed(0)
    model = exp.GPT(_cfg()).eval()

    pooled = _pooled(model, {})

    assert torch.equal(pooled, torch.zeros_like(pooled))


def test_a_missing_sidecar_reads_as_a_live_slot() -> None:
    """``preprocess`` emits ``{name}_mask`` only where a mask fires, so an absent
    sidecar means zero — a slot that holds an item."""
    torch.manual_seed(0)
    model = exp.GPT(_cfg()).eval()

    full = _pooled(model, {0: _LASER, 1: _TURNIP, 2: _LASER, 3: _TURNIP})
    without_sidecars = _pooled(model, {0: _LASER, 1: _TURNIP, 2: _LASER, 3: _TURNIP}, masks=False)

    assert torch.equal(full, without_sidecars)


def test_the_type_clamp_lands_unknown_ids_on_the_last_row() -> None:
    """The stored type is peppi's raw u16 id, which the routed table does not cover."""
    torch.manual_seed(0)
    model = exp.GPT(_cfg()).eval()

    unknown = _pooled(model, {0: {**_LASER, "type": 70_000}})
    last_row = _pooled(model, {0: {**_LASER, "type": model.item_type_emb.num_embeddings - 1}})

    assert torch.equal(unknown, last_row)


def test_context_tokens_accept_the_synthetic_item_context() -> None:
    cfg = _cfg()
    model = exp.GPT(cfg).eval()
    ctx = synthetic_context(cfg, 2, torch.device("cpu"))

    sidecars = {f"{item_column(slot, name)}_mask" for slot in range(ITEM_SLOTS) for name in ITEM_COLUMNS.floats}
    assert set(ctx.features) >= ITEM_INPUT_COLUMNS | sidecars
    with torch.no_grad():
        tokens = model.context_tokens(ctx.features)
    assert tokens.shape == (2, cfg.L_ctx, cfg.d_model)


def test_decode_asks_for_the_item_canonical_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """``canonical_context`` does not self-gate: a decode that forgets ``items`` would
    compile a second program on a different key set."""
    cfg = _cfg()
    model = exp.GPT(cfg).eval()
    seen: list[bool] = []
    real = exp.canonical_context

    def spy(ctx: Context, observation_bundle: str, *, items: bool = False) -> Context:
        seen.append(items)
        return real(ctx, observation_bundle, items=items)

    monkeypatch.setattr(exp, "canonical_context", spy)
    with torch.no_grad():
        chunk = exp.BF16Inference(model, cfg, compiled=False).decode(synthetic_context(cfg, 1, torch.device("cpu")), 4)

    assert seen == [True]
    assert chunk.shape[1:] == (4, A_DIM)


# --- routing ------------------------------------------------------------------


def test_one_routing_feeds_both_observation_paths() -> None:
    cfg = _cfg()
    model = exp.GPT(cfg).eval()

    assert exp._routing(cfg) == (ITEM_COLUMNS, BASE_ITEMS_PROJECTION)
    assert exp._routing(_cfg(item_conditioning=False)) == (None, BASE_ACTION_PROJECTION)
    assert exp._routing(_cfg(item_conditioning=False, observation_bundle="v6_lean")) == (V6_PLAYER_COLUMNS, None)

    loader = exp.loader_kwargs(cfg, {})
    policy = exp.make_policy(model, {}, cfg, device="cpu")
    assert (loader["extra"], loader["projection"]) == (ITEM_COLUMNS, BASE_ITEMS_PROJECTION)
    assert (policy.extra, policy.projection) == (ITEM_COLUMNS, BASE_ITEMS_PROJECTION)
    assert loader["replay_format"] == "policy-world"


def test_the_shipped_config_reads_the_only_source_that_stores_items() -> None:
    """The compact 'policy' decoder builds its dict from POLICY_MDS_COLUMNS, which has
    no item block, so the default source and format must be the policy-world pair."""
    cfg = exp.TrainConfig()

    assert cfg.item_conditioning
    assert cfg.replay_format == "policy-world"
    assert cfg.data_root.endswith("mds-policy-world-v7")


def test_item_conditioning_rejects_a_format_that_drops_items() -> None:
    with pytest.raises(ValueError, match="policy-world"):
        exp.validate_config(_cfg(replay_format="policy"))
    # The control arm reads the same source, so the comparison stays one-axis.
    exp.validate_config(_cfg(item_conditioning=False, replay_format="policy"))
    exp.validate_config(_cfg(replay_format="full"))
    with pytest.raises(ValueError, match="replay_format"):
        exp.validate_config(_cfg(replay_format="compact"))


def test_a_batch_without_item_columns_fails_loud() -> None:
    """A flag-on model handed an item-less observation must name the requirement rather
    than raise a bare KeyError deep inside the encoder."""
    model = exp.GPT(_cfg()).eval()
    item_less = synthetic_context(_cfg(item_conditioning=False), 2, torch.device("cpu")).features

    assert not any(name.startswith("item") for name in item_less)
    with pytest.raises(ValueError, match="policy-world"):
        model.context_tokens(item_less)


def _sliced_replay(source: Mapping[str, object], start: int, length: int) -> dict[str, object]:
    """One shorter replay: every per-frame column is cut, constants pass through."""
    frames = len(np.asarray(source["frame"]))
    out: dict[str, object] = {}
    for name, value in source.items():
        array = np.asarray(value)
        out[name] = array[start : start + length] if array.ndim == 1 and array.shape[0] == frames else value
    return out


def test_a_real_policy_world_window_reaches_context_tokens(tmp_path: Path) -> None:
    """End to end on real data: a v7 replay with live projectiles, encoded to the
    policy-world format, read back through ``loader_kwargs``'s routing, and forwarded."""
    if not _V7_TRAIN.is_dir():
        pytest.skip("local v7 subset is not available")
    cfg = _cfg()
    source = dict(StreamingDataset(local=str(_V7_TRAIN), batch_size=1, shuffle=False)[0])
    live = ~np.isnan(np.asarray(source[item_column(0, "pos_x")], dtype=np.float64))
    # The record is real data, so it decides the fixture: this asserts nothing without a
    # live projectile, and the window sampler needs a replay at least one window long.
    length = min(64, len(live))
    if not live.any() or length < cfg.L_ctx + cfg.sample_chunk_length:
        pytest.skip("the sampled replay is too short or carries no slot-0 projectile")
    start = max(0, min(int(np.flatnonzero(live)[0]), len(live) - length))
    replay = _sliced_replay(source, start, length)
    if np.count_nonzero(~np.isnan(np.asarray(replay[item_column(0, "pos_x")], dtype=np.float64))) < 8:
        pytest.skip("the sliced window carries too few live projectile frames")

    encoded = encode_policy_world_replay(replay, "items-0")
    with MDSWriter(out=str(tmp_path / "train"), columns=POLICY_WORLD_MDS_COLUMNS, compression="zstd") as writer:
        writer.write(encoded)

    kwargs = exp.loader_kwargs(cfg, _parity_stats())
    kwargs |= {"data_root": str(tmp_path), "remote": None, "batch_size": 1}
    batch = next(iter(make_loader(split="train", num_workers=0, windows_per_replay=4, **kwargs)))
    features = batch.context.features

    assert set(features) >= ITEM_INPUT_COLUMNS
    model = exp.GPT(cfg).eval()
    with torch.no_grad():
        tokens = model.context_tokens(features)
    assert tokens.shape == (1, cfg.L_ctx, cfg.d_model)
    assert torch.isfinite(tokens).all()


# --- offline / online parity for one projectile frame -------------------------


def _post(side: int) -> dict[str, object]:
    return {
        "position": {"x": 42.0 - 11.0 * side, "y": -7.5 * side},
        "direction": 1.0 - 2.0 * side,
        "percent": 31.0 + side,
        "shield": 47.5,
        "stock": 3,
        "action": 14 + side,
        "jumps_used": side,
        "airborne": side,
        "hurtbox_state": 1,
        "hitlag_left": 2.0,
    }


def _obs_with_two_items() -> dict[str, object]:
    """One canonical closed-loop frame carrying two live projectiles. Spawn ids are
    ascending, so the laser takes slot 0 and the turnip slot 1."""
    items = [
        {
            "id": 3,
            "type": _LASER["type"],
            "state": _LASER["state"],
            "position": {"x": _LASER["pos_x"], "y": _LASER["pos_y"]},
            "velocity": {"x": _LASER["vel_x"], "y": _LASER["vel_y"]},
            "owner": 0,
        },
        {
            "id": 7,
            "type": _TURNIP["type"],
            "state": _TURNIP["state"],
            "position": {"x": _TURNIP["pos_x"], "y": _TURNIP["pos_y"]},
            "velocity": {"x": _TURNIP["vel_x"], "y": _TURNIP["vel_y"]},
            "owner": 1,
        },
    ]
    return {
        "id": 400,
        "ports": {
            EGO_PORT: {"leader": {"post": _post(0)}, "follower": None},
            OPP_PORT: {"leader": {"post": _post(1)}, "follower": None},
        },
        "items": items,
        "stage": STAGE,
        "_matchup": {"stage": STAGE, "character": {EGO_PORT: 1, OPP_PORT: 22}},
    }


def _parity_stats() -> dict[str, FeatureStats]:
    """Asymmetric stats so standardize and min-max both do real work."""
    rng = np.random.default_rng(41)
    names = ["position_x", "position_y", "percent", "shield", "direction", "hitlag_left"]
    keys = names + [f"nana_{name}" for name in names] + [f"item_{name}" for name in _ITEM_FLOATS]
    out: dict[str, FeatureStats] = {}
    for key in keys:
        low = float(rng.normal(-50, 10))
        out[key] = FeatureStats(
            mean=float(rng.normal(0, 20)),
            std=float(abs(rng.normal(0, 10)) + 0.5),
            min=low,
            max=low + float(abs(rng.normal(0, 60)) + 1.0),
        )
    return out


def _offline_batch(flat: Mapping[str, float], action: np.ndarray) -> dict[str, np.ndarray]:
    """The same frame in the OFFLINE MDS dtypes: item ids are int32 with ``MASK_INT32``
    in an empty slot, while the closed loop hands every item field over as a float NaN
    sentinel column."""
    out: dict[str, np.ndarray] = {}
    for key, value in flat.items():
        if key == "frame":
            continue
        if key in _ITEM_CAT_COLUMNS:
            out[key] = np.array([MASK_INT32 if math.isnan(value) else int(value)], dtype=np.int32)
        elif isinstance(value, int):
            out[key] = np.array([value], dtype=np.int32)
        else:
            out[key] = np.array([value], dtype=np.float32)
    for index, channel in enumerate(ACTION_CHANNELS):
        name = f"{EGO_PREFIX}_{channel}"
        if channel.startswith("button_"):
            out[name] = np.array([action[index] > 0.5], dtype=np.int32)
        else:
            out[name] = np.array([action[index]], dtype=np.float32)
    return relabel_ego(out, EGO_PREFIX)


def test_ring_item_rows_match_preprocess_on_the_offline_dtypes() -> None:
    """One two-projectile frame through the closed-loop ring builder reproduces
    ``preprocess`` on the equivalent MDS-dtype arrays, tensor for tensor."""
    cfg = _cfg()
    extra, projection = exp._routing(cfg)
    stats = _parity_stats()
    flat = flatten_canonical_frame(_obs_with_two_items())
    action = np.linspace(-1.0, 1.0, A_DIM, dtype=np.float32)

    layout = _build_layout(flat, EGO_PREFIX, stats, extra, projection)
    rings = _Rings(layout, cfg.L_ctx)
    rings.gather(flat, action)
    rings.push(None)
    newest = rings.window(1)
    values = dict(zip(layout.value_names, rings.values[:, newest][:, 0], strict=True))
    cats = dict(zip(layout.cat_names, rings.cats[:, newest][:, 0], strict=True))
    masks = dict(zip(layout.mask_names, rings.masks[:, newest][:, 0], strict=True))

    reference = preprocess(_offline_batch(flat, action), stats, extra=extra, projection=projection)

    assert set(values) | set(cats) == {name for name in reference if not name.endswith("_mask")}
    assert set(values) | set(cats) >= ITEM_INPUT_COLUMNS
    for name, value in values.items():
        assert float(value) == pytest.approx(float(reference[name][0]), rel=1e-6, abs=1e-6), name
    for name, value in cats.items():
        assert int(value) == int(reference[name][0]), name
    for name, value in masks.items():
        expected = reference.get(name)
        assert float(value) == (0.0 if expected is None else float(expected[0])), name

    # The frame is not degenerate: two live slots, two empty ones.
    assert int(cats[item_column(0, "type")]) == _LASER["type"]
    assert int(cats[item_column(1, "type")]) == _TURNIP["type"]
    assert int(cats[item_column(2, "type")]) == 0
    for slot in range(ITEM_SLOTS):
        empty = 1.0 if slot >= 2 else 0.0
        assert masks[f"{item_column(slot, 'pos_x')}_mask"] == empty
    assert values[item_column(0, "pos_x")] != 0.0
    assert all(values[item_column(slot, name)] == 0.0 for slot in (2, 3) for name in _ITEM_FLOATS)
    assert [name for name in values if name.endswith("_owner")] == []


# --- configuration ------------------------------------------------------------


def test_config_round_trips_and_rejects_an_026_checkpoint() -> None:
    cfg = _cfg()
    values = asdict(cfg)
    item_fields = {"item_conditioning", "item_type_dim", "item_state_dim", "item_hidden_dim", "item_dim"}

    assert item_fields <= exp._CHECKPOINT_ARCH_FIELDS
    assert exp.config_from_state(values) == cfg
    with pytest.raises(ValueError, match="item_dim"):
        exp.config_from_state({name: value for name, value in values.items() if name not in item_fields})


def test_validate_config_rejects_the_dead_bundle_and_non_positive_dims() -> None:
    exp.validate_config(_cfg())
    exp.validate_config(_cfg(item_conditioning=False, observation_bundle="v6_lean"))

    with pytest.raises(ValueError, match="v6_lean"):
        exp.validate_config(_cfg(observation_bundle="v6_lean"))
    for name in ("item_type_dim", "item_state_dim", "item_hidden_dim", "item_dim"):
        with pytest.raises(ValueError, match=name):
            exp.validate_config(_cfg(**{name: 0}))


def test_model_tag_records_the_arm() -> None:
    assert exp.model_tag(_cfg()).startswith("proj041-")
    assert exp.model_tag(_cfg()).endswith("-base-items")
    assert exp.model_tag(_cfg(item_conditioning=False)).endswith("-base-noitems")
