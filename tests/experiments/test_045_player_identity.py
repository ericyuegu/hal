"""Contracts for O45 player-style representation learning."""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch
from melee import Stage
from streaming import MDSWriter


def _load() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "experiments" / "045_player_identity.py"
    spec = importlib.util.spec_from_file_location("test_exp045", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


exp = _load()


def _cfg(**overrides) -> exp.TrainConfig:
    values = {
        "d_model": 32,
        "n_layers": 1,
        "n_heads": 4,
        "logical_batch_size": 8,
        "anonymous_pairs": 4,
        "professional_pairs": 4,
        "professional_identities_per_batch": 2,
        "professional_replays_per_identity": 2,
        "online_microbatch": 2,
        "target_microbatch": 4,
        "warmup_updates": 2,
        "max_updates": 4,
        "attention_backend": "dense_sdpa",
        "wandb_mode": "disabled",
    }
    return exp.TrainConfig(**{**values, **overrides})


def _metadata(
    index: int,
    identity: str | None = "aklo",
    *,
    character: int = 1,
    stage: int = 2,
    opponent: int = 3,
    descriptor_offset: float = 0.0,
) -> exp.WindowMetadata:
    return exp.WindowMetadata(
        replay_id=f"replay-{index}",
        identity=identity,
        ego_character=character,
        stage=stage,
        opponent_character=opponent,
        descriptor=np.full(exp.DESCRIPTOR_DIM, index + descriptor_offset, dtype=np.float32),
    )


def _raw_window() -> dict[str, np.ndarray]:
    length = exp.WINDOW_LENGTH
    frame = np.arange(length, dtype=np.float32)
    return {
        "stage": np.full(length, Stage.BATTLEFIELD.value, dtype=np.int32),
        "ego_position_x": frame / 10,
        "opp_position_x": -frame / 20,
        "ego_position_y": frame / 30,
        "opp_position_y": frame / 40,
        "ego_percent": frame,
        "opp_percent": frame * 2,
        "ego_stock": np.full(length, 4, dtype=np.int32),
        "opp_stock": np.full(length, 3, dtype=np.int32),
    }


class _FakeDataset:
    cache_usage = 2**30


class _FakeLoader:
    """Cursor-bearing loader used to isolate PairLoader's experiment state."""

    def __init__(self, make_batch) -> None:
        self.dataset = _FakeDataset()
        self._make_batch = make_batch
        self._cursor = 0

    def __iter__(self):
        return self

    def __next__(self):
        batch = self._make_batch(self._cursor)
        self._cursor += 1
        return batch

    def state_dict(self) -> dict[str, int]:
        return {"cursor": self._cursor}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self._cursor = state["cursor"]


def _fake_pair_loader(cfg: exp.TrainConfig, identities: tuple[str, ...]) -> exp.PairLoader:
    anonymous_context = exp.synthetic_context(cfg, 2, seed=41)

    def anonymous_batch(cursor: int) -> exp.PairBatch:
        online = tuple(_metadata(2 * cursor + index, identity=None) for index in range(2))
        target = tuple(_metadata(10_000 + 2 * cursor + index, identity=None) for index in range(2))
        return exp.PairBatch(
            anonymous_context,
            anonymous_context,
            exp.PairMetadata(online, target, torch.zeros(2, dtype=torch.bool)),
        )

    professional_context = exp.synthetic_context(cfg, len(identities), seed=42)

    def professional_batch(cursor: int) -> exp.ProfessionalCandidates:
        metadata = tuple(
            _metadata(cursor * len(identities) + index, identity, character=index)
            for index, identity in enumerate(identities)
        )
        return exp.ProfessionalCandidates(professional_context, metadata, len(identities))

    return exp.PairLoader(
        cfg,
        _FakeLoader(anonymous_batch),
        _FakeLoader(professional_batch),
        identities=identities,
        anonymous_batches_per_update=2,
    )


def test_frozen_configuration_and_identity_partition() -> None:
    cfg = exp.TrainConfig()
    cfg.validate()
    assert cfg.logical_batch_size == 1024
    assert cfg.anonymous_pairs == 768
    assert cfg.professional_pairs == 256
    assert len(exp.DEVELOPMENT_IDENTITIES) == 28
    assert len(exp.SEALED_IDENTITIES) == 10
    assert not set(exp.SEALED_IDENTITIES) & set(exp.DEVELOPMENT_IDENTITIES)
    assert set(exp.SEALED_IDENTITIES) | set(exp.DEVELOPMENT_IDENTITIES) == set(exp.streams.PROFESSIONAL_PLAYER_SLUGS)


def test_frozen_loader_geometry() -> None:
    cfg = exp.TrainConfig()
    assert exp.ANONYMOUS_IO_BATCH == 64
    assert exp.ANONYMOUS_BATCHES_PER_UPDATE == 12
    assert cfg.anonymous_pairs == exp.ANONYMOUS_IO_BATCH * exp.ANONYMOUS_BATCHES_PER_UPDATE
    assert exp.PROFESSIONAL_INPUT_BATCH == len(exp.DEVELOPMENT_IDENTITIES) == 28
    assert cfg.professional_pairs == 16 * 16
    assert exp.LOADER_WORKERS == 8
    assert exp.LOADER_PREFETCH_FACTOR == 2
    assert cfg.attention_backend == "varlen_flash"


def test_source_tagging_and_stratified_batches_use_both_toy_mds_streams(tmp_path: Path) -> None:
    roots = (tmp_path / "first", tmp_path / "second")
    for source_index, root in enumerate(roots):
        with MDSWriter(out=str(root / "train"), columns={"value": "int"}) as writer:
            for value in range(3):
                writer.write({"value": 10 * source_index + value})
    dataset = exp.SourceTaggedStreamingDataset(
        streams=[exp.Stream(local=str(root), split="train", proportion=0.5, keep_zip=False) for root in roots],
        source_slugs=("first", "second"),
        batch_size=2,
        batching_method="stratified",
        shuffle=True,
        shuffle_algo="py1e",
        shuffle_seed=5,
        shuffle_block_size=8,
        cache_limit=None,
        keep_zip=False,
    )

    rows = list(dataset)

    assert len(rows) == 6
    for start in range(0, len(rows), 2):
        assert {row["_o45_source"] for row in rows[start : start + 2]} == {"first", "second"}
    assert dataset.batching_method == "stratified"
    assert dataset.shuffle_algo == "py1e"
    assert dataset.shuffle_block_size == 8


def test_tagged_stratified_mds_cursor_resumes_exactly(tmp_path: Path) -> None:
    def make(root: Path) -> exp.O45StreamingDataLoader:
        source_roots = (root / "first", root / "second")
        for source_index, source_root in enumerate(source_roots):
            with MDSWriter(out=str(source_root / "train"), columns={"value": "int"}) as writer:
                for value in range(8):
                    writer.write({"value": 100 * source_index + value})
        dataset = exp.SourceTaggedStreamingDataset(
            streams=[
                exp.Stream(local=str(source_root), split="train", proportion=0.5) for source_root in source_roots
            ],
            source_slugs=("first", "second"),
            batch_size=2,
            batching_method="stratified",
            shuffle=True,
            shuffle_seed=19,
            shuffle_block_size=32,
        )
        return exp.O45StreamingDataLoader(dataset, batch_size=2, num_workers=0)

    uninterrupted = make(tmp_path / "uninterrupted")
    iterator = iter(uninterrupted)
    next(iterator)
    state = uninterrupted.state_dict()
    expected = [next(iterator) for _ in range(3)]
    resumed = make(tmp_path / "resumed")
    resumed.load_state_dict(state)
    actual_iterator = iter(resumed)
    actual = [next(actual_iterator) for _ in range(3)]

    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got["value"], want["value"])
        assert got["_o45_source"] == want["_o45_source"]


def test_replay_window_randomness_uses_seed_epoch_replay_and_source() -> None:
    first = exp._stable_replay_rng(3, 4, "replay", "aklo").integers(2**31, size=8)
    same = exp._stable_replay_rng(3, 4, "replay", "aklo").integers(2**31, size=8)
    assert np.array_equal(first, same)
    assert not np.array_equal(first, exp._stable_replay_rng(3, 5, "replay", "aklo").integers(2**31, size=8))
    assert not np.array_equal(first, exp._stable_replay_rng(3, 4, "replay", "amsa").integers(2**31, size=8))


def test_pair_loader_assembles_exact_batch_with_distinct_professional_replays() -> None:
    identities = tuple(exp.DEVELOPMENT_IDENTITIES[:4])
    cfg = _cfg(professional_identities_per_batch=2, professional_replays_per_identity=2)
    loader = _fake_pair_loader(cfg, identities)

    batch = loader.sample()

    assert batch.batch_size == 8
    assert batch.metadata.professional_mask.tolist() == [False] * 4 + [True] * 4
    for identity in set(item.identity for item in batch.metadata.online[4:]):
        replays = [item.replay_id for item in batch.metadata.online[4:] if item.identity == identity]
        assert len(replays) == len(set(replays)) == 2
    assert all(len(queue) <= exp.PROFESSIONAL_QUEUE_LIMIT for queue in loader.queues.values())
    assert loader.last_metrics.cache_gib == 2


def test_pair_loader_selects_identities_uniformly() -> None:
    identities = tuple(exp.DEVELOPMENT_IDENTITIES[:4])
    cfg = _cfg(professional_identities_per_batch=2, professional_replays_per_identity=2)
    loader = _fake_pair_loader(cfg, identities)
    counts = {identity: 0 for identity in identities}

    for _ in range(200):
        batch = loader.sample()
        for identity in set(item.identity for item in batch.metadata.online[4:]):
            counts[str(identity)] += 1

    assert all(75 < count < 125 for count in counts.values())


def test_pair_loader_state_resumes_exactly() -> None:
    identities = tuple(exp.DEVELOPMENT_IDENTITIES[:4])
    cfg = _cfg(professional_identities_per_batch=2, professional_replays_per_identity=2)
    uninterrupted = _fake_pair_loader(cfg, identities)
    uninterrupted.sample()
    state = uninterrupted.state_dict()
    expected = uninterrupted.sample()

    resumed = _fake_pair_loader(cfg, identities)
    resumed.load_state_dict(state)
    actual = resumed.sample()

    def fingerprint(metadata) -> list[tuple[str, str | None]]:
        return [(item.replay_id, item.identity) for item in metadata]

    assert fingerprint(actual.metadata.online) == fingerprint(expected.metadata.online)
    assert fingerprint(actual.metadata.target) == fingerprint(expected.metadata.target)
    for name in actual.online.features:
        torch.testing.assert_close(actual.online.features[name], expected.online.features[name])
        torch.testing.assert_close(actual.target.features[name], expected.target.features[name])


def test_uniform_anchor_support_includes_middle_and_endpoints() -> None:
    rng = np.random.default_rng(7)
    final_start = 2048 - exp.WINDOW_LENGTH
    starts = np.asarray([exp.sample_anonymous_anchor(2048, rng) for _ in range(100_000)])
    assert starts.min() == 0
    assert starts.max() == final_start
    thirds = np.histogram(starts, bins=[0, final_start / 3, 2 * final_start / 3, final_start + 1])[0]
    assert np.max(np.abs(thirds / thirds.sum() - 1 / 3)) < 0.01


@pytest.mark.parametrize(
    ("frames", "expected"),
    [(1543, 429), (3328, 1024), (10_000, 1024)],
)
def test_adaptive_separation(frames: int, expected: int) -> None:
    assert exp.adaptive_minimum_separation(frames) == expected
    rng = np.random.default_rng(4)
    for _ in range(100):
        online, target = exp.sample_anonymous_starts(frames, rng)
        assert abs(online - target) >= expected


def test_distant_sampling_is_distance_squared() -> None:
    frames = 3328
    anchor = 1024
    rng = np.random.default_rng(9)
    starts = np.asarray([exp.sample_distant_start(frames, anchor, rng) for _ in range(30_000)])
    assert np.all(np.abs(starts - anchor) >= 1024)
    far = np.mean(starts >= 2048)
    near = np.mean(starts <= 0)
    assert far > near


def test_anonymous_roles_exchange_without_changing_pair_contract() -> None:
    rng = np.random.default_rng(12)
    pairs = [exp.sample_anonymous_starts(4096, rng) for _ in range(2000)]
    assert all(0 <= left <= 4096 - exp.WINDOW_LENGTH for left, _ in pairs)
    assert all(0 <= right <= 4096 - exp.WINDOW_LENGTH for _, right in pairs)
    assert all(abs(left - right) >= 1024 for left, right in pairs)
    assert any(left < right for left, right in pairs)
    assert any(left > right for left, right in pairs)


def test_professional_split_is_stable_and_sealed_players_never_train() -> None:
    first = [exp.professional_split("aklo", f"replay-{index}") for index in range(10_000)]
    second = [exp.professional_split("aklo", f"replay-{index}") for index in range(10_000)]
    assert first == second
    counts = {name: first.count(name) for name in ("train", "gallery", "query")}
    assert 7700 < counts["train"] < 8300
    assert 800 < counts["gallery"] < 1200
    assert 800 < counts["query"] < 1200
    assert all(exp.professional_split(identity, "anything") == "sealed" for identity in exp.SEALED_IDENTITIES)


def test_exactly_one_professional_side_is_required() -> None:
    pro = int(exp.Rank.PRO)
    amateur = int(exp.Rank.PLATINUM)
    assert exp.professional_ego_side({"p1_rank": pro, "p2_rank": amateur}) == "p1"
    assert exp.professional_ego_side({"p1_rank": amateur, "p2_rank": pro}) == "p2"
    assert exp.professional_ego_side({"p1_rank": pro, "p2_rank": pro}) is None
    assert exp.professional_ego_side({"p1_rank": amateur, "p2_rank": amateur}) is None


def test_game_state_descriptor_has_fixed_safe_contents() -> None:
    window = _raw_window()
    descriptor = exp.game_state_descriptor(window)
    assert descriptor.shape == (80,)
    assert descriptor.dtype == np.float32
    first_bin = descriptor[:10]
    assert first_bin[4] == pytest.approx(np.mean(np.arange(32)) / 100)
    assert first_bin[6] == 4
    assert first_bin[7] == 3


def test_professional_assignment_is_cross_replay_derangement() -> None:
    items = [_metadata(index, character=index, stage=index, opponent=index) for index in range(16)]
    pairing = exp.professional_derangement(items)
    assert sorted(pairing.tolist()) == list(range(16))
    assert np.all(pairing != np.arange(16))
    assert all(items[left].replay_id != items[right].replay_id for left, right in enumerate(pairing))


def test_professional_assignment_prefers_nuisance_changes() -> None:
    items = [
        _metadata(0, character=0, stage=0, opponent=0),
        _metadata(1, character=0, stage=0, opponent=0),
        _metadata(2, character=1, stage=1, opponent=1),
        _metadata(3, character=1, stage=1, opponent=1),
    ]
    pairing = exp.professional_derangement(items)
    assert all(items[left].ego_character != items[right].ego_character for left, right in enumerate(pairing))
    assert all(items[left].stage != items[right].stage for left, right in enumerate(pairing))


def test_pair_metadata_keeps_anonymous_out_of_professional_set() -> None:
    anonymous = _metadata(0, identity=None)
    professional = _metadata(1, identity="aklo")
    metadata = exp.PairMetadata(
        (anonymous, professional),
        (anonymous, professional),
        torch.tensor([False, True]),
    )
    assert [item.identity for item, keep in zip(metadata.online, metadata.professional_mask, strict=True) if keep] == [
        "aklo"
    ]
    with pytest.raises(ValueError, match="professional positives"):
        exp.PairMetadata((anonymous,), (anonymous,), torch.tensor([True]))


def test_negative_selection_enforces_character_and_different_identity() -> None:
    anchors = [
        _metadata(0, "aklo", character=1, stage=2, opponent=3),
        _metadata(1, "amsa", character=2, stage=2, opponent=3),
    ]
    candidates = [
        _metadata(2, "amsa", character=1, stage=2, opponent=3),
        _metadata(3, "aklo", character=2, stage=2, opponent=3),
    ]
    z = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    selected = exp.select_professional_negatives(z, z, anchors, candidates)
    assert selected.tolist() == [0, 1]
    for anchor, index in zip(anchors, selected.tolist(), strict=True):
        assert candidates[index].identity != anchor.identity
        assert candidates[index].ego_character == anchor.ego_character


def test_negative_selection_prefers_metadata_tiers_then_batch_hard() -> None:
    anchor = _metadata(0, "aklo", character=1, stage=2, opponent=3)
    candidates = [
        _metadata(1, "amsa", character=1, stage=2, opponent=4),
        _metadata(2, "axe", character=1, stage=2, opponent=3),
        _metadata(3, "cody", character=1, stage=2, opponent=3),
    ]
    anchor_z = torch.tensor([[1.0, 0.0]])
    candidate_z = torch.tensor([[1.0, 0.0], [0.1, 1.0], [0.9, 0.1]])
    selected = exp.select_professional_negatives(
        anchor_z,
        candidate_z,
        [anchor],
        candidates,
    )
    assert selected.item() == 2


def test_negative_selection_skips_when_no_safe_candidate() -> None:
    anchor = _metadata(0, "aklo", character=1)
    candidate = _metadata(1, "amsa", character=2)
    selected = exp.select_professional_negatives(
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[1.0, 0.0]]),
        [anchor],
        [candidate],
    )
    assert selected.item() == -1


def test_one_way_byol_stops_target_gradients() -> None:
    prediction = torch.randn(4, 8, requires_grad=True)
    target = torch.randn(4, 8, requires_grad=True)
    exp.one_way_byol_loss(prediction, target).mean().backward()
    assert prediction.grad is not None
    assert target.grad is None


def test_target_branch_starts_equal_and_has_no_trainable_parameters() -> None:
    model = exp.BYOL(_cfg(), prefer_flex=False)
    assert all(not parameter.requires_grad for parameter in model.target_parameters())
    online = model.online_encoder.state_dict()
    target = model.target_encoder.state_dict()
    assert online.keys() == target.keys()
    assert all(torch.equal(online[name], target[name]) for name in online)


def test_ema_update_is_exact_and_schedule_includes_endpoints() -> None:
    cfg = _cfg()
    model = exp.BYOL(cfg, prefer_flex=False)
    with torch.no_grad():
        for parameter in model.online_encoder.parameters():
            parameter.fill_(2.0)
        for parameter in model.target_encoder.parameters():
            parameter.fill_(1.0)
    model.update_target(0.75)
    assert all(
        torch.equal(parameter, torch.full_like(parameter, 1.25)) for parameter in model.target_encoder.parameters()
    )
    assert exp.ema_tau(0, cfg) == cfg.ema_start
    assert exp.ema_tau(cfg.max_updates - 1, cfg) == cfg.ema_end


def test_learning_rate_warmup_and_cosine_endpoints() -> None:
    cfg = replace(_cfg(), warmup_updates=2, max_updates=5)
    assert exp.learning_rate(0, cfg) == pytest.approx(cfg.learning_rate / 2)
    assert exp.learning_rate(1, cfg) == pytest.approx(cfg.learning_rate)
    assert exp.learning_rate(2, cfg) == pytest.approx(cfg.learning_rate)
    assert exp.learning_rate(4, cfg) == pytest.approx(cfg.minimum_learning_rate)


def test_encoder_omits_broadcast_stage_and_character_inputs() -> None:
    cfg = _cfg()
    model = exp.PlayerEncoder(cfg, prefer_flex=False).eval()
    context = exp.synthetic_context(cfg, 2)
    changed = dict(context.features)
    changed["stage"] = torch.full_like(changed["stage"], 20)
    changed["ego_character"] = torch.full_like(changed["ego_character"], 17)
    changed["opp_character"] = torch.full_like(changed["opp_character"], 18)
    with torch.no_grad():
        first = model(context)
        second = model(exp.Context(changed, context.ctx_pad))
    assert torch.equal(first, second)


def test_preprojector_export_is_independent_of_training_heads() -> None:
    cfg = _cfg()
    model = exp.BYOL(cfg, prefer_flex=False).eval()
    exported = model.export_encoder()
    context = exp.synthetic_context(cfg, 2)
    with torch.no_grad():
        expected = exported.normalized(context)
        for parameter in model.online_projector.parameters():
            parameter.normal_(mean=10, std=5)
        for parameter in model.predictor.parameters():
            parameter.zero_()
        actual = exported.normalized(context)
    assert torch.equal(expected, actual)
    assert not any("projector" in name or "predictor" in name for name, _ in exported.named_parameters())


def test_retrieval_knn_probe_and_prototypes_on_separable_data() -> None:
    identities = ["a", "a", "b", "b", "c", "c"]
    replays = [f"r{index}" for index in range(6)]
    z = np.asarray(
        [[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0.1, 0.9, 0], [0, 0, 1], [0, 0.1, 0.9]],
        dtype=np.float32,
    )
    retrieval = exp.retrieval_metrics(z, identities, replays, z, identities, replays)
    assert retrieval["recall_at_1"] == 1
    assert retrieval["mrr"] == 1
    knn = exp.knn_identification(z, identities, z, identities, k_values=(1,))
    assert knn["knn_1"] == 1
    probe = exp.linear_probe_metrics(z, identities, z, identities)
    assert probe["linear_probe"] == 1
    prototypes = exp.prototype_identification(z, identities, replays, z, identities, shots=(1, 2))
    assert prototypes["prototype_1"] == 1
    assert prototypes["prototype_2"] == 1


def test_distance_gap_and_nuisance_subsets() -> None:
    query = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
    positive = np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
    negative = np.asarray([[0, 1], [1, 0]], dtype=np.float32)
    query_meta = [_metadata(0, character=0, stage=0), _metadata(1, character=1, stage=1)]
    positive_meta = [_metadata(2, character=1, stage=1), _metadata(3, character=0, stage=0)]
    metrics = exp.nuisance_controlled_metrics(query, positive, negative, query_meta, positive_meta)
    assert metrics["distance_gap"] > 0
    assert metrics["roc_auc"] == 1
    assert metrics["triplet_accuracy"] == 1
    assert metrics["character_crossing/coverage"] == 2
    assert metrics["stage_crossing/coverage"] == 2


def test_nuisance_triplets_prefer_crossing_and_match_negative_character() -> None:
    metadata = [
        _metadata(0, "aklo", character=1, stage=2, opponent=3),
        _metadata(1, "aklo", character=1, stage=2, opponent=4),
        _metadata(2, "aklo", character=2, stage=3, opponent=4),
        _metadata(3, "amsa", character=1, stage=4, opponent=5, descriptor_offset=0.01),
        _metadata(4, "axe", character=2, stage=2, opponent=3),
    ]
    queries, positives, negatives = exp.select_nuisance_triplets(metadata, [0])
    assert queries.tolist() == [0]
    assert positives.tolist() == [2]
    assert negatives.tolist() == [3]
    assert metadata[negatives[0]].ego_character == metadata[queries[0]].ego_character
    assert metadata[negatives[0]].identity != metadata[queries[0]].identity


def test_sealed_support_query_split_is_stable_and_nontrivial() -> None:
    identity = exp.SEALED_IDENTITIES[0]
    first = [exp.sealed_replay_split(identity, f"replay-{index}") for index in range(100)]
    second = [exp.sealed_replay_split(identity, f"replay-{index}") for index in range(100)]
    assert first == second
    assert set(first) == {"support", "query"}
    with pytest.raises(ValueError, match="sealed"):
        exp.sealed_replay_split(exp.DEVELOPMENT_IDENTITIES[0], "replay")


def test_collapse_diagnostics_distinguish_full_rank_from_collapse() -> None:
    full_rank = np.eye(64, dtype=np.float32)
    collapsed = np.ones((64, 64), dtype=np.float32)
    healthy = exp.collapse_diagnostics(full_rank)
    failed = exp.collapse_diagnostics(collapsed)
    assert healthy["effective_rank"] > 32
    assert failed["effective_rank"] < 2
    assert healthy["mean_coordinate_std"] > failed["mean_coordinate_std"]


def test_fixed_heldout_cache_runs_existing_metrics_on_cpu() -> None:
    cfg = _cfg(target_microbatch=16)
    identities = exp.DEVELOPMENT_IDENTITIES[:2]
    context = exp.synthetic_context(cfg, 64, seed=71)
    examples = []
    for identity_index, identity in enumerate(identities):
        for split_index in range(32):
            index = identity_index * 32 + split_index
            metadata = _metadata(
                index,
                identity,
                character=split_index % 2,
                stage=2 + split_index % 2,
                descriptor_offset=identity_index / 10,
            )
            examples.append(exp.ProfessionalExample(exp._context_row(context, index), metadata))
    gallery = tuple(examples[identity_index * 32 + offset] for identity_index in range(2) for offset in range(16))
    query = tuple(examples[identity_index * 32 + offset] for identity_index in range(2) for offset in range(16, 32))
    model = exp.BYOL(cfg, prefer_flex=False)

    metrics = exp.heldout_metrics(model, exp.HeldoutCache(gallery, query), cfg, torch.device("cpu"))

    assert metrics["retrieval/queries"] == 32
    assert math_is_finite(metrics["nuisance/distance_gap"])
    assert math_is_finite(metrics["collapse/effective_rank"])


def test_bootstrap_lower_bound() -> None:
    assert exp.bootstrap_lower_bound(np.ones(100), samples=1000) == 1
    assert exp.bootstrap_lower_bound(np.ones(100), chance=0.5, samples=1000) == 0.5


def test_cpu_end_to_end_step_updates_online_and_ema() -> None:
    cfg = _cfg()
    model = exp.BYOL(cfg, prefer_flex=False)
    optimizer = exp.make_optimizer(model, cfg)
    online_before = [parameter.detach().clone() for parameter in model.online_encoder.parameters()]
    target_before = [parameter.detach().clone() for parameter in model.target_encoder.parameters()]
    metrics = exp.train_step(model, exp.synthetic_pair_batch(cfg), optimizer, cfg, 0, torch.device("cpu"))
    assert math_is_finite(metrics.loss)
    assert metrics.valid_triplets == cfg.professional_pairs
    assert any(
        not torch.equal(before, after)
        for before, after in zip(online_before, model.online_encoder.parameters(), strict=True)
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(target_before, model.target_encoder.parameters(), strict=True)
    )
    assert all(parameter.grad is None for parameter in model.target_parameters())


def test_train_step_has_only_one_online_and_one_target_pass_per_example() -> None:
    cfg = _cfg()
    model = exp.BYOL(cfg, prefer_flex=False)
    optimizer = exp.make_optimizer(model, cfg)
    counts = {"online": 0, "target": 0}

    def online_hook(_module, inputs, _output) -> None:
        counts["online"] += inputs[0].batch

    def target_hook(_module, inputs, _output) -> None:
        counts["target"] += inputs[0].batch

    online_handle = model.online_encoder.register_forward_hook(online_hook)
    target_handle = model.target_encoder.register_forward_hook(target_hook)
    try:
        exp.train_step(model, exp.synthetic_pair_batch(cfg), optimizer, cfg, 0, torch.device("cpu"))
    finally:
        online_handle.remove()
        target_handle.remove()
    assert counts == {"online": cfg.logical_batch_size, "target": cfg.logical_batch_size}


def test_checkpoint_contains_complete_training_state(tmp_path: Path) -> None:
    cfg = _cfg()
    model = exp.BYOL(cfg, prefer_flex=False)
    optimizer = exp.make_optimizer(model, cfg)
    pair_loader = _fake_pair_loader(cfg, tuple(exp.DEVELOPMENT_IDENTITIES[:4]))
    pair_loader.sample()
    path = tmp_path / "checkpoint.pt"
    stats = {"position_x": exp.FeatureStats(mean=0, std=1, min=-1, max=1)}
    exp.save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        cfg=cfg,
        stats=stats,
        step=1,
        pair_loader=pair_loader,
        wandb_id="run-id",
    )
    state = torch.load(path, weights_only=False)
    required = {
        "online_encoder",
        "online_projector",
        "predictor",
        "ema_encoder",
        "ema_projector",
        "optimizer",
        "scheduler",
        "ema_schedule_position",
        "rng",
        "pair_loader",
        "professional_split",
        "config",
        "feature_statistics",
        "step",
        "wandb_id",
    }
    assert required <= state.keys()
    assert state["schema"] == 2
    assert state["pair_loader"]["anonymous_mds"] == {"cursor": 2}
    assert state["pair_loader"]["professional_mds"] == {"cursor": 2}
    assert state["ema_schedule_position"] == 1
    assert state["wandb_id"] == "run-id"
    _, _, loaded_cfg, loaded_stats, loaded_state = exp.load_checkpoint(path, device=torch.device("cpu"))
    assert loaded_cfg == cfg
    assert loaded_stats == stats
    assert loaded_state["step"] == 1


def math_is_finite(value: float) -> bool:
    return not np.isnan(value) and not np.isinf(value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_smoke_step() -> None:
    cfg = _cfg(online_microbatch=4, target_microbatch=4)
    device = torch.device("cuda")
    model = exp.BYOL(cfg, prefer_flex=False).to(device)
    optimizer = exp.make_optimizer(model, cfg)
    metrics = exp.train_step(model, exp.synthetic_pair_batch(cfg), optimizer, cfg, 0, device)
    assert math_is_finite(metrics.loss)
