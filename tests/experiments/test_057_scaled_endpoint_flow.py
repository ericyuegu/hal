"""Focused contracts for O57's scaled categorical endpoint flow."""

import copy
import importlib.util
import sys
from dataclasses import asdict
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _load():
    path = Path(__file__).resolve().parents[2] / "experiments" / "057_scaled_endpoint_flow.py"
    name = "test_exp057"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


exp = _load()


def _tiny_cfg(**changes):
    architecture = {
        **asdict(exp.Architecture()),
        "d_model": 32,
        "n_layers": 1,
        "n_heads": 4,
        "L_ctx": 8,
        "flow_d_model": 64,
        "flow_layers": 1,
        "flow_heads": 1,
        "flow_ff_dim": 192,
        "group_head_dim": 32,
        "value_hidden_dim": 64,
        "item_hidden_dim": 8,
        "item_dim": 8,
    }
    values = dict(
        arch=exp.Architecture(**architecture),
        batch_size=2,
        execution_batch_size=1,
        compile_trunk=False,
        compile_flow=False,
        inference_mode="eager",
        num_workers=0,
        push_to_r2=False,
    )
    return exp.TrainConfig(**{**values, **changes})


def test_study_configuration_and_rank1_selection_are_fixed() -> None:
    production = exp.TrainConfig()
    proxy = exp.proxy_config()

    exp.validate_config(production)
    exp.validate_config(proxy)
    assert proxy.max_steps == production.max_steps == 16_384
    assert proxy.warmup_steps == production.warmup_steps == 512
    assert proxy.arch.L_ctx == production.arch.L_ctx == 256
    assert proxy.arch.flow_d_model == 128
    assert production.arch.flow_d_model == 512
    assert production.arch.head_offsets == (1, 2, 3, 4, 5, 6, 9, 12, 16, 20)
    assert production.source_names == ("ranked-anonymized-1-policy-world-v8",)
    assert production.replay_slots == 112_128
    assert production.minimum_replay_gap_batches == 195
    assert production.delay_frames == 0
    assert production.replan_interval_frames == 2
    assert production.exec_horizon == 2
    assert production.awr_enabled is False
    assert production.automatic_evaluation is True
    assert [exp.study_arm_for_config(exp.config_for_arm(arm)) for arm in exp.STUDY_ARM_ORDER] == list(
        exp.STUDY_ARM_ORDER
    )
    with pytest.raises(ValueError, match="fixed training settings"):
        exp.validate_config(replace(production, adam_lr=1e-3))


def test_validation_summary_uses_final_configured_offset() -> None:
    cfg = _tiny_cfg()
    values = {
        "loss_unweighted": 1.0,
        "flow_loss_near_unweighted": 2.0,
        "flow_loss_far_unweighted": 3.0,
        "exact_frame_acc": 0.1,
        "dense_four_sequence_acc": 0.2,
        "change_f1": 0.3,
        "sampled_transition_rate": 0.4,
        "shuffled_context_relative_hidden_change": 0.5,
        "multi_noise_plan_diversity": 0.6,
    }
    final_offset = cfg.arch.head_offsets[-1]
    for group, name in enumerate(exp.CONTROLLER_GROUP_NAMES, start=1):
        values[f"rollout_nll_o{final_offset:02d}_{name}"] = float(group)
        values[f"exposure_gap_o{final_offset:02d}_{name}"] = float(10 * group)

    summary = exp._validation_wandb_metrics(values, cfg)

    assert summary["rollout_nll"] == 10.0
    assert summary["exposure_gap"] == 100.0


def test_run_provenance_records_and_validates_persisted_identities(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HAL_GIT_SHA", "a" * 40)
    monkeypatch.setattr(exp, "checkpoint_sha256", lambda _path: "b" * 64)
    provenance = exp.run_provenance(exp.proxy_config())

    assert provenance["git_sha"] == "a" * 40
    assert provenance["source_statistics_sha256"] == {"ranked-anonymized-1-policy-world-v8": "b" * 64}
    exp.validate_resume_provenance(provenance, provenance)
    changed = copy.deepcopy(provenance)
    changed["git_sha"] = "c" * 40
    with pytest.raises(ValueError, match="git_sha"):
        exp.validate_resume_provenance(changed, provenance)


def test_flow_shapes_and_zero_initialization() -> None:
    cfg = _tiny_cfg()
    model = exp.GPT(cfg)
    hidden = torch.randn(2, cfg.arch.d_model)
    targets = torch.stack([torch.randint(vocab, (2, 10)) for vocab in exp.CONTROLLER_GROUP_VOCABS], dim=-1)
    endpoint = model.flow.endpoint_states(targets)
    logits = model.flow(hidden, endpoint, torch.rand(2), torch.zeros(2, 10, dtype=torch.bool))

    assert {name: value.shape for name, value in logits.items()} == {
        name: (2, 10, vocab)
        for name, vocab in zip(exp.CONTROLLER_GROUP_NAMES, exp.CONTROLLER_GROUP_VOCABS, strict=True)
    }
    assert model.flow.action_projections[0].out_features == cfg.arch.flow_d_model // 4
    assert not model.player_projection.weight.any()
    for block in model.flow.blocks:
        assert not block.modulation.weight.any()
        assert not block.modulation.bias.any()


def test_projectile_pooling_and_additive_identity_masking() -> None:
    cfg = _tiny_cfg()
    model = exp.GPT(cfg)
    context = exp.synthetic_context(cfg, 1, torch.device("cpu"))
    empty_features = {name: value.clone() for name, value in context.features.items()}
    for slot in range(exp.ITEM_SLOTS):
        empty_features[f"{exp.item_column(slot, exp.ITEM_PRESENCE_SUFFIX)}_mask"].fill_(1)
    assert not model._item_features(empty_features).any()

    pooled = []
    for live_slot in (0, exp.ITEM_SLOTS - 1):
        features = {name: value.clone() for name, value in empty_features.items()}
        for name in exp.ITEM_FLOATS:
            column = exp.item_column(live_slot, name)
            features[column].fill_(0.25)
            features[f"{column}_mask"].zero_()
        features[exp.item_column(live_slot, "type")].fill_(7)
        features[exp.item_column(live_slot, "state")].fill_(3)
        pooled.append(model._item_features(features))
    torch.testing.assert_close(*pooled)

    with torch.no_grad():
        model.player_embedding.weight[1].fill_(1)
        model.player_projection.weight.fill_(0.01)
    masked = model.context_tokens(context.features)
    identified_features = dict(context.features)
    identified_features["ego_player_id"] = torch.ones_like(identified_features["ego_player_id"])
    identified = model.context_tokens(identified_features)
    assert not torch.equal(masked, identified)
    with pytest.raises(ValueError, match="opponent identity"):
        model.context_tokens({**context.features, "opp_player_id": torch.zeros_like(context.ctx_pad[:, None])})


@pytest.mark.parametrize(
    ("schedule", "evaluations", "updates"),
    [("uniform4", 4, 4), ("uniform8", 8, 8), ("historical4", 4, 3)],
)
def test_solver_counts_and_committed_tokens(schedule: str, evaluations: int, updates: int) -> None:
    cfg = _tiny_cfg()
    model = exp.GPT(cfg)
    hidden = torch.randn(2, cfg.arch.d_model)
    committed_indices = torch.stack([torch.randint(vocab, (2, 10)) for vocab in exp.CONTROLLER_GROUP_VOCABS], dim=-1)
    committed = torch.zeros(2, 10, dtype=torch.bool)
    committed[:, :3] = True
    result = model.flow.solve(
        hidden,
        committed_indices=committed_indices,
        committed=committed,
        schedule=schedule,
        generator=torch.Generator().manual_seed(4),
    )

    assert (result.evaluations, result.updates) == (evaluations, updates)
    assert torch.equal(result.indices[:, :3], committed_indices[:, :3])
    for group, state in enumerate(result.states):
        expected = torch.nn.functional.one_hot(
            committed_indices[:, :3, group], exp.CONTROLLER_GROUP_VOCABS[group]
        ).float()
        torch.testing.assert_close(state[:, :3], expected)


def test_uniform_four_finishes_at_fourth_endpoint_probabilities() -> None:
    model = exp.GPT(_tiny_cfg())
    result = model.flow.solve(
        torch.randn(2, 32),
        schedule="uniform4",
        generator=torch.Generator().manual_seed(9),
    )

    for state, logits in zip(result.states, result.logits, strict=True):
        probabilities = torch.softmax(logits.float(), dim=-1)
        torch.testing.assert_close(state, probabilities)
        assert torch.equal(state.argmax(-1), logits.argmax(-1))


def test_solver_caches_trunk_skip_across_all_evaluations() -> None:
    model = exp.GPT(_tiny_cfg())
    calls = {name: 0 for name in exp.CONTROLLER_GROUP_NAMES}
    handles = []
    for name in exp.CONTROLLER_GROUP_NAMES:
        handles.append(
            model.flow.trunk_outputs[name].register_forward_hook(
                lambda _module, _inputs, _output, name=name: calls.__setitem__(name, calls[name] + 1)
            )
        )
    try:
        result = model.flow.solve(torch.randn(2, 32), schedule="uniform8")
    finally:
        for handle in handles:
            handle.remove()

    assert result.evaluations == 8
    assert calls == {name: 1 for name in exp.CONTROLLER_GROUP_NAMES}


def test_random_streams_are_split_invariant_and_width_independent() -> None:
    offsets = exp.Architecture().head_offsets
    full_random = exp.FlowTrainingRandom(7, 11)
    full = full_random.draw((4, 3), offsets, torch.device("cpu"))
    split_random = exp.FlowTrainingRandom(7, 11)
    first = split_random.draw((2, 3), offsets, torch.device("cpu"))
    second = split_random.draw((2, 3), offsets, torch.device("cpu"))

    torch.testing.assert_close(full.tau, torch.cat((first.tau, second.tau)))
    assert torch.equal(full.committed, torch.cat((first.committed, second.committed)))
    for complete, one, two in zip(full.noise, first.noise, second.noise, strict=True):
        torch.testing.assert_close(complete, torch.cat((one, two)))


def test_train_step_draws_one_logical_batch_before_splitting(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _tiny_cfg(execution_batch_size=1)
    model = exp.GPT(cfg)
    batch = exp.synthetic_awr_batch(cfg, torch.device("cpu"))
    valid = cfg.batch_size * (cfg.arch.L_ctx - cfg.arch.direct_loss_start)
    draw_prefixes: list[tuple[int, ...]] = []
    original_draw = exp.FlowTrainingRandom.draw

    def record_draw(
        random: exp.FlowTrainingRandom,
        prefix: tuple[int, ...],
        offsets: tuple[int, ...],
        device: torch.device,
    ) -> exp.FlowRandomDraws:
        draw_prefixes.append(prefix)
        return original_draw(random, prefix, offsets, device)

    monkeypatch.setattr(exp.FlowTrainingRandom, "draw", record_draw)
    optimizer = exp.make_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, exp.lr_schedule(cfg))

    exp.train_step(
        model,
        batch,
        cfg,
        step=0,
        update=1,
        valid_prefixes=valid,
        trunk_fn=model.forward,
        flow_fn=model.flow.training_statistics,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    assert draw_prefixes == [(cfg.batch_size, cfg.arch.L_ctx - cfg.arch.direct_loss_start)]


def test_slot_flow_random_is_order_independent_and_resets_per_slot() -> None:
    def context(slot_ids: list[int], reset: list[bool]):
        rows = len(slot_ids)
        return exp.Context(
            features={"probe": torch.zeros(rows, 1)},
            ctx_pad=torch.zeros(rows, dtype=torch.long),
            slot_ids=torch.tensor(slot_ids),
            reset=torch.tensor(reset),
        )

    ordered = exp.SlotFlowRandom(19)
    ordered.begin(context([3, 7], [True, True]))
    first = ordered.noise(10)
    reordered = exp.SlotFlowRandom(19)
    reordered.begin(context([7, 3], [True, True]))
    second = reordered.noise(10)
    for one, two in zip(first, second, strict=True):
        torch.testing.assert_close(one[0], two[1])
        torch.testing.assert_close(one[1], two[0])

    ordered.advance()
    ordered.begin(context([3], [False]))
    advanced = ordered.noise(10)
    assert not torch.equal(first[0][0], advanced[0][0])
    ordered.begin(context([3], [True]))
    reset = ordered.noise(10)
    assert not torch.equal(advanced[0][0], reset[0][0])


def test_optimizer_roles_are_exhaustive_for_adamw_and_muon() -> None:
    cfg = _tiny_cfg()
    model = exp.GPT(cfg)
    for optimizer_name in ("adamw", "muon"):
        treatment = replace(cfg, optimizer=optimizer_name)
        roles = exp.optimizer_roles(model, treatment)
        optimizer = exp.make_optimizer(model, treatment)
        members = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        assert set(roles) == dict(model.named_parameters()).keys()
        assert len(members) == len({id(parameter) for parameter in members})
        assert {id(parameter) for parameter in members} == {id(parameter) for parameter in model.parameters()}
    muon_roles = exp.optimizer_roles(model, replace(cfg, optimizer="muon"))
    assert muon_roles["flow.blocks.0.qkv.weight"].logical_splits == 3
    assert muon_roles["flow.blocks.0.ffn_up.weight"].logical_splits == 2
    assert muon_roles["flow.blocks.0.modulation.weight"].optimizer == "adamw"
    assert muon_roles["flow.action_fusion_in.weight"].optimizer == "muon"
    optimizer = exp.make_optimizer(model, replace(cfg, optimizer="muon"))
    muon_groups = [group for group in optimizer.param_groups if group["use_muon"]]
    adam_groups = [group for group in optimizer.param_groups if not group["use_muon"]]
    assert {group["lr"] for group in muon_groups} == {0.014}
    assert {group["momentum"] for group in muon_groups} == {0.95}
    assert {group["muon_scale_clamp_min_one"] for group in muon_groups} == {False}
    assert {group["betas"] for group in adam_groups} == {(0.9, 0.95)}
    assert {group["eps"] for group in adam_groups} == {1e-12}

    model.zero_grad()
    sum(parameter.sum() for parameter in model.parameters()).backward()
    gradient_norms = exp._subsystem_gradient_norms(model)
    assert gradient_norms.keys() == {"trunk", "flow_decoder", "group_heads", "value_head", "other"}
    assert all(float(value) > 0 for value in gradient_norms.values())


def test_awr_is_off_by_default_and_weights_whole_plans_without_renormalizing() -> None:
    advantage = torch.tensor([[0.0, 199.5]], requires_grad=True)
    eligible = torch.ones_like(advantage, dtype=torch.bool)
    weights, _ = exp.advantage_weights(advantage, eligible, beta=199.5, weight_max=3.5)

    torch.testing.assert_close(weights, torch.tensor([[1.0, torch.e]]))
    assert not weights.requires_grad
    nll = torch.ones(1, 2, 3, 4)
    committed = torch.zeros(1, 2, 3, dtype=torch.bool)
    _, _, loss = exp.flow_objective_parts(
        nll,
        weights,
        committed,
        valid_prefixes=2,
        valid=torch.ones(1, 2, dtype=torch.bool),
    )
    torch.testing.assert_close(loss, weights.mean())


def test_value_head_is_detached_from_history_and_flow_metrics_are_unweighted() -> None:
    model = exp.GPT(_tiny_cfg())
    hidden = torch.randn(2, 3, model.cfg.arch.d_model, requires_grad=True)
    value = exp.detached_value_prediction(model, hidden)
    value.square().mean().backward()
    assert hidden.grad is None
    assert model.value_head.down.weight.grad is not None

    mean_nll = torch.cat((torch.ones(6, 4), torch.full((4, 4), 3.0)))
    metrics = exp.nll_mean_metrics(mean_nll, model.head_offsets)
    assert metrics["loss_unweighted"] == pytest.approx(7.2 / np.log(2))


def test_flow_diagnostics_exclude_padding_and_committed_actions() -> None:
    nll = torch.ones(1, 3, 10, 4)
    nll[:, 2].fill_(100)
    correct = torch.ones_like(nll, dtype=torch.bool)
    valid = torch.tensor([[True, True, False]])
    committed = torch.zeros(1, 3, 10, dtype=torch.bool)
    committed[:, 0, :2] = True
    tau = torch.tensor([[0.1, 0.6, 0.9]])

    values = exp._flow_metric_sums(nll, correct, valid=valid, committed=committed, tau=tau)
    assert torch.equal(values.count_by_offset, torch.tensor([1, 1, 2, 2, 2, 2, 2, 2, 2, 2]))
    torch.testing.assert_close(
        values.nll_by_offset_group,
        values.count_by_offset[:, None].expand(-1, 4).float(),
    )
    assert values.count_by_time.sum() == values.count_by_offset.sum() * 4
    assert values.count_by_commitment.sum() == values.count_by_offset.sum() * 4


def test_training_metric_accumulator_uses_exact_flow_denominators() -> None:
    cfg = _tiny_cfg()
    offset_count = torch.arange(1, 11).float()
    time_count = torch.arange(1, 5).float()
    commitment_count = torch.arange(1, 6).float()
    flow = exp.FlowMetricSums(
        nll_by_offset_group=offset_count[:, None].expand(-1, 4),
        correct_by_offset_group=0.5 * offset_count[:, None].expand(-1, 4),
        count_by_offset=offset_count,
        nll_by_time=time_count,
        correct_by_time=0.5 * time_count,
        count_by_time=time_count,
        nll_by_commitment=commitment_count,
        correct_by_commitment=0.5 * commitment_count,
        count_by_commitment=commitment_count,
    )
    result = exp.TrainStepResult(flow, torch.tensor(0.25), {"probe": torch.tensor(2.0)}, 1e-3)
    accumulator = exp._TrainingMetricAccumulator()
    accumulator.add(result, valid_prefixes=17)
    metrics, updates, prefixes = accumulator.flush(cfg, update=1)

    assert (updates, prefixes) == (1, 17)
    assert metrics["train/nll"] == pytest.approx(4 / np.log(2))
    assert metrics["train/acc_o01_buttons"] == pytest.approx(0.5)
    assert metrics["flow/nll_tau_bin_3"] == pytest.approx(1 / np.log(2))
    assert metrics["flow/acc_committed_4"] == pytest.approx(0.5)


def test_padding_is_excluded_from_final_128_prefixes() -> None:
    cfg = _tiny_cfg()
    model = exp.GPT(cfg)
    batch = exp.synthetic_awr_batch(cfg, torch.device("cpu"))
    context = replace(batch.context, ctx_pad=torch.tensor([0, 7]))
    padded = exp.AWRBatch(replace(batch.batch, context=context), batch.returns, batch.eligible)
    history, targets, valid = exp.prepared_targets(model, padded)

    assert history.shape == (2, 4, exp.CONTROLLER_GROUP_COUNT)
    assert targets.shape == (2, 4, 10, exp.CONTROLLER_GROUP_COUNT)
    assert torch.equal(valid, torch.tensor([[True, True, True, True], [False, False, False, True]]))
    full_actions = torch.cat((exp.stack_actions(padded.context.features), padded.target), dim=1)
    quantized = model.codec.quantize(full_actions)
    for depth, offset in enumerate(model.head_offsets):
        assert torch.equal(targets[:, :, depth], quantized[:, 4 + offset : 8 + offset])


def test_history_shuffle_preserves_padding_and_current_frame() -> None:
    context = exp.Context(
        features={"probe": torch.arange(12).view(2, 6)},
        ctx_pad=torch.tensor([0, 2]),
    )
    shuffled = exp.shuffled_history_context(context, seed=5)
    values = shuffled.features["probe"]

    assert values[0, -1] == context.features["probe"][0, -1]
    assert torch.equal(values[1, :2], context.features["probe"][1, :2])
    assert values[1, -1] == context.features["probe"][1, -1]
    assert set(values[0, :-1].tolist()) == set(context.features["probe"][0, :-1].tolist())
    assert set(values[1, 2:-1].tolist()) == set(context.features["probe"][1, 2:-1].tolist())


def test_delayed_policy_conditions_on_pending_actions_and_records_applied_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    commitments = []

    def predict(_context, committed):
        nonlocal calls
        commitments.append(None if committed is None else committed.copy())
        plan = np.zeros((1, 10, exp.A_DIM), dtype=np.float32)
        plan[0, :, 0] = 10 * calls + np.arange(1, 11)
        calls += 1
        return plan

    telemetry = exp.DecodeTelemetry()
    policy = exp.DelayedTruncationPolicy(
        predict_chunk=predict,
        stats={},
        L_ctx=8,
        L_chunk=10,
        s=2,
        d=0,
        delay_frames=2,
        device="cpu",
        replan_telemetry=telemetry,
    )
    slot = exp.Slot(0, 1)
    state = SimpleNamespace(reset_pending=True)
    policy._slots[slot] = state
    policy._ingest = lambda _live, _obs: None

    def context(_live):
        state.reset_pending = False
        return object()

    policy._context = context
    applied = []
    policy._push_ego = lambda _slot, action: applied.append(action.copy())
    monkeypatch.setattr(exp, "action_vec_to_controller", lambda action: action)

    actions = [float(policy(frame, {slot: {}})[slot][0]) for frame in range(6)]
    assert actions == [0.0, 0.0, 3.0, 4.0, 13.0, 14.0]
    assert [values[0, :, 0].tolist() for values in commitments] == [[0.0, 0.0], [3.0, 4.0], [13.0, 14.0]]
    assert [float(action[0]) for action in applied] == actions
    assert policy.overlap_frames == 16
    assert policy.overlap_disagreements == 16
    assert (telemetry.calls, telemetry.rows, telemetry.executed_frames) == (3, 3, 6)

    state.reset_pending = True
    reset_action = policy(6, {slot: {}})[slot]
    assert float(reset_action[0]) == 0.0
    assert commitments[-1][0, :, 0].tolist() == [0.0, 0.0]
    assert policy.overlap_frames == 16
    assert (telemetry.calls, telemetry.executed_frames) == (4, 8)


def test_exact_resume_reproduces_next_adamw_update() -> None:
    cfg = _tiny_cfg(execution_batch_size=2)
    torch.manual_seed(12)
    model = exp.GPT(cfg)
    optimizer = exp.make_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, exp.lr_schedule(cfg))
    batch = exp.synthetic_awr_batch(cfg, torch.device("cpu"))
    valid = cfg.batch_size * (cfg.arch.L_ctx - cfg.arch.direct_loss_start)
    exp.train_step(
        model,
        batch,
        cfg,
        step=0,
        update=1,
        valid_prefixes=valid,
        trunk_fn=model.forward,
        flow_fn=model.flow.training_statistics,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    resumed = exp.GPT(cfg)
    resumed.load_state_dict(copy.deepcopy(model.state_dict()))
    resumed_optimizer = exp.make_optimizer(resumed, cfg)
    resumed_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(resumed_optimizer, exp.lr_schedule(cfg))
    resumed_scheduler.load_state_dict(copy.deepcopy(scheduler.state_dict()))

    for candidate, candidate_optimizer, candidate_scheduler in (
        (model, optimizer, scheduler),
        (resumed, resumed_optimizer, resumed_scheduler),
    ):
        exp.train_step(
            candidate,
            batch,
            cfg,
            step=1,
            update=2,
            valid_prefixes=valid,
            trunk_fn=candidate.forward,
            flow_fn=candidate.flow.training_statistics,
            optimizer=candidate_optimizer,
            scheduler=candidate_scheduler,
        )
    for actual, expected in zip(model.parameters(), resumed.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)


def test_loader_and_identity_rng_resume_the_exact_next_batch() -> None:
    cfg = _tiny_cfg()

    class Loader:
        def __init__(self) -> None:
            self.cursor = 0

        def __iter__(self):
            return self

        def __next__(self):
            batch = exp.synthetic_awr_batch(cfg, torch.device("cpu"))
            batch.context.features["ego_player_id"].fill_(self.cursor + 3)
            batch.target.fill_(self.cursor + 1)
            self.cursor += 1
            return batch

        def state_dict(self):
            return {"cursor": self.cursor}

        def load_state_dict(self, state):
            self.cursor = state["cursor"]

    loader = Loader()
    masker = exp.IdentityMasker(23, 0.5)
    prefetcher = exp.DeviceBatchPrefetcher(loader, cfg, "cpu", masker)
    try:
        for update in range(1, 9):
            prefetcher.next()
            prefetcher.fill_lookahead(0 if update == 8 else 8 - update)
            if update < 8:
                prefetcher.stage_next()
        assert prefetcher.drained
        loader_state = loader.state_dict()
        masker_state = masker.state_dict()
    finally:
        prefetcher.close()

    expected_prefetcher = exp.DeviceBatchPrefetcher(loader, cfg, "cpu", masker)
    try:
        expected_batch, expected_prefixes = expected_prefetcher.next()
    finally:
        expected_prefetcher.close()

    restored_loader = Loader()
    restored_loader.load_state_dict(loader_state)
    restored_masker = exp.IdentityMasker(999, 0.5)
    restored_masker.load_state_dict(masker_state)
    restored_prefetcher = exp.DeviceBatchPrefetcher(restored_loader, cfg, "cpu", restored_masker)
    try:
        actual_batch, actual_prefixes = restored_prefetcher.next()
    finally:
        restored_prefetcher.close()

    assert actual_prefixes == expected_prefixes
    torch.testing.assert_close(actual_batch.target, expected_batch.target)
    torch.testing.assert_close(
        actual_batch.context.features["ego_player_id"],
        expected_batch.context.features["ego_player_id"],
    )
    assert restored_masker.state_dict()["forced"] == masker.state_dict()["forced"]


def test_accumulated_and_unsplit_updates_match() -> None:
    split_cfg = _tiny_cfg(execution_batch_size=1)
    full_cfg = replace(split_cfg, execution_batch_size=2)
    torch.manual_seed(31)
    split_model = exp.GPT(split_cfg)
    full_model = exp.GPT(full_cfg)
    full_model.load_state_dict(copy.deepcopy(split_model.state_dict()))
    batch = exp.synthetic_awr_batch(split_cfg, torch.device("cpu"))
    valid = split_cfg.batch_size * (split_cfg.arch.L_ctx - split_cfg.arch.direct_loss_start)

    for model, cfg in ((split_model, split_cfg), (full_model, full_cfg)):
        optimizer = exp.make_optimizer(model, cfg)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, exp.lr_schedule(cfg))
        exp.train_step(
            model,
            batch,
            cfg,
            step=0,
            update=1,
            valid_prefixes=valid,
            trunk_fn=model.forward,
            flow_fn=model.flow.training_statistics,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    for actual, expected in zip(split_model.parameters(), full_model.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
