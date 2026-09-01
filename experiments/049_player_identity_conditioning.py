"""Experiment 049: ego-player identity conditioning on the O43 policy.

Only the controlling player's identity is visible. Professional examples use
an exact training connect code; ranked-anonymous examples use Platinum,
Diamond, or Master. Missing identities use the fixed zero embedding. Opponent
identity is never present in a model context.

The checkpoint carries its ordered connect-code vocabulary, so selecting a
professional style at inference needs only a string:

    model, cfg, stats, _ = load_checkpoint("runs/<run>/final.pt")
    engine = BF16Inference(model, cfg, player_code="MANG#0")
    actions = engine.decode(context, horizon=1)

Train the matched arms:

    uv run experiments/049_player_identity_conditioning.py --cfg.treatment control
    uv run experiments/049_player_identity_conditioning.py --cfg.treatment conditioned

Evaluate a professional identity:

    uv run experiments/049_player_identity_conditioning.py \
      --eval runs/<run>/final.pt --eval-player-code 'MANG#0'
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Iterable
from collections.abc import Iterator
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from dataclasses import fields
from dataclasses import replace
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import tyro
from torch import Tensor

import wandb
from hal import r2
from hal import streams
from hal.data.feature_stats import FeatureStats
from hal.data.policy_world_schema import decode_policy_world_replay_slices
from hal.data.schema import Rank
from hal.data.schema import check_schema_version
from hal.training.dataloader import _choose_chunk_starts
from hal.training.dataloader import _make_streaming_dataset
from hal.training.dataloader import _make_window
from hal.training.dataloader import collate_train_batch
from hal.training.ego_stats import load_consolidated_mixture_stats
from hal.training.features import BASE_ACTION_PROJECTION
from hal.training.features import Context
from hal.training.features import ExtraColumns
from hal.training.features import FeatureProjection
from hal.training.features import TrainBatch
from hal.training.player_identity import MASKED_PLAYER_ID
from hal.training.player_identity import PlayerIdentitySidecar
from hal.training.player_identity import PlayerVocabulary
from hal.training.player_identity import ReplayPlayerLookup
from hal.training.player_identity import decode_player_codes
from hal.training.player_identity import load_player_identity_sidecar
from hal.training.player_identity import vocabulary_buffer
from hal.training.replay_reservoir import _stable_replay_rng


def _load_o43() -> ModuleType:
    path = Path(__file__).with_name("043_legacy_codec.py")
    name = "_hal_experiment_043_for_049"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_o43 = _load_o43()
_validate_o43_config = _o43.validate_config
_base_eval_suites = _o43.eval_suites
_BaseGPT = _o43.GPT
_base_model_tag = _o43.model_tag
_base_synthetic_context = _o43.synthetic_context

EXPERIMENT_ID = "049_ego_player_identity_v1"
PLAYER_SIDECAR_LOCAL = "data/processed/player-identity-v1/professional-code-v1.jsonl.gz"
PLAYER_SIDECAR_REMOTE = "s3://hal/processed/player-identity-v1/professional-code-v1.jsonl.gz"
# Frozen from the immutable 38-manifest build.
PLAYER_SIDECAR_SHA256 = "54ccf8a2497fe240313117297ca2ea31158e08db2cc53c67e7aa46853a8dac1c"
PLAYER_VOCAB_SHA256 = "c67c97c995ad033ea7f5b2223efce5b061394566439f091ff6e7aaa6a9d1cfd6"
PLAYER_VOCAB_SIZE = 21_181
PLAYER_EMBED_DIM = 32
TRAIN_SOURCE_NAMES = tuple(source.name for source in streams.POLICY_WORLD_V7_SOURCES)

PLAYER_COLUMNS = ExtraColumns(floats={}, cats={"player_id": None})
PLAYER_PROJECTION = FeatureProjection(
    columns=BASE_ACTION_PROJECTION.columns | {"ego_player_id"},
    derive_spatial=False,
)


@dataclass
class TrainConfig(_o43.TrainConfig):
    """The paired O49 protocol; only ``treatment`` differs between arms."""

    treatment: Literal["control", "conditioned"] = "conditioned"
    experiment_id: str = EXPERIMENT_ID
    player_embed_dim: int = PLAYER_EMBED_DIM
    player_vocab_size: int = PLAYER_VOCAB_SIZE
    player_vocab_sha256: str = PLAYER_VOCAB_SHA256
    player_sidecar_local: str = PLAYER_SIDECAR_LOCAL
    player_sidecar_remote: str = PLAYER_SIDECAR_REMOTE
    player_sidecar_sha256: str = PLAYER_SIDECAR_SHA256
    train_source_names: tuple[str, ...] = TRAIN_SOURCE_NAMES
    eval_player_rank: Literal["PLATINUM", "DIAMOND", "MASTER"] = "MASTER"

    data_root: str = "data/processed/player-identity-v1/full-mix"
    replay_format: Literal["policy", "policy-world"] = "policy-world"
    val_data_root: str = "data/processed/ranked-anonymized-1/mds-policy-world-v7"
    val_replay_format: Literal["policy", "policy-world"] = "policy-world"
    cache_limit_gb: int = 1792


def validate_config(cfg: TrainConfig, *, require_frozen_artifact: bool = False) -> None:
    _validate_o43_config(cfg)
    if cfg.experiment_id != EXPERIMENT_ID:
        raise ValueError(f"experiment_id must be {EXPERIMENT_ID!r}")
    if cfg.treatment not in ("control", "conditioned"):
        raise ValueError(f"unknown O49 treatment {cfg.treatment!r}")
    if cfg.ablation_arm != "A":
        raise ValueError("O49 is based only on O43 arm A")
    if cfg.observation_bundle != "base":
        raise ValueError("O49 freezes O43's base observation bundle")
    if cfg.player_embed_dim != PLAYER_EMBED_DIM:
        raise ValueError(f"player_embed_dim must be {PLAYER_EMBED_DIM}")
    if cfg.player_vocab_size != PLAYER_VOCAB_SIZE:
        raise ValueError(f"player_vocab_size must be the frozen value {PLAYER_VOCAB_SIZE}")
    if cfg.train_source_names != TRAIN_SOURCE_NAMES:
        raise ValueError("O49 source names differ from the frozen 44-source mix")
    if cfg.replay_format != "policy-world" or cfg.val_replay_format != "policy-world":
        raise ValueError("O49 requires policy-world train and validation rows")
    for name, value in (
        ("player_vocab_sha256", cfg.player_vocab_sha256),
        ("player_sidecar_sha256", cfg.player_sidecar_sha256),
    ):
        if value and (len(value) != 64 or any(character not in "0123456789abcdef" for character in value)):
            raise ValueError(f"{name} must be an empty string or a lowercase SHA-256 digest")
        if require_frozen_artifact and not value:
            raise ValueError(f"production O49 requires a frozen {name}")
    if require_frozen_artifact and cfg.player_vocab_sha256 != PLAYER_VOCAB_SHA256:
        raise ValueError("player_vocab_sha256 differs from the frozen O49 vocabulary")
    if require_frozen_artifact and cfg.player_sidecar_sha256 != PLAYER_SIDECAR_SHA256:
        raise ValueError("player_sidecar_sha256 differs from the frozen O49 sidecar")


def _download_sidecar(local: Path, remote: str) -> None:
    if not remote.startswith("s3://"):
        raise ValueError(f"player sidecar remote must be s3://, got {remote!r}")
    bucket, _, key = remote[len("s3://") :].partition("/")
    local.parent.mkdir(parents=True, exist_ok=True)
    r2.client().download_file(bucket, key, str(local))


@cache
def _load_sidecar(path: str, remote: str, expected_sha256: str) -> PlayerIdentitySidecar:
    local = Path(path)
    if not local.is_file():
        _download_sidecar(local, remote)
    return load_player_identity_sidecar(local, expected_sha256=expected_sha256 or None)


def load_sidecar(cfg: TrainConfig) -> PlayerIdentitySidecar:
    sidecar = _load_sidecar(cfg.player_sidecar_local, cfg.player_sidecar_remote, cfg.player_sidecar_sha256)
    if sidecar.vocabulary.size != cfg.player_vocab_size:
        raise ValueError(f"player vocabulary size {sidecar.vocabulary.size} != configured {cfg.player_vocab_size}")
    if cfg.player_vocab_sha256 and sidecar.vocabulary.sha256 != cfg.player_vocab_sha256:
        raise ValueError(
            f"player vocabulary SHA-256 {sidecar.vocabulary.sha256} != configured {cfg.player_vocab_sha256}"
        )
    return sidecar


def _vocabulary_from_state(state: dict[str, Tensor]) -> PlayerVocabulary:
    value = state.get("player_code_bytes")
    if value is None:
        raise ValueError("checkpoint has no embedded O49 player-code vocabulary")
    return PlayerVocabulary(decode_player_codes(value.detach().cpu().numpy().tobytes()))


class GPT(_BaseGPT):
    """O43 plus one additive ego-player embedding path."""

    def __init__(self, cfg: TrainConfig, vocabulary: PlayerVocabulary | None = None) -> None:
        super().__init__(cfg)
        vocabulary = load_sidecar(cfg).vocabulary if vocabulary is None else vocabulary
        if vocabulary.size != cfg.player_vocab_size:
            raise ValueError(f"vocabulary size {vocabulary.size} != configured {cfg.player_vocab_size}")
        if cfg.player_vocab_sha256 and vocabulary.sha256 != cfg.player_vocab_sha256:
            raise ValueError("model vocabulary does not match player_vocab_sha256")
        self.player_vocabulary = vocabulary
        self.player_embedding = nn.Embedding(
            cfg.player_vocab_size,
            cfg.player_embed_dim,
            padding_idx=MASKED_PLAYER_ID,
        )
        self.player_projection = nn.Linear(cfg.player_embed_dim, cfg.d_model, bias=False)
        nn.init.zeros_(self.player_projection.weight)
        self.register_buffer(
            "player_code_bytes",
            torch.from_numpy(vocabulary_buffer(vocabulary)),
            persistent=True,
        )

    def context_tokens(self, features: dict[str, Tensor], action_indices: Tensor | None = None) -> Tensor:
        if "opp_player_id" in features:
            raise ValueError("opponent player identity must never enter an O49 context")
        if "ego_player_id" not in features:
            raise KeyError("O49 context is missing ego_player_id")
        tokens = super().context_tokens(features, action_indices)
        player_ids = features["ego_player_id"]
        if self.cfg.treatment == "control":
            player_ids = torch.zeros_like(player_ids)
        return tokens + self.player_projection(self.player_embedding(player_ids))


def make_optimizer(model: GPT, cfg: TrainConfig):
    """Keep the player embedding with O43's other no-decay embeddings."""
    muon = [parameter for parameter in model.trunk.blocks.parameters() if parameter.ndim >= 2]
    muon_ids = {id(parameter) for parameter in muon}
    embedding_modules = (
        model.cat_embeds,
        model.v6_cat_embeds,
        model.char_emb,
        model.stage_emb,
        model.codec.class_embeddings,
        model.temporal.offset_embedding,
        model.player_embedding,
    )
    embedding_ids = {id(parameter) for module in embedding_modules for parameter in module.parameters()}
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for parameter in model.parameters():
        if id(parameter) in muon_ids:
            continue
        (no_decay if parameter.ndim < 2 or id(parameter) in embedding_ids else decay).append(parameter)
    if len(muon) + len(decay) + len(no_decay) != sum(1 for _ in model.parameters()):
        raise RuntimeError("optimizer parameter partition is incomplete")
    adam = dict(betas=(0.9, 0.95), eps=1e-10, use_muon=False)
    return _o43.SingleDeviceMuonWithAuxAdam(
        [
            dict(
                params=muon,
                lr=cfg.muon_lr,
                momentum=0.95,
                weight_decay=cfg.muon_weight_decay,
                use_muon=True,
            ),
            dict(params=decay, lr=cfg.adam_lr, weight_decay=cfg.adam_weight_decay, **adam),
            dict(params=no_decay, lr=cfg.adam_lr, weight_decay=0.0, **adam),
        ]
    )


def _rank_labels(sample: dict[str, object]) -> dict[str, object]:
    """Attach ranked IDs to decoded validation rows before ego relabeling."""
    out = dict(sample)
    for port in (1, 2):
        key = f"p{port}_rank"
        ranks = np.asarray(out[key])
        unique = np.unique(ranks)
        if len(unique) != 1 or int(unique[0]) not in (int(Rank.PLATINUM), int(Rank.DIAMOND), int(Rank.MASTER)):
            raise ValueError(f"ranked validation row has unsupported {key} values {unique.tolist()}")
        out[f"p{port}_player_id"] = ranks.astype(np.int32, copy=False)
    return out


def loader_kwargs(cfg: TrainConfig, stats: dict[str, FeatureStats]) -> dict[str, object]:
    selected = tuple(streams.BY_NAME[name] for name in cfg.train_source_names)
    return dict(
        data_root=None,
        sources=selected,
        cache_limit=f"{cfg.cache_limit_gb}gb",
        shuffle_block_size=cfg.shuffle_block_size,
        shuffle_seed=cfg.seed,
        stats=stats,
        L_ctx=cfg.L_ctx,
        L_chunk=cfg.sample_chunk_length,
        batch_size=_o43.micro_batch_size(cfg),
        seed=cfg.seed,
        schema_version=cfg.mds_schema_version,
        extra=PLAYER_COLUMNS,
        projection=PLAYER_PROJECTION,
    )


def _make_loaders(cfg: TrainConfig, stats: dict[str, FeatureStats]):
    sidecar = load_sidecar(cfg)
    lookup = ReplayPlayerLookup(sidecar.by_replay)
    kwargs = loader_kwargs(cfg, stats)
    train_loader = _o43.make_reservoir_loader(
        split="train",
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor,
        predownload=cfg.predownload,
        windows_per_replay=cfg.windows_per_replay,
        reservoir_capacity=cfg.reservoir_capacity,
        prefetch_batches=cfg.prefetch_batches,
        replay_format="policy-world",
        replay_labels=lookup,
        **kwargs,
    )
    val_source = streams.RANKED_ANONYMIZED_POLICY_WORLD_V7[0]
    val_kwargs = {
        **kwargs,
        "sources": (val_source,),
        "batch_size": cfg.val_batch_size,
        "cache_limit": f"{min(cfg.cache_limit_gb, 160)}gb",
    }
    val_loader = _o43.make_loader(
        split=cfg.val_split,
        num_workers=0,
        replay_format="policy-world",
        replay_transform=_rank_labels,
        **val_kwargs,
    )
    return train_loader, _o43.cache_validation(val_loader, cfg.val_n_samples)


def load_stats(cfg: TrainConfig) -> dict[str, FeatureStats]:
    sources = tuple(streams.BY_NAME[name] for name in cfg.train_source_names)
    weights = tuple(float(streams.POLICY_WORLD_V7_TRAIN_REPLAYS[source.name]) for source in sources)
    return load_consolidated_mixture_stats(
        [source.local_root / "stats.json" for source in sources],
        weights,
        expected_mds_schema_version=cfg.mds_schema_version,
    )


def identity_audit_batches(
    cfg: TrainConfig,
    stats: dict[str, FeatureStats],
) -> Iterator[TrainBatch]:
    """Yield one deterministic validation window for each replay and ego side."""
    selected = tuple(streams.BY_NAME[name] for name in cfg.train_source_names)
    dataset, _ = _make_streaming_dataset(
        None,
        cfg.val_split,
        sources=selected,
        remote=None,
        cache_limit=f"{cfg.cache_limit_gb}gb",
        shuffle=False,
        shuffle_seed=None,
        shuffle_block_size=cfg.shuffle_block_size,
        predownload=cfg.predownload,
    )
    lookup = ReplayPlayerLookup(load_sidecar(cfg).by_replay)
    windows: list[dict[str, np.ndarray]] = []
    for compact in dataset:
        check_schema_version(
            {"schema_version": int(compact["source_schema_version"])},
            expected=cfg.mds_schema_version,
        )
        replay_id = str(compact["replay_id"])
        frames = int(compact["num_frames"])
        rng = _stable_replay_rng(cfg.eval_seed, 0, replay_id)
        starts = _choose_chunk_starts(frames, cfg.L_ctx, cfg.sample_chunk_length, 1, rng)
        if not starts.size:
            continue
        start = int(starts[0]) - cfg.L_ctx
        range_start = max(0, start)
        range_stop = start + cfg.L_ctx + cfg.sample_chunk_length
        sample = decode_policy_world_replay_slices(compact, ((range_start, range_stop),))[0]
        labels = lookup(compact)
        sample.update({name: value[range_start:range_stop] for name, value in labels.items()})
        pad = max(0, -start)
        for ego_prefix in ("p1", "p2"):
            window = _make_window(
                sample,
                ego_prefix=ego_prefix,
                start=0,
                pad=pad,
                length=cfg.L_ctx + cfg.sample_chunk_length,
                projection=PLAYER_PROJECTION,
            )
            window["ctx_pad"] = np.int64(min(pad, cfg.L_ctx))
            windows.append(window)
            if len(windows) == cfg.val_batch_size:
                yield collate_train_batch(
                    windows,
                    stats=stats,
                    L_ctx=cfg.L_ctx,
                    extra=PLAYER_COLUMNS,
                    projection=PLAYER_PROJECTION,
                )
                windows = []
    if windows:
        yield collate_train_batch(
            windows,
            stats=stats,
            L_ctx=cfg.L_ctx,
            extra=PLAYER_COLUMNS,
            projection=PLAYER_PROJECTION,
        )


def _identity_label(vocabulary: PlayerVocabulary, player_id: int) -> str:
    if player_id == MASKED_PLAYER_ID:
        return "<MASK>"
    if player_id in (int(Rank.PLATINUM), int(Rank.DIAMOND), int(Rank.MASTER)):
        return Rank(player_id).name.title()
    index = player_id - 4
    if not 0 <= index < len(vocabulary.codes):
        raise ValueError(f"player ID {player_id} is outside the checkpoint vocabulary")
    return vocabulary.codes[index]


@torch.no_grad()
def identity_validation_metrics(
    model: GPT,
    batches: Iterable[TrainBatch],
    cfg: TrainConfig,
) -> dict[str, object]:
    """Aggregate teacher-forced NLL by exact ego identity."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    sums: dict[int, Tensor] = {}
    prefix_counts: dict[int, int] = {}
    window_counts: dict[int, int] = {}
    try:
        for cpu_batch in batches:
            batch = cpu_batch.to(device)
            player_ids = batch.context.features["ego_player_id"][:, 0]
            if not torch.all(batch.context.features["ego_player_id"] == player_ids[:, None]):
                raise ValueError("identity audit found a player ID that changes within one window")
            history, targets, valid = _o43.prepared_targets(model, batch)
            with _o43.amp_context(cfg, device):
                hidden = model(batch.context.features, batch.context.ctx_pad, history)
                dense_nll = model.temporal.teacher_forced_nll(hidden, history, targets)
            for row, player_id in enumerate(player_ids.tolist()):
                selected = dense_nll[row][valid[row]].double().cpu()
                sums.setdefault(player_id, torch.zeros_like(selected.sum(dim=0)))
                sums[player_id] += selected.sum(dim=0)
                prefix_counts[player_id] = prefix_counts.get(player_id, 0) + selected.shape[0]
                window_counts[player_id] = window_counts.get(player_id, 0) + 1
    finally:
        model.train(was_training)
    if not sums:
        raise RuntimeError("identity validation contained no windows")

    def metrics(player_id: int) -> dict[str, object]:
        means = sums[player_id] / prefix_counts[player_id]
        all_metrics = _o43.nll_mean_metrics(
            means,
            model.head_offsets,
            aux_loss_weight=cfg.aux_loss_weight,
            next_frame_loss_share=cfg.next_frame_loss_share,
        )
        return {
            "player_id": player_id,
            "identity": _identity_label(model.player_vocabulary, player_id),
            "windows": window_counts[player_id],
            "valid_prefixes": prefix_counts[player_id],
            "loss_bits": all_metrics["loss"],
            "joint_nll_bits": {f"o{offset}": all_metrics[f"nll_o{offset:02d}"] for offset in model.head_offsets},
        }

    total_sum = sum(sums.values(), torch.zeros_like(next(iter(sums.values()))))
    total_prefixes = sum(prefix_counts.values())
    global_metrics = _o43.nll_mean_metrics(
        total_sum / total_prefixes,
        model.head_offsets,
        aux_loss_weight=cfg.aux_loss_weight,
        next_frame_loss_share=cfg.next_frame_loss_share,
    )
    total_windows = sum(window_counts.values())
    if total_windows % 2:
        raise RuntimeError(f"identity validation emitted an odd window count {total_windows}")
    return {
        "schema_version": 1,
        "selection": "one deterministic window per replay and each of p1/p2 as ego",
        "opponent_identity_conditioned": False,
        "source_names": list(cfg.train_source_names),
        "total_replays": total_windows // 2,
        "total_windows": total_windows,
        "total_valid_prefixes": total_prefixes,
        "global_loss_bits": global_metrics["loss"],
        "identities": [metrics(player_id) for player_id in sorted(sums)],
    }


def identity_audit_checkpoint(path: str, *, output: str | None = None) -> Path:
    """Write the full 44-source validation NLL artifact for one checkpoint."""
    model, cfg, stats, state = load_checkpoint(path)
    result = identity_validation_metrics(model, identity_audit_batches(cfg, stats), cfg)
    result.update(
        {
            "checkpoint": str(Path(path).resolve()),
            "checkpoint_sha256": _o43._checkpoint_sha256(Path(path)),
            "checkpoint_step": int(state["step"]),
            "treatment": cfg.treatment,
            "player_sidecar_sha256": cfg.player_sidecar_sha256,
            "player_vocab_sha256": cfg.player_vocab_sha256,
        }
    )
    destination = Path(output) if output is not None else Path(path).resolve().parent / "identity_validation.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"[identity-audit] wrote {destination}", flush=True)
    return destination


def _condition_context(ctx: Context, player_id: int) -> Context:
    if "opp_player_id" in ctx.features:
        raise ValueError("opponent player identity must never enter O49 inference")
    reference = ctx.features[next(iter(ctx.features))]
    features = dict(ctx.features)
    features["ego_player_id"] = torch.full(
        reference.shape[:2],
        player_id,
        dtype=torch.long,
        device=reference.device,
    )
    return replace(ctx, features=features)


def synthetic_context(
    cfg: TrainConfig,
    batch_size: int,
    device: torch.device,
    **kwargs,
) -> Context:
    """Build an O43 synthetic context with masked O49 identity."""
    context = _base_synthetic_context(cfg, batch_size, device, **kwargs)
    return _condition_context(context, MASKED_PLAYER_ID)


class BF16Inference(_o43.BF16Inference):
    """O43 inference with one explicit, constant ego identity.

    Pass ``player_code="MANG#0"`` for a professional style or
    ``player_rank=Rank.MASTER`` for a ranked aggregate. The selectors are
    mutually exclusive. An unknown explicit code raises ``KeyError``.
    """

    def __init__(
        self,
        model: GPT,
        cfg: TrainConfig,
        *,
        player_code: str | None = None,
        player_rank: Rank | None = None,
        **kwargs,
    ) -> None:
        if player_code is not None and player_rank is not None:
            raise ValueError("pass player_code or player_rank, not both")
        if player_code is not None:
            player_id = model.player_vocabulary.id_for_code(player_code)
            player_key = player_code.strip()
        else:
            rank = Rank[cfg.eval_player_rank] if player_rank is None else player_rank
            player_id = model.player_vocabulary.id_for_rank(rank)
            player_key = rank.name.title()
        self.player_id = player_id
        self.player_key = player_key
        super().__init__(model, cfg, **kwargs)

    def decode(self, ctx: Context, horizon: int, **kwargs) -> Tensor:
        return super().decode(_condition_context(ctx, self.player_id), horizon, **kwargs)


def eval_suites(*args, replay_dir: Path, inference: BF16Inference, **kwargs):
    suites = _base_eval_suites(*args, replay_dir=replay_dir, inference=inference, **kwargs)
    evidence = {
        "schema_version": 1,
        "ego_player_id": inference.player_id,
        "ego_player_key": inference.player_key,
        "opponent_identity_conditioned": False,
    }
    replay_dir.mkdir(parents=True, exist_ok=True)
    (replay_dir / "player_identity.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    for metrics in suites.values():
        metrics["ego_player_id"] = float(inference.player_id)
    return suites


def model_tag(cfg: TrainConfig) -> str:
    return f"{_base_model_tag(cfg)}-ego-id{cfg.player_embed_dim}-{cfg.treatment}"


def _init_wandb(cfg: TrainConfig, run_name: str, resume_state: dict | None) -> None:
    wandb.init(
        project="hal",
        group="049-ego-player-identity",
        name=run_name,
        id=None if resume_state is None else resume_state.get("wandb_id"),
        resume="allow" if resume_state is not None else None,
        tags=["049", "legacy-codec", "ego-player-identity", cfg.treatment],
        config=asdict(cfg),
        settings=wandb.Settings(x_stats_sampling_interval=5.0, x_stats_track_process_tree=True),
    )
    if wandb.run is None:
        return
    wandb.define_metric("global_step")
    wandb.define_metric("*", step_metric="global_step")
    wandb.run.summary["architecture/treatment"] = cfg.treatment
    wandb.run.summary["architecture/player_embed_dim"] = cfg.player_embed_dim
    wandb.run.summary["architecture/opponent_identity_conditioned"] = False
    wandb.run.summary["data/source_count"] = len(cfg.train_source_names)
    wandb.run.summary["data/player_sidecar_sha256"] = cfg.player_sidecar_sha256
    wandb.run.summary["data/player_vocab_sha256"] = cfg.player_vocab_sha256
    if cfg.wandb_log_code:
        _o43.log_wandb_code(wandb.run)


def config_from_state(values: dict[str, object]) -> TrainConfig:
    known = {item.name for item in fields(TrainConfig)}
    cfg = TrainConfig(**{name: value for name, value in values.items() if name in known})
    validate_config(cfg)
    return cfg


def load_checkpoint(path: str, *, device: str = _o43.DEVICE):
    state = torch.load(path, map_location=device, weights_only=False)
    cfg = config_from_state(state["cfg"])
    vocabulary = _vocabulary_from_state(state["model"])
    if vocabulary.size != cfg.player_vocab_size:
        raise ValueError("checkpoint vocabulary size differs from its configuration")
    if cfg.player_vocab_sha256 and vocabulary.sha256 != cfg.player_vocab_sha256:
        raise ValueError("checkpoint vocabulary hash differs from its configuration")
    model = GPT(cfg, vocabulary=vocabulary).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, cfg, load_stats(cfg), state


def eval_checkpoint(
    path: str,
    *,
    player_code: str | None = None,
    player_rank: Rank | None = None,
    exec_horizon: int | None = None,
    n_matchups: int | None = None,
    eager: bool = False,
    max_parallel: int | None = None,
    output_name: str | None = None,
) -> dict[str, dict[str, float]]:
    model, cfg, stats, state = load_checkpoint(path)
    cfg = replace(
        cfg,
        inference_mode="eager" if eager else cfg.inference_mode,
        eval_max_parallel=cfg.eval_max_parallel if max_parallel is None else max_parallel,
    )
    validate_config(cfg)
    horizon = cfg.exec_horizon if exec_horizon is None else exec_horizon
    matchups = cfg.final_eval_n_matchups if n_matchups is None else n_matchups
    key = player_code if player_code is not None else (player_rank or Rank[cfg.eval_player_rank]).name.lower()
    default_name = f"eval_replays_s{horizon}_{key.replace('#', '-')}"
    replay_dir = Path(path).resolve().parent / (default_name if output_name is None else output_name)
    inference = BF16Inference(
        model,
        cfg,
        player_code=player_code,
        player_rank=player_rank,
    )
    suites = eval_suites(
        model,
        stats,
        cfg,
        n_matchups=matchups,
        replay_dir=replay_dir,
        checkpoint_sha256=_o43._checkpoint_sha256(Path(path)),
        inference=inference,
        exec_horizon=horizon,
    )
    for metrics in suites.values():
        _o43.require_complete_eval(metrics, matchups)
    print(f"[eval] step={int(state['step'])} ego={inference.player_key!r}: {suites}", flush=True)
    return suites


# Reuse O43's training loop after replacing every experiment-specific seam it
# resolves through its module globals.
_o43.__file__ = __file__
_o43.TrainConfig = TrainConfig
_o43.GPT = GPT
_o43.BF16Inference = BF16Inference
_o43.validate_config = validate_config
_o43.make_optimizer = make_optimizer
_o43._make_loaders = _make_loaders
_o43.eval_suites = eval_suites
_o43.model_tag = model_tag
_o43.synthetic_context = synthetic_context
_o43._init_wandb = _init_wandb
_o43.config_from_state = config_from_state
_o43.load_checkpoint = load_checkpoint


@dataclass
class Args:
    cfg: TrainConfig = dataclass_field(default_factory=TrainConfig)
    comment: str = ""
    resume: str | None = None
    eval: str | None = None
    identity_audit: str | None = None
    identity_audit_output: str | None = None
    eval_run: str | None = None
    eval_player_code: str | None = None
    eval_player_rank: Literal["PLATINUM", "DIAMOND", "MASTER"] | None = None
    eval_exec_horizon: int | None = None
    eval_n_matchups: int | None = None
    eval_eager: bool = False
    eval_max_parallel: int | None = None
    eval_output_name: str | None = None
    benchmark: bool = False
    benchmark_iterations: int = 20


def main(args: Args) -> None:
    selected = sum(value is not None for value in (args.eval, args.resume, args.identity_audit))
    if selected > 1:
        raise SystemExit("pass only one of --eval, --resume, or --identity-audit")
    if args.eval_player_code is not None and args.eval_player_rank is not None:
        raise SystemExit("pass --eval-player-code or --eval-player-rank, not both")
    if args.eval is not None:
        checkpoint = _o43._resolve_eval_checkpoint(args.eval, args.eval_run)
        eval_checkpoint(
            str(checkpoint),
            player_code=args.eval_player_code,
            player_rank=None if args.eval_player_rank is None else Rank[args.eval_player_rank],
            exec_horizon=args.eval_exec_horizon,
            n_matchups=args.eval_n_matchups,
            eager=args.eval_eager,
            max_parallel=args.eval_max_parallel,
            output_name=args.eval_output_name,
        )
        return
    if args.identity_audit is not None:
        identity_audit_checkpoint(args.identity_audit, output=args.identity_audit_output)
        return
    if args.benchmark:
        _o43.run_benchmark(args.cfg, iterations=args.benchmark_iterations)
        return
    if args.resume is not None:
        resume_state = _o43.load_for_resume(args.resume, Path("runs") / args.resume, device=_o43.DEVICE)
        if resume_state is None:
            raise SystemExit(f"no latest.pt for run {args.resume!r}")
        cfg = config_from_state(resume_state["cfg"])
        run_name = args.resume
    else:
        cfg = args.cfg
        resume_state = None
        run_name = None
    validate_config(cfg, require_frozen_artifact=True)
    sidecar = load_sidecar(cfg)
    if sidecar.vocabulary.sha256 != cfg.player_vocab_sha256:
        raise ValueError("training sidecar vocabulary is not the frozen O49 vocabulary")
    _o43.train(
        cfg,
        load_stats(cfg),
        comment=args.comment,
        resume_run=run_name,
        resume_state=resume_state,
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
