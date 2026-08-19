"""Run the controlled width by action-decoder factorial around experiment 026.

The four cells change only trunk width (d256 or d384) and action decoder
structure (independent pointwise heads or the exact 026 causal decoder).

Run:
    uv run experiments/037_width_decoder_factorial.py --cell w0d0
    uv run experiments/037_width_decoder_factorial.py --cell w1d1
"""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import shutil
import sys
from collections.abc import Mapping
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from dataclasses import fields
from dataclasses import replace
from pathlib import Path
from typing import Literal
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from torch import Tensor

_BASE_PATH = Path(__file__).with_name("026_temporal_mtp.py")
_SPEC = importlib.util.spec_from_file_location("hal_exp026_for_037", _BASE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
base = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = base
_SPEC.loader.exec_module(base)

_validate_026_config = base.validate_config
_model_tag_026 = base.model_tag
_TrainConfig026 = base.TrainConfig
_GPT026 = base.GPT
_save_checkpoint_026 = base.save_checkpoint

for _name in dir(base):
    if not _name.startswith("__"):
        globals()[_name] = getattr(base, _name)


FactorialCell = Literal["w0d0", "w0d1", "w1d0", "w1d1"]
DecoderStructure = Literal["independent", "causal"]

_EXPERIMENT_ID = "037_width_decoder_factorial_v1"
_PRODUCTION_STEPS = 4_096
_TRAIN_ORDER_PREFLIGHT_BATCHES = 2
_NAMED_CHECKPOINT_UPDATES = (1_024, 2_048, 4_096)
_CELL_GEOMETRY: Mapping[FactorialCell, tuple[int, int, DecoderStructure]] = {
    "w0d0": (256, 4, "independent"),
    "w0d1": (256, 4, "causal"),
    "w1d0": (384, 6, "independent"),
    "w1d1": (384, 6, "causal"),
}


@dataclass
class TrainConfig(base.TrainConfig):
    """The exact 026 recipe plus the two preregistered factorial labels."""

    batch_size: int = 512
    max_steps: int = _PRODUCTION_STEPS
    cache_limit_gb: int = 160
    eval_max_parallel: int | None = 32

    experiment_id: str = _EXPERIMENT_ID
    factorial_cell: FactorialCell = "w1d1"
    decoder_structure: DecoderStructure = "causal"
    independent_layers: int = 4
    independent_ff_dim: int = 264


_SMOKE_OVERRIDE_FIELDS = frozenset(
    {
        "cache_limit_gb",
        "ckpt_every",
        "compile_temporal",
        "compile_trunk",
        "compiled_inference_bucket",
        "eval_every",
        "eval_max_frames",
        "eval_max_parallel",
        "eval_n_matchups",
        "final_diag_n_matchups",
        "final_eval_n_matchups",
        "grad_accum_steps",
        "inference_mode",
        "max_steps",
        "num_workers",
        "predownload",
        "prefetch_batches",
        "prefetch_factor",
        "push_to_r2",
        "val_batch_size",
        "val_every",
        "val_n_samples",
        "wandb_grad_every",
        "wandb_log_code",
    }
)


def config_for_cell(cfg: TrainConfig, cell: FactorialCell) -> TrainConfig:
    """Apply the frozen W and D levels for one factorial cell."""
    d_model, n_heads, decoder_structure = _CELL_GEOMETRY[cell]
    return replace(
        cfg,
        factorial_cell=cell,
        d_model=d_model,
        n_heads=n_heads,
        decoder_structure=decoder_structure,
    )


def _production_config(cell: FactorialCell) -> TrainConfig:
    return config_for_cell(TrainConfig(), cell)


def _config_changes(cfg: TrainConfig, reference: TrainConfig) -> dict[str, tuple[object, object]]:
    return {
        item.name: (getattr(cfg, item.name), getattr(reference, item.name))
        for item in fields(TrainConfig)
        if getattr(cfg, item.name) != getattr(reference, item.name)
    }


def validate_config(cfg: TrainConfig) -> None:
    """Validate the 026 invariants and the frozen factorial assignment."""
    _validate_026_config(cfg)
    if cfg.experiment_id != _EXPERIMENT_ID:
        raise ValueError(f"experiment_id must be {_EXPERIMENT_ID!r}, got {cfg.experiment_id!r}")
    if cfg.factorial_cell not in _CELL_GEOMETRY:
        raise ValueError(f"unknown factorial cell {cfg.factorial_cell!r}")
    d_model, n_heads, decoder_structure = _CELL_GEOMETRY[cfg.factorial_cell]
    actual = (cfg.d_model, cfg.n_heads, cfg.decoder_structure)
    expected = (d_model, n_heads, decoder_structure)
    if actual != expected:
        raise ValueError(f"cell {cfg.factorial_cell} requires geometry {expected}, got {actual}")
    if cfg.independent_layers != 4 or cfg.independent_ff_dim != 264:
        raise ValueError("D0 capacity match is frozen to four d128->d264->d128 blocks")
    if cfg.max_steps > _PRODUCTION_STEPS:
        raise ValueError(f"037 cannot exceed {_PRODUCTION_STEPS} optimizer steps")

    reference = _production_config(cfg.factorial_cell)
    allowed = _SMOKE_OVERRIDE_FIELDS if cfg.max_steps < _PRODUCTION_STEPS else frozenset()
    forbidden = {
        name: values
        for name, values in _config_changes(cfg, reference).items()
        if name not in allowed
    }
    if forbidden:
        mode = "smoke" if cfg.max_steps < _PRODUCTION_STEPS else "production"
        raise ValueError(f"{mode} 037 config changed frozen scientific fields: {forbidden}")


class IndependentFeedForward(nn.Module):
    """One pointwise residual block with no path between action positions."""

    def __init__(self, d_model: int, d_hidden: int) -> None:
        super().__init__()
        self.up = nn.Linear(d_model, d_hidden, bias=False)
        self.down = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.down(F.silu(self.up(base.decoder_rmsnorm(x))))


class IndependentOffsetDecoder(nn.Module):
    """Capacity-matched MTP logits without learned action-to-action edges."""

    def __init__(self, cfg: TrainConfig, codec: base.StructuredControllerCodec) -> None:
        super().__init__()
        self.codec = codec
        self.head_offsets = tuple(cfg.head_offsets)
        self.d_model = cfg.temporal_d_model
        self.offset_embedding = nn.Embedding(cfg.sample_chunk_length + 1, cfg.offset_embed_dim)
        self.token_projection = nn.Linear(cfg.d_model + cfg.offset_embed_dim, self.d_model)
        self.blocks = nn.ModuleList(
            [IndependentFeedForward(self.d_model, cfg.independent_ff_dim) for _ in range(cfg.independent_layers)]
        )
        self.outputs = nn.ModuleDict(
            {
                name: base.NonlinearActionHead(
                    self.d_model,
                    cfg.group_head_dim,
                    base.GROUP_VOCABS[base.GROUP_INDEX[name]],
                )
                for name in base.GROUP_NAMES
            }
        )
        self.trunk_outputs = nn.ModuleDict(
            {
                name: nn.Linear(cfg.d_model, base.GROUP_VOCABS[base.GROUP_INDEX[name]], bias=False)
                for name in base.GROUP_NAMES
            }
        )

    def _states(self, hidden: Tensor, offsets: tuple[int, ...]) -> Tensor:
        if hidden.ndim != 3:
            raise ValueError(f"hidden must be [B, L, d], got {tuple(hidden.shape)}")
        batch, length = hidden.shape[:2]
        offset_indices = torch.tensor(offsets, device=hidden.device)
        offset = self.offset_embedding(offset_indices).view(1, 1, len(offsets), -1)
        offset = offset.expand(batch, length, -1, -1)
        trunk = base.decoder_rmsnorm(hidden)[:, :, None].expand(-1, -1, len(offsets), -1)
        states = self.token_projection(torch.cat((trunk, offset), dim=-1))
        for block in self.blocks:
            states = block(states)
        return base.decoder_rmsnorm(states)

    def raw_logits_by_group(
        self,
        hidden: Tensor,
        offsets: tuple[int, ...] | None = None,
    ) -> dict[str, Tensor]:
        """Return action-independent raw logits before codec support masking."""
        selected = self.head_offsets if offsets is None else offsets
        if selected != self.head_offsets[: len(selected)]:
            raise ValueError("independent decode accepts only a selected-offset prefix")
        states = self._states(hidden, selected)
        trunk = base.decoder_rmsnorm(hidden)
        return {
            name: self.outputs[name](states) + self.trunk_outputs[name](trunk)[:, :, None]
            for name in base.GROUP_NAMES
        }

    @staticmethod
    def _validate_training_shapes(hidden: Tensor, observed: Tensor, targets: Tensor, horizons: int) -> None:
        expected = (*hidden.shape[:2], horizons, base.N_GROUPS)
        if observed.shape != (*hidden.shape[:2], base.N_GROUPS) or targets.shape != expected:
            raise ValueError(
                f"expected observed {(*hidden.shape[:2], base.N_GROUPS)} and targets {expected}, got "
                f"{tuple(observed.shape)} and {tuple(targets.shape)}"
            )

    def teacher_forced_logits_by_group(
        self,
        hidden: Tensor,
        observed: Tensor,
        targets: Tensor,
    ) -> dict[str, Tensor]:
        self._validate_training_shapes(hidden, observed, targets, len(self.head_offsets))
        logits = self.raw_logits_by_group(hidden)
        logits["buttons"] = logits["buttons"].masked_fill(
            self.codec.button_mask(targets[..., base.TRIG_G]),
            float("-inf"),
        )
        return logits

    def teacher_forced_nll(self, hidden: Tensor, observed: Tensor, targets: Tensor) -> Tensor:
        logits = self.teacher_forced_logits_by_group(hidden, observed, targets)
        losses = [
            F.cross_entropy(
                logits[name].float().reshape(-1, base.GROUP_VOCABS[group]),
                targets[..., group].reshape(-1),
                reduction="none",
            ).view(*targets.shape[:-1])
            for group, name in enumerate(base.GROUP_NAMES)
        ]
        return torch.stack(losses, dim=-1)

    def teacher_forced_logits(self, hidden: Tensor, observed: Tensor, targets: Tensor) -> list[dict[str, Tensor]]:
        values = self.teacher_forced_logits_by_group(hidden, observed, targets)
        return [
            {name: logits[..., depth, :] for name, logits in values.items()}
            for depth in range(len(self.head_offsets))
        ]

    def forced_stepwise_logits(self, hidden: Tensor, observed: Tensor, targets: Tensor) -> list[dict[str, Tensor]]:
        if observed.shape != (hidden.shape[0], base.N_GROUPS):
            raise ValueError("stepwise observed action has the wrong shape")
        if targets.shape != (hidden.shape[0], len(self.head_offsets), base.N_GROUPS):
            raise ValueError("stepwise targets have the wrong shape")
        raw = self.raw_logits_by_group(hidden[:, -1:])
        output: list[dict[str, Tensor]] = []
        for depth in range(len(self.head_offsets)):
            logits = {name: values[:, 0, depth] for name, values in raw.items()}
            logits["buttons"] = logits["buttons"].masked_fill(
                self.codec.button_mask(targets[:, depth, base.TRIG_G]),
                float("-inf"),
            )
            output.append(logits)
        return output

    def _sample_from_raw(
        self,
        raw: dict[str, Tensor],
        *,
        argmax: bool,
        uniforms: Tensor | None,
        gen: torch.Generator | None,
    ) -> tuple[Tensor, list[dict[str, Tensor]]]:
        horizon, batch = next(iter(raw.values())).shape[:2]
        frames: list[Tensor] = []
        emitted_logits: list[dict[str, Tensor]] = []
        for depth in range(horizon):
            picks: dict[str, Tensor] = {}
            frame_logits: dict[str, Tensor] = {}
            for name in base.GROUP_ORDER:
                logits = raw[name][depth]
                if name == "buttons":
                    logits = logits.masked_fill(self.codec.button_mask(picks["triggers"]), float("-inf"))
                group = base.GROUP_INDEX[name]
                uniform = None if uniforms is None else uniforms[depth, group]
                picks[name] = base.sample_categorical(logits, argmax=argmax, uniform=uniform, gen=gen)
                frame_logits[name] = logits
            frames.append(torch.stack([picks[name] for name in base.GROUP_NAMES], dim=-1))
            emitted_logits.append(frame_logits)
        return torch.stack(frames, dim=1).reshape(batch, horizon, base.N_GROUPS), emitted_logits

    def sample_indices(
        self,
        hidden: Tensor,
        observed: Tensor,
        offsets: tuple[int, ...],
        *,
        argmax: bool,
        uniforms: Tensor | None = None,
        gen: torch.Generator | None = None,
    ) -> Tensor:
        if offsets not in (self.head_offsets[:4], self.head_offsets[:6]):
            raise ValueError("live decode may compute only the dense four- or six-offset prefix")
        if observed.shape != (hidden.shape[0], base.N_GROUPS):
            raise ValueError("observed action has the wrong shape")
        if uniforms is not None and uniforms.shape != (len(offsets), base.N_GROUPS, hidden.shape[0]):
            raise ValueError("uniform table must be [frames, groups, batch]")
        raw = {
            name: logits[:, -1].transpose(0, 1)
            for name, logits in self.raw_logits_by_group(hidden, offsets).items()
        }
        frames, _ = self._sample_from_raw(raw, argmax=argmax, uniforms=uniforms, gen=gen)
        return frames

    def rollout_conditioned_logits(self, hidden: Tensor, observed: Tensor) -> tuple[list[dict[str, Tensor]], Tensor]:
        if observed.shape != (hidden.shape[0], base.N_GROUPS):
            raise ValueError("observed action has the wrong shape")
        raw = {
            name: logits[:, -1].transpose(0, 1)
            for name, logits in self.raw_logits_by_group(hidden).items()
        }
        frames, logits = self._sample_from_raw(raw, argmax=True, uniforms=None, gen=None)
        return logits, frames


class GPT(base.GPT):
    """Use exact 026 construction for D1 and replace only the decoder for D0."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__(cfg)
        if cfg.decoder_structure == "independent":
            self.temporal = IndependentOffsetDecoder(cfg, self.codec)


def decoder_parameter_count(model: GPT) -> int:
    """Count the complete decoder subsystem, including its codec reference."""
    return sum(parameter.numel() for parameter in model.temporal.parameters())


def sampling_contract(cfg: TrainConfig) -> dict[str, object]:
    """Return the cell-invariant fields that determine replay/window order."""
    return {
        "data_root": cfg.data_root,
        "compact_data": cfg.compact_data,
        "mds_schema_version": cfg.mds_schema_version,
        "L_ctx": cfg.L_ctx,
        "sample_chunk_length": cfg.sample_chunk_length,
        "batch_size": cfg.batch_size,
        "grad_accum_steps": cfg.grad_accum_steps,
        "seed": cfg.seed,
        "shuffle_block_size": cfg.shuffle_block_size,
        "windows_per_replay": cfg.windows_per_replay,
        "reservoir_capacity": cfg.reservoir_capacity,
        "val_split": cfg.val_split,
        "val_n_samples": cfg.val_n_samples,
    }


def sampling_contract_sha256(cfg: TrainConfig) -> str:
    payload = json.dumps(sampling_contract(cfg), separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def config_sha256(cfg: TrainConfig) -> str:
    """Hash the complete scientific and runtime configuration."""
    payload = json.dumps(asdict(cfg), separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def action_random_contract_sha256(cfg: TrainConfig) -> str:
    """Hash the counter-RNG contract shared by all four cells."""
    contract = {
        "algorithm": "SlotGroupRandom-splitmix64-v1",
        "eval_seed": cfg.eval_seed,
        "groups": base.GROUP_NAMES,
        "inference_buckets": cfg.inference_buckets,
        "horizons": (cfg.exec_horizon, cfg.final_diag_exec_horizon),
    }
    payload = json.dumps(contract, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def validation_cache_sha256(batches: list[base.TrainBatch]) -> str:
    """Hash the exact cached validation model inputs in their scored order."""

    def update_framed(*parts: bytes) -> None:
        for part in parts:
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)

    def update_tensor(name: str, value: Tensor) -> None:
        tensor = value.detach().cpu().contiguous()
        metadata = json.dumps(
            {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        update_framed(name.encode(), metadata, tensor.numpy().tobytes())

    digest = hashlib.sha256()
    for batch_index, batch in enumerate(batches):
        update_framed(b"batch", str(batch_index).encode())
        if batch.replay_ids is not None:
            update_framed(b"replay_ids", str(len(batch.replay_ids)).encode())
            for replay_id in batch.replay_ids:
                update_framed(replay_id.encode())
        else:
            update_framed(b"replay_ids_unavailable")
        for name in sorted(batch.context.features):
            update_tensor(f"feature/{name}", batch.context.features[name])
        update_tensor("ctx_pad", batch.context.ctx_pad)
        update_tensor("complete_target", batch.target)
    return digest.hexdigest()


def launch_manifest(
    cfg: TrainConfig,
    validation_sha256: str,
    train_order_sha256: str,
) -> dict[str, object]:
    """Return immutable run identity evidence for one cell."""
    _, _, _, boot_schedule_sha256 = base.matchup_diversity(cfg.final_eval_n_matchups)
    return {
        "schema_version": 1,
        "experiment_id": _EXPERIMENT_ID,
        "factorial_cell": cfg.factorial_cell,
        "config_sha256": config_sha256(cfg),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sampling_contract_sha256": sampling_contract_sha256(cfg),
        "train_order_first_two_batches_sha256": train_order_sha256,
        "validation_cache_sha256": validation_sha256,
        "action_random_contract_sha256": action_random_contract_sha256(cfg),
        "boot_schedule_sha256": boot_schedule_sha256,
        "named_checkpoints": [f"step_{update:06d}.pt" for update in _NAMED_CHECKPOINT_UPDATES],
        "eval_max_parallel": cfg.eval_max_parallel,
        "max_steps": cfg.max_steps,
    }


def model_tag(cfg: TrainConfig) -> str:
    return f"{_model_tag_026(cfg)}-factorial-{cfg.factorial_cell}-{cfg.decoder_structure}"


def config_from_state(values: dict) -> TrainConfig:
    """Restore only a provenance-identified experiment-037 checkpoint."""
    required = base._CHECKPOINT_ARCH_FIELDS | {
        "decoder_structure",
        "experiment_id",
        "factorial_cell",
        "independent_ff_dim",
        "independent_layers",
    }
    missing = required - values.keys()
    if missing:
        raise ValueError(f"checkpoint is not experiment 037; missing {sorted(missing)}")
    if values["experiment_id"] != _EXPERIMENT_ID:
        raise ValueError(f"checkpoint experiment_id {values['experiment_id']!r} != {_EXPERIMENT_ID!r}")
    known = {item.name for item in fields(TrainConfig)}
    cfg = TrainConfig(**{name: value for name, value in values.items() if name in known})
    validate_config(cfg)
    return cfg


def named_checkpoint_path(path: Path, update: int) -> Path | None:
    """Map resumable/final writes to the three persistent scientific snapshots."""
    if update not in _NAMED_CHECKPOINT_UPDATES:
        return None
    if path.name not in ("latest.pt", "final.pt"):
        return None
    return path.with_name(f"step_{update:06d}.pt")


def save_checkpoint(path: Path, **kwargs) -> None:
    """Preserve checkpoint semantics for manual callers outside the trainer."""
    _save_checkpoint_026(path, **kwargs)
    step = int(kwargs["step"])
    named = named_checkpoint_path(path, step)
    if named is None:
        return
    shutil.copy2(path, named)
    print(f"[ckpt] preserved {named}", flush=True)
    uploader = kwargs.get("uploader")
    if uploader is not None:
        uploader.upload(named)


def _save_update_checkpoint(
    path: Path,
    *,
    update: int,
    model: GPT,
    optimizer,
    scheduler,
    cfg: TrainConfig,
    uploader,
) -> None:
    """Save zero-based resume state under an update-counted scientific name."""
    _save_checkpoint_026(
        path,
        step=update - 1,
        model=model,
        opt=optimizer,
        sched=scheduler,
        cfg=asdict(cfg),
        wandb_id=None if base.wandb.run is None else base.wandb.run.id,
        uploader=uploader,
    )
    named = named_checkpoint_path(path, update)
    if named is not None:
        shutil.copy2(path, named)
        print(f"[ckpt] preserved {named}", flush=True)
        if uploader is not None:
            uploader.upload(named)


def _update_train_order_digest(digest, batch: base.TrainBatch) -> None:
    replay_ids = batch.replay_ids
    if replay_ids is None:
        raise ValueError("037 training batches must carry replay IDs")
    for replay_id in replay_ids:
        digest.update(replay_id.encode())
        digest.update(b"\0")
    digest.update(batch.context.ctx_pad.contiguous().numpy().tobytes())
    digest.update(batch.target[:, :1].contiguous().numpy().tobytes())


def train_order_sha256(batches: list[base.TrainBatch]) -> str:
    """Hash actual replay/window evidence in loader order."""
    digest = hashlib.sha256()
    for batch in batches:
        _update_train_order_digest(digest, batch)
    return digest.hexdigest()


def train(
    cfg: TrainConfig,
    stats: dict[str, base.FeatureStats],
    *,
    comment: str = "",
    resume_run: str | None = None,
    resume_state: dict | None = None,
) -> None:
    """Train one cell with update-counted validation and checkpoint boundaries."""
    if resume_run is not None or resume_state is not None:
        raise RuntimeError(
            "037 checkpoints do not contain a verified loader/RNG cursor; discard the interrupted attempt "
            "and restart this cell fresh from update 0"
        )
    validate_config(cfg)
    run_name = resume_run or base.make_run_name(Path(__file__).stem, model_tag(cfg), cfg.data_root, comment)
    uploader = base.BackgroundUploader(run_name) if cfg.push_to_r2 else None
    base.wandb.init(
        project="hal",
        name=run_name,
        id=None,
        resume=None,
        tags=["gpt", "temporal-mtp", "factorial", "037", cfg.factorial_cell],
        config=asdict(cfg),
    )
    if base.wandb.run is not None:
        base.wandb.define_metric("eval/net_stock_lcb", step_metric="global_step")
        base.wandb.define_metric("eval/net_dmg_lcb", step_metric="global_step")
        if cfg.wandb_log_code:
            base.log_wandb_code(base.wandb.run)
    run_dir, replay_dir = base.setup_run_dir(run_name)
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high" if cfg.allow_tf32 else "highest")
    model = GPT(cfg).to(base.DEVICE)
    counts = base.subsystem_parameter_counts(model)
    if base.wandb.run is not None:
        for name, value in counts.items():
            base.wandb.run.summary[f"parameters/{name}"] = value
        base.wandb.run.summary["parameters/decoder_complete"] = decoder_parameter_count(model)
        base.wandb.run.summary["parameters/total"] = sum(parameter.numel() for parameter in model.parameters())
    optimizer = base.make_optimizer(model, cfg)
    scheduler = base.LambdaLR(optimizer, base.lr_schedule(cfg))
    def trunk_fn(features, pad, actions):
        return model(features, pad, actions)

    temporal_fn = model.temporal.teacher_forced_nll
    if base.DEVICE == "cuda" and cfg.compile_trunk:
        trunk_fn = torch.compile(trunk_fn, dynamic=False)
    if base.DEVICE == "cuda" and cfg.compile_temporal:
        temporal_fn = torch.compile(temporal_fn, dynamic=False)

    train_loader, val_cache = base._make_loaders(cfg, stats)
    raw_iterator = iter(train_loader)
    prefetched = [next(raw_iterator) for _ in range(_TRAIN_ORDER_PREFLIGHT_BATCHES)]
    first_train_sha256 = train_order_sha256(prefetched)
    validation_sha256 = validation_cache_sha256(val_cache)
    manifest = launch_manifest(cfg, validation_sha256, first_train_sha256)
    manifest_path = run_dir / "launch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    if uploader is not None:
        uploader.upload(manifest_path)
    if base.wandb.run is not None:
        for name, value in manifest.items():
            if not isinstance(value, list):
                base.wandb.run.summary[f"manifest/{name}"] = value

    iterator = iter(itertools.chain(prefetched, raw_iterator))
    order_digest = hashlib.sha256()
    copy_stream = torch.cuda.Stream() if base.DEVICE == "cuda" else None
    run_started = base.time.monotonic()
    eval_inference = None
    model.train()
    try:
        for step in range(cfg.max_steps):
            update = step + 1
            if base.DEVICE == "cuda":
                torch.cuda.reset_peak_memory_stats()
            loader_started = base.time.monotonic()
            cpu_batches: list[base.TrainBatch] = []
            for _ in range(cfg.grad_accum_steps):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(train_loader)
                    batch = next(iterator)
                base.validate_batch_geometry(batch, cfg, base.micro_batch_size(cfg))
                _update_train_order_digest(order_digest, batch)
                cpu_batches.append(batch)
            loader_wait = base.time.monotonic() - loader_started
            valid_prefixes = sum(int((cfg.L_ctx - batch.context.ctx_pad).sum()) for batch in cpu_batches)
            if valid_prefixes <= 0:
                raise RuntimeError("training accumulation contains no valid context prefixes")
            optimizer.zero_grad()
            nll_sum = torch.zeros(len(cfg.head_offsets), base.N_GROUPS, device=base.DEVICE)
            n_prefixes = 0
            with base.profile("step") as stopwatch:
                for batch in base.device_batches(cpu_batches, base.DEVICE, copy_stream):
                    history, targets, valid = base.prepared_targets(model, batch)
                    with base.amp_context(cfg, base.DEVICE):
                        hidden = trunk_fn(batch.context.features, batch.context.ctx_pad, history)
                        dense_nll = temporal_fn(hidden, history, targets)
                        parts = base.ActionLoss(nll=dense_nll[valid], targets=targets[valid])
                        joint_nll = parts.nll.sum(dim=-1)
                        primary = joint_nll[:, :4].sum() / (valid_prefixes * 4)
                        auxiliary = joint_nll[:, 4:].sum() / (
                            valid_prefixes * (len(cfg.head_offsets) - 4)
                        )
                        loss = primary + cfg.aux_loss_weight * auxiliary
                    loss.backward()
                    nll_sum += parts.nll.detach().sum(dim=0)
                    n_prefixes += parts.nll.shape[0]
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError(f"update {update}: non-finite gradient norm {gradient_norm}")
                optimizer.step()
                scheduler.step()
                if base.DEVICE == "cuda":
                    torch.cuda.synchronize()
            metrics = base.nll_mean_metrics((nll_sum / n_prefixes).cpu(), cfg.head_offsets)
            log = {
                "global_step": update,
                "samples": update * cfg.batch_size,
                **{f"train/{name}": value for name, value in metrics.items()},
                "train/grad_norm": float(gradient_norm),
                "throughput/step_s": stopwatch.elapsed,
                "throughput/loader_wait_s": loader_wait,
                "throughput/samples_per_s": cfg.batch_size / stopwatch.elapsed,
                "throughput/samples_per_wall_s": cfg.batch_size / (stopwatch.elapsed + loader_wait),
                "throughput/prefixes_per_s": n_prefixes / stopwatch.elapsed,
                "lr/muon": next(group["lr"] for group in optimizer.param_groups if group["use_muon"]),
                "lr/adam": next(group["lr"] for group in optimizer.param_groups if not group["use_muon"]),
            }
            if base.DEVICE == "cuda":
                log["hardware/peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 2**30
                log["hardware/peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
            base.wandb.log(log)
            if update <= 10 or update % 50 == 0:
                print(
                    f"[t+{base.time.monotonic() - run_started:.0f}s] update {update}: "
                    f"{metrics['loss']:.3f} bits objective, {cfg.batch_size / stopwatch.elapsed:.0f} samples/s",
                    flush=True,
                )
            named_boundary = update in _NAMED_CHECKPOINT_UPDATES and update < cfg.max_steps
            val_due = cfg.val_every > 0 and named_boundary
            eval_due = cfg.eval_every > 0 and update < cfg.max_steps and update % cfg.eval_every == 0
            ckpt_due = cfg.ckpt_every > 0 and named_boundary
            checkpoint_path = run_dir / "latest.pt"
            if val_due or eval_due or ckpt_due:
                _save_update_checkpoint(
                    checkpoint_path,
                    update=update,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    cfg=cfg,
                    uploader=uploader,
                )
                if base.wandb.run is not None:
                    base.wandb.run.summary[f"train_order_sha256/update_{update:06d}"] = order_digest.hexdigest()
            if val_due:
                values = base.val_metrics(model, val_cache, cfg)
                base.wandb.log({"global_step": update, **{f"val/{name}": value for name, value in values.items()}})
            if eval_due:
                if eval_inference is None:
                    eval_inference = base.BF16Inference(model, cfg)
                values = base.eval_vs_cpu(
                    model,
                    stats,
                    cfg,
                    n_matchups=cfg.eval_n_matchups,
                    replay_dir=replay_dir / f"step_{update:06d}",
                    checkpoint_sha256=base._checkpoint_sha256(checkpoint_path),
                    inference=eval_inference,
                )
                base.wandb.log({"global_step": update, **{f"eval/{name}": value for name, value in values.items()}})

        latest_path = run_dir / "latest.pt"
        _save_update_checkpoint(
            latest_path,
            update=cfg.max_steps,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            uploader=uploader,
        )
        final_path = run_dir / "final.pt"
        shutil.copy2(latest_path, final_path)
        if uploader is not None:
            uploader.upload(final_path)
        checkpoint_sha = base._checkpoint_sha256(final_path)
        final_val = base.val_metrics(model, val_cache, cfg)
        base.wandb.log(
            {"global_step": cfg.max_steps, **{f"val/{name}": value for name, value in final_val.items()}}
        )
        if base.wandb.run is not None:
            base.wandb.run.summary[f"train_order_sha256/update_{cfg.max_steps:06d}"] = order_digest.hexdigest()
        if eval_inference is None:
            eval_inference = base.BF16Inference(model, cfg)
        final_eval = base.eval_vs_cpu(
            model,
            stats,
            cfg,
            n_matchups=cfg.final_eval_n_matchups,
            replay_dir=replay_dir / "final",
            checkpoint_sha256=checkpoint_sha,
            inference=eval_inference,
        )
        base.wandb.log(
            {"global_step": cfg.max_steps, **{f"eval/{name}": value for name, value in final_eval.items()}}
        )
        stride6 = base.eval_vs_cpu(
            model,
            stats,
            cfg,
            n_matchups=cfg.final_diag_n_matchups,
            replay_dir=replay_dir / "final_s6",
            exec_horizon=cfg.final_diag_exec_horizon,
            checkpoint_sha256=checkpoint_sha,
            inference=eval_inference,
        )
        base.wandb.log(
            {"global_step": cfg.max_steps, **{f"eval_s6/{name}": value for name, value in stride6.items()}}
        )
    finally:
        if uploader is not None:
            uploader.upload_tree(replay_dir, base=run_dir)
            uploader.close()
        base.wandb.finish()


@dataclass
class Args(base.Args):
    cfg: TrainConfig = dataclass_field(default_factory=TrainConfig)
    cell: FactorialCell = "w1d1"


def main(args: Args) -> None:
    """Apply a cell only to fresh train/benchmark modes, then use the 026 CLI."""
    if args.resume is not None:
        raise SystemExit(
            "037 checkpoints do not contain a verified loader/RNG cursor; discard the interrupted "
            "attempt and restart this cell fresh from update 0"
        )
    checkpoint_mode = any(value is not None for value in (args.eval, args.self_play_eval))
    if not checkpoint_mode:
        args.cfg = config_for_cell(args.cfg, args.cell)
    base.main(args)


base.TrainConfig = TrainConfig
base.GPT = GPT
base.validate_config = validate_config
base.model_tag = model_tag
base.config_from_state = config_from_state
base.save_checkpoint = save_checkpoint
base.train = train
base.__file__ = __file__


if __name__ == "__main__":
    main(tyro.cli(Args))
