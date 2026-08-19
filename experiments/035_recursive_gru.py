"""Zero-state GRU temporal mixer on the exact experiment-026 policy recipe.

Only the two-layer causal temporal Transformer is replaced. The scene, prior
full-action, and absolute-offset token is unchanged and drives one standard
``torch.nn.GRUCell(128, 128)`` from a fresh zero state for every decode.
"""

import importlib.util
import sys
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from dataclasses import fields
from dataclasses import replace
from pathlib import Path

import torch
import tyro
from torch import Tensor
from torch import nn

_BASE_PATH = Path(__file__).with_name("026_temporal_mtp.py")
_SPEC = importlib.util.spec_from_file_location("hal_exp026_for_035", _BASE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
base = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = base
_SPEC.loader.exec_module(base)

_validate_026_config = base.validate_config
_model_tag_026 = base.model_tag
_train_026 = base.train
_TrainConfig026 = base.TrainConfig
_CHECKPOINT_ARCH_FIELDS = base._CHECKPOINT_ARCH_FIELDS
_EXPERIMENT_ID = "035_recursive_gru_v1"

for _name in dir(base):
    if not _name.startswith("__"):
        globals()[_name] = getattr(base, _name)


@dataclass
class TrainConfig(base.TrainConfig):
    """The eligible 026 production recipe plus explicit 035 identity."""

    batch_size: int = 512
    cache_limit_gb: int = 160
    eval_max_parallel: int | None = 32
    decoder_arch_version: int = 4
    temporal_layers: int = 1
    experiment_id: str = _EXPERIMENT_ID


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
_EVALUATION_OVERRIDE_FIELDS = frozenset({"eval_max_parallel", "inference_mode"})


def _config_changes(cfg: TrainConfig, reference: TrainConfig) -> dict[str, tuple[object, object]]:
    return {
        item.name: (getattr(cfg, item.name), getattr(reference, item.name))
        for item in fields(TrainConfig)
        if getattr(cfg, item.name) != getattr(reference, item.name)
    }


def _validate_gru_contract(cfg: TrainConfig) -> None:
    if cfg.experiment_id != _EXPERIMENT_ID:
        raise ValueError(f"experiment_id must be {_EXPERIMENT_ID!r}, got {cfg.experiment_id!r}")
    if cfg.decoder_arch_version != 4:
        raise ValueError("035 requires decoder_arch_version=4")
    if cfg.temporal_layers != 1:
        raise ValueError("035 contains exactly one GRUCell")


def validate_config(cfg: TrainConfig) -> None:
    """Validate the frozen 026 recipe and the mixer-only 035 treatment."""
    _validate_gru_contract(cfg)
    _validate_026_config(replace(cfg, decoder_arch_version=3, temporal_layers=2))

    reference = TrainConfig()
    if cfg.max_steps > reference.max_steps:
        raise ValueError(f"035 cannot exceed the frozen {reference.max_steps} optimizer steps")
    allowed = _SMOKE_OVERRIDE_FIELDS if cfg.max_steps < reference.max_steps else frozenset()
    changed = _config_changes(cfg, reference)
    forbidden = {name: value for name, value in changed.items() if name not in allowed}
    if forbidden:
        mode = "smoke" if cfg.max_steps < reference.max_steps else "production"
        raise ValueError(f"{mode} 035 config changed frozen scientific fields: {forbidden}")


def model_tag(cfg: TrainConfig) -> str:
    reference = replace(cfg, decoder_arch_version=3, temporal_layers=2)
    return f"{_model_tag_026(reference)}-gru128"


class CausalTemporalDecoder(nn.Module):
    """Sparse-offset autoregressive chain mixed by one zero-state GRUCell."""

    def __init__(self, cfg: TrainConfig, codec: StructuredControllerCodec) -> None:
        super().__init__()
        self.codec = codec
        self.head_offsets = tuple(cfg.head_offsets)
        self.d_model = cfg.temporal_d_model
        controller_width = N_GROUPS * cfg.action_embed_dim
        self.offset_embedding = nn.Embedding(cfg.sample_chunk_length + 1, cfg.offset_embed_dim)
        self.token_projection = nn.Linear(cfg.d_model + controller_width + cfg.offset_embed_dim, self.d_model)
        self.cell = nn.GRUCell(self.d_model, self.d_model)
        self.group_condition = nn.ModuleDict(
            {
                name: nn.Linear(position * cfg.action_embed_dim, 2 * self.d_model)
                for position, name in enumerate(GROUP_ORDER)
                if position
            }
        )
        self.outputs = nn.ModuleDict(
            {
                name: NonlinearActionHead(self.d_model, cfg.group_head_dim, GROUP_VOCABS[GROUP_INDEX[name]])
                for name in GROUP_NAMES
            }
        )
        self.trunk_outputs = nn.ModuleDict(
            {name: nn.Linear(cfg.d_model, GROUP_VOCABS[GROUP_INDEX[name]], bias=False) for name in GROUP_NAMES}
        )

    def _tokens(self, hidden: Tensor, previous: Tensor) -> Tensor:
        batch, length = hidden.shape[:2]
        horizon = len(self.head_offsets)
        action = self.codec.embed_frame(previous)
        offsets = torch.tensor(self.head_offsets, device=hidden.device)
        offset = self.offset_embedding(offsets).view(1, 1, horizon, -1).expand(batch, length, -1, -1)
        trunk = decoder_rmsnorm(hidden)[:, :, None].expand(-1, -1, horizon, -1)
        return self.token_projection(torch.cat((trunk, action, offset), dim=-1))

    def teacher_forced_states(self, hidden: Tensor, observed: Tensor, targets: Tensor) -> Tensor:
        expected = (*hidden.shape[:2], len(self.head_offsets), N_GROUPS)
        if observed.shape != (*hidden.shape[:2], N_GROUPS) or targets.shape != expected:
            raise ValueError(
                f"expected observed {(*hidden.shape[:2], N_GROUPS)} and targets {expected}, got "
                f"{tuple(observed.shape)} and {tuple(targets.shape)}"
            )
        previous = torch.cat((observed[:, :, None], targets[..., :-1, :]), dim=2)
        tokens = self._tokens(hidden, previous).reshape(-1, len(self.head_offsets), self.d_model)
        state = tokens.new_zeros(tokens.shape[0], self.d_model)
        states: list[Tensor] = []
        for token in tokens.unbind(1):
            state = self.cell(token, state)
            states.append(state)
        stacked = torch.stack(states, dim=1).view(*hidden.shape[:2], len(self.head_offsets), self.d_model)
        return decoder_rmsnorm(stacked)

    def group_features(self, states: Tensor, name: str, embedded: dict[str, Tensor]) -> Tensor:
        position = GROUP_ORDER.index(name)
        if position == 0:
            return states
        prefix = torch.cat([embedded[group] for group in GROUP_ORDER[:position]], dim=-1)
        scale, shift = self.group_condition[name](prefix).chunk(2, dim=-1)
        return states * (1.0 + torch.tanh(scale)) + shift

    def teacher_forced_logits_by_group(self, hidden: Tensor, observed: Tensor, targets: Tensor) -> dict[str, Tensor]:
        states = self.teacher_forced_states(hidden, observed, targets)
        embedded = self.codec.embed_groups(targets)
        trunk = decoder_rmsnorm(hidden)
        logits = {
            name: self.outputs[name](self.group_features(states, name, embedded))
            + self.trunk_outputs[name](trunk)[:, :, None]
            for name in GROUP_NAMES
        }
        logits["buttons"] = logits["buttons"].masked_fill(self.codec.button_mask(targets[..., TRIG_G]), float("-inf"))
        return logits

    def teacher_forced_nll(self, hidden: Tensor, observed: Tensor, targets: Tensor) -> Tensor:
        logits = self.teacher_forced_logits_by_group(hidden, observed, targets)
        losses = [
            F.cross_entropy(
                logits[name].float().reshape(-1, GROUP_VOCABS[group]),
                targets[..., group].reshape(-1),
                reduction="none",
            ).view(*targets.shape[:-1])
            for group, name in enumerate(GROUP_NAMES)
        ]
        return torch.stack(losses, dim=-1)

    def teacher_forced_logits(self, hidden: Tensor, observed: Tensor, targets: Tensor) -> list[dict[str, Tensor]]:
        values = self.teacher_forced_logits_by_group(hidden, observed, targets)
        return [
            {name: logits[..., depth, :] for name, logits in values.items()} for depth in range(len(self.head_offsets))
        ]

    def _step(self, trunk: Tensor, previous: Tensor, offset: int, state: Tensor) -> Tensor:
        offset_tensor = torch.full((trunk.shape[0],), offset, device=trunk.device, dtype=torch.long)
        token = self.token_projection(
            torch.cat((trunk, self.codec.embed_frame(previous), self.offset_embedding(offset_tensor)), dim=-1)
        )
        return self.cell(token, state)

    def forced_stepwise_logits(self, hidden: Tensor, observed: Tensor, targets: Tensor) -> list[dict[str, Tensor]]:
        if targets.shape != (hidden.shape[0], len(self.head_offsets), N_GROUPS):
            raise ValueError("stepwise targets have the wrong shape")
        trunk = decoder_rmsnorm(hidden[:, -1])
        trunk_logits = {name: self.trunk_outputs[name](trunk) for name in GROUP_NAMES}
        previous = observed
        state = trunk.new_zeros(trunk.shape[0], self.d_model)
        out: list[dict[str, Tensor]] = []
        for depth, offset in enumerate(self.head_offsets):
            state = self._step(trunk, previous, offset, state)
            normalized = decoder_rmsnorm(state)
            target = targets[:, depth]
            embedded = self.codec.embed_groups(target)
            group_logits = {
                name: self.outputs[name](self.group_features(normalized, name, embedded)) + trunk_logits[name]
                for name in GROUP_NAMES
            }
            group_logits["buttons"] = group_logits["buttons"].masked_fill(
                self.codec.button_mask(target[:, TRIG_G]), float("-inf")
            )
            out.append(group_logits)
            previous = target
        return out

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
        if uniforms is not None and uniforms.shape != (len(offsets), N_GROUPS, hidden.shape[0]):
            raise ValueError("uniform table must be [frames, groups, batch]")
        trunk = decoder_rmsnorm(hidden[:, -1])
        trunk_logits = {name: self.trunk_outputs[name](trunk) for name in GROUP_NAMES}
        previous = observed
        state = trunk.new_zeros(trunk.shape[0], self.d_model)
        frames: list[Tensor] = []
        for depth, offset in enumerate(offsets):
            state = self._step(trunk, previous, offset, state)
            normalized = decoder_rmsnorm(state)
            embedded: dict[str, Tensor] = {}
            picks: dict[str, Tensor] = {}
            for name in GROUP_ORDER:
                logits = self.outputs[name](self.group_features(normalized, name, embedded)) + trunk_logits[name]
                if name == "buttons":
                    logits = logits.masked_fill(self.codec.button_mask(picks["triggers"]), float("-inf"))
                group = GROUP_INDEX[name]
                uniform = None if uniforms is None else uniforms[depth, group]
                pick = sample_categorical(logits, argmax=argmax, uniform=uniform, gen=gen)
                picks[name] = pick
                embedded[name] = self.codec.group_embedding(name, pick)
            indices = torch.stack([picks[name] for name in GROUP_NAMES], dim=-1)
            frames.append(indices)
            previous = indices
        return torch.stack(frames, dim=1)

    def rollout_conditioned_logits(self, hidden: Tensor, observed: Tensor) -> tuple[list[dict[str, Tensor]], Tensor]:
        """Greedily roll out every selected offset for validation diagnostics."""
        trunk = decoder_rmsnorm(hidden[:, -1])
        trunk_logits = {name: self.trunk_outputs[name](trunk) for name in GROUP_NAMES}
        previous = observed
        state = trunk.new_zeros(trunk.shape[0], self.d_model)
        frames: list[Tensor] = []
        all_logits: list[dict[str, Tensor]] = []
        for offset in self.head_offsets:
            state = self._step(trunk, previous, offset, state)
            normalized = decoder_rmsnorm(state)
            embedded: dict[str, Tensor] = {}
            picks: dict[str, Tensor] = {}
            frame_logits: dict[str, Tensor] = {}
            for name in GROUP_ORDER:
                logits = self.outputs[name](self.group_features(normalized, name, embedded)) + trunk_logits[name]
                if name == "buttons":
                    logits = logits.masked_fill(self.codec.button_mask(picks["triggers"]), float("-inf"))
                pick = logits.argmax(dim=-1)
                frame_logits[name] = logits
                picks[name] = pick
                embedded[name] = self.codec.group_embedding(name, pick)
            previous = torch.stack([picks[name] for name in GROUP_NAMES], dim=-1)
            frames.append(previous)
            all_logits.append(frame_logits)
        return all_logits, torch.stack(frames, dim=1)


def subsystem_parameter_counts(model: GPT) -> dict[str, int]:
    groups = {
        "trunk": model.trunk,
        "observation": model.ctx_proj,
        "codec": model.codec,
        "temporal": model.temporal.cell,
        "heads": nn.ModuleList([model.temporal.outputs, model.temporal.trunk_outputs]),
    }
    return {name: sum(parameter.numel() for parameter in module.parameters()) for name, module in groups.items()}


_CHECKPOINT_035_FIELDS = {"experiment_id"}


def config_from_state(values: dict) -> TrainConfig:
    """Restore only an explicitly identified experiment-035 checkpoint."""
    missing = (_CHECKPOINT_ARCH_FIELDS | _CHECKPOINT_035_FIELDS) - values.keys()
    if missing:
        raise ValueError(f"checkpoint is not experiment 035; missing {sorted(missing)}")
    if values["experiment_id"] != _EXPERIMENT_ID:
        raise ValueError(f"checkpoint experiment_id {values['experiment_id']!r} != required {_EXPERIMENT_ID!r}")
    known = {item.name for item in fields(TrainConfig)}
    cfg = TrainConfig(**{name: value for name, value in values.items() if name in known})
    _validate_gru_contract(cfg)
    return cfg


def _validate_evaluation_override(cfg: TrainConfig, checkpoint_cfg: TrainConfig) -> None:
    _validate_gru_contract(cfg)
    _validate_026_config(replace(cfg, decoder_arch_version=3, temporal_layers=2))
    changed = _config_changes(cfg, checkpoint_cfg)
    forbidden = {name: value for name, value in changed.items() if name not in _EVALUATION_OVERRIDE_FIELDS}
    if forbidden:
        raise ValueError(f"evaluation changed checkpoint-scientific fields: {forbidden}")


def eval_checkpoint(
    path: str,
    *,
    exec_horizon: int | None = None,
    n_matchups: int | None = None,
    eager: bool = False,
    max_parallel: int | None = None,
    output_name: str | None = None,
) -> dict[str, float]:
    """Evaluate a provenance-checked 035 checkpoint with runtime overrides."""
    model, checkpoint_cfg, stats, state = load_checkpoint(path)
    cfg = replace(
        checkpoint_cfg,
        inference_mode="eager" if eager else checkpoint_cfg.inference_mode,
        eval_max_parallel=checkpoint_cfg.eval_max_parallel if max_parallel is None else max_parallel,
    )
    _validate_evaluation_override(cfg, checkpoint_cfg)
    horizon = cfg.exec_horizon if exec_horizon is None else exec_horizon
    default_name = "eval_replays_s6" if horizon == 6 else "eval_replays"
    if output_name is not None and (Path(output_name).name != output_name or output_name in ("", ".", "..")):
        raise ValueError(f"evaluation output name must be one directory name, got {output_name!r}")
    replay_dir = Path(path).resolve().parent / (default_name if output_name is None else output_name)
    values = eval_vs_cpu(
        model,
        stats,
        cfg,
        n_matchups=cfg.final_eval_n_matchups if n_matchups is None else n_matchups,
        replay_dir=replay_dir,
        exec_horizon=horizon,
        checkpoint_sha256=_checkpoint_sha256(Path(path)),
    )
    print(f"[eval] step={state['step']} horizon={horizon}: {values}", flush=True)
    return values


def train(
    cfg: TrainConfig,
    stats: dict[str, FeatureStats],
    *,
    comment: str = "",
    resume_run: str | None = None,
    resume_state: dict | None = None,
) -> None:
    """Run the unchanged 026 loop with explicit experiment-035 metadata."""
    original_init = base.wandb.init

    def init_035(*args, **kwargs):
        kwargs["tags"] = ["gpt", "temporal-mtp", "recursive-gru", "035"]
        return original_init(*args, **kwargs)

    base.wandb.init = init_035
    try:
        _train_026(cfg, stats, comment=comment, resume_run=resume_run, resume_state=resume_state)
    finally:
        base.wandb.init = original_init


@dataclass
class Args(base.Args):
    cfg: TrainConfig = dataclass_field(default_factory=TrainConfig)


base.TrainConfig = TrainConfig
base.CausalTemporalDecoder = CausalTemporalDecoder
base.validate_config = validate_config
base.model_tag = model_tag
base.subsystem_parameter_counts = subsystem_parameter_counts
base.config_from_state = config_from_state
base.eval_checkpoint = eval_checkpoint
base.train = train
base.__file__ = __file__


if __name__ == "__main__":
    base.main(tyro.cli(Args))
