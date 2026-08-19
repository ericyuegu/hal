"""Residual-readout GRU temporal mixer on the exact experiment-035 policy.

The sole treatment is decoding from ``RMSNorm(token + raw_gru_state)``. The
zero-state vanilla GRU, projected input token, raw recurrent state, and every
other model, training, data, and evaluation detail remain unchanged.
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

_BASE_PATH = Path(__file__).with_name("035_recursive_gru.py")
_SPEC = importlib.util.spec_from_file_location("hal_exp035_for_036", _BASE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
base035 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = base035
_SPEC.loader.exec_module(base035)

_base026 = base035.base
_TrainConfig035 = base035.TrainConfig
_train_026 = base035._train_026

for _name in dir(base035):
    if not _name.startswith("__"):
        globals()[_name] = getattr(base035, _name)

_EXPERIMENT_ID = "036_residual_gru_v1"


@dataclass
class TrainConfig(base035.TrainConfig):
    decoder_arch_version: int = 5
    experiment_id: str = _EXPERIMENT_ID


def _config_changes(cfg: TrainConfig, reference: TrainConfig) -> dict[str, tuple[object, object]]:
    return {
        item.name: (getattr(cfg, item.name), getattr(reference, item.name))
        for item in fields(TrainConfig)
        if getattr(cfg, item.name) != getattr(reference, item.name)
    }


def _validate_residual_contract(cfg: TrainConfig) -> None:
    if cfg.experiment_id != _EXPERIMENT_ID:
        raise ValueError(f"experiment_id must be {_EXPERIMENT_ID!r}, got {cfg.experiment_id!r}")
    if cfg.decoder_arch_version != 5:
        raise ValueError("036 requires decoder_arch_version=5")
    if cfg.temporal_layers != 1:
        raise ValueError("036 contains exactly one GRUCell")


def validate_config(cfg: TrainConfig) -> None:
    """Validate the frozen 035 recipe and readout-only 036 treatment."""
    _validate_residual_contract(cfg)
    base035._validate_026_config(replace(cfg, decoder_arch_version=3, temporal_layers=2))
    reference = TrainConfig()
    if cfg.max_steps > reference.max_steps:
        raise ValueError(f"036 cannot exceed the frozen {reference.max_steps} optimizer steps")
    allowed = base035._SMOKE_OVERRIDE_FIELDS if cfg.max_steps < reference.max_steps else frozenset()
    changed = _config_changes(cfg, reference)
    forbidden = {name: value for name, value in changed.items() if name not in allowed}
    if forbidden:
        mode = "smoke" if cfg.max_steps < reference.max_steps else "production"
        raise ValueError(f"{mode} 036 config changed frozen scientific fields: {forbidden}")


def model_tag(cfg: TrainConfig) -> str:
    reference = replace(cfg, decoder_arch_version=3, temporal_layers=2)
    return f"{base035._model_tag_026(reference)}-gru128-resread"


class CausalTemporalDecoder(base035.CausalTemporalDecoder):
    """The 035 GRU with an unscaled token-to-readout residual."""

    @staticmethod
    def readout(token: Tensor, state: Tensor) -> Tensor:
        return decoder_rmsnorm(token + state)

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
        readouts: list[Tensor] = []
        for token in tokens.unbind(1):
            state = self.cell(token, state)
            readouts.append(self.readout(token, state))
        return torch.stack(readouts, dim=1).view(*hidden.shape[:2], len(self.head_offsets), self.d_model)

    def _token_and_state(self, trunk: Tensor, previous: Tensor, offset: int, state: Tensor) -> tuple[Tensor, Tensor]:
        offset_tensor = torch.full((trunk.shape[0],), offset, device=trunk.device, dtype=torch.long)
        token = self.token_projection(
            torch.cat((trunk, self.codec.embed_frame(previous), self.offset_embedding(offset_tensor)), dim=-1)
        )
        return token, self.cell(token, state)

    def forced_stepwise_logits(self, hidden: Tensor, observed: Tensor, targets: Tensor) -> list[dict[str, Tensor]]:
        if targets.shape != (hidden.shape[0], len(self.head_offsets), N_GROUPS):
            raise ValueError("stepwise targets have the wrong shape")
        trunk = decoder_rmsnorm(hidden[:, -1])
        trunk_logits = {name: self.trunk_outputs[name](trunk) for name in GROUP_NAMES}
        previous = observed
        state = trunk.new_zeros(trunk.shape[0], self.d_model)
        out: list[dict[str, Tensor]] = []
        for depth, offset in enumerate(self.head_offsets):
            token, state = self._token_and_state(trunk, previous, offset, state)
            readout = self.readout(token, state)
            target = targets[:, depth]
            embedded = self.codec.embed_groups(target)
            group_logits = {
                name: self.outputs[name](self.group_features(readout, name, embedded)) + trunk_logits[name]
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
            token, state = self._token_and_state(trunk, previous, offset, state)
            readout = self.readout(token, state)
            embedded: dict[str, Tensor] = {}
            picks: dict[str, Tensor] = {}
            for group, name in enumerate(GROUP_ORDER):
                logits = self.outputs[name](self.group_features(readout, name, embedded)) + trunk_logits[name]
                if name == "buttons":
                    logits = logits.masked_fill(self.codec.button_mask(picks["triggers"]), float("-inf"))
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
            token, state = self._token_and_state(trunk, previous, offset, state)
            readout = self.readout(token, state)
            embedded: dict[str, Tensor] = {}
            picks: dict[str, Tensor] = {}
            frame_logits: dict[str, Tensor] = {}
            for _group, name in enumerate(GROUP_ORDER):
                logits = self.outputs[name](self.group_features(readout, name, embedded)) + trunk_logits[name]
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


def config_from_state(values: dict) -> TrainConfig:
    """Restore only an explicitly identified experiment-036 checkpoint."""
    missing = (base035._CHECKPOINT_ARCH_FIELDS | {"experiment_id"}) - values.keys()
    if missing:
        raise ValueError(f"checkpoint is not experiment 036; missing {sorted(missing)}")
    if values["experiment_id"] != _EXPERIMENT_ID:
        raise ValueError(f"checkpoint experiment_id {values['experiment_id']!r} != required {_EXPERIMENT_ID!r}")
    known = {item.name for item in fields(TrainConfig)}
    cfg = TrainConfig(**{name: value for name, value in values.items() if name in known})
    _validate_residual_contract(cfg)
    return cfg


def _validate_evaluation_override(cfg: TrainConfig, checkpoint_cfg: TrainConfig) -> None:
    _validate_residual_contract(cfg)
    base035._validate_026_config(replace(cfg, decoder_arch_version=3, temporal_layers=2))
    changed = _config_changes(cfg, checkpoint_cfg)
    forbidden = {name: value for name, value in changed.items() if name not in base035._EVALUATION_OVERRIDE_FIELDS}
    if forbidden:
        raise ValueError(f"evaluation changed checkpoint-scientific fields: {forbidden}")


def train(
    cfg: TrainConfig,
    stats: dict[str, FeatureStats],
    *,
    comment: str = "",
    resume_run: str | None = None,
    resume_state: dict | None = None,
) -> None:
    """Run the unchanged 026 loop with explicit experiment-036 metadata."""
    original_init = _base026.wandb.init

    def init_036(*args, **kwargs):
        kwargs["tags"] = ["gpt", "temporal-mtp", "recursive-gru", "residual-readout", "036"]
        return original_init(*args, **kwargs)

    _base026.wandb.init = init_036
    try:
        _train_026(cfg, stats, comment=comment, resume_run=resume_run, resume_state=resume_state)
    finally:
        _base026.wandb.init = original_init


@dataclass
class Args(base035.Args):
    cfg: TrainConfig = dataclass_field(default_factory=TrainConfig)


base035.TrainConfig = TrainConfig
base035.CausalTemporalDecoder = CausalTemporalDecoder
base035.validate_config = validate_config
base035.model_tag = model_tag
base035.config_from_state = config_from_state
base035._validate_evaluation_override = _validate_evaluation_override
base035.train = train
_base026.TrainConfig = TrainConfig
_base026.CausalTemporalDecoder = CausalTemporalDecoder
_base026.validate_config = validate_config
_base026.model_tag = model_tag
_base026.config_from_state = config_from_state
_base026.train = train
_base026.__file__ = __file__


if __name__ == "__main__":
    _base026.main(tyro.cli(Args))
