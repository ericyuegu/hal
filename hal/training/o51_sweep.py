"""Exact arm generation for the O51 experiment sequence."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import fields
from dataclasses import replace
from pathlib import Path
from typing import Literal

from hal.data.o51 import D0

EXPERIMENT = "experiments/051_correct_parameterization.py"
INIT_STD_GRID = (0.5, 1.0, 2.0)
READOUT_GRID = ("zero", "mup-normal")
MUON_LR_GRID = (0.014, 0.028, 0.056)
ADAM_LR_GRID = (2.125e-4, 4.25e-4, 8.5e-4)
DECAY_GRID = (0.0, 0.001, 0.01)
BATCH_GRID = (128, 256, 512, 1024)
TIER_SCALES = (1, 2, 4, 8)
SUPERVISED_POSITIONS_PER_WINDOW = 128
_ARM_ID = re.compile(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?")

Stage = Literal[
    "initialization-screen",
    "lr",
    "decay",
    "batch",
    "proxy-transfer",
    "mid-search",
    "seed-repeat",
    "duration",
]


@dataclass(frozen=True, slots=True)
class Treatment:
    """One selected center passed between O51 sweep stages."""

    hidden_std_multiplier: float = 1.0
    readout_init: Literal["zero", "mup-normal"] = "zero"
    muon_lr: float = 0.028
    adam_lr: float = 4.25e-4
    muon_weight_decay: float = 0.001
    adam_weight_decay: float = 0.001
    batch_size: int = 512
    depth_alpha: float = 0.5
    muon_batch_scaling: Literal["fixed", "sqrt"] = "fixed"
    muon_duration_scaling: Literal["fixed", "inverse-sqrt"] = "fixed"
    compile_mode: Literal["reduce-overhead", "max-autotune"] = "reduce-overhead"
    temporal_attention_chunk: int | None = 16_384
    num_workers: int = 16
    replay_pack_batch_size: int = 32
    loader_prefetch_factor: int = 2
    predownload: int = 512
    shuffle_algo: Literal["py1s", "py1e"] = "py1s"
    shuffle_block_size: int = 8192

    def __post_init__(self) -> None:
        positive_floats = {
            "hidden_std_multiplier": self.hidden_std_multiplier,
            "muon_lr": self.muon_lr,
            "adam_lr": self.adam_lr,
        }
        invalid_positive = sorted(
            name
            for name, value in positive_floats.items()
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value <= 0
        )
        if invalid_positive:
            raise ValueError(f"treatment values must be positive and finite: {invalid_positive}")
        decays = {
            "muon_weight_decay": self.muon_weight_decay,
            "adam_weight_decay": self.adam_weight_decay,
        }
        invalid_decays = sorted(
            name
            for name, value in decays.items()
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0
        )
        if invalid_decays:
            raise ValueError(f"treatment decays must be non-negative and finite: {invalid_decays}")
        if self.hidden_std_multiplier not in INIT_STD_GRID:
            raise ValueError(f"hidden_std_multiplier must be one of {INIT_STD_GRID}")
        if self.readout_init not in READOUT_GRID:
            raise ValueError(f"readout_init must be one of {READOUT_GRID}")
        if self.batch_size not in (*BATCH_GRID, 2048):
            raise ValueError("batch_size is outside the O51 sweep")
        if self.depth_alpha not in (0.5, 1.0):
            raise ValueError("depth_alpha must be 0.5 or 1.0")
        if self.muon_batch_scaling not in ("fixed", "sqrt"):
            raise ValueError("muon_batch_scaling must be fixed or sqrt")
        if self.muon_duration_scaling not in ("fixed", "inverse-sqrt"):
            raise ValueError("muon_duration_scaling must be fixed or inverse-sqrt")
        if self.compile_mode not in ("reduce-overhead", "max-autotune"):
            raise ValueError("compile_mode is outside the O51 preflight grid")
        if self.temporal_attention_chunk not in (8192, 16_384, 32_768, None):
            raise ValueError("temporal_attention_chunk is outside the O51 preflight grid")
        if self.num_workers not in (8, 16, 24, 32):
            raise ValueError("num_workers is outside the O51 preflight grid")
        if self.replay_pack_batch_size not in (16, 32, 64):
            raise ValueError("replay_pack_batch_size is outside the O51 preflight grid")
        if self.loader_prefetch_factor not in (1, 2, 4):
            raise ValueError("loader_prefetch_factor is outside the O51 preflight grid")
        if self.predownload not in (
            8 * self.replay_pack_batch_size,
            16 * self.replay_pack_batch_size,
        ):
            raise ValueError("predownload must be 8x or 16x the replay-pack batch")
        if self.shuffle_algo not in ("py1s", "py1e"):
            raise ValueError("shuffle_algo must be py1s or py1e")
        if self.shuffle_block_size not in (4096, 8192):
            raise ValueError("shuffle_block_size is outside the O51 preflight grid")

    @classmethod
    def load(cls, path: Path) -> Treatment:
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"O51 treatment must be a JSON object: {path}")
        expected = {field.name for field in fields(cls)}
        extra = sorted(payload.keys() - expected)
        if extra:
            raise ValueError(f"O51 treatment has unknown fields: {extra}")
        return cls(**payload)

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")
        temporary.replace(path)


@dataclass(frozen=True, slots=True)
class SweepArm:
    """One independently launchable O51 training arm."""

    arm_id: str
    stage: Stage
    level: Literal["base", "proxy", "mid", "large"]
    treatment: Treatment
    target_positions: int = D0
    tier_scale: int = 1
    seed: int = 0
    stop_after_update: int | None = None

    def __post_init__(self) -> None:
        if not _ARM_ID.fullmatch(self.arm_id):
            raise ValueError(f"invalid sweep arm ID: {self.arm_id!r}")
        if (
            not isinstance(self.tier_scale, int)
            or isinstance(self.tier_scale, bool)
            or self.tier_scale not in TIER_SCALES
            or not isinstance(self.target_positions, int)
            or isinstance(self.target_positions, bool)
            or self.target_positions != self.tier_scale * D0
        ):
            raise ValueError("each sweep arm must use an exact matched D/U endpoint")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        positions_per_update = self.treatment.batch_size * SUPERVISED_POSITIONS_PER_WINDOW
        max_updates, remainder = divmod(self.target_positions, positions_per_update)
        if remainder:
            raise ValueError("target positions must end on an optimizer update")
        if self.stop_after_update is not None and (
            not isinstance(self.stop_after_update, int)
            or isinstance(self.stop_after_update, bool)
            or not 1 <= self.stop_after_update < max_updates
        ):
            raise ValueError("stop_after_update must select an early optimizer update")
        if self.stage == "initialization-screen" and self.stop_after_update is None:
            raise ValueError("initialization-screen arms require an early stop")
        if self.stage != "initialization-screen" and self.stop_after_update is not None:
            raise ValueError("only initialization-screen arms can stop early")

    @property
    def requires_preflight(self) -> bool:
        """Return whether the O51 train command requires launch evidence."""
        return self.stop_after_update is None

    def argv(self, *, preflight_report: Path | None = None) -> tuple[str, ...]:
        if self.stop_after_update is not None and preflight_report is not None:
            raise ValueError("initialization-screen arms do not use a preflight report")
        cfg = {
            "target-positions": self.target_positions,
            "tier-scale": self.tier_scale,
            "seed": self.seed,
            "hidden-std-multiplier": self.treatment.hidden_std_multiplier,
            "readout-init": self.treatment.readout_init,
            "muon-lr": self.treatment.muon_lr,
            "adam-lr": self.treatment.adam_lr,
            "muon-weight-decay": self.treatment.muon_weight_decay,
            "adam-weight-decay": self.treatment.adam_weight_decay,
            "batch-size": self.treatment.batch_size,
            "depth-alpha": self.treatment.depth_alpha,
            "muon-batch-scaling": self.treatment.muon_batch_scaling,
            "muon-duration-scaling": self.treatment.muon_duration_scaling,
            "compile-mode": self.treatment.compile_mode,
            "temporal-attention-chunk": self.treatment.temporal_attention_chunk,
            "num-workers": self.treatment.num_workers,
            "replay-pack-batch-size": self.treatment.replay_pack_batch_size,
            "loader-prefetch-factor": self.treatment.loader_prefetch_factor,
            "predownload": self.treatment.predownload,
            "shuffle-algo": self.treatment.shuffle_algo,
            "shuffle-block-size": self.treatment.shuffle_block_size,
        }
        command = [
            "uv",
            "run",
            EXPERIMENT,
            "train",
            "--level",
            self.level,
            "--comment",
            self.arm_id,
        ]
        if self.stop_after_update is not None:
            command.extend(
                (
                    "--smoke",
                    "--smoke-eval-matchups",
                    "0",
                    "--stop-after-update",
                    str(self.stop_after_update),
                )
            )
        if preflight_report is not None:
            command.extend(("--preflight-report", str(preflight_report)))
        for name, value in cfg.items():
            command.extend((f"--cfg.{name}", "None" if value is None else str(value)))
        return tuple(command)


def _safe_float(value: float) -> str:
    return f"{value:g}".replace("-", "n").replace(".", "p")


def _arm(
    stage: Stage,
    suffix: str,
    level: Literal["base", "proxy", "mid", "large"],
    treatment: Treatment,
    *,
    target_positions: int = D0,
    tier_scale: int = 1,
    seed: int = 0,
    stop_after_update: int | None = None,
) -> SweepArm:
    return SweepArm(
        f"o51-{stage}-{suffix}",
        stage,
        level,
        treatment,
        target_positions=target_positions,
        tier_scale=tier_scale,
        seed=seed,
        stop_after_update=stop_after_update,
    )


def initialization_screen_arms(center: Treatment | None = None) -> tuple[SweepArm, ...]:
    """Six base-model arms stopped at D0/8 while retaining a D0/U0 contract."""
    center = center or Treatment()
    updates = (D0 // 8) // (center.batch_size * 128)
    return tuple(
        _arm(
            "initialization-screen",
            f"h{_safe_float(hidden)}-{readout}",
            "base",
            replace(center, hidden_std_multiplier=hidden, readout_init=readout),
            stop_after_update=updates,
        )
        for hidden in INIT_STD_GRID
        for readout in READOUT_GRID
    )


def lr_arms(center: Treatment) -> tuple[SweepArm, ...]:
    return tuple(
        _arm(
            "lr",
            f"m{_safe_float(muon)}-a{_safe_float(adam)}",
            "base",
            replace(center, muon_lr=muon, adam_lr=adam),
        )
        for muon in MUON_LR_GRID
        for adam in ADAM_LR_GRID
    )


def decay_arms(center: Treatment) -> tuple[SweepArm, ...]:
    return tuple(
        _arm(
            "decay",
            f"m{_safe_float(muon)}-a{_safe_float(adam)}",
            "base",
            replace(center, muon_weight_decay=muon, adam_weight_decay=adam),
        )
        for muon in DECAY_GRID
        for adam in DECAY_GRID
    )


def batch_arms(center: Treatment, *, include_2048: bool = False) -> tuple[SweepArm, ...]:
    batches = (*BATCH_GRID, 2048) if include_2048 else BATCH_GRID
    return tuple(
        _arm(
            "batch",
            f"b{batch}-muon-{rule}",
            "base",
            replace(center, batch_size=batch, muon_batch_scaling=rule),
        )
        for batch in batches
        for rule in ("fixed", "sqrt")
    )


def proxy_transfer_arms(center: Treatment) -> tuple[SweepArm, ...]:
    return tuple(
        _arm(
            "proxy-transfer",
            f"alpha-{_safe_float(alpha)}",
            "proxy",
            replace(center, depth_alpha=alpha),
        )
        for alpha in (0.5, 1.0)
    )


def mid_search_arms(center: Treatment) -> tuple[SweepArm, ...]:
    if center.muon_weight_decay not in DECAY_GRID or center.adam_weight_decay not in DECAY_GRID:
        raise ValueError("mid-search center decays must come from the O51 decay grid")
    candidates = [
        ("center", center),
        ("muon-half", replace(center, muon_lr=center.muon_lr / 2)),
        ("muon-double", replace(center, muon_lr=center.muon_lr * 2)),
        ("adam-half", replace(center, adam_lr=center.adam_lr / 2)),
        ("adam-double", replace(center, adam_lr=center.adam_lr * 2)),
    ]
    candidates.extend(
        (f"muon-decay-{_safe_float(value)}", replace(center, muon_weight_decay=value))
        for value in DECAY_GRID
        if value != center.muon_weight_decay
    )
    candidates.extend(
        (f"adam-decay-{_safe_float(value)}", replace(center, adam_weight_decay=value))
        for value in DECAY_GRID
        if value != center.adam_weight_decay
    )
    return tuple(_arm("mid-search", name, "mid", treatment) for name, treatment in candidates)


def seed_repeat_arms(center: Treatment) -> tuple[SweepArm, ...]:
    return tuple(_arm("seed-repeat", f"seed-{seed}", "mid", center, seed=seed) for seed in (1, 2))


def duration_arms(center: Treatment) -> tuple[SweepArm, ...]:
    return tuple(
        _arm(
            "duration",
            f"s{scale}-{rule}",
            "proxy",
            replace(center, muon_duration_scaling=rule),
            target_positions=scale * D0,
            tier_scale=scale,
        )
        for scale in (2, 4, 8)
        for rule in ("fixed", "inverse-sqrt")
    )


def stage_arms(stage: Stage, center: Treatment, *, include_2048: bool = False) -> tuple[SweepArm, ...]:
    generators: dict[Stage, Callable[[], tuple[SweepArm, ...]]] = {
        "initialization-screen": lambda: initialization_screen_arms(center),
        "lr": lambda: lr_arms(center),
        "decay": lambda: decay_arms(center),
        "batch": lambda: batch_arms(center, include_2048=include_2048),
        "proxy-transfer": lambda: proxy_transfer_arms(center),
        "mid-search": lambda: mid_search_arms(center),
        "seed-repeat": lambda: seed_repeat_arms(center),
        "duration": lambda: duration_arms(center),
    }
    return generators[stage]()
