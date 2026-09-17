"""Persist sampled controller logits and execution-time likelihoods."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Final

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch import Tensor

from hal.training.controller_codec import CONTROLLER_DECODE_ORDER
from hal.training.controller_codec import CONTROLLER_GROUP_NAMES
from hal.training.controller_codec import CONTROLLER_GROUP_VOCABS

ACTION_TRACE_SCHEMA_VERSION: Final[int] = 1

_TRACE_SCHEMA = pa.schema(
    [
        pa.field("model", pa.string(), nullable=False),
        pa.field("decode_seed", pa.int64(), nullable=False),
        pa.field("slot_id", pa.int64(), nullable=False),
        pa.field("generation", pa.int32(), nullable=False),
        pa.field("replan_index", pa.int32(), nullable=False),
        pa.field("replan_frame", pa.int32(), nullable=False),
        pa.field("execution_frame", pa.int32(), nullable=False),
        pa.field("plan_depth", pa.int16(), nullable=False),
        pa.field("head_offset", pa.int16(), nullable=False),
        pa.field("group", pa.string(), nullable=False),
        pa.field("sampled_index", pa.int16(), nullable=False),
        pa.field("uniform", pa.float64(), nullable=False),
        pa.field("temperature", pa.float32(), nullable=False),
        pa.field("sampled_logit", pa.float32(), nullable=False),
        pa.field("sampled_probability", pa.float32(), nullable=False),
        pa.field("sampled_log_probability", pa.float32(), nullable=False),
        pa.field("base_probability", pa.float32(), nullable=False),
        pa.field("base_log_probability", pa.float32(), nullable=False),
        pa.field("rank", pa.int16(), nullable=False),
        pa.field("entropy_nats", pa.float32(), nullable=False),
        pa.field("base_entropy_nats", pa.float32(), nullable=False),
        pa.field("top_index", pa.int16(), nullable=False),
        pa.field("top_probability", pa.float32(), nullable=False),
        pa.field("action_log_probability", pa.float32(), nullable=False),
        pa.field("action_probability", pa.float32(), nullable=False),
        pa.field("logits", pa.list_(pa.float32()), nullable=False),
    ],
    metadata={"hal_action_trace_schema": str(ACTION_TRACE_SCHEMA_VERSION)},
)


@dataclass
class _SlotClock:
    generation: int = -1
    replan_index: int = 0


class ActionTraceWriter:
    """Write valid Parquet parts while a live policy samples action plans.

    Each row describes one controller group for one action that will execute.
    Logits are the centered, legality-masked values used by the sampler before
    temperature scaling. The four group probabilities multiply to the full
    action probability because the decoder samples groups autoregressively.
    """

    def __init__(self, root: str | Path, *, model: str, flush_rows: int = 8192) -> None:
        if flush_rows < 1:
            raise ValueError(f"flush_rows must be positive, got {flush_rows}")
        self.root = Path(root)
        if self.root.exists():
            raise FileExistsError(f"refusing to overwrite action trace directory {self.root}")
        self.root.mkdir(parents=True)
        self.model = model
        self.flush_rows = flush_rows
        self._columns: dict[str, list[object]] = {field.name: [] for field in _TRACE_SCHEMA}
        self._clocks: dict[tuple[int, int], _SlotClock] = {}
        self._part = 0
        self._closed = False
        manifest = {
            "schema_version": ACTION_TRACE_SCHEMA_VERSION,
            "model": model,
            "controller_decode_order": list(CONTROLLER_DECODE_ORDER),
            "controller_group_names": list(CONTROLLER_GROUP_NAMES),
            "controller_group_vocabs": dict(zip(CONTROLLER_GROUP_NAMES, CONTROLLER_GROUP_VOCABS, strict=True)),
            "logits": "centered legality-masked logits before temperature scaling",
            "probabilities": "softmax(logits / temperature)",
            "action_probability": "product of the four sampled conditional group probabilities",
        }
        (self.root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    def record_plan(
        self,
        *,
        decode_seed: int,
        slot_ids: Tensor,
        resets: Tensor | None,
        indices: Tensor,
        logits: tuple[Tensor, ...],
        uniforms: Tensor,
        head_offsets: tuple[int, ...],
        temperature: float,
        delay_frames: int,
        replan_interval_frames: int,
    ) -> None:
        """Record the plan slice that the delayed policy will execute."""
        if self._closed:
            raise RuntimeError("cannot record to a closed action trace")
        batch, horizon, groups = indices.shape
        expected_logits = tuple((batch, horizon, vocab) for vocab in CONTROLLER_GROUP_VOCABS)
        actual_logits = tuple(tuple(values.shape) for values in logits)
        if groups != len(CONTROLLER_GROUP_NAMES) or actual_logits != expected_logits:
            raise ValueError(f"trace logits have shapes {actual_logits}, expected {expected_logits}")
        if uniforms.shape != (horizon, groups, batch):
            raise ValueError(f"trace uniforms have shape {tuple(uniforms.shape)}, expected {(horizon, groups, batch)}")
        if slot_ids.shape != (batch,):
            raise ValueError(f"trace slot_ids have shape {tuple(slot_ids.shape)}, expected {(batch,)}")
        if resets is not None and resets.shape != (batch,):
            raise ValueError(f"trace resets have shape {tuple(resets.shape)}, expected {(batch,)}")
        stop = delay_frames + replan_interval_frames
        if delay_frames < 0 or replan_interval_frames < 1 or stop > horizon:
            raise ValueError(
                "executed trace slice must fit the sampled horizon: "
                f"delay={delay_frames}, replan={replan_interval_frames}, horizon={horizon}"
            )
        if len(head_offsets) != horizon:
            raise ValueError(f"got {len(head_offsets)} head offsets for horizon {horizon}")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError(f"temperature must be finite and positive, got {temperature!r}")

        indices = indices.detach().cpu().long()
        logits = tuple(values.detach().cpu().float() for values in logits)
        uniforms = uniforms.detach().cpu().double()
        slot_ids = slot_ids.detach().cpu().long()
        reset_values = torch.zeros(batch, dtype=torch.bool) if resets is None else resets.detach().cpu().bool()

        probabilities: list[Tensor] = []
        log_probabilities: list[Tensor] = []
        base_probabilities: list[Tensor] = []
        base_log_probabilities: list[Tensor] = []
        entropies: list[Tensor] = []
        base_entropies: list[Tensor] = []
        ranks: list[Tensor] = []
        scaled_distributions: list[Tensor] = []
        for group, values in enumerate(logits):
            scaled_log_probability = F.log_softmax(values / temperature, dim=-1)
            base_log_probability = F.log_softmax(values, dim=-1)
            scaled_probability = scaled_log_probability.exp()
            base_probability = base_log_probability.exp()
            picked = indices[..., group, None]
            picked_log_probability = scaled_log_probability.gather(-1, picked).squeeze(-1)
            picked_base_log_probability = base_log_probability.gather(-1, picked).squeeze(-1)
            picked_logit = values.gather(-1, picked).squeeze(-1)
            scaled_distributions.append(scaled_probability)
            probabilities.append(picked_log_probability.exp())
            log_probabilities.append(picked_log_probability)
            base_probabilities.append(picked_base_log_probability.exp())
            base_log_probabilities.append(picked_base_log_probability)
            entropies.append(-(scaled_probability * scaled_log_probability).nan_to_num().sum(dim=-1))
            base_entropies.append(-(base_probability * base_log_probability).nan_to_num().sum(dim=-1))
            ranks.append((values > picked_logit[..., None]).sum(dim=-1) + 1)
        action_log_probability = torch.stack(log_probabilities).sum(dim=0)

        for row in range(batch):
            slot_id = int(slot_ids[row])
            clock = self._clock(decode_seed, slot_id, reset=bool(reset_values[row]))
            replan_frame = clock.replan_index * replan_interval_frames
            for depth in range(delay_frames, stop):
                action_logp = float(action_log_probability[row, depth])
                for group, name in enumerate(CONTROLLER_GROUP_NAMES):
                    values = logits[group][row, depth]
                    sampled_index = int(indices[row, depth, group])
                    scaled = scaled_distributions[group][row, depth]
                    self._append(
                        model=self.model,
                        decode_seed=decode_seed,
                        slot_id=slot_id,
                        generation=clock.generation,
                        replan_index=clock.replan_index,
                        replan_frame=replan_frame,
                        execution_frame=replan_frame + depth,
                        plan_depth=depth,
                        head_offset=head_offsets[depth],
                        group=name,
                        sampled_index=sampled_index,
                        uniform=float(uniforms[depth, group, row]),
                        temperature=temperature,
                        sampled_logit=float(values[sampled_index]),
                        sampled_probability=float(probabilities[group][row, depth]),
                        sampled_log_probability=float(log_probabilities[group][row, depth]),
                        base_probability=float(base_probabilities[group][row, depth]),
                        base_log_probability=float(base_log_probabilities[group][row, depth]),
                        rank=int(ranks[group][row, depth]),
                        entropy_nats=float(entropies[group][row, depth]),
                        base_entropy_nats=float(base_entropies[group][row, depth]),
                        top_index=int(scaled.argmax()),
                        top_probability=float(scaled.max()),
                        action_log_probability=action_logp,
                        action_probability=math.exp(action_logp),
                        logits=values.tolist(),
                    )
            clock.replan_index += 1
        if len(self._columns["model"]) >= self.flush_rows:
            self.flush()

    def _clock(self, decode_seed: int, slot_id: int, *, reset: bool) -> _SlotClock:
        key = (decode_seed, slot_id)
        clock = self._clocks.get(key)
        if clock is None:
            clock = _SlotClock(generation=0)
            self._clocks[key] = clock
        elif reset:
            clock.generation += 1
            clock.replan_index = 0
        return clock

    def _append(self, **values: object) -> None:
        if set(values) != set(self._columns):
            raise ValueError("action trace row does not match the persisted schema")
        for name in self._columns:
            self._columns[name].append(values[name])

    def flush(self) -> None:
        """Write one atomic, independently readable Parquet part."""
        if not self._columns["model"]:
            return
        table = pa.Table.from_pydict(self._columns, schema=_TRACE_SCHEMA)
        target = self.root / f"part-{self._part:05d}.parquet"
        temporary = target.with_suffix(".parquet.tmp")
        if target.exists() or temporary.exists():
            raise FileExistsError(f"refusing to overwrite action trace part {target}")
        pq.write_table(table, temporary, compression="zstd")
        temporary.rename(target)
        self._part += 1
        self._columns = {field.name: [] for field in _TRACE_SCHEMA}

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        self._closed = True

    def __enter__(self) -> ActionTraceWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()
