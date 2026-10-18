"""Build episode-safe contexts from native adapter columns on their device."""

import torch
from torch import Tensor

from hal.data.feature_stats import FeatureStats
from hal.data.feature_stats import consolidate_key
from hal.representation.features import BASE_ITEMS_PROJECTION
from hal.representation.features import ITEM_COLUMNS
from hal.representation.features import Context
from hal.representation.features import feature_kind
from hal.representation.features import float_feature_transform
from hal.wire import ACTION_CHANNELS
from hal.wire import MASK_INT32


class ColumnContextBuilder:
    """Resolve column routes and normalization once, outside frame loops.

    Applied actions belong to the observed post-frame, as in BC. At an actual
    reset there is no preceding input, so the ego controller is neutral.
    Player ids can be per-frame when a training window spans different episodes.
    """

    def __init__(self, stats: dict[str, FeatureStats], controlled_port: int) -> None:
        if controlled_port not in (1, 2):
            raise ValueError("controlled port must be 1 or 2")
        self.port = controlled_port
        action_names = {f"ego_{channel}" for channel in ACTION_CHANNELS}
        self.routes: list[tuple[str, str, str, str, float, float]] = []
        for name in sorted(BASE_ITEMS_PROJECTION.columns - action_names):
            source = name
            if name.startswith("ego_"):
                source = f"p{controlled_port}_{name[4:]}"
            elif name.startswith("opp_"):
                source = f"p{3 - controlled_port}_{name[4:]}"
            kind = feature_kind(name, ITEM_COLUMNS)
            transform, center, scale = "", 0.0, 1.0
            if kind == "float":
                stat = stats[consolidate_key(name)]
                transform = float_feature_transform(name, ITEM_COLUMNS)
                center, scale = (
                    (stat.mean, stat.std) if transform == "standardize" else (stat.min, stat.max - stat.min)
                )
            self.routes.append((source, name, kind, transform, center, scale))

    def __call__(
        self,
        columns: dict[str, Tensor],
        applied_actions: Tensor,
        segment_ids: Tensor,
        episode_starts: Tensor,
        player_ids: Tensor,
    ) -> Context:
        if (
            segment_ids.ndim != 2
            or segment_ids.dtype not in (torch.int32, torch.int64)
            or episode_starts.shape != segment_ids.shape
            or episode_starts.dtype != torch.bool
        ):
            raise ValueError("episode starts and segments must be [B, L]")
        batch, length = segment_ids.shape
        if applied_actions.shape != (batch, length, 2, len(ACTION_CHANNELS)):
            raise ValueError("applied actions have the wrong shape")
        if player_ids.shape not in ((batch,), (batch, length)) or player_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("player ids must be integer [B] or [B, L]")
        if not torch.compiler.is_compiling() and not torch.equal(
            episode_starts[:, 1:], segment_ids[:, 1:] != segment_ids[:, :-1]
        ):
            raise ValueError("episode starts disagree with segment boundaries")
        features: dict[str, Tensor] = {}
        for source, name, kind, transform, center, scale in self.routes:
            value = columns[source]
            if value.shape != (batch, length):
                raise ValueError(f"column {source} must be [B, L]")
            if kind == "cat" and value.dtype not in (torch.int32, torch.int64):
                raise ValueError(f"categorical column {source} must be int32 or int64")
            if kind == "float" and value.dtype != torch.float32:
                raise ValueError(f"float column {source} must be float32")
            masked = value.isnan() if value.is_floating_point() else value == MASK_INT32
            if kind == "cat":
                features[name] = torch.where(masked, 0, value).long()
                continue
            if kind != "float":
                raise ValueError(f"unsupported observation column {name}")
            value = value.float()
            if scale == 0:
                value = torch.zeros_like(value)
            elif transform == "standardize":
                value = (value - center) / value.new_tensor(scale)
            else:
                value = 2 * (value - center) / value.new_tensor(scale) - 1
            features[name] = torch.where(masked, 0, value)
            features[f"{name}_mask"] = masked.float()
        actions = torch.where(episode_starts[..., None], 0, applied_actions[:, :, self.port - 1])
        for channel, name in enumerate(ACTION_CHANNELS):
            value = actions[..., channel]
            features[f"ego_{name}"] = (value > 0.5).float() if name.startswith("button_") else value
        features["ego_player_id"] = player_ids[:, None].expand(batch, length) if player_ids.ndim == 1 else player_ids
        return Context(features, torch.zeros(batch, device=segment_ids.device, dtype=torch.long), segment_ids)


class PackedColumnContextBuilder:
    """Build current-episode features from adapter words with batched column operations.

    Each row contains one episode and uses HAL's fixed port convention. Padding
    follows the valid frames. The caller validates sides, identities and history spans.
    Integer and float columns occupy one little-endian 32-bit word each.
    """

    def __init__(self, stats: dict[str, FeatureStats], fields: tuple[str, ...], device: torch.device) -> None:
        routes = [ColumnContextBuilder(stats, port).routes for port in (1, 2)]
        self.names: dict[str, tuple[str, ...]] = {}
        self.indices: dict[str, Tensor] = {}
        self.word_count = len(fields)
        for kind in ("cat", "float"):
            selected = [[route for route in side if route[2] == kind] for side in routes]
            self.names[kind] = tuple(route[1] for route in selected[0])
            self.indices[kind] = torch.tensor(
                [[fields.index(route[0]) for route in side] for side in selected], device=device
            )
        floats = [route for route in routes[0] if route[2] == "float"]
        self.center = torch.tensor([route[4] for route in floats], device=device)
        self.scale = torch.tensor([route[5] if route[5] else 1.0 for route in floats], device=device)
        self.constant = torch.tensor([route[5] == 0 for route in floats], device=device)
        self.standardize = torch.tensor([route[3] == "standardize" for route in floats], device=device)

    def __call__(
        self, records: Tensor, applied_actions: Tensor, sides: Tensor, identities: Tensor, lengths: Tensor
    ) -> dict[str, Tensor]:
        if (
            records.dtype != torch.uint8
            or records.ndim != 3
            or records.shape[-1] < 4 * self.word_count
            or records.shape[-1] % 4
        ):
            raise ValueError("packed observations must contain byte-aligned 32-bit columns")
        batch, length, _ = records.shape
        if applied_actions.dtype != torch.float32 or applied_actions.shape != (batch, length, 2, len(ACTION_CHANNELS)):
            raise ValueError("packed applied actions must be float32 [B, L, 2, channels]")
        if any(
            value.shape != (batch,) or value.dtype not in (torch.int32, torch.int64)
            for value in (sides, identities, lengths)
        ):
            raise ValueError("packed sides, identities, and lengths must be integer [B]")
        words = records[..., : 4 * self.word_count].contiguous()
        index = self.indices["cat"][sides, None].expand(-1, length, -1)
        integers = words.view(torch.int32).gather(-1, index)
        integers = torch.where(integers == MASK_INT32, 0, integers).long()
        index = self.indices["float"][sides, None].expand(-1, length, -1)
        floats = words.view(torch.float32).gather(-1, index)
        missing = floats.isnan()
        centered = floats - self.center
        normalized = torch.where(self.standardize, centered / self.scale, 2 * centered / self.scale - 1)
        normalized = torch.where(missing | self.constant, 0, normalized)
        valid = torch.arange(length, device=records.device)[None] < lengths[:, None]
        integers = torch.where(valid[..., None], integers, 0)
        normalized = torch.where(valid[..., None], normalized, 0)
        masks = (missing & valid[..., None]).float()
        features = dict(zip(self.names["cat"], integers.unbind(-1), strict=True))
        features.update(zip(self.names["float"], normalized.unbind(-1), strict=True))
        features.update(
            (f"{name}_mask", value) for name, value in zip(self.names["float"], masks.unbind(-1), strict=True)
        )
        actions = applied_actions.gather(2, sides[:, None, None, None].expand(batch, length, 1, len(ACTION_CHANNELS)))
        actions = torch.where(valid[..., None], actions.squeeze(2), 0)
        for channel, name in enumerate(ACTION_CHANNELS):
            value = actions[..., channel]
            features[f"ego_{name}"] = (value > 0.5).float() if name.startswith("button_") else value
        features["ego_player_id"] = torch.where(valid, identities[:, None], 0)
        return features
