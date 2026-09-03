"""Column schema for O51 replay bands."""

from __future__ import annotations

from typing import Final

from hal.data.policy_world_schema import POLICY_WORLD_MDS_COLUMNS

O51_MDS_SCHEMA_VERSION: Final[int] = 1
O51_RETURN_SUFFIX: Final[str] = "awr_return"


def o51_mds_columns() -> dict[str, str]:
    columns = dict(POLICY_WORLD_MDS_COLUMNS)
    columns["o51_schema_version"] = "int"
    for port in ("p1", "p2"):
        columns[f"{port}_{O51_RETURN_SUFFIX}"] = "ndarray:float32"
        columns[f"{port}_{O51_RETURN_SUFFIX}_valid"] = "ndarray:uint8"
        columns[f"{port}_player_id"] = "int"
    return columns


O51_MDS_COLUMNS: Final[dict[str, str]] = o51_mds_columns()
