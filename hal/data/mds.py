"""Small helpers for reading local MDS shards."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from streaming.base.compression import decompress
from streaming.base.format import reader_from_json
from streaming.base.format.base.reader import Reader

# Full replay MDS shards use a 2 GiB uncompressed size limit. Compression makes
# stored shards much smaller while this limit keeps writer output consistent.
FULL_MDS_SHARD_SIZE_LIMIT = 1 << 31


def read_shard_index(root: Path, split: str) -> list[dict[str, Any]]:
    path = root / split / "index.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found")
    return json.loads(path.read_text())["shards"]


@contextmanager
def open_shard(root: Path, split: str, info: dict[str, Any], scratch: Path) -> Iterator[Reader]:
    raw_name = info["raw_data"]["basename"]
    if (root / split / raw_name).is_file():
        yield reader_from_json(str(root), split, info)
        return

    zip_data = info.get("zip_data")
    if zip_data is None:
        raise FileNotFoundError(f"{root / split / raw_name} is missing and the shard has no compressed copy")
    with TemporaryDirectory(dir=scratch) as tmp_root:
        tmp_dir = Path(tmp_root) / split
        tmp_dir.mkdir(parents=True)
        compressed = (root / split / zip_data["basename"]).read_bytes()
        (tmp_dir / raw_name).write_bytes(decompress(info["compression"], compressed))
        yield reader_from_json(tmp_root, split, info)
