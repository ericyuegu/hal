"""Build the offline preview's seed.js from a document-store dump (read_db --out_dir) plus an outline JSON.

uv run preview_from_dump.py <dump dir with blocks/*.json> <outline.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
dump, outline_path = Path(sys.argv[1]), Path(sys.argv[2])
outline = json.loads(outline_path.read_text())
outline = outline.get("data", outline)
blocks = {p.stem: json.loads(p.read_text()) for p in (dump / "blocks").glob("*.json")}
blocks = {k: v.get("data", v) for k, v in blocks.items()}
(HERE / "seed").mkdir(exist_ok=True)
(HERE / "seed" / "seed.js").write_text(
    "window.WB_SEED = " + json.dumps({"outline": outline, "blocks": blocks}, ensure_ascii=False) + ";\n"
)
print(f"seed.js from {len(blocks)} blocks · {len(outline['order'])} in outline")
