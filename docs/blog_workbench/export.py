"""Turn a dump of the workbench document store into markdown.

The dump is what `read_db --out_dir <dir>` writes: <dir>/doc/outline.json and
<dir>/blocks/<id>.json (the seed/ directory has the same shape).

    uv run export.py seed export/policy_anatomy.md
    uv run export.py /path/to/db-dump export/policy_anatomy.md --notes

Figures export as the same stand-in blockquotes the draft used, keyed by their
stable registry key; `{{fig:key}}` references become "Fig N".
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REGISTRY = {r["key"]: r for r in json.loads((HERE / "registry.json").read_text())}


def load(dump: Path) -> tuple[dict, dict[str, dict]]:
    outline_path = dump / "doc" / "outline.json"
    if not outline_path.exists():
        outline_path = dump / "outline.json"
    outline = json.loads(outline_path.read_text())
    outline = outline.get("data", outline)
    blocks = {}
    for path in (dump / "blocks").glob("*.json"):
        raw = json.loads(path.read_text())
        blocks[path.stem] = raw.get("data", raw)
    return outline, blocks


def export(outline: dict, blocks: dict[str, dict], notes: bool) -> str:
    visible = [(i, blocks[i]) for i in outline["order"] if i in blocks and not blocks[i].get("hidden")]
    fig_nums: dict[str, int] = {}
    n = 0
    for _, b in visible:
        if b["type"] == "figure":
            n += 1
            fig_nums[b["fig"]] = n
    out: list[str] = []
    for _bid, b in visible:
        if b["type"] == "heading":
            out.append("#" * int(b.get("level", 2)) + " " + re.sub(r"^#+\s*", "", b.get("md", "")))
        elif b["type"] == "figure":
            name = REGISTRY.get(b["fig"], {}).get("name", b["fig"])
            out.append(f"> **FIGURE {fig_nums[b['fig']]} — {name}** `{b['fig']}`")
            if b.get("caption"):
                out.append(">")
                out.append("> " + b["caption"].replace("\n", "\n> "))
        else:
            md = re.sub(
                r"\{\{\s*fig:([a-z0-9-]+)\s*\}\}", lambda m: f"Fig {fig_nums.get(m.group(1), '?')}", b.get("md", "")
            )
            if b["type"] == "callout":
                md = "\n".join("> " + line for line in md.split("\n"))
            out.append(md)
        if notes and b.get("note"):
            out.append("<!--\n" + b["note"] + "\n-->")
        out.append("")
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip() + "\n"


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dump, target = Path(args[0]), Path(args[1])
    outline, blocks = load(dump)
    text = export(outline, blocks, "--notes" in sys.argv)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    print(f"{target}: {len(text.splitlines())} lines from {len(outline['order'])} blocks")
