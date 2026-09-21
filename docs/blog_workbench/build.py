"""Assemble the blog workbench artifact.

Inputs (all under this directory):
  shell.html, engine.{js,css}, tokens.css   the workbench shell
  figures_src/manifest.json + <key>.{html,js} hand-written figures (diagram-kit.js, figures.css shared)
  figures_src/legacy/                       figures frozen from the retired standalone pages, with their
                                            own css/js (anatomy.*, explorer.*) and manifest.json
  evidence.json                             the study table behind the evidence blocks
  ci.json                                   every treatment − baseline interval (ci_stats.py)
  figures_src/figure-data.json              series behind the charts that evidence.json lacks (figure_data.py)
  assets/                                   Charter fonts + controller illustration

Outputs (dist/):
  index.html, figures/{anatomy,explorer,custom}.js, figures/*.svg, fonts/, assets/, data/

Run:  uv run build.py

Static export — the same page with the document baked in, read-only (no store, no editing,
parked blocks and notes left out), for hosting anywhere as a plain folder:

  uv run build.py --static <dump dir with doc/outline.json + blocks/*.json> [--out export/site]

Per-figure export — one standalone page per figure placed in the dump, named by figure key, with
its caption and page number; shared figures/, fonts/ and assets/ beside them:

  uv run build.py --figures <dump dir> [--out figures]
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIST = HERE / "dist"
SRC = HERE / "figures_src"
LEGACY = SRC / "legacy"
ASSETS = HERE / "assets"


def legacy_figures() -> list[dict]:
    """Frozen figures: html fragments (or svg files) plus registry metadata."""
    out = []
    for e in json.loads((LEGACY / "manifest.json").read_text()):
        if e["source"] == "svg":
            html = f'<figure class="svg-figure"><img src="figures/{e["file"]}" alt="{e["name"]}"></figure>'
        else:
            html = (LEGACY / f"{e['key']}.html").read_text()
        out.append({**e, "html": html})
    return out


def custom_figures() -> list[dict]:
    entries = json.loads((SRC / "manifest.json").read_text())
    out = []
    for e in entries:
        js = (SRC / f"{e['key']}.js").read_text()
        if "attach" in e:  # script-only module that re-renders part of a legacy figure
            out.append({**e, "source": "custom", "js": js})
            continue
        html = (SRC / f"{e['key']}.html").read_text()
        out.append({**e, "source": "custom", "html": html, "js": js})
    return out


def registry_json(figures: list[dict]) -> str:
    entries = []
    for f in figures:
        entries.append(
            {
                "key": f["key"],
                "name": f["name"],
                "source": f["source"],
                "caption": f.get("caption", ""),
                "data": f.get("data", []),
                "studies": f["studies"],
            }
        )
    return json.dumps(entries, ensure_ascii=False, indent=1)


def inline_evidence() -> dict:
    """The studies for the data drawer; the per-boot rows stay in data/evidence.json only."""
    evidence = json.loads((HERE / "evidence.json").read_text())
    evidence.pop("raw_evaluation_data", None)
    return evidence


SEED_FIELDS = ("type", "md", "caption", "fig", "level", "sub", "anchor")


def dump_blocks(dump: Path) -> tuple[list[str], dict[str, dict]]:
    outline = json.loads((dump / "doc" / "outline.json").read_text())
    outline = outline.get("data", outline)
    blocks = {}
    for path in (dump / "blocks").glob("*.json"):
        raw = json.loads(path.read_text())
        blocks[path.stem] = raw.get("data", raw)
    return outline["order"], blocks


def static_seed(dump: Path) -> dict:
    """The visible, non-empty blocks of a store dump, in outline order, stripped to what the renderer reads."""
    outline, blocks = dump_blocks(dump)
    order, kept = [], {}
    for bid in outline:
        b = blocks.get(bid)
        if b is None or b.get("hidden"):
            continue
        if b["type"] != "figure" and not (b.get("md") or "").strip():
            continue
        order.append(bid)
        kept[bid] = {k: b[k] for k in SEED_FIELDS if k in b}
    return {"outline": {"order": order}, "blocks": kept}


def figure_seeds(dump: Path) -> list[tuple[str, dict]]:
    """One single-block seed per placed figure, keyed by figure key, numbered as on the page.

    Numbering mirrors engine.js figureNumbers(): figures count in order and a `sub` figure shares the
    previous number as 2a, 2b, ...; the number rides on the block as `num` so a lone block keeps it.
    """
    outline, blocks = dump_blocks(dump)
    placed = [
        blocks[bid]
        for bid in outline
        if bid in blocks and blocks[bid]["type"] == "figure" and not blocks[bid].get("hidden")
    ]
    groups: list[list[dict]] = []
    for b in placed:
        if b.get("sub") and groups:
            groups[-1].append(b)
        else:
            groups.append([b])
    seeds = []
    for n, group in enumerate(groups, start=1):
        for k, b in enumerate(group):
            num = f"{n}{chr(97 + k)}" if len(group) > 1 else str(n)
            block = {key: b[key] for key in SEED_FIELDS if key in b}
            block["num"] = num
            seeds.append((b["fig"], {"outline": {"order": ["fig"]}, "blocks": {"fig": block}}))
    keys = [key for key, _ in seeds]
    assert len(keys) == len(set(keys)), "a figure is placed twice"
    return seeds


def build(
    out: Path = DIST, seed: dict | None = None, page: str = "index.html", chrome: bool = True, fresh: bool = True
) -> None:
    """Assemble `out/`; `seed` bakes the document in (static). `chrome=False` hides the page furniture
    around the document for a single-figure page. Several pages share one folder when only the first
    call is `fresh`; the shared files are rewritten identically each time."""
    static = seed is not None
    if fresh and out.exists():
        shutil.rmtree(out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "fonts").mkdir(exist_ok=True)
    (out / "assets").mkdir(exist_ok=True)

    for face in ["charter_regular", "charter_italic", "charter_bold", "charter_bold_italic"]:
        shutil.copy(ASSETS / "fonts" / f"{face}.woff2", out / "fonts" / f"{face}.woff2")
    shutil.copy(ASSETS / "GCController_Layout.svg", out / "assets")
    if not static:  # the page never fetches it; it ships beside the workbench for download only
        (out / "data").mkdir()
        shutil.copy(HERE / "evidence.json", out / "data" / "evidence.json")
    for svg in LEGACY.glob("*.svg"):
        shutil.copy(svg, out / "figures" / svg.name)

    legacy = legacy_figures()
    modules = custom_figures()
    custom = [f for f in modules if "attach" not in f]
    figures = legacy + custom
    keys = [f["key"] for f in figures]
    assert len(keys) == len(set(keys)), "duplicate figure keys"

    # orphans.html holds containers anatomy.js still renders into but no placed figure shows
    pool = [
        '<div class="fig-src anatomy" data-fig="orphans"><figure>'
        + (LEGACY / "orphans.html").read_text()
        + "</figure></div>"
    ]
    for f in figures:
        cls = "xfig" if f["source"] == "explorer" else f["source"]
        pool.append(f'<div class="fig-src {cls}" data-fig="{f["key"]}">\n{f["html"]}\n</div>')

    shutil.copy(LEGACY / "anatomy.js", out / "figures" / "anatomy.js")
    shutil.copy(LEGACY / "explorer.js", out / "figures" / "explorer.js")
    kit = (SRC / "diagram-kit.js").read_text()
    ledger = "window.HALFIG_AWR_LEDGER = " + (SRC / "awr-ledger.json").read_text().strip() + ";\n"
    series = "window.HALFIG_DATA = " + json.dumps(json.loads((SRC / "figure-data.json").read_text())) + ";\n"
    ci = "window.HALFIG_CI = " + json.dumps(json.loads((HERE / "ci.json").read_text())) + ";\n"
    (out / "figures" / "custom.js").write_text(kit + "\n" + ledger + series + ci + "\n".join(f["js"] for f in modules))

    shell = (HERE / "shell.html").read_text()
    if static:
        shell = re.sub(r"<!--wb-only-->.*?<!--/wb-only-->", "", shell, flags=re.S)
        shell = shell.replace(
            "<!--__SEED__-->",
            "<script>window.WB_STATIC = true; window.WB_SEED = "
            + json.dumps(seed, ensure_ascii=False).replace("</", "<\\/")
            + ";</script>",
        )
        if not chrome:
            shell = shell.replace('<div class="wb-shell">', '<div class="wb-shell wb-figure-page">')
        shell = (
            '<!doctype html>\n<html lang="en">\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            + shell
        )
    html = (
        shell.replace("/*__TOKENS__*/", (HERE / "tokens.css").read_text())
        .replace("/*__MAIN_CSS__*/", (LEGACY / "anatomy.css").read_text())
        .replace("/*__EXPLORER_CSS__*/", (LEGACY / "explorer.css").read_text())
        .replace("/*__ENGINE_CSS__*/", (HERE / "engine.css").read_text() + "\n" + (SRC / "figures.css").read_text())
        .replace("<!--__FIGURE_POOL__-->", "\n".join(pool))
        .replace("/*__REGISTRY__*/", registry_json(figures))
        .replace("/*__EVIDENCE__*/", json.dumps(inline_evidence(), ensure_ascii=False).replace("</", "<\\/"))
        .replace("/*__ENGINE_JS__*/", (HERE / "engine.js").read_text())
    )
    (out / page).write_text(html)
    if not static:
        (HERE / "registry.json").write_text(registry_json(figures))
    placed = ""
    if static:
        n_fig = sum(1 for b in seed["blocks"].values() if b["type"] == "figure")
        placed = f" · {len(seed['outline']['order'])} blocks, {n_fig} figures baked in"
    print(
        f"{out.relative_to(HERE)}/{page} {len(html) // 1024} KB · {len(figures)} figures ({len(custom)} hand-written, {len(figures) - len(custom)} legacy){placed}"
    )


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--static" in args:
        dump = Path(args[args.index("--static") + 1])
        out = Path(args[args.index("--out") + 1]) if "--out" in args else HERE / "export" / "site"
        build(out.resolve(), static_seed(dump))
    elif "--figures" in args:
        dump = Path(args[args.index("--figures") + 1])
        out = (Path(args[args.index("--out") + 1]) if "--out" in args else HERE / "figures").resolve()
        for i, (key, seed) in enumerate(figure_seeds(dump)):
            build(out, seed, page=f"{key}.html", chrome=False, fresh=i == 0)
    else:
        build()
