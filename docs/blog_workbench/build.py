"""Assemble the blog workbench artifact.

Inputs (all under this directory):
  shell.html, engine.{js,css}, tokens.css   the workbench shell
  figures_src/manifest.json + <key>.{html,js} hand-written figures (diagram-kit.js, figures.css shared)
  figures_src/legacy/                       figures frozen from the retired standalone pages, with their
                                            own css/js (anatomy.*, explorer.*) and manifest.json
  evidence.json                             the study table behind the evidence blocks
  ci.json                                   every treatment − baseline interval (ci_stats.py)
  assets/                                   Charter fonts + controller illustration

Outputs (dist/):
  index.html, figures/{anatomy,explorer,custom}.js, figures/*.svg, fonts/, assets/, data/

Run:  uv run build.py
"""

from __future__ import annotations

import json
import shutil
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


def build() -> None:
    if DIST.exists():
        shutil.rmtree(DIST)
    (DIST / "figures").mkdir(parents=True)
    (DIST / "fonts").mkdir()
    (DIST / "assets").mkdir()
    (DIST / "data").mkdir()

    for face in ["charter_regular", "charter_italic", "charter_bold", "charter_bold_italic"]:
        shutil.copy(ASSETS / "fonts" / f"{face}.woff2", DIST / "fonts" / f"{face}.woff2")
    shutil.copy(ASSETS / "GCController_Layout.svg", DIST / "assets")
    shutil.copy(HERE / "evidence.json", DIST / "data" / "evidence.json")
    for svg in LEGACY.glob("*.svg"):
        shutil.copy(svg, DIST / "figures" / svg.name)

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

    shutil.copy(LEGACY / "anatomy.js", DIST / "figures" / "anatomy.js")
    shutil.copy(LEGACY / "explorer.js", DIST / "figures" / "explorer.js")
    kit = (SRC / "diagram-kit.js").read_text()
    ledger = "window.HALFIG_AWR_LEDGER = " + (SRC / "awr-ledger.json").read_text().strip() + ";\n"
    ci = "window.HALFIG_CI = " + json.dumps(json.loads((HERE / "ci.json").read_text())) + ";\n"
    (DIST / "figures" / "custom.js").write_text(kit + "\n" + ledger + ci + "\n".join(f["js"] for f in modules))

    shell = (HERE / "shell.html").read_text()
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
    (DIST / "index.html").write_text(html)
    (HERE / "registry.json").write_text(registry_json(figures))
    print(
        f"index.html {len(html) // 1024} KB · {len(figures)} figures ({len(custom)} hand-written, {len(figures) - len(custom)} legacy)"
    )


if __name__ == "__main__":
    build()
