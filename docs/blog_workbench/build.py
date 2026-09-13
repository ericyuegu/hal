"""Assemble the blog workbench artifact from the two existing interactive pages.

Inputs (../, i.e. docs/):
  policy_anatomy.html                  the interactive page (design-system tokens, 16 catalogued figures)
  policy_anatomy_artifact_export.html  the Claude artifact export (5-stage explorer, 9 named figures)
  blog_experiment_data.json            the study table behind the evidence blocks

Outputs (dist/):
  index.html, figures/anatomy.js, figures/explorer.js, figures/*.svg, fonts/, assets/, data/

Run:  uv run build.py
"""

from __future__ import annotations

import html as htmlmod
import json
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent
DIST = HERE / "dist"
REPO = HERE.parents[1]
DESIGN_SYSTEM = Path("/tmp/ericyuegu.com Design System")
FONTS = DOCS / "policy_anatomy_assets" / "fonts"

MAIN = (DOCS / "policy_anatomy.html").read_text()
EXPLORER = (DOCS / "policy_anatomy_artifact_export.html").read_text()

VOID_TAGS = {
    "br",
    "hr",
    "img",
    "input",
    "meta",
    "link",
    "col",
    "path",
    "circle",
    "rect",
    "line",
    "use",
    "polyline",
    "polygon",
    "ellipse",
    "stop",
}


def extract_element(html: str, start: int) -> str:
    """Return the balanced element whose opening tag begins at `start`."""
    tag_re = re.compile(r"<(/?)([a-zA-Z][\w:-]*)([^>]*?)(/?)>", re.S)
    depth = 0
    pos = start
    while True:
        m = tag_re.search(html, pos)
        if not m:
            raise ValueError("unbalanced markup")
        closing, name, _attrs, self_closing = m.group(1), m.group(2).lower(), m.group(3), m.group(4)
        pos = m.end()
        if name in VOID_TAGS or self_closing:
            if depth == 0 and m.start() == start:
                return html[start:pos]
            continue
        if closing:
            depth -= 1
            if depth == 0:
                return html[start:pos]
        else:
            depth += 1


def element_at(html: str, marker: str, occurrence: int = 0) -> str:
    idx = -1
    for _ in range(occurrence + 1):
        idx = html.index(marker, idx + 1)
    start = idx if marker.startswith("<") else html.rfind("<", 0, idx)
    return extract_element(html, start)


def strip_figcaption(fragment: str) -> tuple[str, str]:
    """Remove the figure's own caption; the workbench renders captions from the document store."""
    m = re.search(r"\s*<figcaption>(.*?)</figcaption>", fragment, re.S)
    if not m:
        return fragment, ""
    caption = htmlmod.unescape(re.sub(r"<[^>]+>", "", m.group(1)))
    caption = re.sub(r"^Fig\s+\d+:\s*", "", caption).strip()
    return fragment[: m.start()] + fragment[m.end() :], caption


# ---------------------------------------------------------------- main page: CSS, figures, script


def main_css() -> str:
    css = re.search(r"<style>(.*?)</style>", MAIN, re.S).group(1)
    css = re.sub(r"\s*@font-face\{[^}]*\}", "", css)
    css = re.sub(r"\s*:root\{.*?\n\s*\}", "", css, count=1, flags=re.S)
    css = css.replace("policy_anatomy_assets/illustrations/", "assets/")
    # the page column belongs to the workbench shell now
    css = re.sub(r"\n\s*main\{[^}]*\}", "", css)
    css = re.sub(r"\n\s*\.site-top[^{]*\{[^}]*\}", "", css)
    css = re.sub(r"\n\s*\.site-name\{[^}]*\}|\n\s*\.site-nav\{[^}]*\}|\n\s*\.site-foot[^{]*\{[^}]*\}", "", css)
    css = re.sub(r"\n\s*html\{[^}]*\}|\n\s*body\{[^}]*\}", "", css)
    return css


MAIN_CATALOG = re.search(r"var figureCatalogEntries = \[(.*?)\];", MAIN, re.S).group(1)


def main_catalog() -> list[dict]:
    entries = []
    for m in re.finditer(r"\{key: '([^']+)', name: '([^']+)', anchor: '([^']+)'\}", MAIN_CATALOG):
        entries.append({"key": m.group(1), "name": m.group(2), "anchor": m.group(3)})
    assert len(entries) == 16, len(entries)
    return entries


def main_figures() -> list[dict]:
    figures = []
    body = MAIN
    for entry in main_catalog():
        anchor = entry["anchor"]
        marker = f'id="{anchor[1:]}"' if anchor.startswith("#") else f'class="{anchor[1:]}'
        idx = body.index(marker)
        fig_start = body.rfind("<figure", 0, idx)
        fragment = extract_element(body, fig_start)
        assert marker in fragment, entry
        fragment, caption = strip_figcaption(fragment)
        fragment = fragment.replace("policy_anatomy_assets/illustrations/", "assets/")
        figures.append({**entry, "source": "anatomy", "html": fragment, "caption": caption})
    return figures


COLOR_VARS = {
    "ink": "var(--ink)",
    "soft": "var(--ink-soft)",
    "faint": "var(--ink-faint)",
    "ghost": "var(--ink-ghost)",
    "oxblood": "var(--oxblood)",
    "rule": "var(--rule)",
    "dotted": "var(--rule-dotted)",
    "paper": "var(--paper)",
    "aside": "var(--surface-aside)",
    "white": "var(--white)",
}

# figures superseded by hand-written modules in figures_src/ (single choice per diagram, 2026-09-12)
RETIRED = {
    "advantage-weighted-behavior-cloning",
    "objective-decoder-branches",
    "replay-diversity",
    "execution-horizon",
    "batch-sweep",
    "lr-sweep",
    "flagship-model-architecture",
    "stage-map",
    "conditioned-action-plan",
    "ar-axes",
    "svg-action-logit-paths",
    "svg-rle-action-spans",
}

# data providers that an attach module in figures_src adds to an extracted figure
CUSTOM_DATA = {"codecEffects", "pastActionEffect"}

# data behind each anatomy figure, exposed for the data drawer
ANATOMY_DATA = {
    "flagship-model-architecture": ["architectureModules", "architectureEdges"],
    "conditioned-action-plan": [],
    "factorization-ablation": ["factorRows"],
    "controller-representations": ["codecEffects", "codecComparisons"],
    "representation-branches": ["representationBranches"],
    "advantage-weighted-behavior-cloning": ["awrRows"],
    "objective-decoder-branches": ["objectiveBranches"],
    "replay-diversity": [],
    "loader-lineage": ["lineage"],
    "execution-horizon": ["horizonSeries"],
    "capacity-against-delay": ["delayModels"],
    "deployment-frontier": ["frontierRows"],
    "nll-closed-loop-play": ["scalingPoints", "rhoByDelay"],
    "iso-flop-fit": ["FIT", "scalingPoints"],
    "experiment-ledger": ["experiments"],
    "closed-loop-training-path": ["production", "adamwProduction"],
}


def main_script() -> str:
    js = re.search(r"<script>\s*(\(function \(\) \{.*\}\(\)\);)\s*</script>", MAIN, re.S).group(1)
    for name, var in COLOR_VARS.items():
        js = re.sub(rf"(\n\s+{name}: )'#[0-9A-Fa-f]{{6}}'", rf"\1'{var}'", js, count=1)
    # rgba oxblood tints become a token-aware mix so ink mode keeps them visible
    js = js.replace("'rgba(123,46,46,.06)'", "'color-mix(in srgb, var(--oxblood) 7%, transparent)'")
    js = js.replace("'rgba(123,46,46,.05)'", "'color-mix(in srgb, var(--oxblood) 6%, transparent)'")
    names = sorted({n for ns in ANATOMY_DATA.values() for n in ns} - CUSTOM_DATA)
    for n in names:
        assert re.search(rf"\bvar {n}\b", js), f"anatomy data constant missing: {n}"
    exposed = ", ".join(f"{n}: function () {{ return {n}; }}" for n in names)
    hook = (
        "\n    window.HALFIG_ANATOMY = {data: {" + exposed + "},"
        " rerender: function () { renderArchitecture(); renderMtpCandidates(); }};\n"
    )
    tail = js.rfind("}());")
    js = js[:tail] + hook + js[tail:]
    js = split_representation_branches(js)
    js = js.replace("var tip = one('#tip');", "var tip = one('#tip');  // shared tooltip element in the shell")
    return js


# outcomes for the branches that had no matched control or failed outright: no numbers (Eric, 2026-09-13)
BRANCH_OUTCOMES = {
    "geometry": "Stopped without a closed-loop result.",
    "cross": "No matched control; the nearest reference changes the trunk, codec, offsets and decoder size.",
    "flat": "No matched control; the two arms use different context lengths and evaluation protocols.",
    "bpe": "The tested package lost, and it also changed execution cadence, parameter count and crash rate.",
    "chording": "Stopped after three diagnostic steps.",
    "state": "Lost with more parameters, twice the examples and more than twice the training work.",
    "delay": "Lost; plan length, replan interval and the future-data buffer changed with the delay.",
}


def split_representation_branches(js: str) -> str:
    """Past-action encoding worked; it gets its own figure and keeps its delta. The rest keep only an outcome."""
    old = """    function renderRepresentation() {
      var target = one('#representation-list');
      empty(target);
      representationBranches.forEach(function (branch) {"""
    assert old in js, "renderRepresentation changed shape"
    outcomes = json.dumps(BRANCH_OUTCOMES)
    new = (
        """    var BRANCH_OUTCOMES = """
        + outcomes
        + """;
    function renderRepresentation() {
      var lists = [
        [one('#past-action-list'), representationBranches.filter(function (b) { return b.id === 'physical'; })],
        [one('#representation-list'), representationBranches.filter(function (b) { return b.id !== 'physical'; }).map(function (b) {
          return Object.assign({}, b, {stock: null, damage: null, noResult: BRANCH_OUTCOMES[b.id] || b.noResult});
        })]
      ];
      lists.forEach(function (pair) {
      var target = pair[0];
      if (!target) return;
      empty(target);
      pair[1].forEach(function (branch) {"""
    )
    js = js.replace(old, new)
    old_tail = """        renderMasterMap(one('.branch-map', row), branch.focus);
        renderBranchMechanism(one('.branch-mechanism', row), branch.diagram);
      });
    }"""
    assert old_tail in js
    js = js.replace(
        old_tail,
        """        renderMasterMap(one('.branch-map', row), branch.focus);
        renderBranchMechanism(one('.branch-mechanism', row), branch.diagram);
      });
      });
    }""",
    )
    return js


# ---------------------------------------------------------------- explorer: CSS, figures, scripts

EXPLORER_TOKEN_MAP = """
.xfig{
  --bg:var(--paper); --panel:var(--surface-card); --inset:var(--surface-aside); --sunk:var(--paper-warm);
  --ink2:var(--ink-soft); --ink3:var(--ink-mono);
  --rule:var(--rule-underline); --hair:var(--rule);
  --accent:var(--oxblood); --accent-soft:color-mix(in srgb, var(--oxblood) 12%, transparent);
  --indigo:var(--ink); --ochre:var(--oxblood);
  --gain:var(--ink); --loss:var(--oxblood);
  --s1:var(--ink-soft); --s2:var(--ink); --s3:var(--ink); --s4:var(--ink-soft); --s5:var(--ink-faint);
  --shadow:var(--shadow-frame-sm);
  --display:var(--font-serif); --body:var(--font-serif); --mono:var(--font-mono);
  font-family:var(--font-serif); font-size:16px; line-height:1.55; color:var(--ink);
}
.xfig .pill[aria-pressed="true"]{background:var(--ink);border-color:var(--ink);color:var(--paper)}
.xfig .pill.k[aria-pressed="true"]{background:var(--oxblood);border-color:var(--oxblood);color:var(--paper)}
.xfig .lbl{font-family:var(--font-mono)!important;font-weight:600}
.xfig svg .lbl{font-family:var(--font-mono)!important}
"""


def scope_css(css: str, prefix: str) -> str:
    """Prefix every selector with `prefix`, keep @media blocks, drop :root token blocks."""
    out = []
    i = 0
    n = len(css)
    while i < n:
        j = css.find("{", i)
        if j == -1:
            break
        selector = css[i:j].strip()
        if selector.startswith("@media") or selector.startswith("@supports"):
            depth = 1
            k = j + 1
            while depth and k < n:
                if css[k] == "{":
                    depth += 1
                elif css[k] == "}":
                    depth -= 1
                k += 1
            inner = css[j + 1 : k - 1]
            out.append(selector + "{" + scope_css(inner, prefix) + "}")
            i = k
            continue
        if selector.startswith("@"):
            k = css.find("}", j)
            i = k + 1
            continue
        k = css.find("}", j)
        body = css[j + 1 : k]
        i = k + 1
        if not selector:
            continue
        if ":root" in selector:
            continue
        if selector in {"*", "html", "body"} or selector.startswith("body{") or selector.startswith("html"):
            continue
        parts = []
        for sel in selector.split(","):
            sel = sel.strip()
            if not sel:
                continue
            if sel in {"*"}:
                parts.append(f"{prefix} *")
            else:
                parts.append(f"{prefix} {sel}")
        body = re.sub(r"border-radius:[^;}]+", "border-radius:0", body)
        out.append(",".join(parts) + "{" + body + "}")
    return "\n".join(out)


def explorer_css() -> str:
    css = re.search(r"</link>|<style>\n(:root\{.*?)</style>", EXPLORER, re.S)
    css = re.search(r"<style>\n(:root\{.*?)\n</style>", EXPLORER, re.S).group(1)
    return EXPLORER_TOKEN_MAP + scope_css(css, ".xfig")


EXPLORER_FIGURES = [
    {"key": "stage-map", "name": "The five stages, on one time axis", "marker": '<div class="stack">'},
    {"key": "plan-timing", "name": "One plan against the transport queue", "marker": 'id="fig-timing"'},
    {"key": "ar-axes", "name": "The three autoregressive orderings", "marker": 'id="fig-ar"'},
    {"key": "io-paths", "name": "Input path and output path, one shared class set", "marker": 'id="fig-codec"'},
    {
        "key": "evidence-ranking",
        "name": "All 24 comparisons by size of effect",
        "marker": '<div class="toolbar">',
        "extra": 'data-name="evidence-ranking"',
    },
    {"key": "capacity-explorer", "name": "Experiment 039: size × data × control regime", "marker": 'id="cap"'},
    {"key": "horizon-panels", "name": "Experiment 037 across executed frames", "marker": 'data-name="horizon-panels"'},
    {"key": "batch-sweep", "name": "Batch size at a fixed example budget", "marker": 'data-name="batch-sweep"'},
    {"key": "lr-sweep", "name": "Paired Muon and AdamW learning rates", "marker": 'data-name="lr-sweep"'},
]


def explorer_figures() -> list[dict]:
    figures = []
    for spec in EXPLORER_FIGURES:
        fragment = element_at(EXPLORER, spec["marker"])
        if "extra" in spec:
            fragment += "\n" + element_at(EXPLORER, spec["extra"])
        fragment, caption = strip_figcaption(fragment) if spec["key"] not in {"horizon-panels"} else (fragment, "")
        fragment = fragment.replace('id="tip"', 'id="xtip"')
        figures.append(
            {"key": spec["key"], "name": spec["name"], "source": "explorer", "html": fragment, "caption": caption}
        )
    return figures


def explorer_scripts() -> str:
    scripts = re.findall(r"<script>\n(.*?)</script>", EXPLORER, re.S)
    assert len(scripts) == 3, len(scripts)
    js = "\n".join(scripts)
    # the shell owns the theme toggle and figure index
    js = re.sub(r"/\* -+ figure names:.*?\n\}\)\(\);\n", "", js, flags=re.S)
    js = re.sub(r"/\* -+ theme toggle.*?\n\}\)\(\);\n", "", js, flags=re.S)
    js = js.replace('document.getElementById("tip")', 'document.getElementById("xtip")')
    js = js.replace(
        'document.getElementById("theme-toggle").addEventListener("click", () => setTimeout(render, 0));',
        'document.addEventListener("hal:theme", () => setTimeout(render, 0));',
    )
    assert "theme-toggle" not in js, "explorer script still references the removed theme toggle"
    assert 'getElementById("fig-index")' not in js
    # expose the evidence tables for the data drawer
    js = js.replace("const EV = [", "const EV = window.HALFIG_EXPLORER_EV = [", 1)
    assert "HALFIG_EXPLORER_EV" in js
    # the whole explorer becomes one closure; the section jump is guarded because sections are prose blocks now
    js = js.replace(
        'document.getElementById(STAGES[key].sec).scrollIntoView({behavior:"smooth", block:"start"});',
        '{ const sec = document.getElementById(STAGES[key].sec); if (sec) sec.scrollIntoView({behavior:"smooth", block:"start"}); }',
    )
    return "(function () {\n" + js + "\n})();\n"


# ---------------------------------------------------------------- hand-written figures


def custom_figures() -> list[dict]:
    src = HERE / "figures_src"
    entries = json.loads((src / "manifest.json").read_text())
    out = []
    for e in entries:
        js = (src / f"{e['key']}.js").read_text()
        if "attach" in e:  # script-only module that re-renders part of an extracted figure
            out.append({**e, "source": "custom", "js": js})
            continue
        html = (src / f"{e['key']}.html").read_text()
        out.append({**e, "source": "custom", "html": html, "js": js})
    return out


# ---------------------------------------------------------------- shell

# evidence.json studies that back each figure (the data drawer's "studies" tab)
FIG_STUDIES = {
    "flagship-model-architecture": ["training_checkpoint_series", "optimizer_full_scale"],
    "conditioned-action-plan": ["decoder_lineage", "action_factorization_o37"],
    "factorization-ablation": ["action_factorization_o37", "within_frame_order"],
    "controller-representations": ["codec_legacy_cartesian", "codec_cartesian_polar"],
    "representation-branches": [
        "past_action_encoding",
        "hand_features",
        "lean_feature_bundle",
        "temporal_head_cross_attention",
        "flattened_action_tokens",
        "rle_bpe_actions",
        "joint_action_chords",
        "future_state_expert",
        "fixed_long_delay",
    ],
    "advantage-weighted-behavior-cloning": ["awr_actor_weighting", "awr_gamma", "rank_weighted_bc"],
    "objective-decoder-branches": ["endpoint_flow", "continuous_action_flow", "attached_iql", "exploration_modeling"],
    "replay-diversity": ["windows_per_replay"],
    "loader-lineage": ["windows_per_replay", "dataset_v8_cleaning"],
    "execution-horizon": ["execution_horizon"],
    "capacity-against-delay": ["capacity_and_latency"],
    "deployment-frontier": ["capacity_and_latency"],
    "nll-closed-loop-play": ["capacity_and_latency", "training_checkpoint_series"],
    "iso-flop-fit": ["capacity_and_latency"],
    "experiment-ledger": [],
    "closed-loop-training-path": ["training_checkpoint_series", "optimizer_full_scale"],
    "stage-map": [],
    "plan-timing": ["rtc", "execution_horizon"],
    "ar-axes": ["decoder_lineage", "action_factorization_o37"],
    "io-paths": ["past_action_encoding", "onehot_button_pruning"],
    "evidence-ranking": [],
    "capacity-explorer": ["capacity_and_latency"],
    "horizon-panels": ["execution_horizon"],
    "batch-sweep": ["batch_size_fixed_examples"],
    "lr-sweep": ["learning_rate_sweep"],
    "svg-codec-bridges": ["codec_legacy_cartesian", "codec_cartesian_polar"],
    "svg-observation-inputs": ["projectile_inputs", "ice_climbers_inputs"],
    "svg-rle-action-spans": ["rle_bpe_actions"],
    "svg-two-axes-action-feedback": ["action_factorization_o37"],
    "svg-action-logit-paths": [
        "o43_action_head_components",
        "film_vs_concatenation",
        "o26_to_o42_package",
        "old_head_capacity",
    ],
    "scaling-points-table": ["capacity_and_latency"],
    "past-action-encoding": ["past_action_encoding", "onehot_button_pruning"],
}


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
                "studies": f["studies"] if "studies" in f else FIG_STUDIES[f["key"]],
            }
        )
    return json.dumps(entries, ensure_ascii=False, indent=1)


def inline_evidence() -> dict:
    """The studies for the data drawer; the per-boot rows stay in data/evidence.json only."""
    evidence = json.loads((DOCS / "blog_experiment_data.json").read_text())
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
        source = FONTS / f"{face}.woff2"
        if not source.exists():
            source = DESIGN_SYSTEM / "assets" / "fonts" / f"{face}.woff2"
        shutil.copy(source, DIST / "fonts" / f"{face}.woff2")
    shutil.copy(REPO / "docs" / "policy_anatomy_assets" / "illustrations" / "GCController_Layout.svg", DIST / "assets")
    shutil.copy(DOCS / "blog_experiment_data.json", DIST / "data" / "evidence.json")
    for svg in (REPO / "docs" / "blog_figures").glob("*.svg"):
        shutil.copy(svg, DIST / "figures" / svg.name)

    anatomy = main_figures()
    for f in anatomy:
        f["data"] = ANATOMY_DATA[f["key"]]
        if f["key"] == "controller-representations":  # the results table is redrawn as differences by codec-effects.js
            f["html"] = f["html"].replace(
                "Two matched comparisons; do not compare bars across pairs.",
                "Two matched comparisons, each as treatment minus baseline with a 95% interval.",
            )
    explorer = explorer_figures()
    svg_figures = [
        {
            "key": "svg-codec-bridges",
            "name": "Codec bridges (draft SVG)",
            "source": "svg",
            "file": "controller-codec-bridges.svg",
            "caption": "Two matched codec tests, drawn as two small panels.",
        },
        {
            "key": "svg-observation-inputs",
            "name": "Observation inputs (draft SVG)",
            "source": "svg",
            "file": "observation-input-path.svg",
            "caption": "How the player blocks, past action, conditioning and pooled projectile slots become one token.",
        },
        {
            "key": "svg-rle-action-spans",
            "name": "Run-length actions (draft SVG)",
            "source": "svg",
            "file": "rle-action-spans.svg",
            "caption": "Frame actions followed by the variable-length span tokens learned by BPE.",
        },
        {
            "key": "svg-two-axes-action-feedback",
            "name": "Two axes of action feedback (draft SVG)",
            "source": "svg",
            "file": "action-factorization.svg",
            "caption": "The O37 two-by-two: across-frame and within-frame feedback.",
        },
        {
            "key": "svg-action-logit-paths",
            "name": "Action logit paths (draft SVG)",
            "source": "svg",
            "file": "action-logit-paths.svg",
            "caption": "Three paths into the action logits and the one-change deltas.",
        },
    ]
    for f in svg_figures:
        f["html"] = f'<figure class="svg-figure"><img src="figures/{f["file"]}" alt="{f["name"]}"></figure>'

    branches = next(f for f in anatomy if f["key"] == "representation-branches")
    past = (
        branches["html"]
        .replace('id="representation-list"', 'id="past-action-list"')
        .replace("Representation branches", "Past-action encoding")
        .replace(
            "Each row locates the changed component, then compares the two mechanisms.",
            "Where the previous controller frame enters, and the two encodings compared.",
        )
        .replace('aria-label="Representation branches"', 'aria-label="Past-action encoding"')
    )
    anatomy.append(
        {
            "key": "past-action-encoding",
            "name": "Past-action encoding",
            "source": "anatomy",
            "html": past,
            "caption": "The previous controller frame as a 256-way one-hot class against the class embedding plus its physical stick and trigger values. Physical − one-hot: +0.26 [+0.09, +0.43] stocks/min, +23.7 [+13.9, +33.1] damage/min, despite fewer examples and fewer parameters.",
            "data": ["pastActionEffect", "representationBranches"],
        }
    )
    branches["caption"] = (
        "Each row locates the changed component and compares the two mechanisms. None of these produced a matched win; the outcome column says why no delta is reported."
    )
    raw = element_at(MAIN, '<details class="raw">')
    raw = raw.replace('<details class="raw">', '<details class="raw" open>')
    anatomy.append(
        {
            "key": "scaling-points-table",
            "name": "The 33 capacity-sweep points",
            "source": "anatomy",
            "html": '<figure class="breakout"><div class="artifact">' + raw + "</div></figure>",
            "caption": "Every healthy model-budget point behind the NLL and iso-FLOP figures, with its run.",
            "data": ["scalingPoints"],
        }
    )
    figures_hidden_pool: list[dict] = []
    modules = custom_figures()
    custom = [f for f in modules if "attach" not in f]
    figures = [f for f in anatomy + explorer + svg_figures if f["key"] not in RETIRED] + custom
    keys = [f["key"] for f in figures]
    assert len(keys) == len(set(keys)), "duplicate figure keys"

    # retired anatomy/explorer figures stay in the hidden pool so their shared scripts still find their ids; they never render
    for f in anatomy + explorer:
        if f["key"] in RETIRED:
            figures_hidden_pool.append(f)
    pool = []
    for f in figures + figures_hidden_pool:
        cls = "xfig" if f["source"] == "explorer" else f["source"]
        pool.append(f'<div class="fig-src {cls}" data-fig="{f["key"]}">\n{f["html"]}\n</div>')

    (DIST / "figures" / "anatomy.js").write_text(main_script())
    (DIST / "figures" / "explorer.js").write_text(explorer_scripts())
    kit = (HERE / "figures_src" / "diagram-kit.js").read_text()
    ledger = "window.HALFIG_AWR_LEDGER = " + (HERE / "figures_src" / "awr-ledger.json").read_text().strip() + ";\n"
    # every treatment − baseline interval the figures draw; produced by ci_stats.py
    ci = "window.HALFIG_CI = " + json.dumps(json.loads((HERE / "ci.json").read_text())) + ";\n"
    (DIST / "figures" / "custom.js").write_text(kit + "\n" + ledger + ci + "\n".join(f["js"] for f in modules))

    shell = (HERE / "shell.html").read_text()
    html = (
        shell.replace("/*__TOKENS__*/", (HERE / "tokens.css").read_text())
        .replace("/*__MAIN_CSS__*/", main_css())
        .replace("/*__EXPLORER_CSS__*/", explorer_css())
        .replace(
            "/*__ENGINE_CSS__*/",
            (HERE / "engine.css").read_text() + "\n" + (HERE / "figures_src" / "figures.css").read_text(),
        )
        .replace("<!--__FIGURE_POOL__-->", "\n".join(pool))
        .replace("/*__REGISTRY__*/", registry_json(figures))
        .replace(
            "/*__EVIDENCE__*/",
            json.dumps(inline_evidence(), ensure_ascii=False).replace("</", "<\\/"),
        )
        .replace("/*__ENGINE_JS__*/", (HERE / "engine.js").read_text())
    )
    (DIST / "index.html").write_text(html)
    (HERE / "registry.json").write_text(registry_json(figures))
    print(
        f"index.html {len(html) // 1024} KB · anatomy.js {len(main_script()) // 1024} KB · explorer.js {len(explorer_scripts()) // 1024} KB · {len(figures)} figures"
    )


if __name__ == "__main__":
    build()
