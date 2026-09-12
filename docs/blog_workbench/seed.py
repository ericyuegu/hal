"""Build the first document for the workbench: one ordered list of blocks.

Sources (in docs/): policy_anatomy_draft.md (spine + figure order),
blog_experiment_blocks.md (evidence blocks), policy_anatomy_artifact_export.html
(glossary, cut branches, open questions), and registry.json from build.py.

Writes seed/outline.json and seed/blocks/<id>.json for `write_db --db_op batch`,
plus seed/seed.js for the offline preview.

Run:  uv run seed.py
"""

from __future__ import annotations

import json
import re
import shutil
from html.parser import HTMLParser
from pathlib import Path

from build import extract_element
from build import main_catalog

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent
OUT = HERE / "seed"
DRAFT = (DOCS / "policy_anatomy_draft.md").read_text()
BLOCKS = (DOCS / "blog_experiment_blocks.md").read_text().split("\n")
EXPLORER = (DOCS / "policy_anatomy_artifact_export.html").read_text()
REGISTRY = json.loads((HERE / "registry.json").read_text())
CATALOG = [e["key"] for e in main_catalog()]  # the draft numbers FIGURE 1..16 against the page's catalog order
CAPTIONS = {r["key"]: r["caption"] for r in REGISTRY}

doc: list[dict] = []
counter = [0]


def add(block: dict) -> dict:
    counter[0] += 1
    block.setdefault("hidden", False)
    block.setdefault("note", "")
    block["_id"] = f"b{counter[0]:03d}"
    doc.append(block)
    return block


def heading(text: str, level: int = 2, **kw) -> dict:
    return add({"type": "heading", "level": level, "md": text, **kw})


def text(md: str, **kw) -> dict:
    md, sources = split_sources(md)
    if sources:
        kw["note"] = (kw.get("note", "") + "\n" if kw.get("note") else "") + sources
    return add({"type": "text", "md": md.strip(), **kw})


def split_sources(md: str) -> tuple[str, str]:
    """Move a trailing/inline `**Sources:** …` paragraph out of the prose; it belongs in the editorial note."""
    m = re.search(r"(?:^|\n\n|\s)\*\*Sources:\*\*\s*(.*?)(?=\n\n|$)", md, re.S)
    if not m:
        m = re.search(r"\sSources: (.*?)(?=\n\n|$)", md, re.S)  # the plain "Sources: …" form in the bullet blocks
        if not m:
            return md, ""
    sources = "Sources: " + re.sub(r"\s*\n\s*", " ", m.group(1).strip()).rstrip(".")
    md = (md[: m.start()] + md[m.end() :]).rstrip()
    return md, sources


def callout(md: str, **kw) -> dict:
    return add({"type": "callout", "md": md.strip(), **kw})


RETIRED = {"flagship-model-architecture", "conditioned-action-plan"}  # remapped in compose()


def figure(key: str, caption: str | None = None, **kw) -> dict:
    assert key in RETIRED or any(r["key"] == key for r in REGISTRY), key
    return add(
        {
            "type": "figure",
            "fig": key,
            "caption": (caption if caption is not None else CAPTIONS.get(key, "")).strip(),
            **kw,
        }
    )


# ---------------------------------------------------------------- blocks.md named blocks


def blocks_md(marker: str) -> str:
    """Return the dedented block that starts at the line containing `marker`."""
    for i, line in enumerate(BLOCKS):
        if marker in line:
            indent = len(line) - len(line.lstrip(" "))
            out = [line.strip()]
            j = i + 1
            while j < len(BLOCKS):
                nxt = BLOCKS[j]
                if nxt.strip() == "":
                    out.append("")
                    j += 1
                    continue
                lead = len(nxt) - len(nxt.lstrip(" "))
                if lead < indent:
                    break
                if lead == indent and (
                    nxt.lstrip().startswith("- ")
                    or nxt.lstrip().startswith("**Block name")
                    or nxt.lstrip().startswith("#")
                ):
                    break
                if "**Block name:" in nxt or "**Figure name:" in nxt:
                    break
                out.append(nxt[indent:])
                j += 1
            body = "\n".join(out).strip()
            m = re.match(r"^\*\*Block name: (.+?)\*\*\s*(.*)$", body, re.S)
            if m:
                rest = m.group(2)
                sep = "\n\n" if re.match(r"^(\||- |\d+\. )", rest) else " "
                body = f"**{m.group(1)}.**{sep}{rest}"
            body = body.replace("(2^{30})", "2^30")
            # repo-relative links were written for docs/; the workbench keeps them as repo paths
            body = body.replace("](../", "](https://github.com/ericyuegu/hal/blob/main/").replace(
                "](experiments/", "](https://github.com/ericyuegu/hal/blob/main/docs/experiments/"
            )
            return body
    raise KeyError(marker)


def fig_note(marker: str) -> str:
    """The figure-name guidance from blocks.md, as an editorial note."""
    for i, line in enumerate(BLOCKS):
        if marker in line:
            out = [line.strip()]
            j = i + 1
            while j < len(BLOCKS) and BLOCKS[j].strip():
                out.append(BLOCKS[j].strip())
                j += 1
            s = " ".join(out)
            s = re.sub(r"\*\*Figure name: (.+?)\.\*\*", r"Figure name: \1.", s)
            s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
            return s
    raise KeyError(marker)


# ---------------------------------------------------------------- explorer html → markdown


class ToMarkdown(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.href: str | None = None
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = a.get("class", "")
        if tag in ("b", "strong"):
            self.out.append("**")
        elif tag in ("em", "i"):
            self.out.append("*")
        elif tag == "code" or (tag == "span" and "mono" in cls.split()):
            self.out.append("`")
        elif tag == "a":
            self.href = a.get("href")
            self.out.append("[")
        elif tag == "br":
            self.out.append("  \n")
        elif tag == "sub":
            self.out.append("_")
        elif tag in ("p", "div", "li", "dd", "dt", "h3", "h4"):
            if tag != "dd" and self.out and not self.out[-1].endswith("\n"):
                self.out.append("\n\n" if tag in ("p", "div", "h3", "h4") else "\n")
            if tag == "li":
                self.out.append("- ")
            if tag == "h4":
                self.out.append("#### ")
            if tag == "h3":
                self.out.append("### ")
            if tag == "dt":
                self.out.append("**")
        elif tag == "svg":
            self.skip += 1

    def handle_endtag(self, tag):
        if tag == "svg":
            self.skip -= 1
        if tag in ("b", "strong"):
            self.out.append("**")
        elif tag in ("em", "i"):
            self.out.append("*")
        elif tag == "code":
            self.out.append("`")
        elif tag == "span":
            if self.out and self.out[-1] != "`" and "`" in "".join(self.out[-6:]):
                pass
        elif tag == "a":
            self.out.append(f"]({self.href})" if self.href else "]")
            self.href = None
        elif tag == "dt":
            self.out.append("** — ")
        elif tag in ("p", "div", "li", "dd", "h3", "h4"):
            self.out.append("\n")

    def handle_data(self, data):
        if self.skip:
            return
        self.out.append(re.sub(r"\s+", " ", data))


def html_to_md(fragment: str) -> str:
    # spans with the mono class close with a backtick; plain spans do not
    fragment = re.sub(r'<span class="mono[^"]*">(.*?)</span>', r"`\1`", fragment, flags=re.S)
    fragment = re.sub(r'<span class="alias">(.*?)</span>', r" · \1", fragment, flags=re.S)
    fragment = re.sub(r'<span class="vs">', "<br>", fragment)
    fragment = re.sub(r"<span[^>]*>|</span>", "", fragment)
    fragment = re.sub(r"<i class=\"dot\"></i>", "", fragment)
    p = ToMarkdown()
    p.feed(fragment)
    md = "".join(p.out)
    md = md.replace("&nbsp;", " ")
    md = re.sub(r"[ \t]+\n", "\n", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    md = re.sub(r" +", " ", md)
    md = re.sub(r"\*\* — \n", "** — ", md)
    return md.strip()


def explorer_section(sec_id: str) -> str:
    m = re.search(rf'<section id="{sec_id}"[^>]*>(.*?)</section>', EXPLORER, re.S)
    return m.group(1)


def explorer_prose(sec_id: str) -> str:
    """Paragraphs and callouts of a stage section, without figures or tables."""
    body = explorer_section(sec_id)
    body = re.sub(r"<div class=\"sec-head\">.*?</div>", "", body, flags=re.S)
    body = re.sub(
        r"<figure.*?</figure>|<div class=\"table-wrap\">.*?</table></div>|<div class=\"panel[^\"]*\".*?</div>\s*</div>\s*</div>|<div class=\"chart-row\">.*?</div>\s*</div>",
        "",
        body,
        flags=re.S,
    )
    body = re.sub(r"<div class=\"toolbar\">.*?</div>\s*</div>", "", body, flags=re.S)
    body = re.sub(r"<div class=\"stack\">.*", "", body, flags=re.S)
    return html_to_md(body)


def explorer_glossary(heading_text: str) -> str:
    sec = explorer_section("terms")
    start = sec.rfind('<h3 class="gloss-head">', 0, sec.index(heading_text))
    end = sec.find('<h3 class="gloss-head">', start + 10)
    chunk = sec[start : end if end != -1 else None]
    chunk = re.sub(r"<figure.*?</figure>", "", chunk, flags=re.S)
    chunk = re.sub(r"<h3[^>]*>.*?</h3>", "", chunk, flags=re.S)
    return html_to_md(chunk)


def explorer_cards() -> str:
    sec = explorer_section("failed")
    cards = [extract_element(sec, m.start()) for m in re.finditer(r'<div class="card">', sec)]
    rows = ["| Branch | Exp. | Stocks / damage (LCB) | What happened | Runs |", "|---|---|---:|---|---|"]
    for c in cards:
        title = re.search(r"<h4>(.*?)</h4>", c).group(1)
        exp = re.search(r'<span class="exp">(.*?)</span>', c).group(1)
        score = html_to_md(re.search(r'<div class="score[^>]*>(.*?)</div>', c, re.S).group(1)).replace("\n", " ")
        para = html_to_md(re.search(r"<p>(.*?)</p>", c, re.S).group(1))
        runs = " ".join(
            f"[{r}](https://wandb.ai/ericyuegu/hal/runs/{r})" for r in re.findall(r"runs/([a-z0-9]{8})", c)
        )
        rows.append(f"| {title} | {exp} | {score} | {para} | {runs} |")
    intro = html_to_md(re.search(r'<p class="sec-sub">(.*?)</p>', explorer_section("failed"), re.S).group(1))
    tail = html_to_md(re.search(r'<div class="callout">(.*?)</div>', explorer_section("failed"), re.S).group(1))
    return intro + "\n\n" + "\n".join(rows) + "\n\n" + tail


def explorer_open_questions() -> str:
    items = re.findall(
        r'<div class="oq"><span class="n">(\d+)</span><div>(.*?)</div></div>', explorer_section("open"), re.S
    )
    out = []
    for n, body in items:
        title = html_to_md(re.search(r"<h4>(.*?)</h4>", body).group(1)).lstrip("# ")
        paras = re.findall(r'<p(?: class="fix")?>(.*?)</p>', body, re.S)
        why = html_to_md(paras[0])
        fix = html_to_md(paras[1]) if len(paras) > 1 else ""
        out.append(f"{n}. **{title}.** {why}" + (f" *Test:* {fix}" if fix else ""))
    return "\n".join(out)


# ---------------------------------------------------------------- draft.md parsing


def draft_chunks() -> list[tuple[str, str]]:
    """(kind, text) chunks: comment | figure | caption | heading | text."""
    src = DRAFT
    chunks: list[tuple[str, str]] = []
    pos = 0
    for m in re.finditer(r"<!--(.*?)-->", src, re.S):
        before = src[pos : m.start()]
        chunks.extend(_split_paragraphs(before))
        chunks.append(("comment", m.group(1).strip()))
        pos = m.end()
    chunks.extend(_split_paragraphs(src[pos:]))
    return chunks


def _split_paragraphs(part: str) -> list[tuple[str, str]]:
    out = []
    for para in re.split(r"\n\s*\n", part):
        para = para.strip("\n")
        if not para.strip():
            continue
        if para.lstrip().startswith("> **FIGURE"):
            out.append(("figure", para))
        elif re.match(r"^\*Fig \d+:", para.strip()):
            out.append(("caption", para.strip().strip("*")))
        elif para.lstrip().startswith("#"):
            out.append(("heading", para.strip()))
        else:
            out.append(("text", para))
    return out


def build_from_draft() -> dict[str, dict]:
    """Walk the draft; return anchors: figure key -> block, heading text -> block."""
    anchors: dict[str, dict] = {}
    last: dict | None = None
    for kind, body in draft_chunks():
        if kind == "heading":
            hashes, title = re.match(r"^(#+)\s*(.*)$", body).groups()
            last = heading(title, level=len(hashes))
            anchors[title] = last
        elif kind == "figure":
            n = int(re.search(r"FIGURE (\d+)", body).group(1))
            key = CATALOG[n - 1]
            last = figure(key)
            anchors[key] = last
        elif kind == "caption":
            assert last and last["type"] == "figure", body
            last["caption"] = re.sub(r"^Fig \d+:\s*", "", body).replace("\n", " ")
        elif kind == "comment":
            note = body
            if note.startswith("WORKING DRAFT"):
                continue
            if last is not None:
                last["note"] = (last["note"] + "\n" if last["note"] else "") + note
        else:
            joined = body if body.lstrip().startswith("-") else re.sub(r"\s*\n\s*", " ", body)
            last = text(joined)
    return anchors


def insert_after(anchor: dict, blocks: list[dict]) -> None:
    """Move freshly added blocks to sit right after `anchor`, keeping their relative order."""
    for b in blocks:
        doc.remove(b)
    i = doc.index(anchor)
    for offset, b in enumerate(blocks):
        doc.insert(i + 1 + offset, b)


def after(anchor: dict, skip_text: bool = False):
    """Collect blocks added inside the `with` and slot them after the anchor.

    With skip_text, a heading anchor also skips the text blocks that follow it, so
    the new blocks land after the section's own opening prose."""

    class _Ctx:
        def __enter__(self):
            self.start = len(doc)
            return self

        def __exit__(self, *exc):
            added = list(doc[self.start :])
            target = anchor
            if skip_text:
                i = doc.index(anchor) + 1
                while i < len(doc) and doc[i]["type"] in ("text", "callout") and doc[i] not in added:
                    target = doc[i]
                    i += 1
            insert_after(target, added)

    return _Ctx()


# ---------------------------------------------------------------- compose


def compose() -> None:
    a = build_from_draft()

    # the title carries the draft's h1; give the intro an anchor block for the release list
    title = a["Anatomy of a *Melee* policy"]
    title["md"] = "Anatomy of a *Melee* policy"
    # the two front-matter headings are not numbered sections
    a["What I’m releasing"]["level"] = 3
    a["How to read the results"]["level"] = 3

    # Fig 1 and Fig 2 are the single-choice rebuilds (2026-09-12): one causal trunk, one causal decoder, cards for detail
    a["flagship-model-architecture"]["fig"] = "architecture"
    a["flagship-model-architecture"]["caption"] = CAPTIONS["architecture"]
    a["conditioned-action-plan"]["fig"] = "action-dependencies"
    a["conditioned-action-plan"]["caption"] = CAPTIONS["action-dependencies"]

    with after(a["How to read the results"], skip_text=True):
        callout(
            html_to_md(re.search(r'<div class="callout key">(.*?)</div>', EXPLORER, re.S).group(1)),
            note="Explorer's framing callout. Overlaps the bullet list above — merge or park.",
        )
        text(
            blocks_md("Use these terms consistently"),
            note="blocks.md terminology table. The plan-timing figure in §2 draws these terms.",
        )

    # ---- encoding and representation
    with after(a["conditioned-action-plan"]):
        text(blocks_md("Block name: From one action to a causal plan"))
    with after(a["factorization-ablation"]):
        figure("svg-two-axes-action-feedback", hidden=True, note=fig_note("Figure name: Two Axes of Action Feedback"))
        text(blocks_md("The current order is C-stick"), note="blocks.md: within-frame controller ordering.")
    with after(a["controller-representations"]):
        text(blocks_md("Block name: Two controller-codec bridges"), note=fig_note("Figure name: Codec Bridges"))
        figure(
            "svg-codec-bridges",
            hidden=True,
            note="Draft SVG from blog_figures/. Parked: Fig 4 already shows both matched tests.",
        )
        figure("observation-token", note="Replaces io-paths + the observation-inputs SVG (2026-09-12, option A).")
        text(blocks_md("Block name: Remember the physical control"))
        text(blocks_md("Block name: One observation token"))
        text(blocks_md("Block name: Nonlinear action readouts matter"))
        text(blocks_md("Block name: Three paths into the action logits"))
        figure("action-logit-paths", note=fig_note("Figure name: Action Logit Paths"))
        text(blocks_md("- **Next-frame loss share.**"), note="From blocks.md 'other important experiments'.")
        text(blocks_md("- **Ego-player identity.**"), note="From blocks.md 'other important experiments'.")
    with after(a["representation-branches"]):
        text(blocks_md("Block name: Derived geometry never reached a valid test"))
        text(blocks_md("Block name: Compress holds into action spans"))
        figure("rle-action-spans", note=fig_note("Figure name: Run-Length Actions"))
        text(blocks_md("Block name: One token for the full controller"))
        text(blocks_md("Block name: Predicting future state did not improve action choice"))
        text(blocks_md("Block name: Let the output head reread history"))
        text(blocks_md("Block name: Put actions inside the main sequence"))
        text(blocks_md("Block name: Predicting twelve frames ahead"))
        text(blocks_md("**Terminology correction:** O34"), note="blocks.md: rank-weighted BC is not AWR.")

    # ---- optimization
    with after(a["advantage-weighted-behavior-cloning"]):
        text(blocks_md("Block name: Weight actions by estimated advantage"))
        text(
            blocks_md("Block name: Reward must travel far enough backward"),
            note=fig_note("Figure name: AWR Weight Explorer"),
        )
        text(blocks_md("- **All AdamW versus Muon plus AdamW.**"))
        text(blocks_md("- **Batch size at fixed examples.**"))
        figure("batch-sweep")
        text(blocks_md("- **Learning-rate pair.**"))
        figure("lr-sweep")
    with after(a["objective-decoder-branches"]):
        text(blocks_md("Block name: Refine a complete sparse plan from noise"))
        text(blocks_md("Block name: Attach critics to the behavior-cloning policy"))
        text(blocks_md("**Terminology correction:** use **Exploration Modeling"))
        text(
            explorer_cards(),
            hidden=True,
            note="Explorer's 'branches we cut' table (O43-era LCB scores). Parked: overlaps Fig 7 and the blocks above; keep for the IQL-with-separate-critics (030) row, which appears nowhere else.",
        )

    # ---- data & infra
    with after(a["replay-diversity"]):
        text(
            blocks_md("Block name: Spend each batch on different games"),
            note=fig_note("Figure name: Replay Diversity Ring"),
        )
    with after(a["loader-lineage"]):
        text(blocks_md("Block name: Training is a storage-locality problem"))
        text(blocks_md("O52's fixed ring has"), note="Ring constants for the systems figure, not the causal claim.")
        text(blocks_md("Block name: A small data-cleaning cut"))
        text(blocks_md("A small current result can sit beside"), note=fig_note("Figure name: Data Scaling Curves"))

    # ---- optimal size vs latency
    with after(a["Optimal model size vs. latency"], skip_text=True):
        text(
            explorer_glossary("Timing and control").split("\n\n", 1)[0],
            note="Explorer glossary intro for the timing figure.",
        )
        figure(
            "plan-timing",
            "One planning step on the frame timeline. Slippi already holds *d* committed actions; the decoder is given them and writes *H − d* new ones; only the first *R* are entered before the next plan replaces the rest, and inference must finish inside those *R* frames.",
            note="Explorer interactive (blocks.md: 'Plan Timing — reuse').",
        )
        text(
            explorer_glossary("Timing and control").split("\n\n", 1)[1],
            note="Explorer glossary: transport delay, self-forcing, horizon, replan, inference delay. Reconcile symbols with the terminology table in 'How to read the results'.",
        )
    with after(a["execution-horizon"]):
        text(blocks_md("Block name: Action feedback protects longer plans"))
        figure(
            "horizon-panels",
            "Experiment 037 across executed frames, split by future conditioning.",
            hidden=True,
            note="Explorer's two-panel version of Fig 10. Parked as a duplicate.",
        )
    with after(a["deployment-frontier"]):
        figure(
            "capacity-explorer",
            "Experiment 039: size × data × control regime.",
            hidden=True,
            note=fig_note("Figure name: Capacity by Control Timing"),
        )
        text(blocks_md("Block name: Replan while actions are already in flight"))

    # ---- scaling
    with after(a["iso-flop-fit"]):
        figure(
            "scaling-points-table",
            hidden=True,
            note="The raw table from policy_anatomy.html. Parked; the same rows sit in the data drawer of Fig 13 and 14.",
        )
    fig13 = a["nll-closed-loop-play"]
    fig13["note"] += "\n" + fig_note("Figure name: Likelihood Is Not Gameplay")
    a["iso-flop-fit"]["note"] += "\n" + fig_note("Figure name: Iso-Compute Learning Curves")

    # ---- current recipe
    with after(a["experiment-ledger"]):
        figure(
            "evidence-ranking",
            "All 24 comparisons by size of effect, with the lower-bound change as a tick and a badge for how much each comparison isolates.",
            hidden=True,
            note="Explorer's ranked table (O43-era). Parked as a duplicate of Fig 15; its 'strong/near/combined/weak' badges are worth borrowing.",
        )
    with after(a["closed-loop-training-path"]):
        text(blocks_md("- **Checkpoint choice.**"))

    # ---- what is still missing
    heading("What is still missing", level=2)
    text(blocks_md("- **Still missing and worth running.**"))
    text(
        explorer_open_questions(),
        note="Explorer's eight open questions. Overlaps the list above; merge into one list.",
    )

    # ---- parked explorer prose (O43-era stage descriptions)
    for sec, name in [
        ("observation", "Observation encoder"),
        ("trunk", "Transformer trunk"),
        ("decoder", "Temporal action decoder"),
        ("groups", "Within-frame group chain"),
        ("codec", "Controller codec"),
        ("timing", "Closed-loop timing"),
        ("recipe", "Training recipe"),
    ]:
        text(
            f"#### {name} (explorer, O43-era)\n\n" + explorer_prose(sec),
            hidden=True,
            note="Explorer stage prose written against the 15.05M O43 model. Parked; mine for sentences.",
        )
    for head in [
        "Autoregressive conditioning — three separate axes",
        "What goes in, what comes out",
        "Training objectives",
        "Measurement",
    ]:
        text(f"#### Glossary · {head}\n\n" + explorer_glossary(head), hidden=True, note="Explorer glossary. Parked.")


def write() -> None:
    compose()
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "blocks").mkdir(parents=True)
    order = []
    blocks = {}
    for b in doc:
        bid = b.pop("_id")
        b["createdAt"] = b["updatedAt"] = "2026-09-12T00:00:00Z"
        order.append(bid)
        blocks[bid] = b
        (OUT / "blocks" / f"{bid}.json").write_text(json.dumps(b, ensure_ascii=False, indent=1))
    outline = {
        "order": order,
        "title": "Anatomy of a Melee policy",
        "updatedAt": "2026-09-12T00:00:00Z",
        "seededFrom": "docs/blog_workbench/seed.py",
    }
    (OUT / "outline.json").write_text(json.dumps(outline, ensure_ascii=False, indent=1))
    (OUT / "seed.js").write_text(
        "window.WB_SEED = " + json.dumps({"outline": outline, "blocks": blocks}, ensure_ascii=False) + ";\n"
    )
    visible = [b for b in blocks.values() if not b["hidden"]]
    figs = [b for b in visible if b["type"] == "figure"]
    parked = len(blocks) - len(visible)
    print(f"{len(blocks)} blocks · {len(visible)} visible · {len(figs)} figures placed · {parked} parked")
    for b in figs:
        print("  fig", b["fig"])


if __name__ == "__main__":
    write()
