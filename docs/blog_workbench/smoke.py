"""Load dist/ headlessly with a document-store dump as the store; report JS errors and per-figure render counts.

    uv run --no-project smoke.py dbdump [baseline.json]

Serves a copy of dist/ with the dump injected as window.WB_SEED, opens it in Playwright's
Chromium, exercises theme toggle + resize + data drawer, then prints every page error and
console.error, and a {figure key: rendered element count} map for every figure in the
outline. With a baseline JSON the map is diffed and a mismatch exits non-zero.

Needs `uvx --from playwright` browsers (chromium) already installed.
"""

from __future__ import annotations

import http.server
import json
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIST = HERE / "dist"

SCRIPT = r"""
import asyncio, json, sys
from playwright.async_api import async_playwright

async def main(url, out):
    errors = []
    async with async_playwright() as p:
        b = await p.chromium.launch()
        page = await b.new_page(viewport={"width": 1400, "height": 900})
        page.on("pageerror", lambda e: errors.append("pageerror: " + str(e)))
        page.on("console", lambda m: errors.append("console." + m.type + ": " + m.text) if m.type == "error" else None)
        await page.goto(url, wait_until="networkidle")
        await page.wait_for_timeout(1500)
        # exercise the re-render hooks
        await page.evaluate("document.getElementById('wb-theme').click()")
        await page.wait_for_timeout(400)
        await page.set_viewport_size({"width": 700, "height": 900})
        await page.wait_for_timeout(400)
        await page.set_viewport_size({"width": 1400, "height": 900})
        await page.wait_for_timeout(400)
        await page.evaluate("document.getElementById('wb-data').click()")
        await page.wait_for_timeout(400)
        counts = await page.evaluate('''() => {
          const out = {};
          document.querySelectorAll('.wb-figure[data-fig], [data-fig]').forEach(el => {
            const k = el.getAttribute('data-fig');
            if (!k || k === 'orphans') return;
            const n = el.querySelectorAll('*').length;
            out[k] = Math.max(out[k] || 0, n);
          });
          return out;
        }''')
        pool = await page.evaluate("Array.from(document.querySelectorAll('.fig-src')).map(e => e.getAttribute('data-fig'))")
        await b.close()
    json.dump({"errors": errors, "counts": counts, "pool": pool}, open(out, "w"), indent=1)

asyncio.run(main(sys.argv[1], sys.argv[2]))
"""


def main() -> None:
    dump = Path(sys.argv[1])
    baseline = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    outline = json.loads((dump / "doc" / "outline.json").read_text())
    outline = outline.get("data", outline)
    blocks = {p.stem: json.loads(p.read_text()) for p in (dump / "blocks").glob("*.json")}
    blocks = {k: v.get("data", v) for k, v in blocks.items()}
    seed = "window.WB_SEED = " + json.dumps({"outline": outline, "blocks": blocks}, ensure_ascii=False) + ";"

    with tempfile.TemporaryDirectory() as tmp:
        site = Path(tmp) / "site"
        shutil.copytree(DIST, site)
        html = (site / "index.html").read_text()
        assert '<script src="figures/anatomy.js"></script>' in html
        html = html.replace(
            '<script src="figures/anatomy.js"></script>',
            "<script>" + seed + "</script>\n" + '<script src="figures/anatomy.js"></script>',
            1,
        )
        (site / "index.html").write_text(html)

        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *a, **k):
                super().__init__(*a, directory=str(site), **k)

            def log_message(self, *a):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{server.server_port}/index.html"
        out = Path(tmp) / "result.json"
        runner = Path(tmp) / "run.py"
        runner.write_text(SCRIPT)
        subprocess.run(["uvx", "--from", "playwright", "python", str(runner), url, str(out)], check=True)
        server.shutdown()
        result = json.loads(out.read_text())

    for e in result["errors"]:
        print("ERROR", e)
    counts = result["counts"]
    visible = [
        blocks[i]["fig"]
        for i in outline["order"]
        if blocks.get(i, {}).get("type") == "figure" and not blocks[i].get("hidden")
    ]
    print(f"{len(result['errors'])} errors · {len(result['pool'])} pooled figures · {len(counts)} rendered")
    for k in visible:
        print(f"  {k:36} {counts.get(k, 0):6}")
    if baseline and baseline.exists():
        base = json.loads(baseline.read_text())
        diff = {
            k: (base["counts"].get(k), counts.get(k))
            for k in set(base["counts"]) | set(counts)
            if base["counts"].get(k) != counts.get(k)
        }
        print("diff vs baseline:", diff or "none")
        sys.exit(1 if diff or result["errors"] else 0)
    if baseline:
        baseline.write_text(json.dumps(result, indent=1))
        print("baseline written", baseline)
    sys.exit(1 if result["errors"] else 0)


if __name__ == "__main__":
    main()
