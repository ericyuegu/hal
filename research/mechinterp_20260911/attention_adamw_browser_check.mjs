import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");
const [fragmentPath, rendererPath, outputDirectory] = process.argv.slice(2);
assert(fragmentPath && rendererPath && outputDirectory, "Pass fragment, renderer, and output directory");
const fragment = await readFile(fragmentPath);
assert(fragment.length < 1_000_000, "Inline fragment exceeds 1 MB");
const measurements = JSON.parse(await readFile(join(dirname(fragmentPath), "attention_detail.json"), "utf8"));
const checkpoints = measurements.checkpoints.filter(record => record.run_id === "tgpuo1be");
assert.deepEqual(checkpoints.map(record => record.boundary_step), [57344, 131072]);
const document = execFileSync("python3", [rendererPath, fragmentPath], {
  encoding: "utf8", maxBuffer: 8 * 1024 * 1024,
});
await mkdir(outputDirectory, { recursive: true });
const browser = await chromium.launch({ headless: true, args: ["--disable-gpu", "--disable-dev-shm-usage"] });
const results = [];
try {
  for (const width of [736, 360]) {
    for (const colorScheme of ["light", "dark"]) {
      const context = await browser.newContext({
        viewport: { width: width + 32, height: 2000 }, colorScheme,
        hasTouch: width === 360, reducedMotion: "reduce",
      });
      const page = await context.newPage();
      const errors = [];
      page.on("pageerror", error => errors.push(error.message));
      await page.setContent(document, { waitUntil: "networkidle" });
      const frame = page.frames().find(candidate => candidate !== page.mainFrame());
      assert(frame);
      const root = frame.locator("#hal-adamw-attention-detail");
      await root.locator("svg").first().waitFor();
      for (const selector of ["svg", "rect[data-chart-frame]", "text.axis-title[data-axis=x]", "text.axis-title[data-axis=y]"]) {
        assert.equal(await root.locator(selector).count(), 4, selector);
      }
      assert.equal(await root.locator(".heat-cell").count(), 512);
      assert.equal(await root.locator(".attention-stem").count(), 512);
      assert.equal(await root.locator("#hal-adamw-attention-status").isVisible(), false);
      assert.match(await root.innerText(), /tgpuo1be.*57,344/);
      assert.match(await root.innerText(), /tgpuo1be.*131,072/);
      const geometry = await root.evaluate(element => {
        const outsideLabels = [];
        for (const svg of element.querySelectorAll("svg")) {
          const bounds = svg.getBoundingClientRect();
          for (const text of svg.querySelectorAll("text")) {
            const label = text.getBoundingClientRect();
            if (label.left < bounds.left - 1 || label.right > bounds.right + 1
                || label.top < bounds.top - 1 || label.bottom > bounds.bottom + 1) outsideLabels.push(text.textContent);
          }
        }
        return {
          width: element.getBoundingClientRect().width,
          overflow: element.scrollWidth - element.clientWidth,
          outsideLabels,
          fontSize: getComputedStyle(element.querySelector("svg text.axis-title")).fontSize,
        };
      });
      assert(Math.abs(geometry.width - width) <= 1);
      assert(geometry.overflow <= 1);
      assert.deepEqual(geometry.outsideLabels, []);
      assert(parseFloat(geometry.fontSize) >= 11);
      const layer = root.locator("#hal-layer-select-adamw");
      const head = root.locator("#hal-head-select-adamw");
      const before = await root.locator("#hal-adamw-detail-0 svg").innerHTML();
      await layer.selectOption("7");
      await head.selectOption("9");
      assert.notEqual(await root.locator("#hal-adamw-detail-0 svg").innerHTML(), before);
      assert.match(await root.locator("#hal-adamw-attention-selection").innerText(), /7.*9/);
      const heatHit = root.locator("#hal-adamw-heat-0 [data-chart-hit]");
      const bounds = await heatHit.boundingBox();
      assert(bounds);
      const position = { x: bounds.width * 3.5 / 16, y: bounds.height * 5.5 / 16 };
      if (width === 360) await heatHit.tap({ position });
      else await heatHit.click({ position });
      assert.equal(await layer.inputValue(), "5");
      assert.equal(await head.inputValue(), "3");
      const tooltip = root.locator("#hal-adamw-attention-tooltip");
      assert(await tooltip.isVisible());
      const heatTooltip = await tooltip.innerText();
      assert.match(heatTooltip, /AdamW.*tgpuo1be.*57,344/);
      assert.match(heatTooltip, /nats/);
      const expectedPeak = (100 * checkpoints[0].summary.per_head.mean_row_max_weight[5][3]).toFixed(2);
      assert(heatTooltip.includes(`${expectedPeak}%`), "Hover peak differs from measured head mean");
      await layer.selectOption("0");
      await head.selectOption("0");
      const detailHit = root.locator("#hal-adamw-detail-0 [data-chart-hit]");
      const detailBounds = await detailHit.boundingBox();
      assert(detailBounds);
      const detailPosition = { x: detailBounds.width / 2, y: detailBounds.height / 2 };
      if (width === 360) await detailHit.tap({ position: detailPosition });
      else await detailHit.hover({ position: detailPosition });
      assert(await tooltip.isVisible());
      const weightTooltip = await tooltip.innerText();
      assert.match(weightTooltip, /frames/);
      assert.match(weightTooltip, /%/);
      await root.screenshot({ path: join(outputDirectory, `${colorScheme}-${width}.png`) });
      assert.deepEqual(errors, []);
      results.push({ width, colorScheme, geometry, heatTooltip, weightTooltip, errors });
      await context.close();
    }
  }
  const result = {
    fragment_sha256: createHash("sha256").update(fragment).digest("hex"),
    fragment_bytes: fragment.length, browser_version: browser.version(),
    playwright_version: require("playwright/package.json").version, results,
  };
  await writeFile(join(outputDirectory, "browser-check.json"), `${JSON.stringify(result, null, 2)}\n`);
  console.log(JSON.stringify(result, null, 2));
} finally {
  await browser.close();
}
