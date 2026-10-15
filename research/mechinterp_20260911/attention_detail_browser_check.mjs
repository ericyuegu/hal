import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { join } from "node:path";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");
const [fragmentPath, rendererPath, outputDirectory] = process.argv.slice(2);
assert(fragmentPath && rendererPath && outputDirectory, "Pass fragment, renderer, and output directory");
const fragment = await readFile(fragmentPath);
assert(fragment.length < 1_000_000, "Inline fragment exceeds 1 MB");
const document = execFileSync("python3", [rendererPath, fragmentPath], {
  encoding: "utf8",
  maxBuffer: 8 * 1024 * 1024,
});
await mkdir(outputDirectory, { recursive: true });
const browser = await chromium.launch({
  headless: true,
  args: ["--disable-gpu", "--disable-dev-shm-usage"],
});
const results = [];
try {
  for (const width of [736, 360]) {
    for (const colorScheme of ["light", "dark"]) {
      const context = await browser.newContext({
        viewport: { width: width + 32, height: 1900 },
        colorScheme,
        hasTouch: width === 360,
        reducedMotion: "reduce",
      });
      const page = await context.newPage();
      const errors = [];
      page.on("pageerror", error => errors.push(error.message));
      await page.setContent(document, { waitUntil: "networkidle" });
      const frame = page.frames().find(candidate => candidate !== page.mainFrame());
      assert(frame, "Standalone renderer did not create its frame");
      const root = frame.locator("#hal-attention-detail");
      await root.locator("svg").first().waitFor();
      assert.equal(await root.locator("svg").count(), 4);
      assert.equal(await root.locator("rect[data-chart-frame]").count(), 4);
      assert.equal(await root.locator("text.axis-title[data-axis=x]").count(), 4);
      assert.equal(await root.locator("text.axis-title[data-axis=y]").count(), 4);
      assert.equal(await root.locator(".heat-cell").count(), 512);
      assert.equal(await root.locator(".attention-stem").count(), 512);
      assert.equal(await root.locator("#hal-attention-status").isVisible(), false);

      const geometry = await root.evaluate(element => {
        const rootBounds = element.getBoundingClientRect();
        const outsideLabels = [];
        for (const svg of element.querySelectorAll("svg")) {
          const bounds = svg.getBoundingClientRect();
          for (const text of svg.querySelectorAll("text")) {
            const label = text.getBoundingClientRect();
            if (label.left < bounds.left - 1 || label.right > bounds.right + 1
                || label.top < bounds.top - 1 || label.bottom > bounds.bottom + 1) {
              outsideLabels.push(text.textContent);
            }
          }
        }
        return {
          width: rootBounds.width,
          overflow: element.scrollWidth - element.clientWidth,
          outsideLabels,
          fontSize: getComputedStyle(element.querySelector("svg text.axis-title")).fontSize,
          foreground: getComputedStyle(element).color,
        };
      });
      assert(Math.abs(geometry.width - width) <= 1, "Unexpected inline figure width");
      assert(geometry.overflow <= 1, `Horizontal overflow: ${geometry.overflow}`);
      assert.deepEqual(geometry.outsideLabels, [], "SVG label clipping");
      assert(parseFloat(geometry.fontSize) >= 11, "Unreadable chart labels");

      const layer = root.locator("#hal-layer-select");
      const head = root.locator("#hal-head-select");
      const before = await root.locator("#hal-detail-0 svg").innerHTML();
      await layer.selectOption("7");
      await head.selectOption("9");
      const after = await root.locator("#hal-detail-0 svg").innerHTML();
      assert.notEqual(after, before, "Selection did not change the detailed attention plot");
      assert.match(await root.locator("#hal-attention-selection").innerText(), /7.*9/);

      const heatHit = root.locator("#hal-heat-0 [data-chart-hit]");
      const heatBounds = await heatHit.boundingBox();
      assert(heatBounds);
      const location = { x: heatBounds.width * 3.5 / 16, y: heatBounds.height * 5.5 / 16 };
      if (width === 360) await heatHit.tap({ position: location });
      else await heatHit.click({ position: location });
      assert.equal(await layer.inputValue(), "5");
      assert.equal(await head.inputValue(), "3");
      const tooltip = root.locator("#hal-attention-tooltip");
      assert(await tooltip.isVisible(), "Heatmap tooltip did not open");
      const heatTooltip = await tooltip.innerText();
      assert.match(heatTooltip, /Muon/);
      assert.match(heatTooltip, /nats/);
      assert.match(heatTooltip, /%/);

      await layer.selectOption("0");
      await head.selectOption("0");
      const detailHit = root.locator("#hal-detail-0 [data-chart-hit]");
      const detailBounds = await detailHit.boundingBox();
      assert(detailBounds);
      const detailPosition = { x: detailBounds.width / 2, y: detailBounds.height / 2 };
      if (width === 360) await detailHit.tap({ position: detailPosition });
      else await detailHit.hover({ position: detailPosition });
      assert(await tooltip.isVisible(), "Frame-weight tooltip did not open");
      const weightTooltip = await tooltip.innerText();
      assert.match(weightTooltip, /frames/);
      assert.match(weightTooltip, /%/);

      await root.screenshot({ path: join(outputDirectory, `${colorScheme}-${width}.png`) });
      assert.deepEqual(errors, [], "Browser JavaScript errors");
      results.push({ width, colorScheme, geometry, heatTooltip, weightTooltip, errors });
      await context.close();
    }
  }
  const result = {
    fragment_sha256: createHash("sha256").update(fragment).digest("hex"),
    fragment_bytes: fragment.length,
    browser_version: browser.version(),
    playwright_version: require("playwright/package.json").version,
    results,
  };
  await writeFile(join(outputDirectory, "browser-check.json"), `${JSON.stringify(result, null, 2)}\n`);
  console.log(JSON.stringify(result, null, 2));
} finally {
  await browser.close();
}
