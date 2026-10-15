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
const payloadMatch = fragment.toString().match(/const HEAD_POSITION_PAYLOAD = (\{.*\});$/m);
assert(payloadMatch, "Inline head-position payload is missing");
const payload = JSON.parse(payloadMatch[1]);
const document = execFileSync("python3", [rendererPath, fragmentPath], {
  encoding: "utf8",
  maxBuffer: 4 * 1024 * 1024,
});
await mkdir(outputDirectory, { recursive: true });
const browser = await chromium.launch({
  headless: true,
  args: ["--disable-gpu", "--disable-dev-shm-usage"],
});
const controlLayerZero = [67, 73, 69, 246, 253, 66, 68, 250, 72, 69, 71, 66, 69, 177, 67, 73];
const parentLayerZero = [6, 6, 250, 208, 253, 135, 129, 160, 219, 68, 245, 209, 65, 178, 177, 250];
const results = [];
try {
  for (const width of [736, 360]) {
    for (const colorScheme of ["light", "dark"]) {
      const context = await browser.newContext({
        viewport: { width: width + 32, height: 1500 },
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
      const root = frame.locator("#hal-head-token-positions");
      await root.waitFor();
      await page.waitForTimeout(200);
      assert.deepEqual(errors, [], "Browser JavaScript errors before chart render");
      const status = root.locator("#hal-head-token-status");
      assert.equal(
        await status.isVisible(),
        false,
        `Chart validation error: ${await status.innerText()}`,
      );
      await root.locator("svg").first().waitFor();

      assert.equal(await root.locator("svg").count(), 2);
      assert.equal(
        await root.locator("#hal-head-token-title").innerText(),
        "Maximum-weight token for each head",
      );
      assert.equal(await root.locator("rect[data-chart-frame]").count(), 2);
      assert.equal(await root.locator("text.axis-title[data-axis=x]").count(), 2);
      assert.equal(await root.locator("text.axis-title[data-axis=y]").count(), 2);
      assert.equal(await root.locator('[data-axis-tick="x"]').count(), 8);
      assert.equal(await root.locator('[data-axis-tick="y"]').count(), 32);
      assert.equal(await root.locator(".head-token-dot").count(), 32);
      assert.equal(await root.locator('[data-chart-hover-overlay="nearest-dot"]').count(), 2);
      assert.equal(await status.isVisible(), false);
      assert.equal(
        await root.locator("#hal-head-token-panel-title-0").innerText(),
        "Half-rate Muon continuation",
      );
      assert.equal(await root.locator("#hal-head-token-panel-meta-0").innerText(), "7dgv85x8 · 57,344 steps");
      assert.equal(
        await root.locator("#hal-head-token-panel-title-1").innerText(),
        "Original Muon",
      );
      assert.equal(await root.locator("#hal-head-token-panel-meta-1").innerText(), "p1fyyp1z · 57,344 steps");
      assert.equal(
        await root.locator("#hal-head-token-summary-0").innerText(),
        "16 heads · 11 distinct maximum-weight token positions",
      );
      assert.equal(
        await root.locator("#hal-head-token-summary-1").innerText(),
        "16 heads · 14 distinct maximum-weight token positions",
      );

      const readPositions = async panel => root
        .locator(`#hal-head-token-chart-${panel} .head-token-dot`)
        .evaluateAll(nodes => nodes.map(node => Number(node.dataset.position)));
      assert.deepEqual(await readPositions(0), controlLayerZero);
      assert.deepEqual(await readPositions(1), parentLayerZero);

      const geometry = await root.evaluate(element => {
        const rootBounds = element.getBoundingClientRect();
        const outsideLabels = [];
        const frameLeftMargins = [];
        const frameTops = [];
        const viewBoxWidths = [];
        for (const svg of element.querySelectorAll("svg")) {
          const bounds = svg.getBoundingClientRect();
          const viewBox = svg.viewBox.baseVal;
          viewBoxWidths.push(viewBox.width);
          const frame = svg.querySelector("rect[data-chart-frame]");
          frameLeftMargins.push(Number(frame.getAttribute("x")));
          frameTops.push(frame.getBoundingClientRect().top);
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
          frameLeftMargins,
          frameTops,
          viewBoxWidths,
          fontSize: getComputedStyle(element.querySelector("svg text.axis-title")).fontSize,
          foreground: getComputedStyle(element).color,
        };
      });
      assert(Math.abs(geometry.width - width) <= 1, "Unexpected inline figure width");
      assert(geometry.overflow <= 1, `Horizontal overflow: ${geometry.overflow}`);
      assert.deepEqual(geometry.outsideLabels, [], "SVG label clipping");
      assert.deepEqual(geometry.frameLeftMargins, [64, 64], "Insufficient y-axis margin");
      if (width === 736) {
        assert(Math.abs(geometry.frameTops[0] - geometry.frameTops[1]) <= 1, "Plot frames are not aligned");
      }
      assert(geometry.viewBoxWidths.every(value => value > 0), "Invalid responsive viewBox");
      assert(parseFloat(geometry.fontSize) >= 11, "Unreadable chart labels");

      const layer = root.locator("#hal-head-token-layer");
      const before = await readPositions(0);
      await layer.selectOption("5");
      assert.match(await root.locator("#hal-head-token-selection").innerText(), /Layer 5.*16 heads/);
      assert.notDeepEqual(await readPositions(0), before, "Layer selection did not update positions");
      assert.match(await root.locator("#hal-head-token-summary-0").innerText(), /^16 heads · 1[1-6] distinct/);
      for (let layerIndex = 0; layerIndex < 16; layerIndex += 1) {
        await layer.selectOption(String(layerIndex));
        for (let panel = 0; panel < 2; panel += 1) {
          const expectedLayer = payload.models[panel].layers[layerIndex];
          assert.deepEqual(await readPositions(panel), expectedLayer.key_positions);
          assert.equal(
            await root.locator(`#hal-head-token-summary-${panel}`).innerText(),
            `16 heads · ${expectedLayer.distinct_positions} distinct maximum-weight token positions`,
          );
        }
      }
      await layer.selectOption("0");

      const dot = root.locator('#hal-head-token-chart-0 .head-token-dot[data-head="2"]');
      const overlay = root.locator("#hal-head-token-chart-0 [data-chart-hit]");
      const dotBounds = await dot.boundingBox();
      const overlayBounds = await overlay.boundingBox();
      assert(dotBounds && overlayBounds);
      const hit = {
        x: dotBounds.x + dotBounds.width / 2 - overlayBounds.x,
        y: dotBounds.y + dotBounds.height / 2 - overlayBounds.y,
      };
      if (width === 360) await overlay.tap({ position: hit });
      else await overlay.hover({ position: hit });
      const tooltip = root.locator("#hal-head-token-tooltip");
      assert(await tooltip.isVisible(), "Nearest-dot tooltip did not open");
      const tooltipText = await tooltip.innerText();
      assert.match(tooltipText, /Half-rate Muon continuation/);
      assert.match(tooltipText, /Layer \/ head\s+0 \/ 2/);
      assert.match(tooltipText, /Key token position\s+69/);
      assert.match(tooltipText, /Age\s+186 frames/);
      assert.match(tooltipText, /Peak weight\s+[\d.]+%/);
      assert.match(tooltipText, /Other heads here\s+9, 12/);

      await root.screenshot({ path: join(outputDirectory, `${colorScheme}-${width}.png`) });
      assert.deepEqual(errors, [], "Browser JavaScript errors");
      results.push({ width, colorScheme, geometry, tooltipText, errors });
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
