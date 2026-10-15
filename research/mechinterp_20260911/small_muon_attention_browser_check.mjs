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
const match = fragment.toString().match(/const SMALL_MUON_PAYLOAD = (\{.*\});$/m);
assert(match, "Inline Small-Muon payload is missing");
const payload = JSON.parse(match[1]);
const expectedSource = {
  attention_json_sha256: "5a6768a41f8799d2ff283e6f863fc8b16840126716ac71c27a648a8bff2fbd65",
  npz_sha256: "c384dbb42481665efc9bd209a2966f68b40daee5b2aed1f0096f468d42075741",
};
assert.deepEqual(payload.source, expectedSource);
assert.deepEqual(payload.models.map(model => model.run_id), ["a117chkw", "5wfk2esf", "cqbbbg77", "1imfy8v3"]);

function decode(model) {
  const bytes = Buffer.from(model.weights_base64, "base64");
  assert.equal(bytes.length, model.layers * model.heads * model.keys * 4);
  assert.equal(createHash("sha256").update(bytes).digest("hex"), model.weights_float32_sha256);
  const values = new Float32Array(model.layers * model.heads * model.keys);
  for (let index = 0; index < values.length; index += 1) values[index] = bytes.readFloatLE(index * 4);
  return values;
}

const weights = payload.models.map(decode);
function row(modelIndex, layer, head) {
  const model = payload.models[modelIndex];
  const start = (layer * model.heads + head) * model.keys;
  return weights[modelIndex].subarray(start, start + model.keys);
}
function maximum(values) {
  let position = 0;
  for (let key = 1; key < values.length; key += 1) {
    if (values[key] > values[position]) position = key;
  }
  return { position, weight: values[position] };
}

const document = execFileSync("python3", [rendererPath, fragmentPath], {
  encoding: "utf8",
  maxBuffer: 4 * 1024 * 1024,
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
        viewport: { width: width + 32, height: 2200 },
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
      const root = frame.locator("#hal-small-muon-attention");
      await root.waitFor();
      await root.locator("svg").first().waitFor();
      await page.waitForTimeout(200);
      assert.deepEqual(errors, [], "Browser JavaScript errors before chart checks");
      const status = root.locator("#hal-small-muon-status");
      assert.equal(await status.isVisible(), false, `Chart validation error: ${await status.innerText()}`);
      assert.equal(await root.locator("svg").count(), 5);
      assert.equal(await root.locator("rect[data-chart-frame]").count(), 4);
      assert.equal(await root.locator("text.axis-title[data-axis=x]").count(), 4);
      assert.equal(await root.locator("text.axis-title[data-axis=y]").count(), 4);
      assert.equal(await root.locator('[data-axis-tick="x"]').count(), 16);
      assert.equal(await root.locator('[data-axis-tick="y"]').count(), 20);
      assert.equal(await root.locator(".attention-cell").count(), 2560);
      assert.equal(await root.locator(".row-max-marker").count(), 20);
      assert.equal(await root.locator('[data-chart-hover-overlay="nearest-cell"]').count(), 4);
      assert.equal(await root.locator("[data-legend-tick]").count(), 5);
      assert.deepEqual(
        await root.locator("[data-legend-tick]").evaluateAll(nodes => nodes.map(node => node.textContent)),
        ["0%", "1%", "10%", "50%", "100%"],
      );
      assert.equal(await root.locator("#hal-small-muon-title").innerText(), "Where small Muon heads attend");
      assert.equal(
        await root.locator("#hal-small-muon-selection").innerText(),
        "Layer 0 · all heads",
      );

      const layerSelect = root.locator("#hal-small-muon-layer");
      for (let layer = 0; layer < 8; layer += 1) {
        await layerSelect.selectOption(String(layer));
        assert.equal(await root.locator("#hal-small-muon-selection").innerText(), `Layer ${layer} · all heads`);
        for (let modelIndex = 0; modelIndex < payload.models.length; modelIndex += 1) {
          const model = payload.models[modelIndex];
          const expected = [];
          const expectedMaxima = [];
          for (let head = 0; head < model.heads; head += 1) {
            const values = row(modelIndex, layer, head);
            expected.push(...values);
            expectedMaxima.push(maximum(values));
          }
          const actual = await root
            .locator(`#hal-small-muon-chart-${modelIndex} .attention-cell`)
            .evaluateAll(nodes => nodes.map(node => ({
              head: Number(node.dataset.head),
              key: Number(node.dataset.key),
              weight: Number(node.dataset.weight),
            })));
          assert.equal(actual.length, model.heads * model.keys);
          actual.forEach((cell, index) => {
            assert.equal(cell.head, Math.floor(index / model.keys));
            assert.equal(cell.key, index % model.keys);
            assert.equal(cell.weight, expected[index]);
          });
          const actualMaxima = await root
            .locator(`#hal-small-muon-chart-${modelIndex} .row-max-marker`)
            .evaluateAll(nodes => nodes.map(node => ({
              head: Number(node.dataset.head),
              key: Number(node.dataset.key),
            })));
          assert.deepEqual(actualMaxima, expectedMaxima.map((entry, head) => ({ head, key: entry.position })));
          const distinct = new Set(expectedMaxima.map(entry => entry.position)).size;
          assert.equal(
            await root.locator(`#hal-small-muon-summary-${modelIndex}`).innerText(),
            `${model.heads} heads · ${distinct} distinct maximum-weight token positions`,
          );
        }
      }
      await layerSelect.selectOption("0");

      const geometry = await root.evaluate(element => {
        const rootBounds = element.getBoundingClientRect();
        const panelSvgs = [...element.querySelectorAll(".small-muon-chart svg")];
        const outsideLabels = [];
        const frameLeftMargins = [];
        const frameTops = [];
        for (const svg of panelSvgs) {
          const bounds = svg.getBoundingClientRect();
          const frame = svg.querySelector("rect[data-chart-frame]");
          frameLeftMargins.push(Number(frame.getAttribute("x")));
          frameTops.push(frame.getBoundingClientRect().top);
          for (const label of svg.querySelectorAll("text")) {
            const textBounds = label.getBoundingClientRect();
            if (textBounds.left < bounds.left - 1 || textBounds.right > bounds.right + 1
                || textBounds.top < bounds.top - 1 || textBounds.bottom > bounds.bottom + 1) {
              outsideLabels.push(label.textContent);
            }
          }
        }
        return {
          width: rootBounds.width,
          overflow: element.scrollWidth - element.clientWidth,
          outsideLabels,
          frameLeftMargins,
          frameTops,
          labelLayouts: panelSvgs.map(svg => svg.dataset.labelLayout),
          fontSize: getComputedStyle(panelSvgs[0].querySelector("text.axis-title")).fontSize,
        };
      });
      assert(Math.abs(geometry.width - width) <= 1, "Unexpected inline figure width");
      assert(geometry.overflow <= 1, `Horizontal overflow: ${geometry.overflow}`);
      assert.deepEqual(geometry.outsideLabels, [], "SVG label clipping");
      assert.deepEqual(geometry.frameLeftMargins, [64, 64, 64, 64], "Insufficient y-axis margin");
      assert(geometry.labelLayouts.every(value => value === "checked"), "Chart reported label clipping");
      assert(parseFloat(geometry.fontSize) >= 11, "Unreadable chart labels");
      if (width === 736) {
        assert(Math.abs(geometry.frameTops[0] - geometry.frameTops[1]) <= 1, "First-row frames are misaligned");
        assert(Math.abs(geometry.frameTops[2] - geometry.frameTops[3]) <= 1, "Second-row frames are misaligned");
      } else {
        assert(geometry.frameTops.every((top, index, values) => index === 0 || top > values[index - 1]));
      }

      const model = payload.models[0];
      const overlay = root.locator("#hal-small-muon-chart-0 [data-chart-hit]");
      const overlayBounds = await overlay.boundingBox();
      assert(overlayBounds);
      const key = 123;
      const head = 0;
      const position = {
        x: ((key + 0.5) / model.keys) * overlayBounds.width,
        y: ((head + 0.5) / model.heads) * overlayBounds.height,
      };
      if (width === 360) await overlay.tap({ position });
      else await overlay.hover({ position });
      const tooltip = root.locator("#hal-small-muon-tooltip");
      assert(await tooltip.isVisible(), "Nearest-cell tooltip did not open");
      const tooltipText = await tooltip.innerText();
      assert.match(tooltipText, /7\.1M · O37 D1 · a117chkw/);
      assert.match(tooltipText, /Layer \/ head\s+0 \/ 0/);
      assert.match(tooltipText, /Key token position\s+123/);
      assert.match(tooltipText, /Age\s+4 frames/);
      assert.match(tooltipText, /Attention weight\s+21\.0126%/);
      assert.match(tooltipText, /Row maximum\s+123 · 21\.0126%/);
      assert.match(tooltipText, /Other head maxima here\s+1/);
      const tooltipGeometry = await root.evaluate(element => {
        const rootBounds = element.getBoundingClientRect();
        const tooltipBounds = element.querySelector("#hal-small-muon-tooltip").getBoundingClientRect();
        return {
          left: tooltipBounds.left - rootBounds.left,
          right: tooltipBounds.right - rootBounds.right,
          top: tooltipBounds.top - rootBounds.top,
          width: tooltipBounds.width,
        };
      });
      assert(tooltipGeometry.left >= -1, "Tooltip extends past the figure's left edge");
      assert(tooltipGeometry.right <= 1, "Tooltip extends past the figure's right edge");
      assert(tooltipGeometry.top >= -1, "Tooltip extends above the figure");
      assert(tooltipGeometry.width <= width + 1, "Tooltip is wider than the figure");
      if (width === 360) {
        await overlay.dispatchEvent("pointerleave", { pointerType: "touch" });
        assert(await tooltip.isVisible(), "Touch detail did not remain pinned");
        await root.locator("#hal-small-muon-title").tap();
      } else {
        await root.locator("#hal-small-muon-title").hover();
      }
      assert.equal(await tooltip.isVisible(), false, "Tooltip did not dismiss outside the plot");

      await root.screenshot({ path: join(outputDirectory, `${colorScheme}-${width}.png`) });
      assert.deepEqual(errors, [], "Browser JavaScript errors");
      results.push({ width, colorScheme, geometry, tooltipGeometry, tooltipText, errors });
      await context.close();
    }
  }
  const result = {
    fragment_sha256: createHash("sha256").update(fragment).digest("hex"),
    fragment_bytes: fragment.length,
    source: expectedSource,
    embedded_float32_sha256: payload.models.map(model => ({
      run_id: model.run_id,
      sha256: model.weights_float32_sha256,
    })),
    browser_version: browser.version(),
    playwright_version: require("playwright/package.json").version,
    results,
  };
  await writeFile(join(outputDirectory, "browser-check.json"), `${JSON.stringify(result, null, 2)}\n`);
  console.log(JSON.stringify(result, null, 2));
} finally {
  await browser.close();
}
