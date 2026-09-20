/* Run with: node check_figures.cjs <playwright module> [chromium executable].
   Loads the built figures locally; no application services or network are needed. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const {chromium} = require(process.argv[2] || 'playwright');

(async () => {
  const entries = JSON.parse(fs.readFileSync(path.join(__dirname, 'figures_src/manifest.json')));
  const keys = entries.filter(entry => !entry.attach).map(entry => entry.key);
  const browser = await chromium.launch(process.argv[3] ? {executablePath: process.argv[3]} : {});
  try {
    const page = await browser.newPage({viewport: {width: 1200, height: 900}, hasTouch: true});
    const errors = [];
    page.on('pageerror', error => errors.push(String(error)));
    await page.route('https://**', route => route.abort());
    const seed = {outline: {order: keys}, blocks: Object.fromEntries(keys.map(key => [key, {type: 'figure', fig: key}]))};
    await page.addInitScript(value => {window.WB_SEED = value;}, seed);
    await page.goto(pathToFileURL(path.join(__dirname, 'dist/index.html')).href);
    for (const key of keys) assert.equal(await page.locator('#fig-' + key + ' .fig-src').count(), 1, key + ' placed once');
    const values = await page.evaluate(() => ({
      codec: window.HALFIG_CUSTOM.data.codecEffects(),
      capacity: window.HALFIG_CUSTOM.data.capacityPanels(),
      iso: window.HALFIG_CUSTOM.data.isoFlop(),
      isoLevels: window.HALFIG_CUSTOM.data.isoFlopLevels()
    }));
    assert.equal(values.capacity.length, 16);
    assert.equal(values.iso.length, 33);
    assert.equal(values.isoLevels.length, 20);
    assert.ok(Math.abs(values.codec[0].stock_delta - 0.29) < 0.01);
    assert.equal(await page.locator('.replay-batches .batch-grid').count(), 2);
    for (const grid of await page.locator('.replay-batches .batch-grid').all()) assert.equal(await grid.locator('span').count(), 16);
    for (const width of [390, 760, 1200, 390]) {
      await page.setViewportSize({width, height: 900});
      await page.waitForTimeout(100);
      assert.equal(await page.locator('#capacity-panels-diagram .effect-pt').count(), 16);
      assert.equal(await page.locator('#tier-prediction-diagram .pred-pt').count(), 4);
      assert.equal(await page.locator('#batch-effects-diagram circle').count(), 10);
      assert.equal(await page.locator('#lr-effects-diagram circle').count(), 12);
    }
    assert.equal(await page.locator('#identity-effects-diagram').isVisible(), false);
    assert.equal(await page.locator('#identity-effects .mobile-forest .effect-ci').count(), 8);
    const stroke = await page.locator('#identity-effects .mobile-forest .effect-ci').first().evaluate(el => getComputedStyle(el).stroke);
    assert.notEqual(stroke, 'none', 'mobile confidence intervals must remain visible');
    await page.locator('#capacity-panels-diagram [role=button]').first().click();
    assert.match(await page.locator('#capacity-panels-readout').innerText(), /5M/);
    await page.locator('#iso-flop details').evaluate(el => {el.open = true;});
    assert.equal(await page.locator('#iso-residual-diagram circle:not(.iso-hit)').count(), 33);
    assert.ok(await page.locator('#iso-compute-diagram polyline[stroke-dasharray="4 4"]').count() > 0);
    assert.equal(await page.locator('#iso-compute-diagram path').count(), 4);
    assert.equal(await page.locator('#iso-optimal-table tbody tr').count(), 6);
    await page.locator('#iso-compute-diagram .iso-hit').last().hover({force: true});
    assert.match(await page.locator('#iso-flop-readout').textContent(), /Predicted minimum/);
    assert.ok(await page.locator('.iso-tip.on').count() === 1, 'tooltip shown on hover');
    assert.equal(await page.locator('#iso-flop .iso-legend li').count(), 7);
    const before = await page.locator('#ledger-svg .line-G').getAttribute('d');
    await page.locator('#ledger-hl').evaluate(el => {el.value = '3'; el.dispatchEvent(new Event('input', {bubbles: true}));});
    assert.notEqual(await page.locator('#ledger-svg .line-G').getAttribute('d'), before);
    await page.locator('#ledger-svg').tap({force: true});
    assert.match(await page.locator('#ledger-readout').innerText(), /stocks/);
    await page.locator('#wb-theme').click();
    assert.deepEqual(errors, []);
    console.log('PASS: 27 figures; desktop/mobile reflow, data counts, fixed batches, visible intervals, tap readouts, AWR controls, fit support and theme.');
  } finally {
    await browser.close();
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
