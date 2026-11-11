// Draw a few frames of the page at full size, at the given times, to look at them before any full render.
// Usage: node preview.js <out dir> <t1> <t2> ...
const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');
(async () => {
  const out = process.argv[2]; fs.mkdirSync(out, { recursive: true });
  const times = process.argv.slice(3).map(Number);
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
  await page.goto('file://' + path.resolve(__dirname, 'page.html'));
  await page.evaluate(() => document.fonts.ready);
  for (const t of times) {
    await page.evaluate(t => window.seek(t), t);
    await page.screenshot({ path: path.join(out, `t${t.toFixed(2)}.png`), type: 'png' });
  }
  await browser.close();
  console.log('drew', times.length, 'frames');
})();
