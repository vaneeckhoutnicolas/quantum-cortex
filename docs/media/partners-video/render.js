// Render the page's scenes to one encoded file each (1920x1080, 30 fps, JPEG frames piped into ffmpeg), then concatenate.
// Usage: node render.js <out dir> [scene ids...]   (no ids: every scene)
const { chromium } = require('playwright');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const FPS = 30;
const out = process.argv[2]; fs.mkdirSync(out, { recursive: true });
const only = process.argv.slice(3);

async function renderScene(page, sc) {
  const f0 = Math.round(sc.t0 * FPS), f1 = Math.round(sc.t1 * FPS);
  const tmp = path.join(out, `tmp_${sc.id}.mp4`), fin = path.join(out, `${sc.id}.mp4`);
  const ff = spawn('ffmpeg', ['-y', '-v', 'error', '-f', 'image2pipe', '-framerate', String(FPS), '-i', '-', '-c:v', 'libx264', '-crf', '18', '-preset', 'veryfast', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', tmp]);
  let err = ''; ff.stderr.on('data', d => err += d);
  const done = new Promise((res, rej) => ff.on('close', c => c === 0 ? res() : rej(new Error(`ffmpeg ${c}: ${err}`))));
  for (let f = f0; f < f1; f++) {
    const t = f / FPS;
    await page.evaluate(t => window.seek(t), t);
    const buf = await page.screenshot({ type: 'jpeg', quality: 92 });
    if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once('drain', r));
  }
  ff.stdin.end();
  await done;
  fs.renameSync(tmp, fin);
  return f1 - f0;
}

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
  await page.goto('file://' + path.resolve(__dirname, 'page.html'));
  await page.evaluate(() => document.fonts.ready);
  let scenes = await page.evaluate(() => [...document.querySelectorAll('.scene')].map(s => ({ id: s.id, t0: +s.dataset.t0, t1: +s.dataset.t1 })));
  fs.writeFileSync(path.join(out, 'scenes.json'), JSON.stringify(scenes, null, 1));
  if (only.length) scenes = scenes.filter(s => only.includes(s.id));
  const start = Date.now();
  for (const sc of scenes) {
    const n = await renderScene(page, sc);
    console.log(sc.id, n, 'frames', ((Date.now() - start) / 1000).toFixed(0) + 's');
  }
  await browser.close();
})();
