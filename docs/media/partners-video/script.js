const { chromium } = require('playwright'); const path = require('path'); const fs = require('fs');
(async () => {
  const b = await chromium.launch(); const p = await b.newPage({ viewport: { width: 1920, height: 1080 } });
  await p.goto('file://' + path.resolve(__dirname, 'page.html'));
  const md = await p.evaluate(() => {
    const lines = ['# The script of the v3 video, every on screen text with its scene and its local time (seconds)'];
    for (const s of document.querySelectorAll('.scene')) {
      lines.push(`\n## ${s.id}  (${s.dataset.t0} s to ${s.dataset.t1} s, ${(s.dataset.t1 - s.dataset.t0).toFixed(1)} s)`);
      for (const e of s.querySelectorAll('.el')) {
        const own = [...e.childNodes].filter(n => !(n.nodeType === 1 && n.classList && n.classList.contains('el'))).map(n => n.textContent).join(' ');
        let txt = own.replace(/\s+/g, ' ').trim();
        const inner = [...e.querySelectorAll('.el')]; // nested elements are listed on their own
        if (inner.length) txt = [...e.childNodes].filter(n => !(n.nodeType === 1 && (n.classList.contains('el') || n.querySelector('.el')))).map(n => n.textContent).join(' ').replace(/\s+/g, ' ').trim();
        if (txt) lines.push(`- [${e.dataset.in}${e.dataset.out ? '–' + e.dataset.out : ''}] ${txt}`);
      }
    }
    return lines.join('\n') + '\n';
  });
  fs.writeFileSync(path.join(__dirname, 'script.md'), md); await b.close(); console.log(md.split('\n').length, 'lines');
})();
