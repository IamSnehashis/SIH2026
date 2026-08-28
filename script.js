// ---------- mobile nav ----------
document.addEventListener('DOMContentLoaded', () => {
  const toggle = document.querySelector('.nav-toggle');
  const links = document.querySelector('.nav-links');
  if (toggle && links) {
    toggle.addEventListener('click', () => {
      const open = links.classList.toggle('is-open');
      toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
  }

  // ---------- scroll reveal ----------
  const revealEls = document.querySelectorAll('.reveal');
  if ('IntersectionObserver' in window && revealEls.length) {
    const io = new IntersectionObserver((entries) => {
      entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add('is-visible'); io.unobserve(e.target); } });
    }, { threshold: 0.15 });
    revealEls.forEach(el => io.observe(el));
  } else {
    revealEls.forEach(el => el.classList.add('is-visible'));
  }

  // ---------- compare sliders ----------
  document.querySelectorAll('.compare').forEach(initCompare);
});

function initCompare(root) {
  const after = root.querySelector('.compare-after');
  const handle = root.querySelector('.compare-handle');
  let dragging = false;

  const setPos = (clientX) => {
    const rect = root.getBoundingClientRect();
    let pct = ((clientX - rect.left) / rect.width) * 100;
    pct = Math.max(0, Math.min(100, pct));
    after.style.clipPath = `inset(0 0 0 ${pct}%)`;
    handle.style.left = `${pct}%`;
  };

  const move = (e) => {
    if (!dragging) return;
    const x = e.touches ? e.touches[0].clientX : e.clientX;
    setPos(x);
  };

  root.addEventListener('pointerdown', (e) => { dragging = true; setPos(e.clientX); });
  window.addEventListener('pointermove', move);
  window.addEventListener('pointerup', () => dragging = false);

  // keyboard accessibility
  root.setAttribute('tabindex', '0');
  root.setAttribute('role', 'slider');
  root.setAttribute('aria-label', 'Drag to compare medium-resolution input and super-resolved output');
  root.addEventListener('keydown', (e) => {
    const rect = root.getBoundingClientRect();
    const current = parseFloat(handle.style.left) || 50;
    if (e.key === 'ArrowLeft') setPos(rect.left + rect.width * (current - 5) / 100);
    if (e.key === 'ArrowRight') setPos(rect.left + rect.width * (current + 5) / 100);
  });
}

// ---------- demo page: procedural sample tiles + pixelation ----------
function drawSampleTile(ctx, w, h, kind, seed) {
  let s = seed;
  const rand = () => { s = (s * 16807) % 2147483647; return (s - 1) / 2147483646; };

  if (kind === 'farmland') {
    ctx.fillStyle = '#1c3a24';
    ctx.fillRect(0, 0, w, h);
    const cols = 6, rows = 5;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const greens = ['#2f5d34', '#3f7a3f', '#6a8f3a', '#8ea63c', '#4a6b2c'];
        ctx.fillStyle = greens[Math.floor(rand() * greens.length)];
        const pad = 3;
        ctx.fillRect(c * (w / cols) + pad, r * (h / rows) + pad, w / cols - pad * 2, h / rows - pad * 2);
      }
    }
    ctx.strokeStyle = 'rgba(0,0,0,0.25)'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(0, h * 0.15); ctx.lineTo(w, h * 0.1); ctx.stroke();
  } else if (kind === 'river') {
    ctx.fillStyle = '#274a2e'; ctx.fillRect(0, 0, w, h);
    for (let i = 0; i < 400; i++) {
      ctx.fillStyle = rand() > 0.5 ? '#2f5d34' : '#3c6b3a';
      ctx.fillRect(rand() * w, rand() * h, 6, 6);
    }
    ctx.strokeStyle = '#2b6fa3'; ctx.lineWidth = h * 0.14; ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(0, h * 0.2);
    ctx.bezierCurveTo(w * 0.3, h * 0.5, w * 0.6, h * 0.15, w, h * 0.6);
    ctx.stroke();
    ctx.strokeStyle = 'rgba(255,255,255,0.15)'; ctx.lineWidth = h * 0.02;
    ctx.stroke();
  } else {
    ctx.fillStyle = '#2a2f3a'; ctx.fillRect(0, 0, w, h);
    const cell = w / 12;
    for (let r = 0; r < h / cell; r++) {
      for (let c = 0; c < w / cell; c++) {
        if (rand() > 0.3) {
          ctx.fillStyle = `rgb(${60 + rand() * 60},${63 + rand() * 55},${74 + rand() * 55})`;
          ctx.fillRect(c * cell + 2, r * cell + 2, cell - 4, cell - 4);
        }
      }
    }
    ctx.strokeStyle = 'rgba(255,180,84,0.5)'; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(0, h * 0.5); ctx.lineTo(w, h * 0.5); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(w * 0.5, 0); ctx.lineTo(w * 0.5, h); ctx.stroke();
  }
}

function pixelateCanvas(srcCanvas, destCanvas, factor) {
  const w = srcCanvas.width, h = srcCanvas.height;
  const small = document.createElement('canvas');
  small.width = Math.max(1, Math.floor(w / factor));
  small.height = Math.max(1, Math.floor(h / factor));
  const sctx = small.getContext('2d');
  sctx.imageSmoothingEnabled = true;
  sctx.drawImage(srcCanvas, 0, 0, small.width, small.height);

  const dctx = destCanvas.getContext('2d');
  destCanvas.width = w; destCanvas.height = h;
  dctx.imageSmoothingEnabled = false;
  dctx.drawImage(small, 0, 0, w, h);
}
