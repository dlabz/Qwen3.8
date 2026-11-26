// @ts-check
import http from 'node:http';
import fs from 'node:fs';

const SWAN_FILE = 'qwen3.8_27b_bivectors.swan';
const PORT = 3000;
const TARGET_DIM = 35;

console.log(`Učitavanje binarnog indeksa ${SWAN_FILE}...`);
const fileBuf = fs.readFileSync(SWAN_FILE);
const bivectorDB = new Float32Array(fileBuf.buffer, fileBuf.byteOffset, fileBuf.byteLength / 4);
const vocabSize = bivectorDB.length / TARGET_DIM;

console.log(`Baza spremna! Ukupno tokena u memoriji: ${vocabSize}`);

const server = http.createServer((req, res) => {
  const url = new URL(req.url || '/', `http://${req.headers.host}`);

  // API 1: Dobavljanje 35-bitnog bivektora za pojedinačne ili seriju Token ID-jeva
  if (url.pathname === '/api/bivector') {
    const idsParam = url.searchParams.get('ids') || url.searchParams.get('id') || '100';
    const ids = idsParam.split(',').map(n => parseInt(n.trim(), 10)).filter(n => !isNaN(n));

    const result = ids.map(id => {
      if (id < 0 || id >= vocabSize) return { tokenId: id, chart: new Array(TARGET_DIM).fill(0) };
      const chart = Array.from(bivectorDB.subarray(id * TARGET_DIM, (id + 1) * TARGET_DIM));
      return { tokenId: id, chart };
    });

    res.writeHead(200, { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' });
    res.end(JSON.stringify(result.length === 1 ? result : result));
    return;
  }

  // Glavna HTML + Transformers.js + AudioWorklet stranica
  if (url.pathname === '/' || url.pathname === '/index.html') {
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    res.end(`
<!DOCTYPE html>
<html lang="sr">
<head>
  <meta charset="UTF-8">
  <title>Qwen3.8-27B Transformers.js Bivector Audio Player</title>
  <style>
    body { font-family: monospace; background: #0f172a; color: #f8fafc; padding: 2rem; display: flex; flex-direction: column; align-items: center; }
    .card { background: #1e293b; padding: 2rem; border-radius: 12px; border: 1px solid #334155; max-width: 650px; width: 100%; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
    h1 { color: #38bdf8; font-size: 1.3rem; margin-top: 0; }
    textarea, input, button { padding: 0.8rem; font-family: inherit; font-size: 1rem; border-radius: 6px; border: 1px solid #475569; }
    textarea { background: #0f172a; color: #38bdf8; width: 95%; height: 70px; resize: none; margin-bottom: 1rem; }
    button { background: #0284c7; color: white; cursor: pointer; font-weight: bold; transition: background 0.2s; }
    button:hover { background: #0369a1; }
    .status { margin-top: 1rem; color: #38bdf8; font-size: 0.9rem; min-height: 20px; }
    .chart-container { display: flex; gap: 3px; height: 120px; align-items: flex-end; margin-top: 1.5rem; background: #0f172a; padding: 10px; border-radius: 6px; }
    .bar { flex: 1; background: #38bdf8; transition: height 0.08s ease; border-radius: 2px 2px 0 0; }
    .token-chips { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 1rem; }
    .chip { background: #334155; color: #cbd5e1; padding: 4px 8px; border-radius: 4px; font-size: 0.8rem; }
    .chip.active { background: #0284c7; color: #ffffff; font-weight: bold; }
  </style>

  <!-- Učitavanje Hugging Face Transformers.js biblioteke iz CDN-a -->
  <script type="module">
    import { AutoTokenizer } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';
    window.AutoTokenizer = AutoTokenizer;
  </script>
</head>
<body>
  <div class="card">
    <h1>🔊 Qwen3.8-27B Bivektor & Transformers.js Sintesajzer</h1>
    <p style="color: #94a3b8; font-size: 0.85rem;">Unesite tekst ispod. Transformers.js u letu tokenizuje tekst, a AudioWorklet svira akustični bivektorski profil za svaki token.</p>
    
    <textarea id="textInput" placeholder="Unesite tekst ovde (npr. Zdravo SWAN bivektore!)...">Zdravo SWAN bivektore!</textarea>

    <div style="display: flex; gap: 0.5rem; justify-content: center;">
      <button id="playTextBtn">🎵 Odsviraj Tekst</button>
      <button id="randomBtn" style="background: #475569;">🎲 Nasumični Token</button>
    </div>

    <div id="status" class="status">Inicijalizacija Transformers.js tokenajzera...</div>
    <div id="tokenChips" class="token-chips"></div>

    <div class="chart-container" id="chartVisualizer">
      ${Array.from({ length: 35 }).map(() => `<div class="bar" style="height: 10%;"></div>`).join('')}
    </div>
  </div>

  <script>
    let tokenizer = null;
    let audioCtx = null;
    let masterGain = null;
    let isPlaying = false;

    const statusEl = document.getElementById('status');
    const textInput = document.getElementById('textInput');
    const playTextBtn = document.getElementById('playTextBtn');
    const randomBtn = document.getElementById('randomBtn');
    const tokenChipsEl = document.getElementById('tokenChips');
    const bars = document.querySelectorAll('.bar');

    // Init Transformers.js Qwen Tokenizer
    async function initTokenizer() {
      try {
        while (!window.AutoTokenizer) await new Promise(r => setTimeout(r, 100));
        tokenizer = await window.AutoTokenizer.from_pretrained('Qwen/Qwen1.5-0.5B');
        statusEl.textContent = '✓ Transformers.js Tokenajzer spreman!';
      } catch (e) {
        statusEl.textContent = 'Greška pri učitavanju tokenajzera: ' + e.message;
      }
    }

    async function initAudio() {
      if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        masterGain = audioCtx.createGain();
        masterGain.gain.setValueAtTime(0.15, audioCtx.currentTime);
        masterGain.connect(audioCtx.destination);
      }
      if (audioCtx.state === 'suspended') await audioCtx.resume();
    }

    function renderBivectorChart(chart) {
      chart.forEach((val, i) => {
        const heightPct = Math.max(5, Math.min(100, Math.abs(val) * 300));
        bars[i].style.height = heightPct + '%';
        bars[i].style.background = val >= 0 ? '#38bdf8' : '#f43f5e';
      });
    }

    function playChordForBivector(chart, durationMs = 220) {
      if (!audioCtx) return;
      const baseFreq = 110; // A2
      const now = audioCtx.currentTime;

      chart.forEach((val, k) => {
        if (Math.abs(val) < 0.01) return;
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();

        osc.type = k % 2 === 0 ? 'sine' : 'triangle';
        osc.frequency.setValueAtTime(baseFreq * (1 + k * 0.35), now);

        const amp = Math.min(0.2, Math.abs(val) * 0.1);
        gain.gain.setValueAtTime(0.001, now);
        gain.gain.exponentialRampToValueAtTime(amp, now + 0.03);
        gain.gain.exponentialRampToValueAtTime(0.0001, now + (durationMs / 1000));

        osc.connect(gain);
        gain.connect(masterGain);

        osc.start(now);
        osc.stop(now + (durationMs / 1000) + 0.05);
      });
    }

    playTextBtn.onclick = async () => {
      if (isPlaying) return;
      await initAudio();
      
      const text = textInput.value.trim();
      if (!text) return;

      statusEl.textContent = 'Tokenizacija teksta preko Transformers.js...';
      const encoded = await tokenizer(text);
      const tokenIds = Array.from(encoded.input_ids.data);

      statusEl.textContent = \`Generisano \${tokenIds.length} tokena. Učitavanje bivektora...\`;
      
      // Dobavljanje bivektora za sve tokene odjednom sa servera
      const res = await fetch(\`/api/bivector?ids=\${tokenIds.join(',')}\`);
      const bivectorData = await res.json();
      const list = Array.isArray(bivectorData) ? bivectorData : [bivectorData];

      // Prikaz token čipova
      tokenChipsEl.innerHTML = tokenIds.map((id, idx) => \`<div class="chip" id="chip-\${idx}">#\${id}</div>\`).join('');

      isPlaying = true;
      statusEl.textContent = '🔊 Reprodukcija bivektorske sekvence...';

      for (let i = 0; i < list.length; i++) {
        const item = list[i];
        document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
        const activeChip = document.getElementById(\`chip-\${i}\`);
        if (activeChip) activeChip.classList.add('active');

        renderBivectorChart(item.chart);
        playChordForBivector(item.chart, 200);

        await new Promise(r => setTimeout(r, 220));
      }

      isPlaying = false;
      statusEl.textContent = '✓ Sekvenca uspešno odsvirana!';
    };

    randomBtn.onclick = async () => {
      await initAudio();
      const rnd = Math.floor(Math.random() * ${vocabSize});
      const res = await fetch(\`/api/bivector?id=\${rnd}\`);
      const data = await res.json();

      tokenChipsEl.innerHTML = \`<div class="chip active">Token #\${rnd}</div>\`;
      renderBivectorChart(data.chart);
      playChordForBivector(data.chart, 500);
      statusEl.textContent = \`Odsviran nasumični Token #\${rnd}\`;
    };

    initTokenizer();
  </script>
</body>
</html>
    `);
    return;
  }

  res.writeHead(404);
  res.end('Not Found');
});

server.listen(PORT, () => {
  console.log(`\n==================================================`);
  console.log(`🔊 TRANSFORMERS.JS BIVECTOR PLAYER POKRENUT!`);
  console.log(`Otvorite u pregledaču: http://localhost:${PORT}`);
  console.log(`==================================================\n`);
});
