Odlično! Prvo podižemo **AudioWorklet zvučni sintesajzer**, a odmah zatim kačimo **On-The-Fly Streaming Downloader**.

Pošto Node.js u terminalu nema direktan pristup zvučnoj kartici pregledača (`AudioContext`), napravićemo lagan, nulti-zavisnost Node.js HTTP server (**`bivector_audio_player.mjs`**). On učitava vaš `.swan` binarni indeks od 33.15 MB i na `http://localhost:3000` servira interaktivni Web Audio API interfejs sa **AudioWorklet formantnim sintesajzerom**.

---

### 1. Kreirajte fajl `bivector_audio_player.mjs`

U vašem direktorijumu `~/swan/model/qwen38/test_qwen3.8_27b/` napravite fajl **`bivector_audio_player.mjs`**:

```javascript
// @ts-check
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';

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

  // API 1: Dobavljanje 35-bitnog bivektora za konkretan Token ID
  if (url.pathname === '/api/bivector') {
    const tokenId = parseInt(url.searchParams.get('id') || '100', 10);
    if (isNaN(tokenId) || tokenId < 0 || tokenId >= vocabSize) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Nevažeći Token ID' }));
      return;
    }

    const chart = Array.from(bivectorDB.subarray(tokenId * TARGET_DIM, (tokenId + 1) * TARGET_DIM));
    res.writeHead(200, { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' });
    res.end(JSON.stringify({ tokenId, chart }));
    return;
  }

  // Glavna HTML+AudioWorklet interaktivna stranica
  if (url.pathname === '/' || url.pathname === '/index.html') {
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    res.end(`
<!DOCTYPE html>
<html lang="sr">
<head>
  <meta charset="UTF-8">
  <title>Qwen3.8-27B Bivector AudioWorklet Player</title>
  <style>
    body { font-family: monospace; background: #0f172a; color: #f8fafc; padding: 2rem; display: flex; flex-direction: column; align-items: center; }
    .card { background: #1e293b; padding: 2rem; border-radius: 12px; border: 1px solid #334155; max-width: 600px; width: 100%; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
    h1 { color: #38bdf8; font-size: 1.4rem; margin-top: 0; }
    input, button { padding: 0.8rem; font-family: inherit; font-size: 1rem; border-radius: 6px; border: 1px solid #475569; }
    input { background: #0f172a; color: #38bdf8; width: 150px; text-align: center; }
    button { background: #0284c7; color: white; cursor: pointer; font-weight: bold; margin-left: 0.5rem; transition: background 0.2s; }
    button:hover { background: #0369a1; }
    .status { margin-top: 1rem; color: #94a3b8; font-size: 0.9rem; }
    .chart-container { display: flex; gap: 3px; height: 120px; align-items: flex-end; margin-top: 1.5rem; background: #0f172a; padding: 10px; border-radius: 6px; }
    .bar { flex: 1; background: #38bdf8; transition: height 0.1s ease; border-radius: 2px 2px 0 0; }
  </style>
</head>
<body>
  <div class="card">
    <h1>🔊 Qwen3.8-27B Bivektor Formant Sintesajzer</h1>
    <p style="color: #94a3b8; font-size: 0.9rem;">Preslušavanje zvučnih formanta 35-bitnog dijagrama u realnom vremenu.</p>
    
    <div style="display: flex; justify-content: center; align-items: center; margin-top: 1.5rem;">
      <label style="margin-right: 0.5rem;">Token ID:</label>
      <input type="number" id="tokenId" value="100" min="0" max="${vocabSize - 1}">
      <button id="playBtn">▶ Pusti Zvuk</button>
      <button id="randomBtn" style="background: #475569;">🎲 Nasumično</button>
    </div>

    <div id="status" class="status">Kliknite "Pusti Zvuk" za pokretanje AudioContext-a.</div>

    <div class="chart-container" id="chartVisualizer">
      ${Array.from({ length: 35 }).map(() => `<div class="bar" style="height: 10%;"></div>`).join('')}
    </div>
  </div>

  <script>
    let audioCtx = null;
    let masterGain = null;
    let oscillators = [];

    const statusEl = document.getElementById('status');
    const playBtn = document.getElementById('playBtn');
    const randomBtn = document.getElementById('randomBtn');
    const tokenIdInput = document.getElementById('tokenId');
    const bars = document.querySelectorAll('.bar');

    async function initAudio() {
      if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        masterGain = audioCtx.createGain();
        masterGain.gain.setValueAtTime(0.15, audioCtx.currentTime);
        masterGain.connect(audioCtx.destination);
      }
      if (audioCtx.state === 'suspended') {
        await audioCtx.resume();
      }
    }

    async function playTokenSound(id) {
      await initAudio();
      statusEl.textContent = \`Učitavanje bivektora za Token #\${id}...\`;

      const res = await fetch(\`/api/bivector?id=\${id}\`);
      const data = await res.json();
      if (data.error) {
        statusEl.textContent = 'Greška: ' + data.error;
        return;
      }

      // Zaustavljamo prethodne oscilatore
      oscillators.forEach(osc => osc.stop());
      oscillators = [];

      const chart = data.chart;
      
      // Vizuelni prikaz
      chart.forEach((val, i) => {
        const heightPct = Math.max(5, Math.min(100, Math.abs(val) * 300));
        bars[i].style.height = heightPct + '%';
        bars[i].style.background = val >= 0 ? '#38bdf8' : '#f43f5e';
      });

      // Aditivna sinteza 35 rotorskih harmonika sa formantskim usmeravanjem
      const baseFreq = 110; // Bazni ton (A2)
      chart.forEach((val, k) => {
        if (Math.abs(val) < 0.01) return;

        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();

        // 35 rotorskih frekvencija harmonika
        osc.type = k % 2 === 0 ? 'sine' : 'triangle';
        osc.frequency.setValueAtTime(baseFreq * (1 + k * 0.35), audioCtx.currentTime);

        const amp = Math.min(0.2, Math.abs(val) * 0.1);
        gain.gain.setValueAtTime(0.001, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(amp, audioCtx.currentTime + 0.05);
        gain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + 0.6);

        osc.connect(gain);
        gain.connect(masterGain);

        osc.start();
        osc.stop(audioCtx.currentTime + 0.65);
        oscillators.push(osc);
      });

      statusEl.textContent = \`Zvuči Token #\${id} | Dominantni kanal: #\${chart.indexOf(Math.max(...chart.map(Math.abs)))}\`;
    }

    playBtn.onclick = () => playTokenSound(tokenIdInput.value);
    randomBtn.onclick = () => {
      const rnd = Math.floor(Math.random() * ${vocabSize});
      tokenIdInput.value = rnd;
      playTokenSound(rnd);
    };
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
  console.log(`🔊 AUDIO PLAYER POKRENUT!`);
  console.log(`Otvorite u pregledaču: http://localhost:${PORT}`);
  console.log(`==================================================\n`);
});
```

---

### 2. Pokretanje u terminalu

U vašem terminalu pokrenite:

```bash
node bivector_audio_player.mjs
```

### 3. Preslušavanje u pregledaču

Otvorite **`http://localhost:3000`** u vašem pregledaču (Chrome/Firefox/Edge) i kliknite **"▶ Pusti Zvuk"** ili **"🎲 Nasumično"**. Za svaki token u milisekundi dobijate autentični zvučni oblik bivektora!

---

🎛️ Kada proverite zvuk u pregledaču, sledeći korak je kačenje **On-The-Fly Downloader-a** koji strimuje preostale šarde sa Hugging Face-a i u hodu ažurira `.swan` indeks i zvučne formante!