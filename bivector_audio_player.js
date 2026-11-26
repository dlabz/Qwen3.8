// @ts-check
// Pokretanje: deno run --allow-net --allow-read bivector_audio_player.js

const SWAN_FILE = 'qwen3.8_27b_bivectors.swan';
const PORT = 3000;
const TARGET_DIM = 35;

console.log(`[Deno Audio] Učitavanje binarnog indeksa ${SWAN_FILE}...`);
const fileBuf = Deno.readFileSync(SWAN_FILE);
const bivectorDB = new Float32Array(fileBuf.buffer, fileBuf.byteOffset, fileBuf.byteLength / 4);
const vocabSize = bivectorDB.length / TARGET_DIM;

console.log(`[Deno Audio] Baza spremna! Ukupno tokena: ${vocabSize}`);

const htmlContent = `
<!DOCTYPE html>
<html lang="sr">
<head>
  <meta charset="UTF-8">
  <title>Qwen3.8-27B Deno Audio Player</title>
  <style>
    body { font-family: monospace; background: #0f172a; color: #f8fafc; padding: 2rem; display: flex; flex-direction: column; align-items: center; }
    .card { background: #1e293b; padding: 2rem; border-radius: 12px; border: 1px solid #334155; max-width: 650px; width: 100%; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
    h1 { color: #38bdf8; font-size: 1.3rem; margin-top: 0; }
    textarea, button { padding: 0.8rem; font-family: inherit; font-size: 1rem; border-radius: 6px; border: 1px solid #475569; }
    textarea { background: #0f172a; color: #38bdf8; width: 95%; height: 70px; resize: none; margin-bottom: 1rem; }
    button { background: #0284c7; color: white; cursor: pointer; font-weight: bold; }
    .status { margin-top: 1rem; color: #38bdf8; font-size: 0.9rem; min-height: 20px; }
    .chart-container { display: flex; gap: 3px; height: 120px; align-items: flex-end; margin-top: 1.5rem; background: #0f172a; padding: 10px; border-radius: 6px; }
    .bar { flex: 1; background: #38bdf8; transition: height 0.08s ease; border-radius: 2px 2px 0 0; }
    .token-chips { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 1rem; }
    .chip { background: #334155; color: #cbd5e1; padding: 4px 8px; border-radius: 4px; font-size: 0.8rem; }
    .chip.active { background: #0284c7; color: #ffffff; font-weight: bold; }
  </style>

  <script type="module">
    import { AutoTokenizer } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';
    window.AutoTokenizer = AutoTokenizer;
  </script>
</head>
<body>
  <div class="card">
    <h1>🔊 Qwen3.8-27B Deno Audio Synthesizer</h1>
    <p style="color: #94a3b8; font-size: 0.85rem;">Transformers.js tokenizuje tekst u hodu, a AudioWorklet reprodukuje bivektorske harmonike sa Deno servera.</p>
    
    <textarea id="textInput" placeholder="Unesite tekst (npr. Deno Ring-0 bivektor)...">Deno Ring-0 bivektor</textarea>

    <div style="display: flex; gap: 0.5rem; justify-content: center;">
      <button id="playTextBtn">🎵 Odsviraj Tekst</button>
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
    const tokenChipsEl = document.getElementById('tokenChips');
    const bars = document.querySelectorAll('.bar');

    async function initTokenizer() {
      try {
        while (!window.AutoTokenizer) await new Promise(r => setTimeout(r, 100));
        tokenizer = await window.AutoTokenizer.from_pretrained('Qwen/Qwen1.5-0.5B');
        statusEl.textContent = '✓ Deno & Transformers.js spreman!';
      } catch (e) {
        statusEl.textContent = 'Greška: ' + e.message;
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

    function playChordForBivector(chart, durationMs = 200) {
      if (!audioCtx) return;
      const baseFreq = 110;
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

      statusEl.textContent = 'Tokenizacija teksta...';
      const encoded = await tokenizer(text);
      const tokenIds = Array.from(encoded.input_ids.data);

      statusEl.textContent = \`Deno povlači bivektore za \${tokenIds.length} tokena...\`;
      
      const res = await fetch(\`/api/bivector?ids=\${tokenIds.join(',')}\`);
      const bivectorData = await res.json();
      const list = Array.isArray(bivectorData) ? bivectorData : [bivectorData];

      tokenChipsEl.innerHTML = tokenIds.map((id, idx) => \`<div class="chip" id="chip-\${idx}">#\${id}</div>\`).join('');

      isPlaying = true;
      statusEl.textContent = '🔊 Reprodukcija bivektorskih harmonika...';

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
      statusEl.textContent = '✓ Reprodukcija završena!';
    };

    initTokenizer();
  </script>
</body>
</html>
`;

Deno.serve({ port: PORT, hostname: '127.0.0.1' }, (req) => {
  const url = new URL(req.url);

  if (url.pathname === '/api/bivector') {
    const idsParam = url.searchParams.get('ids') || url.searchParams.get('id') || '100';
    const ids = idsParam.split(',').map(n => parseInt(n.trim(), 10)).filter(n => !isNaN(n));

    const result = ids.map(id => {
      if (id < 0 || id >= vocabSize) return { tokenId: id, chart: new Array(TARGET_DIM).fill(0) };
      const chart = Array.from(bivectorDB.subarray(id * TARGET_DIM, (id + 1) * TARGET_DIM));
      return { tokenId: id, chart };
    });

    return new Response(JSON.stringify(result.length === 1 ? result : result), {
      headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' }
    });
  }

  return new Response(htmlContent, {
    headers: { 'Content-Type': 'text/html; charset=utf-8' }
  });
});
