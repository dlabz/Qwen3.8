// @ts-check
// Pokretanje: deno run -A qwen_full_pipeline.js "Napiši kratku pesmu o bivektorima."

import { AutoTokenizer, env } from 'npm:@xenova/transformers@2.17.2';

env.allowLocalModels = false;

const promptText = Deno.args.join(' ') || 'Napiši kratku pesmu o bivektorima.';
const TARGET_DIM = 35;

/**
 * Čita .swan fajl sa podsetnikom na potencijalno JSON zaglavlje
 * @param {string} filePath 
 */
function loadSwanModule(filePath) {
  try {
    const fileBuf = Deno.readFileSync(filePath);
    const view = new DataView(fileBuf.buffer, fileBuf.byteOffset, fileBuf.byteLength);
    const headerLen = Number(view.getBigUint64(0, true));

    if (headerLen > 0 && headerLen < fileBuf.byteLength) {
      const jsonBuf = fileBuf.subarray(8, 8 + headerLen);
      const catalog = JSON.parse(new TextDecoder().decode(jsonBuf));
      const dataBuf = fileBuf.subarray(8 + headerLen);
      const bivectors = new Float32Array(dataBuf.buffer, dataBuf.byteOffset, dataBuf.byteLength / 4);
      return { catalog, bivectors };
    } else {
      const bivectors = new Float32Array(fileBuf.buffer, fileBuf.byteOffset, fileBuf.byteLength / 4);
      return { catalog: null, bivectors };
    }
  } catch (_e) {
    return null;
  }
}

console.log('====================================================');
console.log(' 🌐 QWEN3.8 MULTI-LAYER SWAN DEEP PIPELINE');
console.log('====================================================\n');

// 1. Učitavanje Embed tokens i Lm Head baze
console.log('[1/3] Učitavanje ulaznih i izlaznih bivektorskih matrica...');
const embedModule = loadSwanModule('embed_tokens.swan');
const lmHeadModule = loadSwanModule('lm_head.swan') || loadSwanModule('qwen3.8_27b_bivectors.swan');

if (!lmHeadModule) {
  console.error('❌ Greška: lm_head.swan ili qwen3.8_27b_bivectors.swan nije pronađen!');
  Deno.exit(1);
}

const vocabSize = Math.floor(lmHeadModule.bivectors.length / TARGET_DIM);
console.log(`  ✓ Ulazni Embeddings (embed_tokens.swan): ${embedModule ? 'Učitan' : 'Pasivni fallback na lm_head'}`);
console.log(`  ✓ Izlazna glava (lm_head.swan): Učitana (${vocabSize} tokena)`);

// 2. Dinamička detekcija i učitavanje svih slojeva (layer_00.swan, layer_01.swan, ...)
console.log('\n[2/3] Učitavanje slojeva transformacije (layer_XX.swan)...');
const layers = [];

for (let idx = 0; idx < 64; idx++) {
  const pad = String(idx).padStart(2, '0');
  const fileName = `layer_${pad}.swan`;
  const mod = loadSwanModule(fileName);
  if (mod) {
    layers.push({ id: idx, bivectors: mod.bivectors });
  }
}

if (layers.length > 0) {
  console.log(`  ✓ Učitano je ukupno ${layers.length} slojeva transformacije!`);
} else {
  console.log(`  ⚠️ Nisu pronađeni pojedinačni layer_XX.swan fajlovi. Primenjujem ugrađene rotorske rotacije slojeva.`);
}

/**
 * Normalizacija bivektora (RMSNorm)
 * @param {Float32Array} vec 
 */
function applyRMSNorm(vec) {
  let sumSq = 0;
  for (let k = 0; k < TARGET_DIM; k++) sumSq += vec[k] * vec[k];
  const rms = Math.sqrt(sumSq / TARGET_DIM + 1e-6);
  for (let k = 0; k < TARGET_DIM; k++) vec[k] /= rms;
}

/**
 * Prolazak bivektorskog signala kroz sve slojeve (Forward Pass)
 * @param {number} tokenId 
 * @param {Set<number>} recentTokens 
 * @param {number} temperature 
 */
function fullForwardPass(tokenId, recentTokens, temperature = 0.7) {
  // 1. Ekstrakcija ulaznog bivektora iz embed_tokens
  const state = new Float32Array(TARGET_DIM);
  const srcBuf = embedModule ? embedModule.bivectors : lmHeadModule.bivectors;
  if (tokenId >= 0 && tokenId < vocabSize) {
    state.set(srcBuf.subarray(tokenId * TARGET_DIM, (tokenId + 1) * TARGET_DIM));
  }

  // Initial RMSNorm
  applyRMSNorm(state);

  // 2. Sekvencijalni prolaz kroz sve slojeve (Layer-by-Layer Transformation)
  if (layers.length > 0) {
    for (const layer of layers) {
      const nextState = new Float32Array(TARGET_DIM);
      const lBuf = layer.bivectors;
      const matrixSize = TARGET_DIM * TARGET_DIM;

      if (lBuf.length >= matrixSize) {
        // Matrica transformacije sloja W [35 x 35]
        for (let i = 0; i < TARGET_DIM; i++) {
          let sum = 0;
          for (let j = 0; j < TARGET_DIM; j++) {
            sum += state[j] * lBuf[i * TARGET_DIM + j];
          }
          nextState[i] = sum;
        }
      } else {
        // Rotorska transformacija ako sloj drži skup bivektorskih sidara
        for (let k = 0; k < TARGET_DIM; k++) {
          const shiftK = (k + 1) % TARGET_DIM;
          nextState[k] = state[k] * 0.85 + lBuf[shiftK % lBuf.length] * 0.15;
        }
      }

      // Residual connection + Layer RMSNorm
      for (let k = 0; k < TARGET_DIM; k++) state[k] += nextState[k];
      applyRMSNorm(state);
    }
  } else {
    // Sintetička 36-slojna rezonantna transformacija u odsustvu zasebnih slojeva
    for (let l = 0; l < 36; l++) {
      for (let k = 0; k < TARGET_DIM; k++) {
        const nextK = (k + (l % 3) + 1) % TARGET_DIM;
        state[k] = state[k] * 0.9 + state[nextK] * 0.1 * ((l % 2 === 0) ? 1 : -1);
      }
      applyRMSNorm(state);
    }
  }

  // 3. Projekcija na lm_head i uzorkovanje narednog tokena
  const scores = [];
  const lmDB = lmHeadModule.bivectors;

  for (let id = 100; id < vocabSize; id++) {
    if (recentTokens.has(id)) continue; // Repetition penalty

    let dot = 0;
    const offset = id * TARGET_DIM;
    for (let k = 0; k < TARGET_DIM; k++) {
      dot += state[k] * lmDB[offset + k];
    }
    scores.push({ id, score: dot });
  }

  // Top-K (K=12) sampling sa temperaturom
  scores.sort((a, b) => b.score - a.score);
  const candidates = scores.slice(0, 12);

  const expScores = candidates.map(c => Math.exp(c.score / temperature));
  const sumExp = expScores.reduce((a, b) => a + b, 0);

  let rand = Math.random() * sumExp;
  for (let i = 0; i < candidates.length; i++) {
    rand -= expScores[i];
    if (rand <= 0) return candidates[i].id;
  }

  return candidates.id;
}

async function runInference() {
  console.log('\n[3/3] Tokenizacija i pokretanje pune generacije...');
  const tokenizer = await AutoTokenizer.from_pretrained('Qwen/Qwen1.5-0.5B');

  const encoded = await tokenizer(promptText);
  const inputIds = Array.from(encoded.input_ids.data).map(id => Number(id));

  console.log(`  ✓ Ulazni tokeni (${inputIds.length}):`, inputIds);
  console.log('\n--- ODGOVOR MODELA ---');

  const generatedIds = [...inputIds];
  const recentTokens = new Set(inputIds.slice(-6));
  const maxNewTokens = 40;

  for (let step = 0; step < maxNewTokens; step++) {
    const lastTokenId = generatedIds[generatedIds.length - 1];

    const nextTokenId = fullForwardPass(lastTokenId, recentTokens, 0.7);
    generatedIds.push(nextTokenId);

    recentTokens.add(nextTokenId);
    if (recentTokens.size > 12) {
      const firstVal = recentTokens.values().next().value;
      recentTokens.delete(firstVal);
    }

    const chunkText = tokenizer.decode([nextTokenId]);
    Deno.stdout.writeSync(new TextEncoder().encode(chunkText));

    if (nextTokenId === 151643 || nextTokenId === 0) break;
  }

  console.log('\n\n====================================================');
  console.log('✓ Puni bivektorski forward-pass uspešno završen!');
  console.log('====================================================\n');
}

runInference();
