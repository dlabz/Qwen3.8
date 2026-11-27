// @ts-check
// Pokretanje: deno run -A stream_all_shards.js

import { direct2bivectorStream } from './direct2bivector_stream.mjs';

const REPO_BASE = 'https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main';
const TOTAL_SHARDS = 18;
const MAX_RETRIES = 5;
const CHECKPOINT_FILE = 'processed_shards.json';

console.log('====================================================');
console.log(` 🚀 FAULT-TOLERANT BIVECTOR STREAMER (18 SHARDS)`);
console.log('====================================================\n');

// 1. Učitavanje checkpoint stanja (koji šardovi su završeni)
let processedShards = new Set();
try {
  const data = Deno.readTextFileSync(CHECKPOINT_FILE);
  processedShards = new Set(JSON.parse(data));
  console.log(`✓ Pronađen checkpoint: ${processedShards.size}/${TOTAL_SHARDS} šardova već obrađeno.\n`);
} catch (_e) {
  console.log(`ℹ️ Započinjem novu sesiju strimovanja od šarda 1.\n`);
}

function saveCheckpoint() {
  Deno.writeTextFileSync(CHECKPOINT_FILE, JSON.stringify(Array.from(processedShards)));
}

/**
 * Pomoćna funkcija za bezbedan upis bivektora u .swan fajl
 */
function appendToSwanFile(swanFileName, bivectorChart) {
  const bytes = new Uint8Array(bivectorChart.buffer, bivectorChart.byteOffset, bivectorChart.byteLength);
  Deno.writeFileSync(swanFileName, bytes, { append: true, create: true });
}

/**
 * Određuje ciljni .swan fajl na osnovu imena tenzora
 */
function resolveSwanTarget(tensorName) {
  if (tensorName.includes('embed_tokens')) return 'embed_tokens.swan';
  if (tensorName.includes('lm_head')) return 'lm_head.swan';
  
  const layerMatch = tensorName.match(/layers\.(\d+)\./);
  if (layerMatch) {
    const layerNum = String(layerMatch).padStart(2, '0');
    return `layer_${layerNum}.swan`;
  }
  return 'qwen3.8_27b_bivectors.swan';
}

/**
 * Strimovanje pojedinačnog šarda sa automatskim ponavljanjem u slučaju greške
 */
async function processShardWithRetry(shardIdx) {
  const shardStr = String(shardIdx).padStart(5, '0');
  const fileName = `model-${shardStr}-of-00018.safetensors`;
  const shardUrl = `${REPO_BASE}/${fileName}`;

  if (processedShards.has(fileName)) {
    console.log(`⏩ [${shardIdx}/${TOTAL_SHARDS}] Šard ${fileName} je već obrađen. Preskačem.`);
    return true;
  }

  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    console.log(`\n⏳ [${shardIdx}/${TOTAL_SHARDS}] Pokušaj ${attempt}/${MAX_RETRIES}: Strimovanje ${fileName}...`);
    
    try {
      const pendingWrites = [];

      const stream = direct2bivectorStream({
        modelUrl: shardUrl,
        useWebGPU: false,
        targetDim: 35,
        hiddenDim: 5120,
        fetchOptions: {
          headers: { 'User-Agent': 'SWAN-Deno-Streamer/1.0' },
          redirect: 'follow'
        }
      });

      let tensorCount = 0;
      for await (const { tensorName, numVectors, bivectorChart } of stream) {
        tensorCount++;
        const targetSwan = resolveSwanTarget(tensorName);
        pendingWrites.push({ targetSwan, bivectorChart });
        console.log(`  ✓ Tenzor: ${tensorName.padEnd(42)} ➔ ${targetSwan.padEnd(18)} (${numVectors} vec)`);
      }

      // Upisujemo na disk tek kada šard stigne do 100% uspeha
      for (const { targetSwan, bivectorChart } of pendingWrites) {
        appendToSwanFile(targetSwan, bivectorChart);
      }

      processedShards.add(fileName);
      saveCheckpoint();

      console.log(`✓ Šard ${fileName} uspešno završen! (${tensorCount} tenzora)`);
      return true;

    } catch (err) {
      console.error(`⚠️ Greška pri strimovanju ${fileName} (Pokušaj ${attempt}): ${err.message}`);
      
      if (attempt < MAX_RETRIES) {
        const backoffMs = attempt * 3000;
        console.log(`  ⏳ Pauza ${backoffMs / 1000}s pre ponovnog pokušaja...`);
        await new Promise(r => setTimeout(r, backoffMs));
      } else {
        console.error(`❌ Šard ${fileName} nije uspeo nakon ${MAX_RETRIES} pokušaja.`);
        return false;
      }
    }
  }

  return false;
}

// Glavna petlja
let allSuccess = true;
for (let shardIdx = 1; shardIdx <= TOTAL_SHARDS; shardIdx++) {
  const ok = await processShardWithRetry(shardIdx);
  if (!ok) {
    allSuccess = false;
    console.log(`\n⚠️ Prekidam dalje strimovanje zbog mrežne greške na šardu ${shardIdx}. Pokrenite skriptu ponovo da nastavite tačno odavde.`);
    break;
  }
}

if (allSuccess) {
  console.log('\n====================================================');
  console.log('✓ SVIH 18 ŠARDOVA SU USPEŠNO STRIMOVANI I PAKOVANI!');
  console.log('====================================================\n');
}
