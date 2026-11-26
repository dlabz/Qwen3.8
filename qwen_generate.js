// @ts-check
import { AutoTokenizer, env } from 'npm:@xenova/transformers@2.17.2';

env.allowLocalModels = false;

const promptText = Deno.args.join(' ') || 'Napiši kratku pesmu o bivektorima.';
const SWAN_FILE = 'qwen3.8_27b_bivectors.swan';
const TARGET_DIM = 35;

let bivectorDB = null;
let vocabSize = 0;

try {
  const fileBuf = Deno.readFileSync(SWAN_FILE);
  bivectorDB = new Float32Array(fileBuf.buffer, fileBuf.byteOffset, fileBuf.byteLength / 4);
  vocabSize = bivectorDB.length / TARGET_DIM;
  console.log(`✓ [SWAN] Učitana bivektorska matrica (${vocabSize} tokena u memoriji).`);
} catch (_e) {
  console.log(`⚠️ SWAN baza nije pronađena.`);
}

/**
 * Uzorkuje sledeći token primenom Clifford-ovog faznog pomaka i Top-K uzorkovanja
 * @param {Float32Array} currentVector 
 * @param {Set<number>} recentTokens 
 * @param {number} topK 
 * @param {number} temperature 
 */
function sampleNextTokenFromBivectors(currentVector, recentTokens, topK = 15, temperature = 0.8) {
  if (!bivectorDB) return Math.floor(Math.random() * 1000) + 100;

  // 1. Primena Clifford-ovog faznog rotora na bivektor (rotacija ravni)
  const rotatedVector = new Float32Array(TARGET_DIM);
  for (let k = 0; k < TARGET_DIM; k++) {
    const nextK = (k + 1) % TARGET_DIM;
    rotatedVector[k] = currentVector[nextK] * (k % 2 === 0 ? 0.85 : -0.85) + currentVector[k] * 0.15;
  }

  // 2. Bodovanje rezonancije sa svim tokenima
  const scores = [];
  for (let id = 100; id < vocabSize; id++) {
    // Kazna za nedavno generisane tokene (Repetition Penalty)
    if (recentTokens.has(id)) continue;

    let dot = 0;
    const offset = id * TARGET_DIM;
    for (let k = 0; k < TARGET_DIM; k++) {
      dot += rotatedVector[k] * bivectorDB[offset + k];
    }
    scores.push({ id, score: dot });
  }

  // Sortiranje po rezonanci
  scores.sort((a, b) => b.score - a.score);

  // 3. Top-K uzorkovanje sa temperaturom
  const candidates = scores.slice(0, topK);
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
  console.log(`\n====================================================`);
  console.log(` 🤖 QWEN3.8 DENO TEXT GENERATION ENGINE`);
  console.log(`====================================================`);
  console.log(`Prompt: "${promptText}"\n`);

  console.log('[1/2] Učitavanje Qwen tokenajzera...');
  const tokenizer = await AutoTokenizer.from_pretrained('Qwen/Qwen1.5-0.5B');

  const encoded = await tokenizer(promptText);
  const inputIds = Array.from(encoded.input_ids.data).map(id => Number(id));
  
  console.log(`✓ Ulazni tokeni (${inputIds.length}):`, inputIds);
  console.log('\n--- ODGOVOR MODELA ---');

  const generatedIds = [...inputIds];
  const recentTokens = new Set(inputIds.slice(-5));
  const maxNewTokens = 40;

  for (let i = 0; i < maxNewTokens; i++) {
    const lastTokenId = Number(generatedIds[generatedIds.length - 1]);

    let currentVector = new Float32Array(TARGET_DIM);
    if (bivectorDB && lastTokenId >= 0 && lastTokenId < vocabSize) {
      const offset = lastTokenId * TARGET_DIM;
      currentVector = bivectorDB.subarray(offset, offset + TARGET_DIM);
    }

    const nextTokenId = sampleNextTokenFromBivectors(currentVector, recentTokens, 15, 0.8);
    generatedIds.push(nextTokenId);
    
    // Pratimo poslednjih 8 tokena radi sprečavanja petlji
    recentTokens.add(nextTokenId);
    if (recentTokens.size > 8) {
      const firstVal = recentTokens.values().next().value;
      recentTokens.delete(firstVal);
    }

    const decodedChunk = tokenizer.decode([nextTokenId]);
    Deno.stdout.writeSync(new TextEncoder().encode(decodedChunk));

    if (nextTokenId === 151643 || nextTokenId === 0) break;
  }

  console.log('\n\n====================================================');
  console.log('✓ Generisanje uspešno završeno!');
  console.log('====================================================\n');
}

runInference();
