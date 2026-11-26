// @ts-check
import fs from 'node:fs';

/**
 * Pomoćna funkcija za bivektor-sendvič projekciju vektora D=5120
 * @param {Float32Array} vector Vektor skrivene dimenzije 5120
 * @param {number} [targetDim=35]
 * @returns {Float32Array} 35-bitni normalizovani dijagram
 */
export function projectVectorToSandwichChart(vector, targetDim = 35) {
  const halfDim = Math.floor(vector.length / 2); // 2560
  const stride = Math.floor(halfDim / targetDim); // 73

  const topSlice = vector.subarray(0, halfDim);
  const bottomSlice = vector.subarray(halfDim, vector.length);
  const chart = new Float32Array(targetDim);

  let sumSq = 0;
  for (let k = 0; k < targetDim; k++) {
    let sumTop = 0, sumBottom = 0;
    const base = k * stride;
    for (let i = 0; i < stride; i++) {
      sumTop += topSlice[base + i] || 0;
      sumBottom += bottomSlice[base + i] || 0;
    }
    const ax = sumTop / stride;
    const by = sumBottom / stride;

    const denom = ax + by;
    const rho = Math.abs(denom) > 1e-12 ? (4 * ax * by) / (denom * denom) : 0;
    const val = ax * by * rho;

    chart[k] = val;
    sumSq += val * val;
  }

  // RMS normalizacija
  const norm = Math.sqrt(sumSq) || 1.0;
  for (let k = 0; k < targetDim; k++) {
    chart[k] /= norm;
  }

  return chart;
}

/**
 * Klasa za munjevitu pretragu bivektora u memoriji
 */
export class BivectorLookupEngine {
  /**
   * @param {string} swanFilePath Putanja do .swan binarnog indeksa
   * @param {number} [targetDim=35] Dimenzija bivektorskog dijagrama
   */
  constructor(swanFilePath, targetDim = 35) {
    this.targetDim = targetDim;
    
    console.log(`Učitavanje binarnog indeksa ${swanFilePath} u RAM...`);
    const fileBuf = fs.readFileSync(swanFilePath);
    
    // Direktno mapiranje u Float32Array (Zero-Copy u V8)
    this.bivectorDB = new Float32Array(
      fileBuf.buffer,
      fileBuf.byteOffset,
      fileBuf.byteLength / 4
    );

    this.vocabSize = this.bivectorDB.length / this.targetDim;
    console.log(`Baza spremna! Ukupno indeksirano tokena: ${this.vocabSize}`);
  }

  /**
   * Pronalazi Top-K najbližih token ID-jeva za dati 35-bitni dijagram
   * @param {Float32Array} queryChart 35-bitni dijagram upita
   * @param {number} [topK=10] Broj najsličnijih rezultata
   * @returns {{ tokenId: number, similarity: number }[]}
   */
  findNearestTokens(queryChart, topK = 10) {
    const scores = new Float32Array(this.vocabSize);
    const targetDim = this.targetDim;
    const db = this.bivectorDB;

    // Fast SIMD-friendly dot product petlja nad 248.320 bivektora
    for (let t = 0; t < this.vocabSize; t++) {
      let dot = 0;
      const offset = t * targetDim;
      for (let k = 0; k < targetDim; k++) {
        dot += queryChart[k] * db[offset + k];
      }
      scores[t] = dot; // Kosinusna sličnost pošto su vektori već normalizovani
    }

    // Pronalaženje Top-K elemenata
    /** @type {{ tokenId: number, similarity: number }[]} */
    const candidates = [];
    for (let t = 0; t < this.vocabSize; t++) {
      candidates.push({ tokenId: t, similarity: scores[t] });
    }

    candidates.sort((a, b) => b.similarity - a.similarity);
    return candidates.slice(0, topK);
  }
}

// === LOKALNI TEST PRONALAŽENJA ===
if (process.argv[1]?.endsWith('bivector_lookup.mjs')) {
  const swanFile = 'qwen3.8_27b_bivectors.swan';
  const engine = new BivectorLookupEngine(swanFile);

  // Pročitamo bivektor tokena 100 i potražimo njegove najbliže komšije
  const queryTokenId = 100;
  const sampleChart = engine.bivectorDB.subarray(queryTokenId * 35, (queryTokenId + 1) * 35);

  console.log(`\n--- Pretraga za Bivektor Tokena #${queryTokenId} ---`);
  const startTime = performance.now();
  const results = engine.findNearestTokens(sampleChart, 10);
  const elapsed = (performance.now() - startTime).toFixed(3);

  console.log(`Vreme pretrage nad 248.320 tokena: ${elapsed} ms\n`);
  console.log('Top 10 Najbližih Token ID-jeva:');
  results.forEach((r, rank) => {
    console.log(` #${rank + 1}: Token ID = ${r.tokenId.toString().padStart(6, ' ')} | Sličnost = ${r.similarity.toFixed(4)}`);
  });
}