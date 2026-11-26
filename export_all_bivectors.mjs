// @ts-check
import fs from 'node:fs';
import { parseSafetensorsHeader, bfloat16ToFloat32 } from '../qwen_converter.mjs';

/**
 * Projekcija i RMS normalizacija bivektor-sendviča za jedan vektor D=5120
 * @param {Float32Array} vector 
 * @param {Float32Array} outputBuffer 
 * @param {number} outOffset 
 * @param {number} [targetDim=35] 
 */
function processBivectorSandwichDirect(vector, outputBuffer, outOffset, targetDim = 35) {
  const halfDim = Math.floor(vector.length / 2); // 2560
  const stride = Math.floor(halfDim / targetDim); // 73

  const topSlice = vector.subarray(0, halfDim);
  const bottomSlice = vector.subarray(halfDim, vector.length);

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

    outputBuffer[outOffset + k] = val;
    sumSq += val * val;
  }

  // RMS Normalizacija
  const norm = Math.sqrt(sumSq) || 1.0;
  for (let k = 0; k < targetDim; k++) {
    outputBuffer[outOffset + k] /= norm;
  }
}

async function exportAllBivectors() {
  const inputFile = 'model-00018.safetensors';
  const outputFile = 'qwen3.8_27b_bivectors.swan';

  console.log('=== STRIMING EKSTRAKCIJA BIVEKTORA ZA SVE TOKENE ===');
  console.log(`Ulaz:  ${inputFile}`);
  console.log(`Izlaz: ${outputFile}\n`);

  const fdIn = fs.openSync(inputFile, 'r');
  const fdOut = fs.openSync(outputFile, 'w');

  // 1. Parsiranje Safetensors zaglavlja
  const headerBuf = Buffer.alloc(1024 * 1024);
  const bytesRead = fs.readSync(fdIn, headerBuf, 0, 1024 * 1024, 0);
  const { header, dataOffset } = parseSafetensorsHeader(headerBuf.subarray(0, bytesRead));

  const tensorName = 'lm_head.weight';
  const info = header[tensorName];
  if (!info) {
    throw new Error(`Tenzor ${tensorName} nije pronađen u ${inputFile}`);
  }

  // Eksplicitno uzimamo dimenzije iz niza shape
  const [vocabSize, hiddenDim] = info.shape;
  const targetDim = 35;
  const [startOffset] = info.data_offsets;

  console.log(`Učitane dimenzije: ${vocabSize} tokena x ${hiddenDim} skrivene dimenzije`);
  console.log(`Očekivana veličina binarnog indeksa: ${(vocabSize * targetDim * 4 / (1024 * 1024)).toFixed(2)} MB\n`);

  const batchTokens = 4096; // Blok od 4.096 tokena (~42 MB pročitana u krugu)
  const bytesPerToken = hiddenDim * 2; // BF16 -> 10.240 bajtova po tokenu
  const readBuf = Buffer.alloc(batchTokens * bytesPerToken);

  const outFloatBuf = new Float32Array(batchTokens * targetDim);
  const outByteBuf = new Uint8Array(outFloatBuf.buffer);

  const startTime = Date.now();

  for (let tokenIdx = 0; tokenIdx < vocabSize; tokenIdx += batchTokens) {
    const currentBatch = Math.min(batchTokens, vocabSize - tokenIdx);
    const readBytes = currentBatch * bytesPerToken;
    const currentFileOffset = dataOffset + startOffset + tokenIdx * bytesPerToken;

    // Čitanje serije težina sa diska bez preopterećenja RAM-a
    fs.readSync(fdIn, readBuf, 0, readBytes, currentFileOffset);

    const u16View = new Uint16Array(readBuf.buffer, readBuf.byteOffset, currentBatch * hiddenDim);
    const f32View = bfloat16ToFloat32(u16View);

    for (let b = 0; b < currentBatch; b++) {
      const tokenVec = f32View.subarray(b * hiddenDim, (b + 1) * hiddenDim);
      processBivectorSandwichDirect(tokenVec, outFloatBuf, b * targetDim, targetDim);
    }

    // Upis sklopljene serije direktno u .swan binarni indeks
    const validOutputBytes = currentBatch * targetDim * 4;
    fs.writeSync(fdOut, outByteBuf.subarray(0, validOutputBytes));

    const progress = Math.min(100, ((tokenIdx + currentBatch) / vocabSize * 100)).toFixed(1);
    process.stdout.write(`\r[Napredak]: ${progress}% (${tokenIdx + currentBatch} / ${vocabSize} tokena)`);
  }

  fs.closeSync(fdIn);
  fs.closeSync(fdOut);

  const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);
  const stat = fs.statSync(outputFile);

  console.log(`\n\nUSPEŠNO ZAVRŠENO!`);
  console.log(`Vreme obrade: ${elapsed} s`);
  console.log(`Kreiran binarni indeks: ${outputFile} (${(stat.size / (1024 * 1024)).toFixed(2)} MB)`);
}

exportAllBivectors();