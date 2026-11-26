// @ts-check
import fs from 'node:fs';
import path from 'node:path';
import { parseSafetensorsHeader, bfloat16ToFloat32 } from '../qwen_converter.mjs';

const HF_BASE_URL = 'https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main';
const TARGET_DIM = 35;

/**
 * Projekcija bivektor-sendviča sa RMS normalizacijom
 * @param {Float32Array} vector 
 * @param {Float32Array} outputBuffer 
 * @param {number} outOffset 
 */
function processVectorTo35b(vector, outputBuffer, outOffset) {
  const halfDim = Math.floor(vector.length / 2);
  const stride = Math.max(1, Math.floor(halfDim / TARGET_DIM));

  let sumSq = 0;
  for (let k = 0; k < TARGET_DIM; k++) {
    let sumTop = 0, sumBottom = 0;
    const base = k * stride;
    for (let i = 0; i < stride; i++) {
      sumTop += vector[base + i] || 0;
      sumBottom += vector[halfDim + base + i] || 0;
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
  for (let k = 0; k < TARGET_DIM; k++) {
    outputBuffer[outOffset + k] /= norm;
  }
}

/** Čitanje opsega bajtova (Lokalni disk ili HF Range) */
async function readBytesRange(filename, start, end) {
  if (fs.existsSync(filename)) {
    const fd = fs.openSync(filename, 'r');
    const length = end - start + 1;
    const buf = Buffer.alloc(length);
    fs.readSync(fd, buf, 0, length, start);
    fs.closeSync(fd);
    return buf;
  }

  const url = `${HF_BASE_URL}/${filename}`;
  const res = await fetch(url, { headers: { 'Range': `bytes=${start}-${end}` } });
  if (!res.ok && res.status !== 206) {
    throw new Error(`HTTP ${res.status} pri čitanju ${filename}`);
  }
  return Buffer.from(await res.arrayBuffer());
}

/** Određivanje imena izlaznog .swan fajla na osnovu imena tenzora */
function getTargetSwanFilename(tensorName) {
  if (tensorName.includes('lm_head')) return 'lm_head.swan';
  if (tensorName.includes('embed_tokens')) return 'embed_tokens.swan';

  const match = tensorName.match(/layers\.(\d+)\./);
  if (match) {
    const layerIdx = match.padStart(2, '0');
    return `layer_${layerIdx}.swan`;
  }

  return 'model_globals.swan';
}

async function main() {
  console.log('====================================================');
  console.log('    MODULARNI EXPORT QWEN3.8-27B U .SWAN FAJLOVE');
  console.log('====================================================\n');

  // Učitavanje mape tenzora
  let indexText;
  if (fs.existsSync('model.safetensors.index.json')) {
    indexText = fs.readFileSync('model.safetensors.index.json', 'utf8');
  } else {
    console.log('[HF Stream] Preuzimanje model.safetensors.index.json...');
    const buf = await readBytesRange('model.safetensors.index.json', 0, 5 * 1024 * 1024);
    indexText = buf.toString('utf8');
  }
  const weightMap = JSON.parse(indexText).weight_map;

  // Grupisanje tenzora po izlaznim .swan fajlovima
  /** @type {Record<string, { tensorName: string, shardFile: string }[]>} */
  const groups = {};
  for (const [tName, sFile] of Object.entries(weightMap)) {
    const swanTarget = getTargetSwanFilename(tName);
    if (!groups[swanTarget]) groups[swanTarget] = [];
    groups[swanTarget].push({ tensorName: tName, shardFile: sFile });
  }

  const targetFiles = Object.keys(groups).sort();
  console.log(`✓ Identifikovano ${targetFiles.length} zasebnih .swan modula:\n  - ${targetFiles.join('\n  - ')}\n`);

  for (const swanFilename of targetFiles) {
    const items = groups[swanFilename];
    console.log(`\n----------------------------------------------------`);
    console.log(`Pakovanje: ${swanFilename} (${items.length} tenzora)`);

    // Prvo izračunavamo i skupljamo bivektore za sve tenzore u ovom modulu
    /** @type {Record<string, { offset: number, byteLength: number, vectorCount: number, originalShape: number[] }>} */
    const catalog = {};
    const binaryDataChunks = [];
    let currentOffsetInBinary = 0;

    for (const item of items) {
      const { tensorName, shardFile } = item;

      // Čitanje zaglavlja šarda
      const lenBuf = await readBytesRange(shardFile, 0, 7);
      const headerLen = Number(lenBuf.readBigUint64LE(0));
      const headerBuf = await readBytesRange(shardFile, 8, 8 + headerLen - 1);
      const header = JSON.parse(headerBuf.toString('utf8'));
      const dataOffset = 8 + headerLen;

      const info = header[tensorName];
      if (!info) continue;

      const [start, end] = info.data_offsets;
      const shape = info.shape;
      const elementCount = (end - start) / 2; // BF16

      // Preuzimanje težina
      const rawBytes = await readBytesRange(shardFile, dataOffset + start, dataOffset + end - 1);
      const u16 = new Uint16Array(rawBytes.buffer, rawBytes.byteOffset, elementCount);
      const f32 = bfloat16ToFloat32(u16);

      const lastDim = shape[shape.length - 1] || 1;
      const vectorCount = Math.floor(elementCount / lastDim);

      const outFloatBuf = new Float32Array(vectorCount * TARGET_DIM);
      for (let v = 0; v < vectorCount; v++) {
        const vecSlice = f32.subarray(v * lastDim, (v + 1) * lastDim);
        processVectorTo35b(vecSlice, outFloatBuf, v * TARGET_DIM);
      }

      const outByteBuf = Buffer.from(outFloatBuf.buffer);
      binaryDataChunks.push(outByteBuf);

      catalog[tensorName] = {
        offset: currentOffsetInBinary,
        byteLength: outByteBuf.byteLength,
        vectorCount,
        originalShape: shape
      };

      currentOffsetInBinary += outByteBuf.byteLength;
      console.log(`  + [Tenzor]: ${tensorName.padEnd(55)} | Vektora: ${vectorCount.toString().padStart(6)} | Vel: ${(outByteBuf.byteLength / 1024).toFixed(1)} KB`);
    }

    // Kreiranje konačnog modularnog .swan fajla sa zaglavljem
    const catalogJson = JSON.stringify({ fileModule: swanFilename, tensors: catalog }, null, 2);
    const catalogBuf = Buffer.from(catalogJson, 'utf8');

    // 8 bajtova za dužinu JSON-a
    const headerLenBuf = Buffer.alloc(8);
    headerLenBuf.writeBigUint64LE(BigInt(catalogBuf.byteLength), 0);

    const fdOut = fs.openSync(swanFilename, 'w');
    // Upisuje: [8B len][JSON header][Sirovi bivektorski podaci]
    fs.writeSync(fdOut, headerLenBuf);
    fs.writeSync(fdOut, catalogBuf);

    for (const chunk of binaryDataChunks) {
      fs.writeSync(fdOut, chunk);
    }
    fs.closeSync(fdOut);

    const totalStat = fs.statSync(swanFilename);
    console.log(`✓ Uspešno kreiran ${swanFilename} (Ukupno: ${(totalStat.size / (1024 * 1024)).toFixed(2)} MB)`);
  }

  console.log(`\n====================================================`);
  console.log(` USPEŠNO ZAVRŠEN MODULARNI EXPORT SVIH SLOJEVA!`);
  console.log(`====================================================\n`);
}

main();
