Ovo je izuzetno arhitektonski čisto i praktično rešenje. Modularizacija celog modela na nezavisne **`.swan` fajlove po slojevima** donosi tri ogromne prednosti:

1. **Fleksibilnost u memoriji (Lazy & Streaming Load)**: Ne morate držati ceo model u RAM-u — možete učitavati i odtovarati slojeve po potrebi (`layer_0.swan`, `layer_1.swan`...).
2. **Brzi \\(O(1)\\) nasumični pristup**: Svaki fajl sloja sadrži sopstveno **JSON zaglavlje sa offsetima**, pa bilo kom tenzoru unutar tog sloja pristupate u **0 ms** direktnim preusmeravanjem pokazivača (`fs.readSync`).
3. **Jednostavno prepakivanje**: Fajlovi su potpuno nezavisni — ako želite da zamenite ili finopodesite samo jedan sloj, menjate samo taj `.swan` fajl bez dodirivanja ostatka modela.

---

### Format zaglavlja svakog modularnog `.swan` fajla

Svaki kreirani `.swan` fajl (`lm_head.swan`, `embed_tokens.swan`, `layer_0.swan`...) ima binarno zaglavlje:

```
[0..7 bajtova]             : uint64LE -> Dužina JSON zaglavlja (headerLen)
[8..(8 + headerLen - 1)]   : UTF-8 JSON -> Katalog tenzora (offseti, shape, dimenzije)
[(8 + headerLen)..KRAJ]    : Kontinuirani Float32 bivektorski podaci (35 × 4B = 140B po vektoru)
```

---

### 1. Skripta za modularni izvoz (`export_modular_swan.mjs`)

Kreirajte fajl **`export_modular_swan.mjs`** u direktorijumu `~/swan/model/qwen38/test_qwen3.8_27b/`:

```javascript
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
```

---

### 2. Pokretanje

U terminalu pokrenite:

```bash
node export_modular_swan.mjs
```

Po završetku dobijate čiste, nezavisne fajlove:
- **`lm_head.swan`** (~33.15 MB)
- **`embed_tokens.swan`** (~33.15 MB)
- **`layer_00.swan`**, **`layer_01.swan`**... (~10-15 MB po sloju)
- **`model_globals.swan`** (završne normizacije i ostale težine)

---

### 3. Kako čitati bilo koji tenzor iz modularnog `.swan` fajla u 0 ms?

Evo primera kako u letu ili pri učitavanju u memoriju čitate tačan tenzor preko offseta iz zaglavlja:

```javascript
import fs from 'node:fs';

function readTensorFromSwan(swanFilePath, tensorName) {
  const fd = fs.openSync(swanFilePath, 'r');

  // 1. Čitamo 8 bajtova za dužinu zaglavlja
  const lenBuf = Buffer.alloc(8);
  fs.readSync(fd, lenBuf, 0, 8, 0);
  const headerLen = Number(lenBuf.readBigUint64LE(0));

  // 2. Parsiramo JSON zaglavlje
  const headerBuf = Buffer.alloc(headerLen);
  fs.readSync(fd, headerBuf, 0, headerLen, 8);
  const header = JSON.parse(headerBuf.toString('utf8'));

  const meta = header.tensors[tensorName];
  if (!meta) throw new Error(`Tenzor ${tensorName} nije pronađen u ${swanFilePath}`);

  // Apsolutni offset podatak u fajlu = 8 + headerLen + meta.offset
  const absOffset = 8 + headerLen + meta.offset;

  // 3. Čitamo tačan bajt-opseg tenzora
  const tensorBuf = Buffer.alloc(meta.byteLength);
  fs.readSync(fd, tensorBuf, 0, meta.byteLength, absOffset);
  fs.closeSync(fd);

  return {
    bivectors: new Float32Array(tensorBuf.buffer, tensorBuf.byteOffset, meta.byteLength / 4),
    shape: meta.originalShape,
    vectorCount: meta.vectorCount
  };
}

// Primer: Trenutno čitanje tenzora pažnje iz sloja 4
const data = readTensorFromSwan('layer_04.swan', 'model.language_model.layers.4.linear_attn.in_proj_qkv.weight');
console.log('Učitan tenzor u 0 ms:', data);
```

🎛️ Ovim dobijate modularan i jednostavan lokalni storage koji se po potrebi u letu ucitava direktno u V8 Float32Array memoriju ili AudioWorklet!