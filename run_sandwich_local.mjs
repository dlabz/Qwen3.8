// @ts-check
import fs from 'node:fs';
import { parseSafetensorsHeader, bfloat16ToFloat32 } from '../qwen_converter.mjs';

/**
 * Bivektor-sendvič projekcija sa koherencijom bisektora (ρ)
 * deli vektor D=5120 na Top (0..2559) i Bottom (2560..5119) kriške
 * @param {Float32Array} vector 
 * @param {number} [targetDim=35] 
 */
function processBivectorSandwich(vector, targetDim = 35) {
  const halfDim = Math.floor(vector.length / 2); // 2560
  const stride = Math.floor(halfDim / targetDim); // 73 elemenata po kanalu

  const topSlice = vector.subarray(0, halfDim);
  const bottomSlice = vector.subarray(halfDim, vector.length);

  const topChart = new Float32Array(targetDim);
  const bottomChart = new Float32Array(targetDim);
  const rhoCoherence = new Float32Array(targetDim);
  const sandwichChart = new Float32Array(targetDim);

  for (let k = 0; k < targetDim; k++) {
    let sumTop = 0, sumBottom = 0;
    const base = k * stride;
    for (let i = 0; i < stride; i++) {
      sumTop += topSlice[base + i] || 0;
      sumBottom += bottomSlice[base + i] || 0;
    }
    const ax = sumTop / stride;
    const by = sumBottom / stride;

    topChart[k] = ax;
    bottomChart[k] = by;

    // Bezdimenzionalna koherencija bisektora: ρ = 4*ax*by / (ax + by)^2
    const denom = ax + by;
    const rho = Math.abs(denom) > 1e-12 ? (4 * ax * by) / (denom * denom) : 0;
    rhoCoherence[k] = rho;

    // Sendvič projekcija filtrirana koherencijom
    sandwichChart[k] = ax * by * rho;
  }

  // RMS normalizacija izlaznog dijagrama
  let sumSq = 0;
  for (let k = 0; k < targetDim; k++) sumSq += sandwichChart[k] * sandwichChart[k];
  const norm = Math.sqrt(sumSq) || 1.0;
  for (let k = 0; k < targetDim; k++) sandwichChart[k] /= norm;

  return { topChart, bottomChart, rhoCoherence, sandwichChart };
}

// 2. Čitanje prvog vektora iz lm_head.weight sa diska
const filename = 'model-00018.safetensors';
const fd = fs.openSync(filename, 'r');

const headerBuf = Buffer.alloc(1024 * 1024);
const bytesRead = fs.readSync(fd, headerBuf, 0, 1024 * 1024, 0);
const { header, dataOffset } = parseSafetensorsHeader(headerBuf.subarray(0, bytesRead));

const tensorName = 'lm_head.weight';
const info = header[tensorName];
const [start] = info.data_offsets;

// Čitamo samo prvih 10.240 bajtova (1 vektor D=5120 u BF16)
const bytesToRead = 5120 * 2;
const tensorBuf = Buffer.alloc(bytesToRead);
fs.readSync(fd, tensorBuf, 0, bytesToRead, dataOffset + start);

const u16 = new Uint16Array(tensorBuf.buffer, tensorBuf.byteOffset, 5120);
const f32 = bfloat16ToFloat32(u16);

// 3. Pokretanje sendvič projekcije
const res = processBivectorSandwich(f32);

console.log('=== BIVECTOR SANDWICH PROJEKCIJA (Pravi lm_head.weight) ===');
console.log('Top Slice (prvih 5 kanala):       ', Array.from(res.topChart.subarray(0, 5)).map(n => n.toFixed(4)).join(', '));
console.log('Bottom Slice (prvih 5 kanala):    ', Array.from(res.bottomChart.subarray(0, 5)).map(n => n.toFixed(4)).join(', '));
console.log('Koherencija ρ (prvih 5 kanala):   ', Array.from(res.rhoCoherence.subarray(0, 5)).map(n => n.toFixed(4)).join(', '));
console.log('\n--- Sendvič Dijagram od 35 Polja (prvih 10 kanala) ---');
console.log(Array.from(res.sandwichChart.subarray(0, 10)).map(n => n.toFixed(4)).join(', ') + ' ...');

fs.closeSync(fd);