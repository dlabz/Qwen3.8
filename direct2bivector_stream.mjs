// @ts-check

/**
 * @file direct2bivector_stream.mjs
 * Vanilla JS ES Module za striming sklapanje Qwen3.8-27B tenzora (D=5120) u 35-bitne bivektore.
 */

/**
 * @typedef {Object} TensorMeta
 * @property {string} dtype
 * @property {number[]} shape
 * @property {number[]} data_offsets
 */

/**
 * @typedef {Object} HeaderParseResult
 * @property {Record<string, TensorMeta>} header
 * @property {number} dataOffset
 */

/**
 * @typedef {Object} Direct2BivectorOptions
 * @property {string} modelUrl HTTP/HTTPS URL do `.safetensors` fajla
 * @property {number} [chunkSizeBytes] Veličina bloka za preuzimanje (default: 32MB)
 * @property {boolean} [useWebGPU] Omogućava WebGPU akceleraciju ako je dostupna
 * @property {number} [targetDim] Dimenzija bivektorskog dijagrama (default: 35)
 * @property {number} [hiddenDim] Skrivena dimenzija Qwen3.8-27B modela (default: 5120)
 */

// ============================================================================
// 1. WEBGPU WGSL COMPUTE SHADER (Zero-Copy GPU Reduction)
// ============================================================================

export const DIRECT_2_BIVECTOR_WGSL = /* wgsl */ `
struct Params {
    dim: u32,
    targetDim: u32,
    stride: u32,
    totalVectors: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> inputBF16: array<u32>; // U32 pakuje 2x BF16
@group(0) @binding(2) var<storage, read_write> outputCharts: array<f32>;

// Bitwise dekodiranje bfloat16 u float32 direktno na GPU
fn bf16_to_f32(raw_u16: u32) -> f32 {
    let u32_val = raw_u16 << 16u;
    return bitcast<f32>(u32_val);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let vector_idx = global_id.x;
    if (vector_idx >= params.totalVectors) {
        return;
    }

    let input_base_elements = vector_idx * params.dim;
    let output_base = vector_idx * params.targetDim;

    var sum_sq: f32 = 0.0;

    // Sklapanje D=5120 na 35 polja (stride = 5120 / 35 = 146 elemenata po polju)
    for (var k: u32 = 0u; k < params.targetDim; k = k + 1u) {
        let chunk_start = input_base_elements + k * params.stride;
        var chunk_sum: f32 = 0.0;

        for (var i: u32 = 0u; i < params.stride; i = i + 1u) {
            let elem_idx = chunk_start + i;
            let u32_idx = elem_idx / 2u;
            let is_high = (elem_idx % 2u) == 1u;
            let packed_val = inputBF16[u32_idx];
            
            var raw_bf16: u32;
            if (is_high) {
                raw_bf16 = packed_val >> 16u;
            } else {
                raw_bf16 = packed_val & 0xFFFFu;
            }

            chunk_sum = chunk_sum + bf16_to_f32(raw_bf16);
        }

        let mean_val = chunk_sum / f32(params.stride);
        outputCharts[output_base + k] = mean_val;
        sum_sq = sum_sq + (mean_val * mean_val);
    }

    // RMS Normalizacija dijagrama
    let norm = sqrt(sum_sq);
    let inv_norm = select(1.0, 1.0 / norm, norm > 1e-12);

    for (var k: u32 = 0u; k < params.targetDim; k = k + 1u) {
        outputCharts[output_base + k] = outputCharts[output_base + k] * inv_norm;
    }
}
`;

// ============================================================================
// 2. VANILLA JS PARSER ZAGLAVLJA I PREUZIMANJE
// ============================================================================

/**
 * Parsira Safetensors JSON zaglavlje iz uvodnog bajt bafera.
 * @param {Uint8Array} buffer 
 * @returns {HeaderParseResult}
 */
export function parseSafetensorsHeader(buffer) {
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  const headerLength = Number(view.getBigUint64(0, true));
  const jsonBytes = buffer.subarray(8, 8 + headerLength);
  const jsonText = new TextDecoder('utf-8').decode(jsonBytes);
  const header = JSON.parse(jsonText);

  return {
    header,
    dataOffset: 8 + headerLength
  };
}

/**
 * Brza CPU konverzija bfloat16 Uint16Array u Float32Array pomoću bitwise operacija.
 * @param {Uint16Array} uint16Array 
 * @returns {Float32Array}
 */
export function bfloat16ToFloat32CPU(uint16Array) {
  const float32 = new Float32Array(uint16Array.length);
  const u32View = new Uint32Array(float32.buffer);
  for (let i = 0; i < uint16Array.length; i++) {
    u32View[i] = uint16Array[i] << 16;
  }
  return float32;
}

/**
 * CPU fallback za sklapanje vektora D=5120 na 35 polja sa RMS normalizacijom.
 * @param {Float32Array} tensorData Pozadinsko stablo tenzora (dužina = numVectors * dim)
 * @param {number} numVectors Broj vektora u tenzoru
 * @param {number} [dim=5120] Skrivena dimenzija
 * @param {number} [targetDim=35]
 * @returns {Float32Array}
 */
export function processDirect2BivectorCPU(tensorData, numVectors, dim = 5120, targetDim = 35) {
  const stride = Math.max(1, Math.floor(dim / targetDim));
  const output = new Float32Array(numVectors * targetDim);

  for (let v = 0; v < numVectors; v++) {
    const vBaseInput = v * dim;
    const vBaseOutput = v * targetDim;

    let sumSq = 0;
    for (let k = 0; k < targetDim; k++) {
      const chunkBase = vBaseInput + k * stride;
      let chunkSum = 0;
      for (let i = 0; i < stride; i++) {
        chunkSum += tensorData[chunkBase + i] || 0;
      }
      const meanVal = chunkSum / stride;
      output[vBaseOutput + k] = meanVal;
      sumSq += meanVal * meanVal;
    }

    const norm = Math.sqrt(sumSq) || 1.0;
    for (let k = 0; k < targetDim; k++) {
      output[vBaseOutput + k] /= norm;
    }
  }

  return output;
}

// ============================================================================
// 3. WEBGPU COMPUTATION ENGINE
// ============================================================================

export class WebGPUBivectorEngine {
  /** @type {GPUDevice | null} */
  device = null;
  /** @type {GPUComputePipeline | null} */
  pipeline = null;

  async init() {
    if (typeof navigator === 'undefined' || !navigator.gpu) return false;
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) return false;
      this.device = await adapter.requestDevice();

      const shaderModule = this.device.createShaderModule({ code: DIRECT_2_BIVECTOR_WGSL });
      this.pipeline = await this.device.createComputePipeline({
        layout: 'auto',
        compute: { module: shaderModule, entryPoint: 'main' }
      });
      return true;
    } catch (e) {
      console.warn('WebGPU nedostupan, prelazak na CPU fallback:', e);
      return false;
    }
  }

  /**
   * Spušta sirove BF16 bajtove direktno na GPU i vraća sklopljene bivektore.
   * @param {Uint8Array} rawBytes 
   * @param {number} numVectors 
   * @param {number} [dim=5120] 
   * @param {number} [targetDim=35] 
   * @returns {Promise<Float32Array>}
   */
  async process(rawBytes, numVectors, dim = 5120, targetDim = 35) {
    if (!this.device || !this.pipeline) throw new Error('WebGPU engine nije inicijalizovan.');

    const device = this.device;
    const stride = Math.floor(dim / targetDim);

    const paramsArray = new Uint32Array([dim, targetDim, stride, numVectors]);
    const paramsBuffer = device.createBuffer({
      size: paramsArray.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(paramsBuffer, 0, paramsArray);

    const inputBuffer = device.createBuffer({
      size: rawBytes.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(inputBuffer, 0, rawBytes);

    const outputByteLength = numVectors * targetDim * 4;
    const outputBuffer = device.createBuffer({
      size: outputByteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    const stagingBuffer = device.createBuffer({
      size: outputByteLength,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    const bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuffer } },
        { binding: 1, resource: { buffer: inputBuffer } },
        { binding: 2, resource: { buffer: outputBuffer } }
      ]
    });

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(numVectors / 64), 1, 1);
    passEncoder.end();

    commandEncoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, outputByteLength);
    device.queue.submit([commandEncoder.finish()]);

    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(stagingBuffer.getMappedRange().slice(0));
    stagingBuffer.unmap();

    paramsBuffer.destroy();
    inputBuffer.destroy();
    outputBuffer.destroy();
    stagingBuffer.destroy();

    return result;
  }
}

// ============================================================================
// 4. STRIMING KOORDINATOR (Async Generator)
// ============================================================================


/**
 * Preuzima samo specificirani opseg bajtova sa servera bez učitavanja celog fajla
 * @param {string} url 
 * @param {number} startByte 
 * @param {number} endByte 
 */
async function fetchByteRange(url, startByte, endByte) {
  const res = await fetch(url, {
    headers: { 'Range': `bytes=${startByte}-${endByte}` }
  });
  if (!res.ok && res.status !== 206) {
    throw new Error(`HTTP Range request failed (${res.status})`);
  }
  return await res.arrayBuffer();
}

/**
 * Strimovanje bivektora uz pomoć HTTP Range čitanja (Niska potrošnja memorije)
 */
export async function* direct2bivectorStream({ modelUrl, targetDim = 35, hiddenDim = 5120 }) {
  // 1. Čitanje prva 8 bajtova za dužinu JSON zaglavlja
  const headerLenBuffer = await fetchByteRange(modelUrl, 0, 7);
  const headerLenView = new DataView(headerLenBuffer);
  const headerLen = Number(headerLenView.getBigUint64(0, true));

  // 2. Čitanje samo JSON zaglavlja sa opisom tenzora
  const jsonBuffer = await fetchByteRange(modelUrl, 8, 8 + headerLen - 1);
  const jsonText = new TextDecoder().decode(jsonBuffer);
  const headerJson = JSON.parse(jsonText);

  const dataOffsetBase = 8 + headerLen;

  // 3. Prolazak kroz tenzore i preuzimanje samo njihovih bajtova po potrebi
  for (const [tensorName, info] of Object.entries(headerJson)) {
    if (tensorName === '__metadata__') continue;

    const [start, end] = info.data_offsets;
    const absStart = dataOffsetBase + start;
    const absEnd = dataOffsetBase + end - 1;

    // Preuzimamo samo bajtove ovog konkretnog tenzora
    const tensorBuffer = await fetchByteRange(modelUrl, absStart, absEnd);
    const floatData = new Float32Array(tensorBuffer);

    const numVectors = Math.floor(floatData.length / hiddenDim) || 1;
    const bivectorChart = new Float32Array(numVectors * targetDim);

    // Bivektorska projekcija tenzora
    for (let v = 0; v < numVectors; v++) {
      const srcOffset = v * hiddenDim;
      const dstOffset = v * targetDim;
      for (let k = 0; k < targetDim; k++) {
        bivectorChart[dstOffset + k] = floatData[srcOffset + (k % hiddenDim)] || 0;
      }
    }

    yield { tensorName, numVectors, bivectorChart };
  }
}

