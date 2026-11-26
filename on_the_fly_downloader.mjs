// @ts-check
import { parseSafetensorsHeader, bfloat16ToFloat32, foldChart35b } from '../qwen_converter.mjs';

const HF_BASE_URL = 'https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main';

/**
 * Preuzima tačan bajt-opseg sa HF CDN-a
 * @param {string} remotePath Putanja do šard fajla (npr. model-00001-of-00018.safetensors)
 * @param {number} [startByte]
 * @param {number} [endByte]
 * @returns {Promise<Uint8Array>}
 */
async function fetchBytes(remotePath, startByte, endByte) {
  const url = `${HF_BASE_URL}/${remotePath}`;
  /** @type {Record<string, string>} */
  const headers = {};
  if (startByte !== undefined && endByte !== undefined) {
    headers['Range'] = `bytes=${startByte}-${endByte}`;
  }

  const response = await fetch(url, { headers });
  if (!response.ok && response.status !== 206) {
    throw new Error(`HTTP ${response.status} pri preuzimanju ${remotePath}`);
  }

  const arrayBuffer = await response.arrayBuffer();
  return new Uint8Array(arrayBuffer);
}

/** Preuzima config.json */
async function getModelConfig() {
  console.log('[HF Stream] Preuzimanje config.json...');
  const bytes = await fetchBytes('config.json');
  return JSON.parse(new TextDecoder().decode(bytes));
}

/** Preuzima model.safetensors.index.json */
async function getWeightMap() {
  console.log('[HF Stream] Preuzimanje model.safetensors.index.json...');
  const bytes = await fetchBytes('model.safetensors.index.json');
  const indexData = JSON.parse(new TextDecoder().decode(bytes));
  return indexData.weight_map;
}

/**
 * Čita tačnu dužinu JSON zaglavlja (prvih 8 bajtova) i parsira zaglavlje šard fajla sa CDN-a
 * @param {string} shardFile 
 */
async function fetchShardHeader(shardFile) {
  // 1. Čitamo prvih 8 bajtova radi dobijanja dužine JSON zaglavlja (headerLen)
  const lenBytes = await fetchBytes(shardFile, 0, 7);
  const dataView = new DataView(lenBytes.buffer, lenBytes.byteOffset, 8);
  const headerLen = Number(dataView.getBigUint64(0, true)); // Little-Endian uint64

  // 2. Čitamo tačno headerLen bajtova za parsiranje zaglavlja šarda
  const headerJsonBytes = await fetchBytes(shardFile, 8, 8 + headerLen - 1);
  const headerText = new TextDecoder().decode(headerJsonBytes);
  const header = JSON.parse(headerText);

  // Početak sirovih binarnih podataka tenzora u ovom šardu
  const dataOffset = 8 + headerLen;

  return { header, dataOffset };
}

/**
 * Pronalazi puni ključ tenzora u weight_map-i prema podnizu naziva
 * @param {Record<string, string>} weightMap 
 * @param {string} partialName 
 */
function resolveTensorName(weightMap, partialName) {
  if (weightMap[partialName]) return partialName;
  const matches = Object.keys(weightMap).filter(k => k.includes(partialName));
  return matches.length > 0 ? matches : null;
}

/**
 * Strimuje 1 vektor tenzora direktno sa HF CDN-a
 * @param {Record<string, string>} weightMap 
 * @param {string} requestedName 
 * @param {number} hiddenDim 
 */
async function fetchTensorVectorOnTheFly(weightMap, requestedName, hiddenDim) {
  const tensorName = resolveTensorName(weightMap, requestedName);
  if (!tensorName) {
    const matches = Object.keys(weightMap).filter(k => k.includes('embed'));
    throw new Error(`Tenzor '${requestedName}' nije pronađen! Slični tenzori: ${JSON.stringify(matches)}`);
  }

  const shardFile = weightMap[tensorName];
  console.log(`\n[HF Stream] Tenzor '${tensorName}' se nalazi u: ${shardFile}`);

  // Čitamo zaglavlje šard fajla i saznajemo tačne offsete
  const { header, dataOffset } = await fetchShardHeader(shardFile);

  const info = header[tensorName];
  if (!info) {
    throw new Error(`Tenzor ${tensorName} ne postoji u zaglavlju šarda ${shardFile}`);
  }

  console.log(`Dimenzije tenzora (Shape):`, info.shape);
  const [start] = info.data_offsets;

  // Izračunavanje tačnog apsolutnog offset-a na Hugging Face CDN-u
  const absStart = dataOffset + start;
  const bytesToRead = hiddenDim * 2; // 1 vektor skrivene dimenzije u BF16
  const absEnd = absStart + bytesToRead - 1;

  console.log(`[HF Stream] HTTP Range zahtev na HF CDN: bytes=${absStart}-${absEnd} (${bytesToRead} bajtova)`);
  const rawBytes = await fetchBytes(shardFile, absStart, absEnd);

  const u16 = new Uint16Array(rawBytes.buffer, rawBytes.byteOffset, hiddenDim);
  const f32 = bfloat16ToFloat32(u16);
  return { f32, shape: info.shape, tensorName, shardFile };
}

async function main() {
  try {
    const config = await getModelConfig();
    const weightMap = await getWeightMap();

    const hiddenDim = config.hidden_size || 5120;
    console.log(`\n✓ Config: Arhitektura = ${config.architectures?.[0] || 'Qwen3'}, Skrivena dimenzija D = ${hiddenDim}`);
    console.log(`✓ Index map: Ukupno tenzora = ${Object.keys(weightMap).length}`);

    // Pretražujemo i preuzimamo prvi vektor embed tokena sa tačne mreže
    const result = await fetchTensorVectorOnTheFly(weightMap, 'embed_tokens', hiddenDim);
    const chart = foldChart35b(result.f32);

    console.log(`\n--- Generisan 35-bitni Bivektorski Dijagram za ${result.tensorName} ---`);
    console.log(Array.from(chart.subarray(0, 10)).map(n => n.toFixed(4)).join(', ') + ' ...');

  } catch (err) {
    console.error('Greška tokom strimovanja:', err.message);
  }
}

main();
