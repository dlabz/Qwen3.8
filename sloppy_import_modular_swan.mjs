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
