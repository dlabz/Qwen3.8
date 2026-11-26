// @ts-check
// Pokretanje: deno run --allow-net --allow-read RElay.js

const MAGIC_DOL0 = 0x444F4C30;

const CMD = {
  NONE:            0,
  REBOOT:          1 << 0,
  EMIT_WIDGET:     1 << 1,
  COCHLEAR_TOGGLE: 1 << 2,
  INJECT_PROMPT:   1 << 3,
  INSPECT_ROSETTE: 1 << 4,
};

const SWAN_FILE = 'qwen3.8_27b_bivectors.swan';
let bivectorDB = null;
let vocabSize = 0;

try {
  const fileBuf = Deno.readFileSync(SWAN_FILE);
  bivectorDB = new Float32Array(fileBuf.buffer, fileBuf.byteOffset, fileBuf.byteLength / 4);
  vocabSize = bivectorDB.length / 35;
  console.log(`[RElay Deno] SWAN bivektorski indeks spreman u memoriji (${vocabSize} tokena).`);
} catch (_e) {
  console.log(`[RElay Deno] Pasivni režim bez ${SWAN_FILE}`);
}

const replyBuffer = new ArrayBuffer(64);
const replyView = new DataView(replyBuffer);

console.log(`\n====================================================`);
console.log(` ⚡ DOLDOC64 × SWAN RING-0 DENO RELAY ACTIVE`);
console.log(` Listening on: ws://127.0.0.1:9001`);
console.log(` Protocol: Native Deno.upgradeWebSocket (64B ArrayBuffer)`);
console.log(`====================================================\n`);

Deno.serve({ port: 9001, hostname: '127.0.0.1' }, (req) => {
  if (req.headers.get("upgrade") !== "websocket") {
    return new Response("Ring-0 Deno Binary RElay Active", { status: 200 });
  }

  const { socket, response } = Deno.upgradeWebSocket(req);

  socket.onopen = () => console.log("[Deno RElay] Klijent konektovan.");

  socket.onmessage = (event) => {
    if (typeof event.data === "string") {
      console.log(`[Deno RElay Text Prompt]: ${event.data}`);
      return;
    }

    const buf = event.data;
    if (buf.byteLength < 64) return;

    const view = new DataView(buf);
    const magic = view.getUint32(0, true);
    if (magic !== MAGIC_DOL0) return;

    const cmdFlags  = view.getUint32(4, true);
    const layerIdx  = view.getUint32(8, true);
    const rosetteId = view.getUint32(12, true);
    const zDepth    = view.getFloat32(16, true);
    const vzSpeed   = view.getFloat32(20, true);
    const center21  = view.getFloat32(24, true);
    const tokenId   = view.getUint32(28, true);

    let topVal = 0;
    if (bivectorDB && tokenId < vocabSize) {
      topVal = bivectorDB[tokenId * 35];
    }

    // Upis 64B odziva bez alokacije novih objekata
    replyView.setUint32(0, MAGIC_DOL0, true);
    replyView.setUint32(4, cmdFlags, true);
    replyView.setUint32(8, layerIdx, true);
    replyView.setUint32(12, rosetteId, true);
    replyView.setFloat32(16, zDepth + vzSpeed * 0.016, true);
    replyView.setFloat32(20, vzSpeed, true);
    replyView.setFloat32(24, center21, true);
    replyView.setUint32(28, tokenId, true);
    replyView.setFloat32(32, topVal, true);

    socket.send(replyBuffer);
  };

  return response;
});
