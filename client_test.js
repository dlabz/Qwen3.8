// @ts-check
// Pokretanje: deno run --allow-net client_test.js

const MAGIC_DOL0 = 0x444F4C30;

console.log('Spajanje na Deno Ring-0 Binary RElay (ws://127.0.0.1:9001)...');
const ws = new WebSocket('ws://127.0.0.1:9001');
ws.binaryType = 'arraybuffer';

ws.onopen = () => {
  console.log('✓ Konekcija uspostavljena preko nativnog Deno WebSocket-a!\n');

  const buffer = new ArrayBuffer(64);
  const view = new DataView(buffer);

  view.setUint32(0, MAGIC_DOL0, true);
  view.setUint32(4, 0x14, true); // INSPECT_ROSETTE | COCHLEAR_TOGGLE
  view.setUint32(8, 4, true);    // Layer Index 4
  view.setUint32(12, 155, true); // Rosette ID 155
  view.setFloat32(16, 1.234, true);
  view.setFloat32(20, 0.05, true);
  view.setFloat32(24, 21.0, true);
  view.setUint32(28, 100, true);  // Token ID 100

  const t0 = performance.now();
  ws.send(buffer);

  ws.onmessage = (event) => {
    const elapsed = (performance.now() - t0).toFixed(3);
    const replyView = new DataView(event.data);

    console.log(`====================================================`);
    console.log(`✓ PRIMLJEN ZERO-COPY ODZIV U DENO-U U ${elapsed} ms!`);
    console.log(`====================================================`);
    console.log(` Magic Header  : 0x${replyView.getUint32(0, true).toString(16).toUpperCase()}`);
    console.log(` Cmd Flags     : 0x${replyView.getUint32(4, true).toString(16)}`);
    console.log(` Layer Index   : ${replyView.getUint32(8, true)}`);
    console.log(` Rosette ID    : ${replyView.getUint32(12, true)}`);
    console.log(` Novo Z (Depth): ${replyView.getFloat32(16, true).toFixed(4)}`);
    console.log(` Token ID      : ${replyView.getUint32(28, true)}`);
    console.log(` SWAN Bivektor : ${replyView.getFloat32(32, true).toFixed(4)}`);
    console.log(`====================================================\n`);

    ws.close();
  };
};
