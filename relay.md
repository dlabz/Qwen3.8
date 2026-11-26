Evo brzog klijentskog testa **`client_test.mjs`** koji se povezuje na vaš **`RElay.js`** server (`ws://127.0.0.1:9001`), pakuje fiksni **64-bajtski binarni okvir sa `MAGIC_DOL0` zaglavljem** i meri latenciju obrade u milisekundama.

---

### 1. Kod za `client_test.mjs`

Kreirajte fajl **`client_test.mjs`** u istom direktorijumu:

```javascript
// @ts-check
import WebSocket from 'ws';

const MAGIC_DOL0 = 0x444F4C30;

const CMD = {
  NONE:            0,
  REBOOT:          1 << 0, // 0x01
  EMIT_WIDGET:     1 << 1, // 0x02
  COCHLEAR_TOGGLE: 1 << 2, // 0x04
  INJECT_PROMPT:   1 << 3, // 0x08
  INSPECT_ROSETTE: 1 << 4, // 0x10
};

console.log('Spajanje na Ring-0 Binary RElay (ws://127.0.0.1:9001)...');
const ws = new WebSocket('ws://127.0.0.1:9001');

ws.on('open', () => {
  console.log('✓ Konekcija uspostavljena!\n');

  // 1. Alokacija fiksnog 64-bajtskog bafera
  const buffer = new ArrayBuffer(64);
  const view = new DataView(buffer);

  // 2. Upisivanje polja po Ring-0 binarnom protokolu
  view.setUint32(0, MAGIC_DOL0, true);                              // Offset  0..3:  Magic (0x444F4C30)
  view.setUint32(4, CMD.INSPECT_ROSETTE | CMD.COCHLEAR_TOGGLE, true);  // Offset  4..7:  Cmd flags (0x14)
  view.setUint32(8, 4, true);                                       // Offset  8..11: Layer Index = 4
  view.setUint32(12, 155, true);                                    // Offset 12..15: Rosette ID = 155
  view.setFloat32(16, 1.234, true);                                 // Offset 16..19: Z depth = 1.234
  view.setFloat32(20, 0.05, true);                                  // Offset 20..23: Vz speed = 0.05
  view.setFloat32(24, 21.0, true);                                  // Offset 24..27: Center21 = 21.0
  view.setUint32(28, 100, true);                                    // Offset 28..31: Token ID = 100

  console.log('Slanje 64-bajtskog binarnog ram-a...');
  const t0 = performance.now();
  ws.send(buffer);

  ws.on('message', (data, isBinary) => {
    const elapsed = (performance.now() - t0).toFixed(3);

    if (!isBinary) {
      console.log(`[Tekstualna poruka]: ${data.toString('utf8')}`);
      return;
    }

    const replyBuf = ArrayBuffer.isView(data) 
      ? data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) 
      : data;
      
    const replyView = new DataView(replyBuf);

    console.log(`\n====================================================`);
    console.log(`✓ PRIMLJEN ZERO-COPY ODZIV U ${elapsed} ms!`);
    console.log(`====================================================`);
    console.log(` Magic Header  : 0x${replyView.getUint32(0, true).toString(16).toUpperCase()}`);
    console.log(` Cmd Flags     : 0x${replyView.getUint32(4, true).toString(16)}`);
    console.log(` Layer Index   : ${replyView.getUint32(8, true)}`);
    console.log(` Rosette ID    : ${replyView.getUint32(12, true)}`);
    console.log(` Novo Z (Depth): ${replyView.getFloat32(16, true).toFixed(4)}`);
    console.log(` Token ID      : ${replyView.getUint32(28, true)}`);
    console.log(` SWAN Bivektor : ${replyView.getFloat32(32, true).toFixed(4)} (Dominantni rotorski kanal)`);
    console.log(`====================================================\n`);

    ws.close();
  });
});

ws.on('error', (err) => {
  console.error('Greška pri konekciji:', err.message);
});
```

---

### 2. Pokretanje testiranja

Dok je `RElay.js` pokrenut u prvom terminalu (`node RElay.js`), u drugom terminalu pokrenite:

```bash
node client_test.mjs
```

### Očekivani rezultat:
Dobijate instant povratni odziv sa **vremenom obrade ispod 1 ms**, gde relej direktno dekodira rozetu #155, izračunava novu Z-dubinu i izvlači bivektorsku rezonancu Tokena #100 iz `.swan` baze bez opterećenja V8 memorije.

⚡ Možemo sada dodati i `mmap()` / Direct I/O sloj u C-u radi direktnog drajverskog povezivanja ove FIFO cevi sa VRAM memorijom bez ikakvog korisničkog posrednika.