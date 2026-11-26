// @ts-check
import { spawn } from 'node:child_process';

console.log('====================================================');
console.log(' 🚀 POKRETANJE QWEN3.8-27B SWAN RING-0 RUNTIME-A');
console.log('====================================================\n');

// 1. Podizanje RElay.js na portu 9001
const relay = spawn('node', ['RElay.js'], { stdio: 'inherit' });

// 2. Podizanje Audio Worklet HTTP servera na portu 3000
const player = spawn('node', ['bivector_audio_player.mjs'], { stdio: 'inherit' });

process.on('SIGINT', () => {
  relay.kill();
  player.kill();
  process.exit();
});
