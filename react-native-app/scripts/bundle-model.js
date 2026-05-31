#!/usr/bin/env node
/**
 * bundle-model.js — copy the deployable int8 recognizer into the native asset
 * locations so it ships inside the app binary. Run after `npm run bootstrap`
 * (which generates android/ and ios/) and any time the model is re-exported.
 *
 *   node scripts/bundle-model.js
 *
 * Source of truth is model-pipeline/weights/w600k_mbf_int8_static.onnx — the
 * artifact chosen in the model-pipeline decision (3.68 MB, 99.7% / EER 0.006).
 */

const fs = require('fs');
const path = require('path');

const MODEL = 'w600k_mbf_int8_static.onnx';
const SRC = path.resolve(__dirname, '../../model-pipeline/weights', MODEL);

const TARGETS = [
  path.resolve(__dirname, '../android/app/src/main/assets', MODEL),
  path.resolve(__dirname, '../ios', MODEL),
];

if (!fs.existsSync(SRC)) {
  console.error(`✗ model not found at ${SRC}`);
  console.error('  Re-export it from model-pipeline (finalize_int8.py) first.');
  process.exit(1);
}

const sizeMb = (fs.statSync(SRC).size / 1e6).toFixed(2);
let copied = 0;
for (const dest of TARGETS) {
  const dir = path.dirname(dest);
  if (!fs.existsSync(dir)) {
    console.warn(`• skip ${dest} (dir missing — run npm run bootstrap first)`);
    continue;
  }
  fs.copyFileSync(SRC, dest);
  console.log(`✓ ${path.relative(path.resolve(__dirname, '..'), dest)} (${sizeMb} MB)`);
  copied++;
}

console.log(copied ? `\nBundled into ${copied} target(s).` : '\nNo targets written.');
console.log('iOS: also add the .onnx to the app target under "Copy Bundle Resources".');
