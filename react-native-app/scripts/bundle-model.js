#!/usr/bin/env node
/**
 * bundle-model.js — copy the deployable int8 recognizer into assets/models/ so
 * Metro bundles it as an app asset (resolved at runtime via expo-asset in
 * src/core/embedder.ts). Run this BEFORE the first build/prebuild and any time
 * the model is re-exported:
 *
 *   npm run bundle:model
 *
 * Source of truth: model-pipeline/weights/w600k_mbf_int8_static.onnx
 * (3.68 MB, 99.7% / EER 0.006).
 */

const fs = require('fs');
const path = require('path');

const MODEL = 'w600k_mbf_int8_static.onnx';
const SRC = path.resolve(__dirname, '../../model-pipeline/weights', MODEL);
const DEST_DIR = path.resolve(__dirname, '../assets/models');
const DEST = path.join(DEST_DIR, MODEL);

if (!fs.existsSync(SRC)) {
  console.error(`✗ model not found at ${SRC}`);
  console.error('  Re-export it from model-pipeline (finalize_int8.py) first.');
  process.exit(1);
}

fs.mkdirSync(DEST_DIR, { recursive: true });
fs.copyFileSync(SRC, DEST);
const sizeMb = (fs.statSync(DEST).size / 1e6).toFixed(2);
console.log(`✓ assets/models/${MODEL} (${sizeMb} MB)`);
console.log('  Metro will bundle it; embedder.ts loads it via expo-asset.');
