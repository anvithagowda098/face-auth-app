/**
 * embedder.ts — the live ONNX Runtime session that runs the int8 recognizer
 * on-device, producing a 512-d L2-normalised embedding from an aligned face.
 *
 * This is the single module that touches the native runtime
 * (onnxruntime-react-native). Everything it depends on (preprocess, match) is
 * pure and unit-tested; this layer is a thin, well-typed wrapper so the
 * device-only surface stays small.
 *
 * Model bundling (Expo): the .onnx is a Metro asset (see metro.config.js adds
 * the 'onnx' extension) under assets/models/. We resolve it with expo-asset,
 * read the bytes with expo-file-system, and hand ORT the Uint8Array — this
 * avoids any platform-specific file-path / scheme quirks. Run
 * `npm run bundle:model` once to copy the artifact into assets/models/.
 */

import { InferenceSession, Tensor } from 'onnxruntime-react-native';
import { Platform } from 'react-native';
import { Asset } from 'expo-asset';
import * as FileSystem from 'expo-file-system';

import type { DetectedFace, Embedding } from './types';
import { EMBEDDING_DIM } from './constants';
import { toNCHW, INPUT_SHAPE } from './preprocess';
import { l2normalize } from './match';

// Static require so Metro bundles the model. Keep the path in sync with
// scripts/bundle-model.js (TARGET) — it copies the artifact here.
// eslint-disable-next-line @typescript-eslint/no-var-requires
const MODEL_ASSET = require('../../assets/models/w600k_mbf_int8_static.onnx');

let _session: InferenceSession | null = null;
let _inputName = 'input.1';
let _outputName = 'embedding';
let _loading: Promise<void> | null = null;

function base64ToBytes(b64: string): Uint8Array {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
  const clean = b64.replace(/=+$/, '');
  const out = new Uint8Array(Math.floor((clean.length * 3) / 4));
  let bits = 0;
  let acc = 0;
  let p = 0;
  for (let i = 0; i < clean.length; i++) {
    const idx = chars.indexOf(clean[i]!);
    if (idx < 0) continue;
    acc = (acc << 6) | idx;
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out[p++] = (acc >> bits) & 0xff;
    }
  }
  return p === out.length ? out : out.subarray(0, p);
}

/** Resolve + read the bundled model into a byte buffer ORT can load directly. */
async function loadModelBytes(): Promise<Uint8Array> {
  const asset = Asset.fromModule(MODEL_ASSET);
  if (!asset.downloaded) await asset.downloadAsync();
  const uri = asset.localUri ?? asset.uri;
  const b64 = await FileSystem.readAsStringAsync(uri, {
    encoding: FileSystem.EncodingType.Base64,
  });
  return base64ToBytes(b64);
}

/** Load the session once. Safe to call repeatedly (idempotent). */
export async function init(): Promise<void> {
  if (_session) return;
  if (_loading) return _loading;
  _loading = (async () => {
    const bytes = await loadModelBytes();
    const session = await InferenceSession.create(bytes, {
      // CoreML on iOS / XNNPACK on Android where available; CPU fallback.
      executionProviders:
        Platform.OS === 'ios' ? ['coreml', 'cpu'] : ['xnnpack', 'cpu'],
      graphOptimizationLevel: 'all',
      intraOpNumThreads: 2,
    });
    _session = session;
    _inputName = session.inputNames[0] ?? _inputName;
    _outputName = session.outputNames[0] ?? _outputName;
  })();
  try {
    await _loading;
  } finally {
    _loading = null;
  }
}

export function isReady(): boolean {
  return _session !== null;
}

/**
 * Run the recognizer on one aligned face and return its unit embedding.
 * Caller must have run init(). Throws if the output shape is unexpected.
 */
export async function embed(face: DetectedFace): Promise<Embedding> {
  if (!_session) {
    throw new Error('embedder.embed: session not initialised — call init() first');
  }
  const input = toNCHW(face.rgb);
  const tensor = new Tensor('float32', input, INPUT_SHAPE as number[]);
  const outputs = await _session.run({ [_inputName]: tensor });
  const raw = outputs[_outputName]?.data as Float32Array | undefined;
  if (!raw || raw.length !== EMBEDDING_DIM) {
    throw new Error(
      `embedder.embed: expected ${EMBEDDING_DIM}-d output, got ${raw?.length ?? 'none'}`,
    );
  }
  // The model is not guaranteed to emit unit vectors; normalise so cosine == dot.
  return l2normalize(raw);
}

/** Release the session (e.g. on logout / low-memory). */
export async function dispose(): Promise<void> {
  if (_session) {
    await _session.release();
    _session = null;
  }
}

export const Embedder = { init, isReady, embed, dispose };
export default Embedder;
