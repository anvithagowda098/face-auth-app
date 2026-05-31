/**
 * embedder.ts — the live ONNX Runtime session that runs the int8 recognizer
 * on-device, producing a 512-d L2-normalised embedding from an aligned face.
 *
 * This is the single module that touches the native runtime
 * (onnxruntime-react-native). Everything it depends on (preprocess, match) is
 * pure and unit-tested; this layer is a thin, well-typed wrapper so the
 * device-only surface stays small.
 *
 * Model bundling:
 *   Android -> android/app/src/main/assets/<MODEL_FILE>
 *   iOS     -> added to the app bundle (Copy Bundle Resources)
 * See docs/INTEGRATION.md. The path is resolved by react-native-fs / the ORT
 * asset loader at init().
 */

import { InferenceSession, Tensor } from 'onnxruntime-react-native';
import { Platform } from 'react-native';

import type { DetectedFace, Embedding } from './types';
import { MODEL_FILE, EMBEDDING_DIM } from './constants';
import { toNCHW, INPUT_SHAPE } from './preprocess';
import { l2normalize } from './match';

let _session: InferenceSession | null = null;
let _inputName = 'input.1';
let _outputName = 'embedding';
let _loading: Promise<void> | null = null;

/**
 * Resolve the bundled model to an absolute path ORT can open. On Android the
 * asset is copied out of the APK on first run; on iOS it lives in the bundle.
 */
async function resolveModelPath(): Promise<string> {
  // react-native-fs is the de-facto way to reach bundled assets cross-platform.
  // Kept as a dynamic import so pure-logic tests never pull native modules.
  const RNFS = require('react-native-fs');
  if (Platform.OS === 'android') {
    const dest = `${RNFS.DocumentDirectoryPath}/${MODEL_FILE}`;
    const exists = await RNFS.exists(dest);
    if (!exists) {
      await RNFS.copyFileAssets(MODEL_FILE, dest); // from android assets/
    }
    return dest;
  }
  // iOS: bundled resource.
  return `${RNFS.MainBundlePath}/${MODEL_FILE}`;
}

/** Load the session once. Safe to call repeatedly (idempotent). */
export async function init(): Promise<void> {
  if (_session) return;
  if (_loading) return _loading;
  _loading = (async () => {
    const path = await resolveModelPath();
    const session = await InferenceSession.create(path, {
      // XNNPACK on Android / CoreML on iOS where available; CPU fallback.
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
