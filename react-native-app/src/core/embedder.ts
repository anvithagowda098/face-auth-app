/**
 * embedder.ts — the live TFLite session that runs the int8 recognizer
 * on-device, producing a 512-d L2-normalised embedding from an aligned face.
 *
 * This is the single module that touches the native runtime
 * (react-native-fast-tflite). Everything it depends on (preprocess, match) is
 * pure and unit-tested; this layer is a thin, well-typed wrapper so the
 * device-only surface stays small.
 *
 * Model bundling (Expo): the .tflite is a Metro asset (see metro.config.js adds
 * the 'tflite' extension) under assets/models/. We resolve it with expo-asset,
 * read the bytes with expo-file-system, and hand react-native-fast-tflite the
 * Uint8Array — this avoids any platform-specific file-path / scheme quirks.
 * Run `npm run bundle:model` once to copy the artifact into assets/models/.
 */

import { loadTensorflowModel } from 'react-native-fast-tflite';

import type { DetectedFace, Embedding } from './types';
import { EMBEDDING_DIM } from './constants';
import { toNCHW, INPUT_SHAPE } from './preprocess';
import { l2normalize } from './match';

// Static require so Metro bundles the model. Keep the path in sync with
// scripts/bundle-model.js (TARGET) — it copies the artifact here.
// eslint-disable-next-line @typescript-eslint/no-var-requires
const MODEL_ASSET = require('../../assets/models/model.tflite');
//const faceRecognition = await loadTensorflowModel(MODEL_ASSET, [])
//const model = faceRecognition.state === 'loaded' ? faceRecognition.model : undefined

//let _session: InferenceSession | null = null;
let _inputName = 'input.1';
let _outputName = 'embedding';

/**
 * Run the recognizer on one aligned face and return its unit embedding.
 * Caller must have run init(). Throws if the output shape is unexpected.
 */
export async function embed(face: DetectedFace): Promise<Embedding> {
  /*const input = toNCHW(face.rgb);
  const tensor = new Tensor('float32', input, INPUT_SHAPE as number[]);
  const outputs = await _session.run({ [_inputName]: tensor });
  const raw = outputs[_outputName]?.data as Float32Array | undefined;
  if (!raw || raw.length !== EMBEDDING_DIM) {
    throw new Error(
      `embedder.embed: expected ${EMBEDDING_DIM}-d output, got ${raw?.length ?? 'none'}`,
    );
  }
  // The model is not guaranteed to emit unit vectors; normalise so cosine == dot.
  return l2normalize(raw);*/
}

export const Embedder = { embed };
export default Embedder;
