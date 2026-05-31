/**
 * preprocess.ts — turn an aligned 112x112 RGB byte buffer into the model's
 * input tensor: 1x3x112x112 float32, NCHW, normalised (px - 127.5) / 127.5.
 *
 * Must match model-pipeline/benchmark.py:load_crop exactly (RGB order, mean/std,
 * channel-first layout) or embeddings will not match the calibrated thresholds.
 */

import { INPUT_SIZE, PIXEL_MEAN, PIXEL_STD } from './constants';

const HW = INPUT_SIZE * INPUT_SIZE;

/**
 * @param rgb row-major RGB, length 112*112*3, values 0..255.
 * @returns Float32Array length 3*112*112 in NCHW (R-plane, G-plane, B-plane).
 */
export function toNCHW(rgb: Uint8Array | Uint8ClampedArray): Float32Array {
  const expected = HW * 3;
  if (rgb.length !== expected) {
    throw new Error(`toNCHW: expected ${expected} bytes, got ${rgb.length}`);
  }
  const out = new Float32Array(expected);
  // De-interleave RGBRGB... -> RRR...GGG...BBB... and normalise in one pass.
  for (let i = 0; i < HW; i++) {
    const j = i * 3;
    out[i] = (rgb[j] - PIXEL_MEAN) / PIXEL_STD; // R plane
    out[HW + i] = (rgb[j + 1] - PIXEL_MEAN) / PIXEL_STD; // G plane
    out[2 * HW + i] = (rgb[j + 2] - PIXEL_MEAN) / PIXEL_STD; // B plane
  }
  return out;
}

/** Tensor shape for the model input, exported for the ORT session. */
export const INPUT_SHAPE: readonly number[] = [1, 3, INPUT_SIZE, INPUT_SIZE];
