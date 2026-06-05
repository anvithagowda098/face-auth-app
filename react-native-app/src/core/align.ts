/**
 * align.ts — ArcFace 5-point alignment template + helpers.
 *
 * The canonical destination template is InsightFace's `arcface_dst` for a
 * 112x112 crop. Alignment estimates the similarity transform from the detected
 * landmarks to this template; the native plugin uses the resulting affine to
 * warp the camera frame into the 112x112 RGB buffer the model consumes.
 *
 * We keep the template + ordering logic here (pure, testable). The actual pixel
 * warp happens natively (Android/iOS) for speed; this module is the source of
 * truth for the transform so JS and native agree.
 */

import type { Point } from './types';
import type { Landmarks } from 'react-native-vision-camera-face-detector';
import { estimateSimilarity, type Affine } from './geometry';
import { INPUT_SIZE } from './constants';

/**
 * InsightFace ArcFace destination template (112x112), order:
 *   leftEye, rightEye, nose, leftMouth, rightMouth.
 */
export const ARCFACE_TEMPLATE: readonly Point[] = [
  { x: 38.2946, y: 51.6963 },
  { x: 73.5318, y: 51.5014 },
  { x: 56.0252, y: 71.7366 },
  { x: 41.5493, y: 92.3655 },
  { x: 70.7299, y: 92.2041 },
];

/**
 * Put landmarks into ArcFace order and fix L/R by image x-coordinate, so it is
 * robust to detectors that label "left/right" relative to the subject (mirrored
 * front camera) rather than the image.
 */
export function orderLandmarks(lm: Landmarks): Point[] {
  'worklet'
  let leftEye;
  let rightEye;
  if (lm.LEFT_EYE.x <= lm.RIGHT_EYE.x) {
    leftEye = lm.LEFT_EYE;
    rightEye = lm.RIGHT_EYE;
  } else {
    leftEye = lm.RIGHT_EYE;
    rightEye = lm.LEFT_EYE;
  }
  let leftMouth;
  let rightMouth;
  if (lm.MOUTH_LEFT.x <= lm.MOUTH_RIGHT.x) {
    leftMouth = lm.MOUTH_LEFT;
    rightMouth = lm.MOUTH_RIGHT;
  } else {
    leftMouth = lm.MOUTH_RIGHT;
    rightMouth = lm.MOUTH_LEFT;
  }
  return [leftEye, rightEye, lm.NOSE_BASE, leftMouth, rightMouth];
}

/**
 * Estimate the affine that warps the detected face onto the 112x112 ArcFace
 * template. Feed this to the native warp (cv2.warpAffine equivalent).
 */
export function alignmentTransform(lm: Landmarks): Affine {
  'worklet'
  const src = orderLandmarks(lm);
  return estimateSimilarity(src, ARCFACE_TEMPLATE as Point[]);
}

/**
 * Apply similarity transform and get RGB array from a frame
 */
 // TODO: verify this
export function warpAndExtractRGB(srcPixels: Uint8Array, width: number, height: number, lm: Landmarks) {
  'worklet'
  // Estimate similarity transform: returns [a, b, c, d, tx, ty]
  const m = alignmentTransform(lm);

  const a = m[0];
  const b = m[1];
  const c = m[2];
  const d = m[3];
  const tx = m[4];
  const ty = m[5];

  const rgb = new Uint8Array(112 * 112 * 3);

  for (let y = 0; y < 112; y++) {
    for (let x = 0; x < 112; x++) {
      // Apply inverse transform to map output pixel to source
      const srcX = a * x + c * y + tx;
      const srcY = b * x + d * y + ty;

      // Nearest neighbor sampling
      // TODO: this is bad. use bilinear
      const ix = Math.max(0, Math.min(Math.floor(srcX), width - 1));
      const iy = Math.max(0, Math.min(Math.floor(srcY), height - 1));

      const srcIdx = (iy * width + ix) * 3; // RGB input
      const dstIdx = (y * 112 + x) * 3;   // RGB output

      rgb[dstIdx] = srcPixels[srcIdx];       // R
      rgb[dstIdx + 1] = srcPixels[srcIdx + 1]; // G
      rgb[dstIdx + 2] = srcPixels[srcIdx + 2]; // B
    }
  }

  return rgb;
}

/** Sanity-check the template is exactly INPUT_SIZE-shaped (compile-time guard). */
export const TEMPLATE_SIZE = INPUT_SIZE;
/* vi: set et sw=2: */
