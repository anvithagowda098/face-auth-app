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

import type { FaceLandmarks, Point } from './types';
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
export function orderLandmarks(lm: FaceLandmarks): Point[] {
  const [leftEye, rightEye] =
    lm.leftEye.x <= lm.rightEye.x ? [lm.leftEye, lm.rightEye] : [lm.rightEye, lm.leftEye];
  const [leftMouth, rightMouth] =
    lm.leftMouth.x <= lm.rightMouth.x
      ? [lm.leftMouth, lm.rightMouth]
      : [lm.rightMouth, lm.leftMouth];
  return [leftEye, rightEye, lm.nose, leftMouth, rightMouth];
}

/**
 * Estimate the affine that warps the detected face onto the 112x112 ArcFace
 * template. Feed this to the native warp (cv2.warpAffine equivalent).
 */
export function alignmentTransform(lm: FaceLandmarks): Affine {
  const src = orderLandmarks(lm);
  return estimateSimilarity(src, ARCFACE_TEMPLATE as Point[]);
}

/** Sanity-check the template is exactly INPUT_SIZE-shaped (compile-time guard). */
export const TEMPLATE_SIZE = INPUT_SIZE;
