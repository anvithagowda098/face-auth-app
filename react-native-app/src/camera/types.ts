/**
 * types.ts — the shape the native frame-processor plugin returns to JS.
 *
 * Per frame the plugin returns a lightweight FaceStatus (used for the live
 * overlay, quality gating, and liveness). When called with { capture: true } it
 * additionally warps the face to the 112x112 ArcFace template and returns the
 * RGB bytes as base64 (`alignedRgbB64`), which we decode into a DetectedFace.
 */

import type { FaceLandmarks } from '../core/types';

export interface FaceStatus {
  /** Whether exactly one usable face is in frame. */
  found: boolean;
  confidence: number;
  faceRatio: number;
  yaw: number;
  pitch: number;
  roll: number;
  /** Normalised landmark positions (0..1 of frame) for drawing the overlay. */
  landmarks: FaceLandmarks | null;
  /** Liveness signals derived natively from the mesh (see LivenessEngine). */
  leftEyeOpen: number; // 0..1 probability eye is open
  rightEyeOpen: number;
  smiling: number; // 0..1
  /** Present only on a capture frame: base64 of 112*112*3 RGB bytes. */
  alignedRgbB64?: string;
}
