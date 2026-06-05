/**
 * Shared types for the face-recognition core.
 *
 * The core is split into pure-logic modules (geometry, align, preprocess,
 * match) that import nothing from react-native — so they are unit-testable on
 * plain Node — and a thin runtime module (embedder) that owns the
 * onnxruntime-react-native session.
 */

/** A 2-D point in image pixel coordinates. */
export interface Point {
  x: number;
  y: number;
}

import { Landmarks } from 'react-native-vision-camera-face-detector';

/** Output of the native frame-processor plugin for a single detected face. */
export interface DetectedFace {
  /** Aligned 112x112 RGB image, row-major, 3 bytes/pixel (0-255). */
  rgb: Uint8Array;
  /** Detected keypoints in the ORIGINAL frame, for overlay + quality checks. */
  landmarks: Landmarks;
  /** Estimated pitch / roll / yaw in degrees, for pose-quality gating. */
  pitchAngle: number;
  rollAngle: number;
  yawAngle: number;
}

/** A 512-d L2-normalised face embedding. */
export type Embedding = Float32Array;

/** One enrolled identity: its averaged template plus per-shot embeddings. */
export interface GalleryEntry {
  workerId: string;
  /** Mean of the enrolment shots, re-normalised. The matching template. */
  template: Embedding;
  enrolledAt: string;
  shots: number;
}

/** Result of matching a probe embedding against the gallery. */
export interface MatchResult {
  matched: boolean;
  /** Best worker id, or null if nothing cleared the threshold. */
  workerId: string | null;
  /** Cosine similarity to the best template, in [-1, 1]. */
  score: number;
  /** Cosine to the runner-up — margin = score - secondScore signals ambiguity. */
  secondScore: number;
  threshold: number;
}
