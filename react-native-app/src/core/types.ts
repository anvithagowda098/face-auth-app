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

/**
 * The five ArcFace keypoints, in canonical order:
 *   leftEye, rightEye, nose, leftMouth, rightMouth
 * "left"/"right" are relative to the IMAGE (not the subject), i.e. leftEye has
 * the smaller x. The native detector plugin normalises to this convention.
 */
export interface FaceLandmarks {
  leftEye: Point;
  rightEye: Point;
  nose: Point;
  leftMouth: Point;
  rightMouth: Point;
}

/** Output of the native frame-processor plugin for a single detected face. */
export interface DetectedFace {
  /** Aligned 112x112 RGB image, row-major, 3 bytes/pixel (0-255). */
  rgb: Uint8Array;
  /** Detected keypoints in the ORIGINAL frame, for overlay + quality checks. */
  landmarks: FaceLandmarks;
  /** Detector confidence 0..1. */
  confidence: number;
  /** Fraction of the frame the face bbox occupies (proxy for distance). */
  faceRatio: number;
  /** Estimated yaw / pitch / roll in degrees, for pose-quality gating. */
  yaw: number;
  pitch: number;
  roll: number;
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
