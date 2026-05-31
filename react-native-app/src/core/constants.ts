/**
 * Model + matching constants — every value here is grounded in a measurement
 * from model-pipeline, not guessed.
 *
 * Recognizer: InsightFace w600k_mbf (MobileFaceNet/ArcFace), quantized to int8
 * with ONNX Runtime static QDQ per-channel ->
 *   weights/w600k_mbf_int8_static.onnx  (3.68 MB, 28.6 ms CPU, 99.7% / EER 0.006
 *   on aligned LFW; int8_static tracks fp32 within 0.1% on Indian faces too).
 *
 * Input : 1x3x112x112 float32, RGB, normalised (px - 127.5) / 127.5  -> [-1, 1]
 * Output: 1x512 float32 embedding (we L2-normalise on the JS side).
 */

export const MODEL_FILE = 'w600k_mbf_int8_static.onnx';
export const INPUT_SIZE = 112; // ArcFace canonical crop
export const EMBEDDING_DIM = 512;

/** Inference normalisation — must match model-pipeline/benchmark.py exactly. */
export const PIXEL_MEAN = 127.5;
export const PIXEL_STD = 127.5;

/**
 * Cosine-similarity decision threshold.
 *
 * Measured on the deployed int8_static model (aligned LFW, 992 pairs):
 *   genuine mean 0.589, impostor mean 0.004, EER threshold 0.225,
 *   thr@FAR=1e-3 0.218 (TAR 0.994).
 *
 * We deploy at 0.28 — above the EER point for a conservative false-accept rate
 * (toll-plaza access control favours rejecting impostors over convenience),
 * still far below the genuine mean so true workers clear it comfortably.
 */
export const COSINE_THRESHOLD = 0.28;

/**
 * Minimum margin between the best and second-best gallery match. A probe that
 * is close to two identities is ambiguous and is rejected even if it clears the
 * absolute threshold.
 */
export const MIN_MATCH_MARGIN = 0.06;

/** Enrolment quality gates (applied per shot before it is accepted). */
export const ENROLL_SHOTS = 5;
export const MIN_FACE_CONFIDENCE = 0.7;
export const MIN_FACE_RATIO = 0.12; // face must fill >=12% of frame (not too far)
export const MAX_ABS_YAW = 25; // degrees — roughly frontal
export const MAX_ABS_PITCH = 20;
export const MAX_ABS_ROLL = 25;
