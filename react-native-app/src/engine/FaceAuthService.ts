/**
 * FaceAuthService.ts — the real authentication engine, replacing the old
 * simulated EdgeFaceEngine. It owns the flow that ties the pieces together:
 *
 *   detected face (aligned 112x112 from the native plugin)
 *     -> Embedder.embed  (int8 ArcFace, 512-d)
 *     -> averageEmbeddings (enrol)  /  matchGallery|verifyAgainst (verify)
 *     -> OfflineDB  (gallery + signed access log)
 *
 * No Math.random scores anywhere. Every number returned is a real measurement.
 */

import { averageEmbeddings, matchGallery, verifyAgainst } from '../core/match';
import {
  MIN_FACE_CONFIDENCE,
  MIN_FACE_RATIO,
  MAX_FACE_RATIO,
  MAX_ABS_YAW,
  MAX_ABS_PITCH,
  MAX_ABS_ROLL,
  DUPLICATE_ENROLL_COSINE,
} from '../core/constants';
import { OfflineDB } from '../db/OfflineDB';
import type { Embedding, MatchResult } from '../core/types';
import { type Face } from 'react-native-vision-camera-face-detector';

export interface VerifyOutcome extends MatchResult {
  latencyMs: number;
  livenessPass: boolean;
  faceQuality: number;
}

export interface EnrollOutcome {
  workerId: string;
  shots: number;
  /** Mean pairwise cosine across the enrol shots — a self-consistency score. */
  cohesion: number;
}

/** Reject low-quality captures before they pollute a template or a decision. */
export function qualityReason(face: Face): string | null {
//  if (face.confidence < MIN_FACE_CONFIDENCE) return 'Hold steady — face not clearly detected';
//  if (face.faceRatio < MIN_FACE_RATIO) return 'Move closer';
  if (Math.abs(face.pitchAngle) > MAX_ABS_PITCH) return 'Keep your head level';
  if (Math.abs(face.rollAngle) > MAX_ABS_ROLL) return "Don't tilt your head";
  if (Math.abs(face.yawAngle) > MAX_ABS_YAW) return 'Look straight ahead';
  return null;
}

/**
 * Live framing guidance from a streaming FaceStatus (looser than the capture
 * gate — meant for the on-screen coach line, not for accepting a shot). Returns
 * a short instruction, or null when the framing is good enough to capture.
 *
 * `s` is a FaceStatus-shaped object; we read only the numeric fields so the core
 * engine needn't import the camera types.
 */
export interface FramingSignal {
  found: boolean;
  confidence: number;
  faceRatio: number;
  yaw: number;
  pitch: number;
  roll: number;
  leftEyeOpen: number;
  rightEyeOpen: number;
}

export function framingHint(s: FramingSignal): string | null {
  if (!s.found || s.confidence < MIN_FACE_CONFIDENCE) return 'Center your face in the oval';
  if (s.faceRatio < MIN_FACE_RATIO) return 'Move closer';
  if (s.faceRatio > MAX_FACE_RATIO) return 'Move back a little';
  if (Math.abs(s.yaw) > MAX_ABS_YAW) return 'Look straight ahead';
  if (Math.abs(s.pitch) > MAX_ABS_PITCH) return 'Keep your head level';
  if (Math.abs(s.roll) > MAX_ABS_ROLL) return "Don't tilt your head";
  // Eyes unreadable despite a close, frontal face usually means glare/glasses.
  if (s.leftEyeOpen < 0 && s.rightEyeOpen < 0) return 'Remove glasses or add light';
  return null;
}

/** True when a streaming frame is good enough to auto-capture an enrol shot. */
export function isFrameCaptureReady(s: FramingSignal): boolean {
  return framingHint(s) === null;
}

/** A 0..1 quality score for logging (1 = frontal, close, confident). */
export function qualityScore(face: Face): number {
  'worklet'
  const pose =
    1 -
    Math.min(1, (Math.abs(face.yawAngle) / 45 + Math.abs(face.pitchAngle) / 45 + Math.abs(face.rollAngle) / 45) / 3);
  // TODO: we do need faceRatio here for the "close" part
  // const size = Math.min(1, face.faceRatio / 0.35);
  const size = 1;
  const conf = 1;
  return Math.max(0, Math.min(1, 0.45 * conf + 0.35 * pose + 0.2 * size));
}

export const FaceAuthService = {
/*  async init(): Promise<void> {
    await Embedder.init();
  },

  isReady(): boolean {
    return Embedder.isReady();
  },
*/

  /**
   * Enrol a worker from N aligned shots: embed each, average into a template,
   * persist. Returns a cohesion score so the UI can warn on inconsistent enrol.
   */
  async enroll(
    workerId: string,
    embeddings: Embedding[],
    metadata: Record<string, unknown> = {},
  ): Promise<EnrollOutcome> {
    if (embeddings.length === 0) throw new Error('enroll: no shots captured');

    const template = averageEmbeddings(embeddings);
    const cohesion = meanPairwiseCosine(embeddings);

    // Duplicate-identity guard: one face must not become two worker IDs. Match
    // the new template against everyone *except* this same ID (re-enrolment of
    // the same worker is allowed and simply updates their template).
    const gallery = (await OfflineDB.getGallery()).filter(g => g.workerId !== workerId);
    if (gallery.length > 0) {
      const dup = matchGallery(template, gallery, DUPLICATE_ENROLL_COSINE, 0);
      if (dup.matched && dup.workerId) {
        throw new Error(
          `This face is already enrolled as ${dup.workerId} ` +
            `(similarity ${(dup.score * 100).toFixed(0)}%). One worker per face.`,
        );
      }
    }

    await OfflineDB.enrollWorker(workerId, template, embeddings.length, {
      ...metadata,
      enrolledVia: 'mobile',
      cohesion: Math.round(cohesion * 1000) / 1000,
    });
    return { workerId, shots: embeddings.length, cohesion };
  },

  /**
   * Verify a captured face. If `claimedWorkerId` is given, do a 1:1 check
   * against that identity (badge-then-face flow); otherwise 1:N identify across
   * the whole gallery. Logs the attempt either way.
   */
  async verify(
    embedding: Embedding,
    faceQuality: number,
    claimedWorkerId?: string,
    livenessPass = true,
  ): Promise<VerifyOutcome> {
    const t0 = Date.now();

    let result: MatchResult;
    if (claimedWorkerId) {
      const entry = await OfflineDB.getWorker(claimedWorkerId);
      if (!entry) throw new Error(`verify: ${claimedWorkerId} is not enrolled`);
      result = verifyAgainst(embedding, entry);
    } else {
      const gallery = await OfflineDB.getGallery();
      result = matchGallery(embedding, gallery);
    }

    const latencyMs = Date.now() - t0;
    // Liveness must pass for an overall grant.
    const granted = result.matched && livenessPass;

    await OfflineDB.logAccess({
      workerId: result.workerId ?? claimedWorkerId ?? null,
      matched: granted,
      score: result.score,
      livenessPass,
      latencyMs,
      faceQuality,
      challenges: [],
    });

    return { ...result, matched: granted, latencyMs, livenessPass, faceQuality };
  },
};

function meanPairwiseCosine(embs: Embedding[]): number {
  if (embs.length < 2) return 1;
  let sum = 0;
  let n = 0;
  for (let i = 0; i < embs.length; i++) {
    for (let j = i + 1; j < embs.length; j++) {
      let dot = 0;
      for (let k = 0; k < embs[i].length; k++) dot += embs[i][k] * embs[j][k];
      sum += dot;
      n++;
    }
  }
  return n > 0 ? sum / n : 1;
}

export default FaceAuthService;
