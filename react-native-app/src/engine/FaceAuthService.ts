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

import Embedder from '../core/embedder';
import { averageEmbeddings, matchGallery, verifyAgainst } from '../core/match';
import {
  MIN_FACE_CONFIDENCE,
  MIN_FACE_RATIO,
  MAX_ABS_YAW,
  MAX_ABS_PITCH,
  MAX_ABS_ROLL,
} from '../core/constants';
import { OfflineDB } from '../db/OfflineDB';
import type { DetectedFace, Embedding, MatchResult } from '../core/types';

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
export function qualityReason(face: DetectedFace): string | null {
  if (face.confidence < MIN_FACE_CONFIDENCE) return 'Hold steady — face not clearly detected';
  if (face.faceRatio < MIN_FACE_RATIO) return 'Move closer';
  if (Math.abs(face.yaw) > MAX_ABS_YAW) return 'Look straight ahead';
  if (Math.abs(face.pitch) > MAX_ABS_PITCH) return 'Keep your head level';
  if (Math.abs(face.roll) > MAX_ABS_ROLL) return "Don't tilt your head";
  return null;
}

/** A 0..1 quality score for logging (1 = frontal, close, confident). */
function qualityScore(face: DetectedFace): number {
  const conf = Math.min(1, face.confidence);
  const pose =
    1 -
    Math.min(1, (Math.abs(face.yaw) / 45 + Math.abs(face.pitch) / 45 + Math.abs(face.roll) / 45) / 3);
  const size = Math.min(1, face.faceRatio / 0.35);
  return Math.max(0, Math.min(1, 0.45 * conf + 0.35 * pose + 0.2 * size));
}

export const FaceAuthService = {
  async init(): Promise<void> {
    await Embedder.init();
  },

  isReady(): boolean {
    return Embedder.isReady();
  },

  /**
   * Enrol a worker from N aligned shots: embed each, average into a template,
   * persist. Returns a cohesion score so the UI can warn on inconsistent enrol.
   */
  async enroll(
    workerId: string,
    faces: DetectedFace[],
    metadata: Record<string, unknown> = {},
  ): Promise<EnrollOutcome> {
    if (faces.length === 0) throw new Error('enroll: no shots captured');
    const embeddings: Embedding[] = [];
    for (const f of faces) embeddings.push(await Embedder.embed(f));

    const template = averageEmbeddings(embeddings);
    const cohesion = meanPairwiseCosine(embeddings);

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
    face: DetectedFace,
    claimedWorkerId?: string,
    livenessPass = true,
  ): Promise<VerifyOutcome> {
    const t0 = Date.now();
    const probe = await Embedder.embed(face);

    let result: MatchResult;
    if (claimedWorkerId) {
      const entry = await OfflineDB.getWorker(claimedWorkerId);
      if (!entry) throw new Error(`verify: ${claimedWorkerId} is not enrolled`);
      result = verifyAgainst(probe, entry);
    } else {
      const gallery = await OfflineDB.getGallery();
      result = matchGallery(probe, gallery);
    }

    const latencyMs = Date.now() - t0;
    const faceQuality = qualityScore(face);
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
