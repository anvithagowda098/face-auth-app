/**
 * LivenessEngine.ts — active-liveness challenges driven by the signals the
 * native detector actually provides (ML Kit eye-open / smile probabilities and
 * head Euler angles). This replaces the previous version, which assumed a
 * 468-point MediaPipe mesh that the FAST detector does not emit.
 *
 * A challenge is a small state machine fed the per-frame FaceStatus. Randomising
 * which challenges run, and their order, defeats simple photo/replay attacks.
 */

import type { FaceStatus } from '../camera/types';

export type ChallengeId = 'blink' | 'smile' | 'turn_left' | 'turn_right';

export const CHALLENGE_PROMPT: Record<ChallengeId, string> = {
  blink: 'Blink slowly',
  smile: 'Smile',
  turn_left: 'Turn your head left',
  turn_right: 'Turn your head right',
};

const EYE_CLOSED = 0.35;
const EYE_OPEN = 0.7;
const SMILE_ON = 0.7;
const YAW_DEG = 18;

/** One challenge's progress detector. Returns true once satisfied. */
class Detector {
  private armed = false; // for blink: have we seen eyes closed yet
  constructor(private id: ChallengeId) {}

  feed(s: FaceStatus): boolean {
    if (!s.found) return false;
    switch (this.id) {
      case 'blink': {
        const eye = Math.min(
          s.leftEyeOpen < 0 ? 1 : s.leftEyeOpen,
          s.rightEyeOpen < 0 ? 1 : s.rightEyeOpen,
        );
        if (eye < EYE_CLOSED) this.armed = true;
        return this.armed && eye > EYE_OPEN; // closed then re-opened
      }
      case 'smile':
        return s.smiling >= SMILE_ON;
      case 'turn_left':
        return s.yaw > YAW_DEG;
      case 'turn_right':
        return s.yaw < -YAW_DEG;
    }
  }
}

export interface LivenessProgress {
  current: ChallengeId | null;
  index: number;
  total: number;
  done: boolean;
  prompt: string;
}

export class LivenessSession {
  private detectors: Detector[];
  private idx = 0;

  constructor(public readonly challenges: ChallengeId[]) {
    this.detectors = challenges.map(c => new Detector(c));
  }

  /** Feed a frame; advances the sequence. Returns current progress. */
  feed(s: FaceStatus): LivenessProgress {
    if (this.idx < this.detectors.length && this.detectors[this.idx].feed(s)) {
      this.idx++;
    }
    return this.progress();
  }

  progress(): LivenessProgress {
    const done = this.idx >= this.challenges.length;
    const current = done ? null : this.challenges[this.idx];
    return {
      current,
      index: this.idx,
      total: this.challenges.length,
      done,
      prompt: current ? CHALLENGE_PROMPT[current] : 'Liveness confirmed',
    };
  }
}

const ALL: ChallengeId[] = ['blink', 'smile', 'turn_left', 'turn_right'];

/**
 * Pick `n` distinct challenges in a fresh order each session. `seed` lets tests
 * be deterministic; production passes a per-session value.
 */
export function pickChallenges(n = 2, seed = Date.now()): ChallengeId[] {
  const pool = [...ALL];
  let s = seed >>> 0;
  const out: ChallengeId[] = [];
  while (out.length < n && pool.length > 0) {
    s = (s * 1664525 + 1013904223) >>> 0;
    const i = s % pool.length;
    out.push(pool.splice(i, 1)[0]);
  }
  return out;
}
