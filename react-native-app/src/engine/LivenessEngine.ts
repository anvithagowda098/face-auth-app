/**
 * LivenessEngine.ts — active-liveness challenges driven by the signals the
 * native detector actually provides (ML Kit eye-open / smile probabilities and
 * head Euler angles). This replaces the previous version, which assumed a
 * 468-point MediaPipe mesh that the FAST detector does not emit.
 *
 * A challenge is a small state machine fed the per-frame Face. Randomising
 * the order of challenges, defeats simple photo/replay attacks.
 */

import {
  Face,
} from 'react-native-vision-camera-face-detector';
import * as Crypto from 'expo-crypto';

export type ChallengeId = 'blink_left_eye' | 'blink_right_eye' | 'smile' | 'turn_left' | 'turn_right';

export const CHALLENGE_PROMPT: Record<ChallengeId, string> = {
  blink_left_eye: 'Blink your left eye slowly',
  blink_right_eye: 'Blink your right eye slowly',
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

  feed(s: Face): boolean {
    'worklet'
    switch (this.id) {
      case 'blink_left_eye': {
        const eye = s.leftEyeOpenProbability
        if (eye < EYE_CLOSED) this.armed = true;
        return this.armed && eye > EYE_OPEN; // closed then re-opened
      }
      case 'blink_right_eye': {
        const eye = s.rightEyeOpenProbability
        if (eye < EYE_CLOSED) this.armed = true;
        return this.armed && eye > EYE_OPEN; // closed then re-opened
      }
      case 'smile':
        return s.smilingProbability >= SMILE_ON;
      case 'turn_left':
        return s.yawAngle > YAW_DEG;
      case 'turn_right':
        return s.yawAngle < -YAW_DEG;
    }
  }
}

export interface LivenessProgress {
  current: ChallengeId | null;
  index: number;
  challenges: ChallengeId[];
  done: boolean;
  prompt: string;
}

// Secure random integer in [0, max)
function secureRandomInt(max: number) {
    const maxUint32 = 0xFFFFFFFF;
    const limit = maxUint32 - (maxUint32 % max);

    while (true) {
        const bytes = Crypto.getRandomBytes(4);

        const value =
            (bytes[0] << 24) |
            (bytes[1] << 16) |
            (bytes[2] << 8) |
            bytes[3];

        const unsigned = value >>> 0; // convert to unsigned 32-bit

        if (unsigned < limit) {
            // TODO: use rejection sampling instead of modulo
            return unsigned % max;
        }
    }
}

export function shuffleSecure(arr: number[]) {
    const a = [...arr];

    for (let i = a.length - 1; i > 0; i--) {
        const j = secureRandomInt(i + 1);
        [a[i], a[j]] = [a[j], a[i]];
    }

    return a;
}

const ALL: ChallengeId[] = ['blink_left_eye', 'blink_right_eye', 'smile', 'turn_left', 'turn_right'];

export class LivenessSession {
  private detectors: Detector[];
  private idx = 0;
  private challenges = shuffleSecure(ALL);

  constructor() {
    this.detectors = this.challenges.map(c => new Detector(c));
  }

  /** Feed a frame; advances the sequence. Returns true if advanced. */
  feed(s: Face): boolean {
    'worklet'
    if (this.idx < this.detectors.length && this.detectors[this.idx].feed(s)) {
      this.idx++;
      return true;
    }
    return false;
  }

  progress(): LivenessProgress {
    const done = this.idx >= this.challenges.length;
    const current = done ? null : this.challenges[this.idx];
    return {
      current,
      index: this.idx,
      challenges: this.challenges,
      done,
      prompt: current ? CHALLENGE_PROMPT[current] : 'Liveness confirmed',
    };
  }
}
/* vi: set et sw=2: */
