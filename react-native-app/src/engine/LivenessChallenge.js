/**
 * LivenessChallenge.js
 * Active liveness detection using MediaPipe Face Mesh landmarks.
 *
 * Challenges:
 *   blink      - Eye Aspect Ratio (EAR) < 0.22 for 2+ consecutive frames
 *   smile      - Mouth width / height ratio > 2.8
 *   turn_left  - Yaw angle > +18° (nose tip shifts right in image = face turns left)
 *   turn_right - Yaw angle < -18°
 *   nod        - Pitch change > 12° between 3 frames
 *
 * Landmark indices follow MediaPipe 468-point Face Mesh spec.
 */

// ─── EAR landmarks (left eye, right eye) ───────────────────────────────────
// Left eye:  p1=362,p2=385,p3=387,p4=263,p5=373,p6=380
// Right eye: p1=33, p2=160,p3=158,p4=133,p5=153,p6=144
const EYE_L = { p1:362,p2:385,p3:387,p4:263,p5:373,p6:380 };
const EYE_R = { p1:33, p2:160,p3:158,p4:133,p5:153,p6:144 };

// Mouth landmarks for smile
const MOUTH = { left:61, right:291, top:13, bottom:14 };

// Nose tip + eye centres for yaw
const NOSE_TIP   = 1;
const LEFT_EYE_C = 468;  // left eye centre (MediaPipe iris)
const RIGHT_EYE_C= 473;

function dist(a, b) {
  const dx = a.x - b.x, dy = a.y - b.y;
  return Math.sqrt(dx*dx + dy*dy);
}

function ear(lm, eye) {
  const p1=lm[eye.p1], p2=lm[eye.p2], p3=lm[eye.p3];
  const p4=lm[eye.p4], p5=lm[eye.p5], p6=lm[eye.p6];
  return (dist(p2,p6) + dist(p3,p5)) / (2.0 * dist(p1,p4) + 1e-6);
}

export class LivenessChallengeEngine {
  constructor() {
    this.reset();
  }

  reset() {
    this._earHistory   = [];   // last 5 EAR values
    this._yawHistory   = [];
    this._pitchHistory = [];
    this._blinkCount   = 0;
    this._smileDetected= false;
  }

  /**
   * Feed a MediaPipe landmarks array (468 points, each {x,y,z}).
   * Returns object with detected events this frame.
   */
  analyze(landmarks) {
    if (!landmarks || landmarks.length < 468) return {};

    const lm = landmarks;
    const events = {};

    // ── Blink (EAR) ────────────────────────────────────────────────────────
    const earVal = (ear(lm, EYE_L) + ear(lm, EYE_R)) / 2.0;
    this._earHistory.push(earVal);
    if (this._earHistory.length > 5) this._earHistory.shift();

    // Blink = EAR drops below threshold in any two consecutive frames
    for (let i = 1; i < this._earHistory.length; i++) {
      if (this._earHistory[i-1] > 0.25 && this._earHistory[i] < 0.22) {
        this._blinkCount++;
        events.blink = true;
        break;
      }
    }

    // ── Smile ───────────────────────────────────────────────────────────────
    const mouthW = dist(lm[MOUTH.left], lm[MOUTH.right]);
    const mouthH = dist(lm[MOUTH.top],  lm[MOUTH.bottom]) + 1e-6;
    const smileRatio = mouthW / mouthH;
    if (smileRatio > 2.8) {
      events.smile = true;
      this._smileDetected = true;
    }

    // ── Yaw (head turn) via nose-eye geometry ───────────────────────────────
    // Use nose tip x offset relative to midpoint of both eye centres
    const noseX   = lm[NOSE_TIP].x;
    let eyeMidX;
    if (lm[LEFT_EYE_C] && lm[RIGHT_EYE_C]) {
      eyeMidX = (lm[LEFT_EYE_C].x + lm[RIGHT_EYE_C].x) / 2.0;
    } else {
      // Fallback: eye corner midpoints
      eyeMidX = (lm[EYE_L.p1].x + lm[EYE_R.p1].x) / 2.0;
    }
    // Normalise to image width (landmarks are 0–1)
    const yawProxy = (noseX - eyeMidX) * 100;  // approx degrees
    this._yawHistory.push(yawProxy);
    if (this._yawHistory.length > 8) this._yawHistory.shift();

    if (yawProxy > 18)  events.turn_left  = true;
    if (yawProxy < -18) events.turn_right = true;

    // ── Nod (pitch) via nose-tip vertical movement ──────────────────────────
    const noseY = lm[NOSE_TIP].y;
    this._pitchHistory.push(noseY);
    if (this._pitchHistory.length > 6) this._pitchHistory.shift();
    if (this._pitchHistory.length >= 4) {
      const delta = Math.abs(
        this._pitchHistory[this._pitchHistory.length-1] -
        this._pitchHistory[this._pitchHistory.length-4]
      ) * 100;
      if (delta > 12) events.nod = true;
    }

    return events;
  }

  get blinkCount()    { return this._blinkCount; }
  get smileDetected() { return this._smileDetected; }
}

// ─── Challenge Sequencer ──────────────────────────────────────────────────────

const ALL_CHALLENGES = ['blink', 'smile', 'turn_left', 'turn_right'];
const CHALLENGE_LABELS = {
  blink:      '👁  Please BLINK',
  smile:      '😊  Please SMILE',
  turn_left:  '⬅️  Turn HEAD LEFT',
  turn_right: '➡️  Turn HEAD RIGHT',
};

export function pickChallenges(n = 2) {
  // Shuffle and pick n random challenges — different each session (anti-replay)
  const shuffled = [...ALL_CHALLENGES].sort(() => Math.random() - 0.5);
  return shuffled.slice(0, n);
}

export function challengeLabel(challenge) {
  return CHALLENGE_LABELS[challenge] || challenge;
}
