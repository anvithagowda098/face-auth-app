/**
 * EdgeFaceEngine.js
 * JS-side wrapper for the native TFLite inference module.
 * On real device: calls NativeModules.EdgeFaceEngine
 * In web/demo mode: uses pure-JS simulation for UI development
 */

import { NativeModules, Platform } from 'react-native';
import { v4 as uuidv4 } from 'uuid';

const { EdgeFaceEngine: NativeEngine } = NativeModules;

// Cosine similarity threshold (from blueprint §5.3)
const COSINE_THRESHOLD = 0.62;
const LIVENESS_THRESHOLD = 0.78;

/**
 * Simulated inference result for development/demo mode.
 * Returns realistic timing distributions matching blueprint benchmarks.
 */
function simulateInference(mode, userId) {
  // Realistic timing for mid-range Android (SD678 class)
  const detectMs  = 35 + Math.random() * 30;   // 35–65 ms
  const livenessMs = 85 + Math.random() * 70;  // 85–155 ms
  const embedMs   = 380 + Math.random() * 150; // 380–530 ms
  const totalMs   = detectMs + livenessMs + embedMs;

  const score       = 0.72 + Math.random() * 0.22; // 0.72–0.94
  const livenessScore = 0.80 + Math.random() * 0.18;

  return {
    ok: score > COSINE_THRESHOLD && livenessScore > LIVENESS_THRESHOLD,
    score: parseFloat(score.toFixed(4)),
    livenessScore: parseFloat(livenessScore.toFixed(4)),
    latencyMs: parseFloat(totalMs.toFixed(1)),
    breakdown: {
      detectMs:   parseFloat(detectMs.toFixed(1)),
      livenessMs: parseFloat(livenessMs.toFixed(1)),
      embedMs:    parseFloat(embedMs.toFixed(1)),
    },
    eventId: uuidv4(),
    userId: userId || 'unknown',
    timestamp: new Date().toISOString(),
    mode,
    platform: Platform.OS,
    simulated: !NativeEngine,
  };
}

export const EdgeFaceEngine = {
  /**
   * Enrol a new identity from a camera frame.
   * @param {string} frameBase64 - JPEG frame as base64
   * @param {string} userId - unique worker ID
   * @returns {Promise<{ok, eventId, timestamp}>}
   */
  async enroll(frameBase64, userId) {
    if (NativeEngine) {
      return NativeEngine.enroll(frameBase64, userId);
    }
    // Simulated enrol
    await new Promise(r => setTimeout(r, 400)); // simulate processing
    return {
      ok: true,
      eventId: uuidv4(),
      userId,
      timestamp: new Date().toISOString(),
      simulated: true,
    };
  },

  /**
   * Verify a face against the enrolled gallery.
   * @param {string} frameBase64 - JPEG frame as base64
   * @param {string} userId - expected worker ID
   * @returns {Promise<FaceAuthResult>}
   */
  async verify(frameBase64, userId) {
    if (NativeEngine) {
      return NativeEngine.verify(frameBase64, userId);
    }
    await new Promise(r => setTimeout(r, 620)); // simulate ~620ms latency
    return simulateInference('verify', userId);
  },

  /**
   * Run liveness challenge verification.
   * @param {string[]} completedChallenges - e.g. ['blink', 'turn_right']
   * @param {string[]} required - challenges that must be completed
   * @returns {boolean}
   */
  validateChallenges(completedChallenges, required) {
    return required.every(c => completedChallenges.includes(c));
  },

  /**
   * Check if native engine is loaded (real device) or simulated.
   */
  isNative() {
    return !!NativeEngine;
  },
};

export default EdgeFaceEngine;
