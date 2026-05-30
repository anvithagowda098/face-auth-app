/**
 * VerifyScreen.js
 * Main verification screen. Flow:
 *   1. Camera opens → face detected → liveness challenges
 *   2. All challenges passed → face embedding compared to gallery
 *   3. Result card shown → logged to SQLite → dismissed
 */

import React, { useState, useRef, useCallback, useEffect } from 'react';
import {
  View, Text, StyleSheet, TouchableOpacity,
  SafeAreaView, Alert, ActivityIndicator,
} from 'react-native';
import { Camera, useCameraDevice, useFrameProcessor } from 'react-native-vision-camera';
import LivenessOverlay from '../components/LivenessOverlay';
import ResultCard from '../components/ResultCard';
import { EdgeFaceEngine } from '../engine/EdgeFaceEngine';
import { pickChallenges } from '../engine/LivenessChallenge';
import { OfflineDB } from '../db/OfflineDB';

const CHALLENGE_TIMEOUT_MS = 8000;

export default function VerifyScreen({ route, navigation }) {
  const { workerId } = route?.params || { workerId: 'EMP_001' };
  const device = useCameraDevice('front');

  const [phase, setPhase]         = useState('ready');   // ready | liveness | verifying | done
  const [challenges, setChallenges] = useState([]);
  const [completedSet, setCompleted] = useState(new Set());
  const [currentIdx, setCurrentIdx] = useState(0);
  const [timeLeftMs, setTimeLeftMs] = useState(CHALLENGE_TIMEOUT_MS);
  const [result, setResult]        = useState(null);
  const [hasCamPerm, setHasCamPerm] = useState(false);

  const timerRef   = useRef(null);
  const cameraRef  = useRef(null);

  useEffect(() => {
    Camera.requestCameraPermission().then(s => setHasCamPerm(s === 'granted'));
  }, []);

  const startLiveness = useCallback(() => {
    const chal = pickChallenges(2);
    setChallenges(chal);
    setCompleted(new Set());
    setCurrentIdx(0);
    setTimeLeftMs(CHALLENGE_TIMEOUT_MS);
    setPhase('liveness');
    startTimer();
  }, []);

  const startTimer = useCallback(() => {
    clearInterval(timerRef.current);
    let remaining = CHALLENGE_TIMEOUT_MS;
    timerRef.current = setInterval(() => {
      remaining -= 200;
      setTimeLeftMs(remaining);
      if (remaining <= 0) {
        clearInterval(timerRef.current);
        Alert.alert('Timeout', 'Liveness check timed out. Please try again.',
          [{ text: 'Retry', onPress: () => startLiveness() }]);
        setPhase('ready');
      }
    }, 200);
  }, []);

  // Simulated: in production this is called from frame processor
  const simulateChallengeComplete = useCallback((challenge) => {
    setCompleted(prev => {
      const next = new Set(prev);
      next.add(challenge);
      setCurrentIdx(ci => {
        const newIdx = ci + 1;
        if (newIdx >= challenges.length) {
          clearInterval(timerRef.current);
          setTimeout(() => runVerify(), 600);
        }
        return newIdx;
      });
      return next;
    });
  }, [challenges]);

  const runVerify = useCallback(async () => {
    setPhase('verifying');
    try {
      // In production: capture frame from camera ref
      // const frame = await cameraRef.current.takePhoto();
      // const b64 = await RNFS.readFile(frame.path, 'base64');
      // const res = await EdgeFaceEngine.verify(b64, workerId);

      // Simulated for demo:
      const res = await EdgeFaceEngine.verify(null, workerId);

      setResult(res);
      setPhase('done');

      await OfflineDB.logAccess({
        workerId,
        matched:           res.ok,
        score:             res.score,
        livenessScore:     res.livenessScore,
        latencyMs:         res.latencyMs,
        faceQuality:       0.85,
        challengesCompleted: challenges,
      });
    } catch (e) {
      Alert.alert('Error', e.message);
      setPhase('ready');
    }
  }, [workerId, challenges]);

  const dismiss = useCallback(() => {
    setPhase('ready');
    setResult(null);
    setChallenges([]);
  }, []);

  if (!hasCamPerm) {
    return (
      <SafeAreaView style={styles.center}>
        <Text style={styles.permText}>Camera permission required</Text>
        <TouchableOpacity style={styles.btn} onPress={() => Camera.requestCameraPermission()}>
          <Text style={styles.btnText}>Grant Permission</Text>
        </TouchableOpacity>
      </SafeAreaView>
    );
  }

  if (!device) {
    return <SafeAreaView style={styles.center}><ActivityIndicator color="#00D97E" /></SafeAreaView>;
  }

  return (
    <View style={styles.root}>
      {/* Camera */}
      <Camera
        ref={cameraRef}
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={phase !== 'done'}
        photo={true}
      />

      {/* Header */}
      <SafeAreaView style={styles.header}>
        <TouchableOpacity onPress={() => navigation?.goBack()} style={styles.backBtn}>
          <Text style={styles.backText}>‹ Back</Text>
        </TouchableOpacity>
        <Text style={styles.workerLabel}>{workerId}</Text>
        <View style={styles.offlineBadge}>
          <Text style={styles.offlineText}>● OFFLINE</Text>
        </View>
      </SafeAreaView>

      {/* Liveness overlay */}
      {phase === 'liveness' && (
        <LivenessOverlay
          challenges={challenges}
          completedSet={completedSet}
          currentIndex={currentIdx}
          timeLeftMs={timeLeftMs}
        />
      )}

      {/* Simulate challenge buttons (for demo recording) */}
      {phase === 'liveness' && currentIdx < challenges.length && (
        <View style={styles.simBtns}>
          <Text style={styles.simLabel}>Demo: tap to simulate challenge</Text>
          <TouchableOpacity
            style={styles.simBtn}
            onPress={() => simulateChallengeComplete(challenges[currentIdx])}
          >
            <Text style={styles.simBtnText}>
              Complete: {challenges[currentIdx].replace('_', ' ').toUpperCase()}
            </Text>
          </TouchableOpacity>
        </View>
      )}

      {/* Verifying spinner */}
      {phase === 'verifying' && (
        <View style={styles.verifyingOverlay}>
          <ActivityIndicator size="large" color="#00D97E" />
          <Text style={styles.verifyingText}>Matching face…</Text>
          <Text style={styles.verifyingSubtext}>On-device • No network used</Text>
        </View>
      )}

      {/* Start button */}
      {phase === 'ready' && (
        <View style={styles.startArea}>
          <Text style={styles.startHint}>Position your face in the oval</Text>
          <TouchableOpacity style={styles.startBtn} onPress={startLiveness}>
            <Text style={styles.startBtnText}>BEGIN VERIFICATION</Text>
          </TouchableOpacity>
        </View>
      )}

      {/* Result card */}
      {phase === 'done' && result && (
        <ResultCard result={result} onDismiss={dismiss} />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  root:   { flex: 1, backgroundColor: '#000' },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#0A0A0A' },

  header: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    paddingHorizontal: 16, paddingTop: 8,
  },
  backBtn:  { padding: 8 },
  backText: { color: '#fff', fontSize: 17 },
  workerLabel: { color: '#fff', fontSize: 15, fontWeight: '600' },
  offlineBadge: { backgroundColor: 'rgba(0,0,0,0.5)', borderRadius: 8, paddingHorizontal: 8, paddingVertical: 4 },
  offlineText:  { color: '#FF9500', fontSize: 11, fontWeight: '700' },

  simBtns:   { position: 'absolute', bottom: 160, left: 20, right: 20, alignItems: 'center' },
  simLabel:  { color: 'rgba(255,255,255,0.5)', fontSize: 12, marginBottom: 8 },
  simBtn:    { backgroundColor: 'rgba(0,217,126,0.15)', borderWidth: 1, borderColor: '#00D97E',
               borderRadius: 12, paddingHorizontal: 24, paddingVertical: 12 },
  simBtnText:{ color: '#00D97E', fontSize: 14, fontWeight: '700' },

  verifyingOverlay: { ...StyleSheet.absoluteFillObject, justifyContent: 'center', alignItems: 'center',
                      backgroundColor: 'rgba(0,0,0,0.75)' },
  verifyingText:    { color: '#fff', fontSize: 18, fontWeight: '700', marginTop: 16 },
  verifyingSubtext: { color: '#666', fontSize: 12, marginTop: 4 },

  startArea:   { position: 'absolute', bottom: 60, left: 0, right: 0, alignItems: 'center' },
  startHint:   { color: 'rgba(255,255,255,0.5)', fontSize: 13, marginBottom: 16 },
  startBtn:    { backgroundColor: '#00D97E', borderRadius: 16, paddingHorizontal: 40, paddingVertical: 16 },
  startBtnText:{ color: '#000', fontSize: 16, fontWeight: '800', letterSpacing: 0.5 },

  permText: { color: '#fff', marginBottom: 20 },
  btn:      { backgroundColor: '#00D97E', borderRadius: 12, padding: 14 },
  btnText:  { color: '#000', fontWeight: '700' },
});
