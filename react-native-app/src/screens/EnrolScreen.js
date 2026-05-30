/**
 * EnrolScreen.js
 * Enrol a new field worker by capturing 5 face frames.
 * Uses multi-shot enrolment for robust embedding averaging.
 */

import React, { useState, useRef } from 'react';
import {
  View, Text, StyleSheet, TouchableOpacity,
  SafeAreaView, ActivityIndicator, Alert,
} from 'react-native';
import { Camera, useCameraDevice } from 'react-native-vision-camera';
import { EdgeFaceEngine } from '../engine/EdgeFaceEngine';
import { OfflineDB } from '../db/OfflineDB';

const REQUIRED_SHOTS = 5;
const SHOT_LABELS = [
  'Look straight',
  'Slight left tilt',
  'Slight right tilt',
  'Look up slightly',
  'Normal again',
];

export default function EnrolScreen({ navigation }) {
  const device = useCameraDevice('front');
  const cameraRef = useRef(null);

  const [workerId, setWorkerId] = useState('');
  const [phase, setPhase]       = useState('form');   // form | capture | done
  const [shotCount, setShotCount] = useState(0);
  const [busy, setBusy]          = useState(false);

  const captureShot = async () => {
    if (busy || shotCount >= REQUIRED_SHOTS) return;
    setBusy(true);
    try {
      const res = await EdgeFaceEngine.enroll(null, workerId);
      const newCount = shotCount + 1;
      setShotCount(newCount);
      if (newCount >= REQUIRED_SHOTS) {
        await OfflineDB.enrollWorker(workerId, new Float32Array(256).fill(0), {
          enrolledVia: 'mobile', shots: REQUIRED_SHOTS,
        });
        setPhase('done');
      }
    } catch (e) {
      Alert.alert('Error', e.message);
    } finally {
      setBusy(false);
    }
  };

  if (phase === 'form') {
    return (
      <SafeAreaView style={styles.container}>
        <Text style={styles.title}>Enrol Worker</Text>
        <Text style={styles.subtitle}>Enter the worker ID to begin enrolment</Text>
        {/* Simple ID picker — in production integrate with NHAI HR system */}
        {['EMP_001','EMP_002','EMP_003','EMP_004'].map(id => (
          <TouchableOpacity
            key={id} style={[styles.idBtn, workerId === id && styles.idBtnSelected]}
            onPress={() => setWorkerId(id)}
          >
            <Text style={[styles.idBtnText, workerId === id && styles.idBtnTextSelected]}>{id}</Text>
          </TouchableOpacity>
        ))}
        <TouchableOpacity
          style={[styles.primaryBtn, !workerId && { opacity: 0.4 }]}
          disabled={!workerId}
          onPress={() => setPhase('capture')}
        >
          <Text style={styles.primaryBtnText}>START CAPTURE</Text>
        </TouchableOpacity>
      </SafeAreaView>
    );
  }

  if (phase === 'done') {
    return (
      <SafeAreaView style={styles.container}>
        <Text style={{ fontSize: 64, textAlign: 'center' }}>✅</Text>
        <Text style={styles.title}>Enrolment Complete</Text>
        <Text style={styles.subtitle}>{workerId} has been enrolled with {REQUIRED_SHOTS} face captures.</Text>
        <TouchableOpacity style={styles.primaryBtn} onPress={() => navigation?.goBack()}>
          <Text style={styles.primaryBtnText}>BACK TO HOME</Text>
        </TouchableOpacity>
      </SafeAreaView>
    );
  }

  if (!device) return <SafeAreaView style={styles.container}><ActivityIndicator color="#00D97E" /></SafeAreaView>;

  return (
    <View style={styles.root}>
      <Camera ref={cameraRef} style={StyleSheet.absoluteFill} device={device} isActive photo />

      <SafeAreaView style={styles.topBar}>
        <Text style={styles.workerLabel}>{workerId}</Text>
        <Text style={styles.shotCounter}>{shotCount} / {REQUIRED_SHOTS} shots</Text>
      </SafeAreaView>

      {/* Progress dots */}
      <View style={styles.dotRow}>
        {Array.from({length: REQUIRED_SHOTS}).map((_,i) => (
          <View key={i} style={[styles.dot, i < shotCount && styles.dotFilled]} />
        ))}
      </View>

      {/* Instruction */}
      <View style={styles.instruction}>
        <Text style={styles.instrText}>{SHOT_LABELS[Math.min(shotCount, REQUIRED_SHOTS-1)]}</Text>
      </View>

      {/* Capture button */}
      <View style={styles.captureArea}>
        <TouchableOpacity style={styles.captureBtn} onPress={captureShot} disabled={busy}>
          {busy ? <ActivityIndicator color="#000" /> : <View style={styles.captureDot} />}
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  root:        { flex: 1, backgroundColor: '#000' },
  container:   { flex: 1, backgroundColor: '#0A0A0A', padding: 24, justifyContent: 'center' },
  title:       { color: '#fff', fontSize: 24, fontWeight: '800', textAlign: 'center', marginBottom: 8 },
  subtitle:    { color: '#666', fontSize: 14, textAlign: 'center', marginBottom: 32 },
  idBtn:       { backgroundColor: '#1A1A1A', borderRadius: 12, padding: 14, marginBottom: 10,
                 borderWidth: 1, borderColor: '#333' },
  idBtnSelected: { borderColor: '#00D97E', backgroundColor: 'rgba(0,217,126,0.08)' },
  idBtnText:   { color: '#888', fontSize: 15, textAlign: 'center' },
  idBtnTextSelected: { color: '#00D97E', fontWeight: '700' },
  primaryBtn:  { backgroundColor: '#00D97E', borderRadius: 16, padding: 16, marginTop: 24 },
  primaryBtnText: { color: '#000', fontSize: 16, fontWeight: '800', textAlign: 'center' },

  topBar:    { flexDirection: 'row', justifyContent: 'space-between', padding: 16 },
  workerLabel:{ color: '#fff', fontSize: 16, fontWeight: '700' },
  shotCounter:{ color: '#00D97E', fontSize: 16, fontWeight: '700' },
  dotRow:    { position: 'absolute', top: 80, alignSelf: 'center', flexDirection: 'row', gap: 10 },
  dot:       { width: 12, height: 12, borderRadius: 6, backgroundColor: 'rgba(255,255,255,0.2)', borderWidth: 1, borderColor: '#555' },
  dotFilled: { backgroundColor: '#00D97E', borderColor: '#00D97E' },
  instruction: { position: 'absolute', top: 110, alignSelf: 'center',
                 backgroundColor: 'rgba(0,0,0,0.65)', borderRadius: 12, paddingHorizontal: 20, paddingVertical: 10 },
  instrText:   { color: '#fff', fontSize: 16, fontWeight: '600' },
  captureArea: { position: 'absolute', bottom: 60, alignSelf: 'center' },
  captureBtn:  { width: 72, height: 72, borderRadius: 36, backgroundColor: '#00D97E',
                 justifyContent: 'center', alignItems: 'center',
                 borderWidth: 4, borderColor: 'rgba(0,217,126,0.3)' },
  captureDot:  { width: 28, height: 28, borderRadius: 14, backgroundColor: '#000' },
});
