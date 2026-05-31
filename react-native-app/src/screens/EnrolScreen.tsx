/**
 * EnrolScreen — capture N aligned shots of a worker, embed each, average into a
 * template, and persist. Quality is gated per shot so a bad frame never enters
 * the template; the enrol cohesion score is surfaced at the end.
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { View, StyleSheet, TextInput, Pressable } from 'react-native';
import { Camera } from 'react-native-vision-camera';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';

import FaceCamera, { type FaceCameraHandle } from '../camera/FaceCamera';
import FaceOverlay from '../components/FaceOverlay';
import { Screen, Text, Button, Icon, ProgressRing } from '../ui';
import { palette, spacing, radius } from '../theme';
import { FaceAuthService, qualityReason, type EnrollOutcome } from '../engine/FaceAuthService';
import { ENROLL_SHOTS } from '../core/constants';
import type { DetectedFace } from '../core/types';
import type { FaceStatus } from '../camera/types';
import type { RootStackParamList } from '../navigation';

type Props = NativeStackScreenProps<RootStackParamList, 'Enrol'>;
type Phase = 'form' | 'capture' | 'saving' | 'done' | 'error';

const SHOT_PROMPTS = [
  'Look straight ahead',
  'Turn slightly left',
  'Turn slightly right',
  'Tilt head up slightly',
  'Look straight again',
];

export default function EnrolScreen({ navigation }: Props) {
  const camera = useRef<FaceCameraHandle>(null);
  const [phase, setPhase] = useState<Phase>('form');
  const [workerId, setWorkerId] = useState('');
  const [shots, setShots] = useState<DetectedFace[]>([]);
  const [hint, setHint] = useState('');
  const [live, setLive] = useState(false); // a usable face is in frame
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<EnrollOutcome | null>(null);
  const [errMsg, setErrMsg] = useState('');

  useEffect(() => {
    if (phase === 'capture') Camera.requestCameraPermission();
  }, [phase]);

  const onStatus = useCallback((s: FaceStatus) => {
    setLive(s.found && s.confidence > 0.6 && s.faceRatio > 0.12);
  }, []);

  const captureShot = useCallback(async () => {
    if (busy) return;
    setBusy(true);
    try {
      const face = await camera.current!.capture(5000);
      const reason = qualityReason(face);
      if (reason) {
        setHint(reason);
        setBusy(false);
        return;
      }
      const next = [...shots, face];
      setShots(next);
      setHint('');
      if (next.length >= ENROLL_SHOTS) {
        setPhase('saving');
        const outcome = await FaceAuthService.enroll(workerId.trim(), next);
        setResult(outcome);
        setPhase('done');
      }
    } catch (e) {
      setErrMsg((e as Error).message);
      setPhase('error');
    } finally {
      setBusy(false);
    }
  }, [busy, shots, workerId]);

  // ── form ──────────────────────────────────────────────────────────────────
  if (phase === 'form') {
    const valid = workerId.trim().length >= 3;
    return (
      <Screen>
        <Pressable onPress={() => navigation.goBack()} style={styles.formBack} hitSlop={8}>
          <Icon name="arrow-left" size={20} color={palette.textSecondary} />
          <Text variant="label" color={palette.textSecondary}>
            Back
          </Text>
        </Pressable>

        <View style={styles.formBody}>
          <Text variant="overline" color={palette.accent}>
            NEW ENROLMENT
          </Text>
          <Text variant="display" style={{ marginBottom: spacing.xs }}>
            Enrol worker
          </Text>
          <Text variant="body" color={palette.textSecondary} style={{ marginBottom: spacing.xxl }}>
            We'll capture {ENROLL_SHOTS} angles to build a robust face template. Nothing leaves the device.
          </Text>

          <Text variant="label" color={palette.textSecondary} style={{ marginBottom: spacing.sm }}>
            WORKER ID
          </Text>
          <TextInput
            value={workerId}
            onChangeText={setWorkerId}
            placeholder="e.g. NHAI-DEL-0427"
            placeholderTextColor={palette.textMuted}
            autoCapitalize="characters"
            autoCorrect={false}
            style={styles.input}
          />
        </View>

        <Button
          label="Start capture"
          icon="face-scan"
          disabled={!valid}
          onPress={() => setPhase('capture')}
        />
      </Screen>
    );
  }

  // ── done ──────────────────────────────────────────────────────────────────
  if (phase === 'done' && result) {
    const strong = result.cohesion >= 0.5;
    return (
      <Screen>
        <View style={styles.center}>
          <View style={styles.doneIcon}>
            <Icon name="check-circle" size={40} color={palette.success} strokeWidth={2} />
          </View>
          <Text variant="h1" center>
            {result.workerId} enrolled
          </Text>
          <Text variant="body" color={palette.textSecondary} center>
            {result.shots} shots · template cohesion {(result.cohesion * 100).toFixed(0)}%
          </Text>
          {!strong && (
            <Text variant="label" color={palette.warning} center>
              Cohesion is low — consider re-enrolling in better lighting.
            </Text>
          )}
          <View style={styles.doneActions}>
            <Button label="Done" onPress={() => navigation.goBack()} />
          </View>
        </View>
      </Screen>
    );
  }

  if (phase === 'error') {
    return (
      <Screen>
        <View style={styles.center}>
          <Icon name="alert" size={36} color={palette.danger} />
          <Text variant="h2" center>
            Enrolment failed
          </Text>
          <Text variant="body" color={palette.textSecondary} center>
            {errMsg}
          </Text>
          <Button label="Back" variant="secondary" onPress={() => navigation.goBack()} />
        </View>
      </Screen>
    );
  }

  // ── capture / saving ──────────────────────────────────────────────────────
  const progress = shots.length / ENROLL_SHOTS;
  const prompt = SHOT_PROMPTS[Math.min(shots.length, SHOT_PROMPTS.length - 1)];

  return (
    <View style={styles.root}>
      <FaceCamera ref={camera} isActive={phase === 'capture'} onStatus={onStatus} />
      <FaceOverlay
        title={workerId}
        subtitle={`Capture ${shots.length} / ${ENROLL_SHOTS}`}
        hint={hint || prompt}
        readiness={live ? 'ready' : 'searching'}
      />

      <View style={styles.captureBar}>
        <ProgressRing progress={progress} size={84} stroke={5}>
          <Pressable
            onPress={captureShot}
            disabled={busy || phase === 'saving' || !live}
            style={({ pressed }) => [
              styles.shutter,
              (!live || busy) && styles.shutterDisabled,
              pressed && { transform: [{ scale: 0.94 }] },
            ]}
          >
            <Icon name="face-scan" size={26} color={palette.textInverse} strokeWidth={2} />
          </Pressable>
        </ProgressRing>
        <Text variant="label" color={palette.textSecondary} center style={{ marginTop: spacing.md }}>
          {phase === 'saving' ? 'Building template…' : live ? 'Tap to capture' : 'Looking for a face…'}
        </Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#000' },
  formBack: { flexDirection: 'row', alignItems: 'center', gap: spacing.xs, marginBottom: spacing.xl },
  formBody: { flex: 1 },
  input: {
    backgroundColor: palette.surfaceAlt,
    borderWidth: 1,
    borderColor: palette.border,
    borderRadius: radius.md,
    paddingHorizontal: spacing.lg,
    height: 52,
    color: palette.text,
    fontSize: 16,
    letterSpacing: 0.5,
  },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center', gap: spacing.md },
  doneIcon: {
    width: 72,
    height: 72,
    borderRadius: radius.lg,
    backgroundColor: palette.successSoft,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: spacing.sm,
  },
  doneActions: { alignSelf: 'stretch', marginTop: spacing.xl },
  captureBar: { position: 'absolute', bottom: spacing.xxxl, left: 0, right: 0, alignItems: 'center' },
  shutter: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: palette.accent,
    alignItems: 'center',
    justifyContent: 'center',
  },
  shutterDisabled: { backgroundColor: palette.borderStrong },
});
