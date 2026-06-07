/**
 * EnrolScreen — frontal auto-capture. The worker holds still and looks at the
 * camera; once framing is good the app grabs ENROLL_SHOTS aligned shots
 * automatically (spaced by ENROLL_SHOT_INTERVAL_MS for a little natural
 * variety), embeds each, averages into a template, and persists. Every shot is
 * quality-gated, and FaceAuthService.enroll blocks a face that is already
 * enrolled under another ID. No fake "turn left/right" prompts — ArcFace
 * alignment normalises pose, so honest frontal shots give the best template.
 */

import { useCallback, useState } from 'react';
import { View, StyleSheet, TextInput, Pressable } from 'react-native';
import { useCameraPermission } from 'react-native-vision-camera';

import FaceCamera from '../camera/FaceCamera';
import FaceOverlay from '../components/FaceOverlay';
import { Screen, Text, Button, Icon, ProgressRing } from '../ui';
import { palette, spacing, radius } from '../theme';
import {
  FaceAuthService,
  type EnrollOutcome,
} from '../engine/FaceAuthService';
import { ENROLL_SHOTS } from '../core/constants';
import type { DetectedFace } from '../core/types';
import type { ScreenProps } from '../navigation';
import { type LivenessProgress } from '../engine/LivenessEngine';
import { type Embedding } from '../core/types';
import { useSQLiteContext } from 'expo-sqlite';

type Props = ScreenProps<'Enrol'>;
type Phase = 'form' | 'capture' | 'saving' | 'done' | 'error';

export default function EnrolScreen({ navigation }: Props) {
  const db = useSQLiteContext();
  const [phase, setPhase] = useState<Phase>('form');
  const [workerId, setWorkerId] = useState('');
  const [shots, setShots] = useState<DetectedFace[]>([]);
  const [hint, setHint] = useState('Center your face in the oval');
  const [result, setResult] = useState<EnrollOutcome | null>(null);
  const [errMsg, setErrMsg] = useState('');
//  const [livenessIdx, setLivenessIdx] = useState(0);

  const runVerify = useCallback(async (embeddings: Embedding[]) => {
    setHint('Hold still — matching');
    try {
      console.log(`Enrolling workerId ${workerId}`);
      const outcome = await FaceAuthService.enroll(db, workerId, embeddings);
      setResult(outcome);
      setPhase('done');
    } catch (e) {
      setErrMsg((e as Error).message);
      setPhase('error');
    }
  }, [workerId]);

  const onLivenessProgress = (p: LivenessProgress, embeddings?: Embedding[], faceQuality?: number) => {
  //    setLivenessIdx(p.index);
      setHint(p.prompt);
      if (p.done && embeddings !== undefined && faceQuality !== undefined) runVerify(embeddings);
  };

  const { hasPermission } = useCameraPermission();
  if (!hasPermission) {
    return (
      <Screen>
        <View style={styles.center}>
          <Icon name="face-scan" size={40} color={palette.textMuted} />
          <Text variant="h2" center>
            Camera access needed
          </Text>
          <Text variant="body" color={palette.textSecondary} center>
            Enable the camera in Settings to enroll workers.
          </Text>
          <Button label="Back" variant="secondary" onPress={() => navigation.goBack()} />
        </View>
      </Screen>
    );
  }

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
            Look straight at the camera and hold still — we'll auto-capture {ENROLL_SHOTS} shots to
            build a robust face template. Nothing leaves the device.
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
            {result.shots} shots · template cohesion {result.cohesion.toFixed(0)}%
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

  return (
    <View style={styles.root}>
      <FaceCamera isActive={phase === 'capture'} onLivenessProgress={onLivenessProgress} />
      <FaceOverlay
        title={workerId}
        subtitle={`Captured ${shots.length} / ${ENROLL_SHOTS}`}
        hint={phase !== 'capture' ? undefined : hint}
        readiness='ready'
      />

      <View style={styles.captureBar}>
        <ProgressRing progress={progress} size={96} stroke={6}>
          <View style={styles.count}>
            <Text variant="h1">{shots.length}</Text>
            <Text variant="label" color={palette.textSecondary}>
              / {ENROLL_SHOTS}
            </Text>
          </View>
        </ProgressRing>
        <Text variant="label" color={palette.textSecondary} center style={{ marginTop: spacing.md }}>
          {phase === 'saving'
            ? 'Building template…'
            : true
              ? 'Auto-capturing — hold still'
              : 'Center your face to begin'}
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
  count: { alignItems: 'center', justifyContent: 'center' },
});
/* vi: set et sw=2: */
