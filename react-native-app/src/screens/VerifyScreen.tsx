/**
 * VerifyScreen — live verification:
 *   acquire a good face -> run randomised liveness challenges -> capture an
 *   aligned crop -> embed (int8 ArcFace) -> match against the gallery -> log.
 *
 * If a workerId is passed it's a 1:1 check (badge-then-face); otherwise it's a
 * 1:N identify across the gallery.
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { View, StyleSheet } from 'react-native';
import { Camera } from 'react-native-vision-camera';

import FaceCamera, { type FaceCameraHandle } from '../camera/FaceCamera';
import FaceOverlay from '../components/FaceOverlay';
import LivenessGuide from '../components/LivenessGuide';
import ResultSheet from '../components/ResultSheet';
import { Screen, Text, Button, Icon } from '../ui';
import { palette, spacing } from '../theme';
import { FaceAuthService, type VerifyOutcome } from '../engine/FaceAuthService';
import { LivenessSession, pickChallenges } from '../engine/LivenessEngine';
import { MIN_FACE_CONFIDENCE, MIN_FACE_RATIO } from '../core/constants';
import type { FaceStatus } from '../camera/types';
import type { ScreenProps } from '../navigation';

type Props = ScreenProps<'Verify'>;
type Phase = 'searching' | 'liveness' | 'verifying' | 'done' | 'error';

export default function VerifyScreen({ route, navigation }: Props) {
  const workerId = route.params?.workerId;
  const camera = useRef<FaceCameraHandle>(null);

  const [granted, setGranted] = useState<boolean | null>(null);
  const [perm, setPerm] = useState<'unknown' | 'granted' | 'denied'>('unknown');
  const [phase, setPhase] = useState<Phase>('searching');
  const [hint, setHint] = useState('Position your face in the frame');
  const [outcome, setOutcome] = useState<VerifyOutcome | null>(null);
  const [livenessIdx, setLivenessIdx] = useState(0);
  const [errMsg, setErrMsg] = useState('');

  const challenges = useMemo(() => pickChallenges(2), []);
  const session = useRef(new LivenessSession(challenges));
  const busy = useRef(false);

  useEffect(() => {
    Camera.requestCameraPermission().then(s =>
      setPerm(s === 'granted' ? 'granted' : 'denied'),
    );
  }, []);

  const runVerify = useCallback(async () => {
    if (busy.current) return;
    busy.current = true;
    setPhase('verifying');
    setHint('Hold still — matching');
    try {
      const face = await camera.current!.capture(6000);
      const res = await FaceAuthService.verify(face, workerId, true);
      setOutcome(res);
      setGranted(res.matched);
      setPhase('done');
    } catch (e) {
      setErrMsg((e as Error).message);
      setPhase('error');
    }
  }, [workerId]);

  const onStatus = useCallback(
    (s: FaceStatus) => {
      if (phase === 'verifying' || phase === 'done' || phase === 'error') return;

      const hasFace = s.found && s.confidence >= MIN_FACE_CONFIDENCE;
      if (!hasFace) {
        setPhase('searching');
        setHint('Position your face in the frame');
        return;
      }
      if (s.faceRatio < MIN_FACE_RATIO) {
        setPhase('searching');
        setHint('Move a little closer');
        return;
      }

      // good face -> drive liveness
      if (phase === 'searching') setPhase('liveness');
      const p = session.current.feed(s);
      setLivenessIdx(p.index);
      setHint(p.prompt);
      if (p.done) runVerify();
    },
    [phase, runVerify],
  );

  const reset = useCallback(() => {
    session.current = new LivenessSession(challenges);
    busy.current = false;
    setOutcome(null);
    setGranted(null);
    setLivenessIdx(0);
    setPhase('searching');
    setHint('Position your face in the frame');
  }, [challenges]);

  if (perm === 'denied') {
    return (
      <Screen>
        <View style={styles.center}>
          <Icon name="face-scan" size={40} color={palette.textMuted} />
          <Text variant="h2" center>
            Camera access needed
          </Text>
          <Text variant="body" color={palette.textSecondary} center>
            Enable the camera in Settings to verify workers.
          </Text>
          <Button label="Back" variant="secondary" onPress={() => navigation.goBack()} />
        </View>
      </Screen>
    );
  }

  const readiness =
    phase === 'liveness' || phase === 'verifying'
      ? 'ready'
      : phase === 'error'
        ? 'error'
        : 'searching';

  return (
    <View style={styles.root}>
      <FaceCamera ref={camera} isActive={phase !== 'done'} onStatus={onStatus} />

      <FaceOverlay
        title={workerId ? `Verify ${workerId}` : 'Verify worker'}
        subtitle={workerId ? '1:1 identity check' : '1:N identify'}
        hint={phase === 'verifying' ? undefined : hint}
        readiness={readiness}
      />

      {/* back button */}
      <Button
        label="Back"
        icon="arrow-left"
        variant="ghost"
        full={false}
        onPress={() => navigation.goBack()}
        style={styles.back}
      />

      {/* liveness strip */}
      {phase === 'liveness' && (
        <View style={styles.bottom}>
          <LivenessGuide challenges={challenges} index={livenessIdx} />
        </View>
      )}

      {phase === 'verifying' && (
        <View style={styles.bottom}>
          <Text variant="label" color={palette.textSecondary} center>
            On-device · no network
          </Text>
        </View>
      )}

      {phase === 'error' && (
        <View style={styles.bottom}>
          <Text variant="body" color={palette.danger} center style={{ marginBottom: spacing.md }}>
            {errMsg}
          </Text>
          <Button label="Try again" onPress={reset} />
        </View>
      )}

      {phase === 'done' && outcome && (
        <ResultSheet
          outcome={outcome}
          onDismiss={() => {
            if (granted) navigation.goBack();
            else reset();
          }}
        />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#000' },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center', gap: spacing.lg },
  back: { position: 'absolute', top: spacing.xxl + 40, left: spacing.sm },
  bottom: { position: 'absolute', left: spacing.xl, right: spacing.xl, bottom: spacing.xxxl },
});
