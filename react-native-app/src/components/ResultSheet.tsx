/**
 * ResultSheet.tsx — verification outcome as a bottom sheet. Shows the verdict,
 * the matched identity, the cosine score relative to the decision threshold (so
 * the result is legible, not a black box), and the real latency.
 */

import React, { useEffect, useRef } from 'react';
import { Animated, Easing, StyleSheet, View } from 'react-native';
import { Text, Icon, Button } from '../ui';
import { palette, spacing, radius, elevation } from '../theme';
import type { VerifyOutcome } from '../engine/FaceAuthService';

interface Props {
  outcome: VerifyOutcome;
  onDismiss: () => void;
}

export default function ResultSheet({ outcome, onDismiss }: Props) {
  const slide = useRef(new Animated.Value(40)).current;
  const fade = useRef(new Animated.Value(0)).current;
  const granted = outcome.matched;
  const accent = granted ? palette.success : palette.danger;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(slide, { toValue: 0, duration: 320, easing: Easing.out(Easing.cubic), useNativeDriver: true }),
      Animated.timing(fade, { toValue: 1, duration: 260, useNativeDriver: true }),
    ]).start();
  }, [fade, slide]);

  // Position the score on a track spanning [0, 1] cosine, marking the threshold.
  const scorePct = clamp01(outcome.score) * 100;
  const thrPct = clamp01(outcome.threshold) * 100;

  return (
    <Animated.View style={[styles.sheet, { opacity: fade, transform: [{ translateY: slide }] }]}>
      <View style={styles.handle} />

      <View style={styles.header}>
        <View style={[styles.verdictIcon, { backgroundColor: granted ? palette.successSoft : palette.dangerSoft }]}>
          <Icon name={granted ? 'check-circle' : 'x-circle'} size={28} color={accent} strokeWidth={2} />
        </View>
        <View style={{ flex: 1 }}>
          <Text variant="h1" color={accent}>
            {granted ? 'Access granted' : 'Access denied'}
          </Text>
          <Text variant="label" color={palette.textSecondary}>
            {outcome.workerId ? outcome.workerId : granted ? 'Verified' : 'No matching identity'}
          </Text>
        </View>
      </View>

      {/* Score vs threshold track */}
      <View style={styles.scoreBlock}>
        <View style={styles.scoreLabels}>
          <Text variant="caption" color={palette.textMuted}>
            MATCH SCORE
          </Text>
          <Text variant="monoSm" color={palette.text} mono>
            {outcome.score.toFixed(3)}
          </Text>
        </View>
        <View style={styles.track}>
          <View style={[styles.fill, { width: `${scorePct}%`, backgroundColor: accent }]} />
          <View style={[styles.threshold, { left: `${thrPct}%` }]} />
        </View>
        <Text variant="caption" color={palette.textMuted} style={{ marginTop: spacing.xs }}>
          Decision threshold {outcome.threshold.toFixed(2)}
        </Text>
      </View>

      <View style={styles.metrics}>
        <Metric label="LATENCY" value={`${outcome.latencyMs}`} unit="ms" />
        <Metric label="LIVENESS" value={outcome.livenessPass ? 'Pass' : 'Fail'} color={outcome.livenessPass ? palette.success : palette.danger} />
        <Metric label="QUALITY" value={`${Math.round(outcome.faceQuality * 100)}`} unit="%" />
      </View>

      <Button label="Done" onPress={onDismiss} variant={granted ? 'primary' : 'secondary'} />
    </Animated.View>
  );
}

function Metric({ label, value, unit, color = palette.text }: { label: string; value: string; unit?: string; color?: string }) {
  return (
    <View style={styles.metric}>
      <Text variant="caption" color={palette.textMuted}>
        {label}
      </Text>
      <View style={{ flexDirection: 'row', alignItems: 'baseline', gap: 2 }}>
        <Text variant="monoSm" color={color} mono>
          {value}
        </Text>
        {unit ? (
          <Text variant="caption" color={palette.textMuted}>
            {unit}
          </Text>
        ) : null}
      </View>
    </View>
  );
}

const clamp01 = (v: number) => Math.max(0, Math.min(1, v));

const styles = StyleSheet.create({
  sheet: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: palette.surface,
    borderTopLeftRadius: radius.xl,
    borderTopRightRadius: radius.xl,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderColor: palette.border,
    padding: spacing.xl,
    paddingBottom: spacing.xxl,
    gap: spacing.xl,
    ...elevation.sheet,
  },
  handle: {
    alignSelf: 'center',
    width: 38,
    height: 4,
    borderRadius: 2,
    backgroundColor: palette.borderStrong,
  },
  header: { flexDirection: 'row', alignItems: 'center', gap: spacing.lg },
  verdictIcon: {
    width: 52,
    height: 52,
    borderRadius: radius.md,
    alignItems: 'center',
    justifyContent: 'center',
  },
  scoreBlock: { gap: spacing.sm },
  scoreLabels: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  track: {
    height: 8,
    borderRadius: 4,
    backgroundColor: palette.surfaceHi,
    overflow: 'visible',
    justifyContent: 'center',
  },
  fill: { position: 'absolute', left: 0, top: 0, bottom: 0, borderRadius: 4 },
  threshold: {
    position: 'absolute',
    width: 2,
    top: -3,
    bottom: -3,
    backgroundColor: palette.textSecondary,
  },
  metrics: { flexDirection: 'row', justifyContent: 'space-between' },
  metric: { gap: spacing.xs },
});
