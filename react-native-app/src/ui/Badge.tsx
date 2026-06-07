/**
 * Badge.tsx — compact status pill (with optional leading dot). Used for the
 * offline indicator, sync state, and inline pass/fail tags.
 */

import React from 'react';
import { View, StyleSheet } from 'react-native';
import Text from './Text';
import { palette, radius, spacing } from '../theme';

type Tone = 'neutral' | 'accent' | 'success' | 'danger' | 'warning';

interface Props {
  label: string;
  tone?: Tone;
  dot?: boolean;
}

const TONES: Record<Tone, { fg: string; bg: string }> = {
  neutral: { fg: palette.textSecondary, bg: palette.surfaceAlt },
  accent: { fg: palette.accent, bg: palette.accentSoft },
  success: { fg: palette.success, bg: palette.successSoft },
  danger: { fg: palette.danger, bg: palette.dangerSoft },
  warning: { fg: palette.warning, bg: palette.warningSoft },
};

export default function Badge({ label, tone = 'neutral', dot }: Props) {
  const t = TONES[tone];
  return (
    <View style={[styles.pill, { backgroundColor: t.bg }]}>
      {dot && <View style={[styles.dot, { backgroundColor: t.fg }]} />}
      <Text variant="caption" color={t.fg} style={styles.label}>
        {label}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  pill: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs + 2,
    paddingHorizontal: spacing.sm + 2,
    paddingVertical: spacing.xs + 1,
    borderRadius: radius.pill,
    alignSelf: 'flex-start',
  },
  dot: { width: 6, height: 6, borderRadius: 3 },
  label: { textTransform: 'uppercase' },
});
