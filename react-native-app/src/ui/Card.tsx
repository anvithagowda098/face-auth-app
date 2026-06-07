/**
 * Card.tsx — the standard surface container. One hairline border, soft
 * elevation, consistent radius. `tone` tints the border/background for state.
 */

import React from 'react';
import { View, StyleSheet, type ViewStyle, type StyleProp } from 'react-native';
import { palette, radius, spacing, elevation } from '../theme';

type Tone = 'default' | 'accent' | 'success' | 'danger';

interface Props {
  children: React.ReactNode;
  tone?: Tone;
  elevated?: boolean;
  style?: StyleProp<ViewStyle>;
}

const toneStyles: Record<Tone, ViewStyle> = {
  default: { backgroundColor: palette.surface, borderColor: palette.border },
  accent: { backgroundColor: palette.surface, borderColor: palette.accentBorder },
  success: { backgroundColor: palette.surface, borderColor: 'rgba(52,211,153,0.32)' },
  danger: { backgroundColor: palette.surface, borderColor: 'rgba(242,109,109,0.32)' },
};

export default function Card({ children, tone = 'default', elevated, style }: Props) {
  return (
    <View style={[styles.card, toneStyles[tone], elevated && elevation.card, style]}>
      {children}
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    borderRadius: radius.lg,
    borderWidth: StyleSheet.hairlineWidth * 2,
    padding: spacing.lg,
  },
});
