/**
 * FaceOverlay.tsx — the camera HUD: a framing reticle whose colour reflects
 * capture readiness, plus a top scrim with the title/offline badge and a bottom
 * hint line. Drawn over the live camera; non-interactive.
 */

import React from 'react';
import { View, StyleSheet, Dimensions } from 'react-native';
import Svg, { Defs, Mask, Rect, Ellipse } from 'react-native-svg';
import { Text, Badge } from '../ui';
import { palette, spacing, radius } from '../theme';

const { width: W, height: H } = Dimensions.get('window');
const OVAL_W = W * 0.72;
const OVAL_H = OVAL_W * 1.3;
const OVAL_CY = H * 0.42;

type Readiness = 'idle' | 'searching' | 'ready' | 'error';

interface Props {
  title: string;
  subtitle?: string;
  hint?: string;
  readiness?: Readiness;
}

const READY_COLOR: Record<Readiness, string> = {
  idle: palette.borderStrong,
  searching: palette.warning,
  ready: palette.accent,
  error: palette.danger,
};

export default function FaceOverlay({ title, subtitle, hint, readiness = 'idle' }: Props) {
  const color = READY_COLOR[readiness];
  return (
    <View style={StyleSheet.absoluteFill} pointerEvents="none">
      {/* Dim everything outside the oval with an SVG mask. */}
      <Svg width={W} height={H} style={StyleSheet.absoluteFill}>
        <Defs>
          <Mask id="hole">
            <Rect x="0" y="0" width={W} height={H} fill="#fff" />
            <Ellipse cx={W / 2} cy={OVAL_CY} rx={OVAL_W / 2} ry={OVAL_H / 2} fill="#000" />
          </Mask>
        </Defs>
        <Rect x="0" y="0" width={W} height={H} fill={palette.scrim} mask="url(#hole)" />
        <Ellipse
          cx={W / 2}
          cy={OVAL_CY}
          rx={OVAL_W / 2}
          ry={OVAL_H / 2}
          stroke={color}
          strokeWidth={3}
          strokeDasharray={readiness === 'ready' ? undefined : '10 8'}
          fill="none"
        />
      </Svg>

      {/* Top bar */}
      <View style={styles.top}>
        <View>
          <Text variant="h2">{title}</Text>
          {subtitle ? (
            <Text variant="label" color={palette.textSecondary} style={{ marginTop: 2 }}>
              {subtitle}
            </Text>
          ) : null}
        </View>
        <Badge label="Offline" tone="warning" dot />
      </View>

      {/* Hint pinned below the oval */}
      {hint ? (
        <View style={[styles.hintWrap, { top: OVAL_CY + OVAL_H / 2 + spacing.xl }]}>
          <View style={[styles.hint, { borderColor: color }]}>
            <Text variant="bodyStrong" color={palette.text} center>
              {hint}
            </Text>
          </View>
        </View>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  top: {
    position: 'absolute',
    top: spacing.xxl,
    left: spacing.xl,
    right: spacing.xl,
    flexDirection: 'row',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
  },
  hintWrap: { position: 'absolute', left: 0, right: 0, alignItems: 'center' },
  hint: {
    backgroundColor: palette.scrim,
    borderWidth: 1,
    borderRadius: radius.pill,
    paddingHorizontal: spacing.xl,
    paddingVertical: spacing.md,
  },
});

export { OVAL_W, OVAL_H, OVAL_CY };
