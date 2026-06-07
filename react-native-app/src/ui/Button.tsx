/**
 * Button.tsx — primary / secondary / ghost / danger actions with a leading
 * icon slot, press feedback, busy + disabled states. One component, no per-
 * screen restyling.
 */

import React from 'react';
import {
  Pressable,
  ActivityIndicator,
  StyleSheet,
  View,
  type ViewStyle,
  type StyleProp,
} from 'react-native';
import Text from './Text';
import Icon, { type IconName } from './Icon';
import { palette, radius, spacing } from '../theme';

type Variant = 'primary' | 'secondary' | 'ghost' | 'danger';

interface Props {
  label: string;
  onPress?: () => void;
  variant?: Variant;
  icon?: IconName;
  busy?: boolean;
  disabled?: boolean;
  full?: boolean;
  style?: StyleProp<ViewStyle>;
}

export default function Button({
  label,
  onPress,
  variant = 'primary',
  icon,
  busy,
  disabled,
  full = true,
  style,
}: Props) {
  const v = VARIANTS[variant];
  const isDisabled = disabled || busy;

  return (
    <Pressable
      onPress={onPress}
      disabled={isDisabled}
      style={({ pressed }) => [
        styles.base,
        { backgroundColor: v.bg, borderColor: v.border },
        full && styles.full,
        pressed && !isDisabled && styles.pressed,
        isDisabled && styles.disabled,
        style,
      ]}
      android_ripple={{ color: 'rgba(255,255,255,0.08)' }}
      hitSlop={6}
    >
      <View style={styles.row}>
        {busy ? (
          <ActivityIndicator size="small" color={v.fg} />
        ) : (
          <>
            {icon && <Icon name={icon} size={19} color={v.fg} strokeWidth={2} />}
            <Text variant="bodyStrong" color={v.fg}>
              {label}
            </Text>
          </>
        )}
      </View>
    </Pressable>
  );
}

const VARIANTS: Record<Variant, { bg: string; fg: string; border: string }> = {
  primary: { bg: palette.accent, fg: palette.textInverse, border: palette.accent },
  secondary: { bg: palette.surfaceAlt, fg: palette.text, border: palette.borderStrong },
  ghost: { bg: 'transparent', fg: palette.textSecondary, border: 'transparent' },
  danger: { bg: palette.dangerSoft, fg: palette.danger, border: 'rgba(242,109,109,0.4)' },
};

const styles = StyleSheet.create({
  base: {
    height: 52,
    borderRadius: radius.md,
    borderWidth: 1,
    paddingHorizontal: spacing.xl,
    alignItems: 'center',
    justifyContent: 'center',
  },
  full: { alignSelf: 'stretch' },
  row: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm },
  pressed: { opacity: 0.85, transform: [{ scale: 0.99 }] },
  disabled: { opacity: 0.45 },
});
