/**
 * Text.tsx — typography component bound to the type scale. Use `variant` instead
 * of ad-hoc fontSize/fontWeight so headings, labels and readouts stay consistent.
 */

import React from 'react';
import { Text as RNText, type TextProps, type TextStyle } from 'react-native';
import { palette, type } from '../theme';
import type { TypeRole } from '../theme';

interface Props extends TextProps {
  variant?: TypeRole;
  color?: string;
  center?: boolean;
  mono?: boolean;
}

export default function Text({
  variant = 'body',
  color = palette.text,
  center,
  mono,
  style,
  ...rest
}: Props) {
  const base = type[variant] as TextStyle;
  return (
    <RNText
      {...rest}
      style={[
        base,
        { color },
        center && { textAlign: 'center' },
        mono && { fontVariant: ['tabular-nums'] as TextStyle['fontVariant'] },
        style,
      ]}
    />
  );
}
