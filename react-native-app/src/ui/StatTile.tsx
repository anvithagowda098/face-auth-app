/**
 * StatTile.tsx — a single metric readout: small overline label, large numeric
 * value, optional unit + leading icon. The dashboard grid is built from these.
 */

import React from 'react';
import { View, StyleSheet } from 'react-native';
import Text from './Text';
import Icon, { type IconName } from './Icon';
import Card from './Card';
import { palette, spacing } from '../theme';

interface Props {
  label: string;
  value: string | number;
  unit?: string;
  icon?: IconName;
  valueColor?: string;
}

export default function StatTile({ label, value, unit, icon, valueColor = palette.text }: Props) {
  return (
    <Card style={styles.tile}>
      <View style={styles.head}>
        <Text variant="overline" color={palette.textMuted}>
          {label}
        </Text>
        {icon && <Icon name={icon} size={16} color={palette.textMuted} />}
      </View>
      <View style={styles.valueRow}>
        <Text variant="mono" color={valueColor} mono>
          {value}
        </Text>
        {unit && (
          <Text variant="monoSm" color={palette.textMuted} style={styles.unit}>
            {unit}
          </Text>
        )}
      </View>
    </Card>
  );
}

const styles = StyleSheet.create({
  tile: { flex: 1, minWidth: '46%', gap: spacing.md },
  head: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  valueRow: { flexDirection: 'row', alignItems: 'baseline', gap: spacing.xs },
  unit: { marginBottom: 1 },
});
