/**
 * Screen.tsx — base page wrapper: app background + safe-area insets. `scroll`
 * wraps children in a ScrollView with consistent padding.
 */

import React from 'react';
import { View, ScrollView, StyleSheet, type ViewStyle, type StyleProp } from 'react-native';
import {
  SafeAreaView,
  type Edge,
} from 'react-native-safe-area-context';
import { palette, spacing } from '../theme';

interface Props {
  children: React.ReactNode;
  scroll?: boolean;
  padded?: boolean;
  edges?: readonly Edge[];
  refreshControl?: React.ReactElement;
  contentStyle?: StyleProp<ViewStyle>;
}

export default function Screen({
  children,
  scroll,
  padded = true,
  edges = ['top', 'bottom'],
  refreshControl,
  contentStyle,
}: Props) {
  const pad = padded ? { padding: spacing.xl } : null;
  return (
    <SafeAreaView style={styles.root} edges={edges}>
      {scroll ? (
        <ScrollView
          contentContainerStyle={[pad, styles.scroll, contentStyle]}
          showsVerticalScrollIndicator={false}
          refreshControl={refreshControl}
        >
          {children}
        </ScrollView>
      ) : (
        <View style={[styles.flex, pad, contentStyle]}>{children}</View>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: palette.bg },
  flex: { flex: 1 },
  scroll: { paddingBottom: spacing.xxxl },
});
