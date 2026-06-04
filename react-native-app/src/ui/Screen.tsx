/**
 * Screen.tsx — base page wrapper: app background + safe-area insets. `scroll`
 * wraps children in a ScrollView with consistent padding.
 *
 * Insets are computed from platform constants instead of pulling in
 * react-native-safe-area-context (we dropped it for the minimal-deps build).
 * Good enough for a kiosk-style app; swap in useSafeAreaInsets later if you add
 * the library back.
 */

import React from 'react';
import {
  View,
  ScrollView,
  StyleSheet,
  Platform,
  StatusBar as RNStatusBar,
  type ViewStyle,
  type StyleProp,
} from 'react-native';
import { palette, spacing } from '../theme';

export type Edge = 'top' | 'bottom' | 'left' | 'right';

const INSET_TOP = Platform.OS === 'android' ? RNStatusBar.currentHeight ?? 24 : 47;
const INSET_BOTTOM = Platform.OS === 'ios' ? 34 : 0;

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
  const insets: ViewStyle = {
    paddingTop: edges.includes('top') ? INSET_TOP : 0,
    paddingBottom: edges.includes('bottom') ? INSET_BOTTOM : 0,
  };
  return (
    <View style={[styles.root, insets]}>
      {scroll ? (
        <ScrollView
          contentContainerStyle={[pad, styles.scroll, contentStyle]}
          showsVerticalScrollIndicator={false}
          refreshControl={refreshControl as React.ReactElement<any>}
        >
          {children}
        </ScrollView>
      ) : (
        <View style={[styles.flex, pad, contentStyle]}>{children}</View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: palette.bg },
  flex: { flex: 1 },
  scroll: { paddingBottom: spacing.xxxl },
});
