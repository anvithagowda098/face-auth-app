/**
 * App root — warms the on-device ORT session, then mounts the tiny stack router
 * (src/navigation.tsx). A boot gate avoids flashing the first screen before the
 * session is ready.
 */

import React, { useEffect, useState } from 'react';
import { View, ActivityIndicator, StyleSheet } from 'react-native';
import { StatusBar } from 'expo-status-bar';

import HomeScreen from './src/screens/HomeScreen';
import VerifyScreen from './src/screens/VerifyScreen';
import EnrolScreen from './src/screens/EnrolScreen';
import { palette } from './src/theme';
import { Router, type ScreenMap } from './src/navigation';

const screens: ScreenMap = {
  Home: HomeScreen,
  Verify: VerifyScreen,
  Enrol: EnrolScreen,
};

export default function App() {
  return (
    <View style={styles.root}>
      <StatusBar style="light" />
      <Router screens={screens} initialRouteName="Home" />
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: palette.bg },
  boot: { flex: 1, backgroundColor: palette.bg, alignItems: 'center', justifyContent: 'center' },
});
/* vi: set et sw=2: */
