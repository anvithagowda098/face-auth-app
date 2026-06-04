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
import { FaceAuthService } from './src/engine/FaceAuthService';
import { palette } from './src/theme';
import { Router, type ScreenMap } from './src/navigation';

const screens: ScreenMap = {
  Home: HomeScreen,
  Verify: VerifyScreen,
  Enrol: EnrolScreen,
};

export default function App() {
  const [booted, setBooted] = useState(false);

  useEffect(() => {
    // Warm the ORT session at launch so the first verify isn't penalised. If the
    // native model isn't present (e.g. running before a build) we still proceed;
    // capture() surfaces a clear error at use time.
    FaceAuthService.init()
      .catch(() => { })
      .finally(() => setBooted(true));
  }, []);

  return (
    <View style={styles.root}>
      <StatusBar style="light" />
      {booted ? (
        <Router screens={screens} initialRouteName="Home" />
      ) : (
        <View style={styles.boot}>
          <ActivityIndicator color={palette.accent} />
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: palette.bg },
  boot: { flex: 1, backgroundColor: palette.bg, alignItems: 'center', justifyContent: 'center' },
});
