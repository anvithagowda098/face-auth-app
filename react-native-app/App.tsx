/**
 * App root — initialises the on-device engine, sets up the dark navigation
 * theme, and mounts the stack. A small boot gate keeps the first screen from
 * flashing before the ORT session is ready.
 */

import React, { useEffect, useState } from 'react';
import { StatusBar, View, ActivityIndicator, StyleSheet } from 'react-native';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { NavigationContainer, type Theme, DefaultTheme } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';

import HomeScreen from './src/screens/HomeScreen';
import VerifyScreen from './src/screens/VerifyScreen';
import EnrolScreen from './src/screens/EnrolScreen';
import { FaceAuthService } from './src/engine/FaceAuthService';
import { palette } from './src/theme';
import type { RootStackParamList } from './src/navigation';

const Stack = createNativeStackNavigator<RootStackParamList>();

const navTheme: Theme = {
  ...DefaultTheme,
  dark: true,
  colors: {
    ...DefaultTheme.colors,
    background: palette.bg,
    card: palette.bg,
    text: palette.text,
    border: palette.border,
    primary: palette.accent,
    notification: palette.accent,
  },
};

export default function App() {
  const [booted, setBooted] = useState(false);

  useEffect(() => {
    // Warm the ORT session at launch so the first verify isn't penalised. If the
    // native model isn't present (e.g. dev without a build) we still proceed;
    // capture() surfaces a clear error at use time.
    FaceAuthService.init()
      .catch(() => {})
      .finally(() => setBooted(true));
  }, []);

  return (
    <SafeAreaProvider>
      <StatusBar barStyle="light-content" backgroundColor={palette.bg} />
      {booted ? (
        <NavigationContainer theme={navTheme}>
          <Stack.Navigator
            initialRouteName="Home"
            screenOptions={{ headerShown: false, contentStyle: { backgroundColor: palette.bg } }}
          >
            <Stack.Screen name="Home" component={HomeScreen} />
            <Stack.Screen name="Verify" component={VerifyScreen} options={{ animation: 'fade' }} />
            <Stack.Screen name="Enrol" component={EnrolScreen} />
          </Stack.Navigator>
        </NavigationContainer>
      ) : (
        <View style={styles.boot}>
          <ActivityIndicator color={palette.accent} />
        </View>
      )}
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  boot: { flex: 1, backgroundColor: palette.bg, alignItems: 'center', justifyContent: 'center' },
});
