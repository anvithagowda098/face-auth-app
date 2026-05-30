/**
 * App.js — EdgeFace Sentinel root
 * NHAI Innovation Hackathon 7.0
 */

import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import HomeScreen   from './src/screens/HomeScreen';
import VerifyScreen from './src/screens/VerifyScreen';
import EnrolScreen  from './src/screens/EnrolScreen';

const Stack = createNativeStackNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator
        initialRouteName="Home"
        screenOptions={{
          headerShown: false,
          animation: 'slide_from_right',
          contentStyle: { backgroundColor: '#0A0A0A' },
        }}
      >
        <Stack.Screen name="Home"   component={HomeScreen} />
        <Stack.Screen name="Verify" component={VerifyScreen} />
        <Stack.Screen name="Enrol"  component={EnrolScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
