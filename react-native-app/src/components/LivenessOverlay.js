/**
 * LivenessOverlay.js
 * Animated challenge prompt displayed over the camera feed.
 * Shows: countdown timer, challenge instruction, completion checkmarks.
 */

import React, { useEffect, useRef, useState } from 'react';
import {
  View, Text, StyleSheet, Animated, Easing,
} from 'react-native';
import { challengeLabel } from '../engine/LivenessChallenge';

const CHALLENGE_TIMEOUT_MS = 8000;  // 8 s per challenge

export default function LivenessOverlay({ challenges, completedSet, currentIndex, timeLeftMs }) {
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const fadeAnim  = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    // Pulse animation on current challenge
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, { toValue: 1.06, duration: 600, easing: Easing.inOut(Easing.ease), useNativeDriver: true }),
        Animated.timing(pulseAnim, { toValue: 1.00, duration: 600, easing: Easing.inOut(Easing.ease), useNativeDriver: true }),
      ])
    ).start();
    Animated.timing(fadeAnim, { toValue: 1, duration: 300, useNativeDriver: true }).start();
  }, [currentIndex]);

  const progress = Math.max(0, timeLeftMs / CHALLENGE_TIMEOUT_MS);
  const ringColor = progress > 0.4 ? '#00D97E' : '#FF6B35';

  return (
    <View style={styles.container} pointerEvents="none">
      {/* Face oval guide */}
      <View style={styles.ovalGuide} />

      {/* Timer arc (simplified as a progress bar at top) */}
      <View style={styles.timerBar}>
        <View style={[styles.timerFill, { width: `${progress * 100}%`, backgroundColor: ringColor }]} />
      </View>

      {/* Current challenge */}
      {currentIndex < challenges.length && (
        <Animated.View style={[styles.challengeBox, { opacity: fadeAnim, transform: [{ scale: pulseAnim }] }]}>
          <Text style={styles.challengeText}>
            {challengeLabel(challenges[currentIndex])}
          </Text>
        </Animated.View>
      )}

      {/* Completed list */}
      <View style={styles.completedList}>
        {challenges.map((c, i) => (
          <Text key={c} style={[styles.completedItem, completedSet.has(c) && styles.done]}>
            {completedSet.has(c) ? '✅' : (i === currentIndex ? '⬜' : '◻️')} {c.replace('_', ' ')}
          </Text>
        ))}
      </View>

      {/* All done */}
      {challenges.length > 0 && challenges.every(c => completedSet.has(c)) && (
        <View style={styles.successBadge}>
          <Text style={styles.successText}>✅  Liveness Verified</Text>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    ...StyleSheet.absoluteFillObject,
    alignItems: 'center',
    justifyContent: 'center',
  },
  ovalGuide: {
    width: 240,
    height: 320,
    borderRadius: 120,
    borderWidth: 2.5,
    borderColor: 'rgba(0,217,126,0.6)',
    backgroundColor: 'transparent',
    position: 'absolute',
    top: '12%',
  },
  timerBar: {
    position: 'absolute',
    top: 8,
    left: 20,
    right: 20,
    height: 4,
    borderRadius: 2,
    backgroundColor: 'rgba(255,255,255,0.2)',
    overflow: 'hidden',
  },
  timerFill: {
    height: 4,
    borderRadius: 2,
  },
  challengeBox: {
    position: 'absolute',
    bottom: '22%',
    backgroundColor: 'rgba(0,0,0,0.72)',
    borderRadius: 16,
    paddingHorizontal: 24,
    paddingVertical: 14,
    borderWidth: 1,
    borderColor: 'rgba(0,217,126,0.4)',
  },
  challengeText: {
    color: '#FFFFFF',
    fontSize: 20,
    fontWeight: '700',
    letterSpacing: 0.3,
  },
  completedList: {
    position: 'absolute',
    bottom: '10%',
    alignItems: 'flex-start',
  },
  completedItem: {
    color: 'rgba(255,255,255,0.5)',
    fontSize: 13,
    marginVertical: 2,
    textTransform: 'capitalize',
  },
  done: {
    color: '#00D97E',
    fontWeight: '600',
  },
  successBadge: {
    position: 'absolute',
    bottom: '22%',
    backgroundColor: '#00D97E',
    borderRadius: 16,
    paddingHorizontal: 28,
    paddingVertical: 14,
  },
  successText: {
    color: '#000',
    fontSize: 18,
    fontWeight: '800',
  },
});
