/**
 * ResultCard.js
 * Animated result panel shown after face verification.
 * Displays: GRANTED / DENIED, identity, confidence bar, timing breakdown.
 */

import React, { useEffect, useRef } from 'react';
import { View, Text, StyleSheet, Animated, Easing } from 'react-native';

export default function ResultCard({ result, onDismiss }) {
  const slideAnim = useRef(new Animated.Value(120)).current;
  const fadeAnim  = useRef(new Animated.Value(0)).current;

  const granted = result?.ok;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(slideAnim, { toValue: 0, duration: 380, easing: Easing.out(Easing.cubic), useNativeDriver: true }),
      Animated.timing(fadeAnim,  { toValue: 1, duration: 280, useNativeDriver: true }),
    ]).start();

    const t = setTimeout(() => onDismiss?.(), 4500);
    return () => clearTimeout(t);
  }, [result]);

  if (!result) return null;

  const pct = Math.round((result.score || 0) * 100);
  const barColor = granted ? '#00D97E' : '#FF4C4C';

  return (
    <Animated.View style={[
      styles.card,
      granted ? styles.grantedCard : styles.deniedCard,
      { transform: [{ translateY: slideAnim }], opacity: fadeAnim },
    ]}>

      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.icon}>{granted ? '✅' : '🚫'}</Text>
        <View>
          <Text style={[styles.verdict, { color: barColor }]}>
            ACCESS {granted ? 'GRANTED' : 'DENIED'}
          </Text>
          <Text style={styles.identity}>
            {result.userId || 'Unknown Identity'}
          </Text>
        </View>
      </View>

      {/* Confidence bar */}
      <View style={styles.barWrapper}>
        <Text style={styles.barLabel}>Match confidence</Text>
        <View style={styles.barBg}>
          <View style={[styles.barFill, { width: `${pct}%`, backgroundColor: barColor }]} />
        </View>
        <Text style={[styles.barPct, { color: barColor }]}>{pct}%</Text>
      </View>

      {/* Breakdown */}
      <View style={styles.grid}>
        <Metric label="Total latency"   value={`${result.latencyMs?.toFixed(0)} ms`}  />
        <Metric label="Liveness"        value={result.liveness_passed !== false ? '✅ Pass' : '❌ Fail'} />
        <Metric label="Liveness score"  value={`${Math.round((result.livenessScore||0)*100)}%`} />
        <Metric label="Mode"            value={result.simulated ? 'Simulated' : 'On-device'} />
      </View>

      {/* Timestamp */}
      <Text style={styles.ts}>{new Date(result.timestamp).toLocaleTimeString()} • NHAI Sentinel v1.0</Text>
    </Animated.View>
  );
}

function Metric({ label, value }) {
  return (
    <View style={styles.metric}>
      <Text style={styles.metricLabel}>{label}</Text>
      <Text style={styles.metricValue}>{value}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    position: 'absolute',
    bottom: 0, left: 0, right: 0,
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    padding: 24,
    paddingBottom: 40,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: -4 },
    shadowOpacity: 0.3,
    shadowRadius: 12,
    elevation: 16,
  },
  grantedCard: { backgroundColor: '#0A1A13' },
  deniedCard:  { backgroundColor: '#1A0A0A' },
  header: { flexDirection: 'row', alignItems: 'center', marginBottom: 16, gap: 14 },
  icon:   { fontSize: 40 },
  verdict:{ fontSize: 20, fontWeight: '800', letterSpacing: 1 },
  identity:{ color: '#999', fontSize: 13, marginTop: 2 },

  barWrapper: { marginBottom: 16 },
  barLabel:   { color: '#666', fontSize: 12, marginBottom: 6 },
  barBg:      { height: 8, backgroundColor: '#222', borderRadius: 4, overflow: 'hidden' },
  barFill:    { height: 8, borderRadius: 4 },
  barPct:     { fontSize: 12, fontWeight: '700', marginTop: 4, textAlign: 'right' },

  grid: {
    flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginBottom: 16,
  },
  metric: {
    flex: 1, minWidth: '45%',
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 10, padding: 10,
  },
  metricLabel: { color: '#666', fontSize: 11, marginBottom: 3 },
  metricValue: { color: '#fff', fontSize: 14, fontWeight: '600' },

  ts: { color: '#444', fontSize: 11, textAlign: 'center' },
});
