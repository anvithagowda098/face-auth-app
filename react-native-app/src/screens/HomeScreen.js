/**
 * HomeScreen.js
 * Dashboard: enrolled workers, verification stats, sync status, quick actions.
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  View, Text, StyleSheet, TouchableOpacity,
  SafeAreaView, ScrollView, RefreshControl,
} from 'react-native';
import { OfflineDB } from '../db/OfflineDB';
import { syncManager } from '../sync/SyncManager';

export default function HomeScreen({ navigation }) {
  const [stats, setStats]       = useState(null);
  const [syncStatus, setSyncStatus] = useState('idle');
  const [pendingSync, setPending]   = useState(0);
  const [refreshing, setRefreshing] = useState(false);

  const loadStats = useCallback(async () => {
    try {
      const s = await OfflineDB.getStats();
      setStats(s);
      setPending(s.pendingSync);
    } catch (_) {}
  }, []);

  useEffect(() => {
    loadStats();
    syncManager.start();
    const unsub = syncManager.onStatusChange(({ status, count }) => {
      setSyncStatus(status);
      if (status === 'synced') loadStats();
    });
    return () => { unsub(); syncManager.stop(); };
  }, []);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await loadStats();
    setRefreshing(false);
  }, []);

  const syncColor = { idle:'#666', offline:'#FF9500', syncing:'#00A8FF', synced:'#00D97E', error:'#FF4C4C' };

  return (
    <SafeAreaView style={styles.root}>
      <ScrollView
        contentContainerStyle={styles.scroll}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor="#00D97E" />}
      >
        {/* Header */}
        <View style={styles.header}>
          <View>
            <Text style={styles.headerTitle}>NHAI Sentinel</Text>
            <Text style={styles.headerSub}>EdgeFace v1.0 • 100% Offline</Text>
          </View>
          <View style={styles.nhaiLogo}>
            <Text style={styles.nhaiLogoText}>NHAI</Text>
          </View>
        </View>

        {/* Stats grid */}
        <View style={styles.grid}>
          <StatCard label="Enrolled Workers" value={stats?.enrolledWorkers ?? '—'} icon="👷" />
          <StatCard label="Verifications"    value={stats?.totalVerifications ?? '—'} icon="🔍" />
          <StatCard label="Granted"          value={stats?.totalGranted ?? '—'} icon="✅" color="#00D97E" />
          <StatCard label="Avg Latency"      value={stats ? `${stats.avgLatencyMs} ms` : '—'} icon="⚡" />
        </View>

        {/* Sync status */}
        <View style={styles.syncCard}>
          <View style={styles.syncRow}>
            <View style={[styles.syncDot, { backgroundColor: syncColor[syncStatus] || '#666' }]} />
            <Text style={styles.syncLabel}>
              {syncStatus === 'idle'    && 'Waiting for connectivity'}
              {syncStatus === 'offline' && 'Offline — data stored locally'}
              {syncStatus === 'syncing' && 'Syncing to AWS…'}
              {syncStatus === 'synced'  && 'All records synced to AWS'}
              {syncStatus === 'error'   && 'Sync failed — will retry'}
            </Text>
          </View>
          {pendingSync > 0 && (
            <Text style={styles.pendingText}>{pendingSync} records pending upload</Text>
          )}
          <TouchableOpacity style={styles.syncBtn} onPress={() => syncManager.forcSync()}>
            <Text style={styles.syncBtnText}>Force Sync Now</Text>
          </TouchableOpacity>
        </View>

        {/* Model info */}
        <View style={styles.modelCard}>
          <Text style={styles.modelTitle}>On-device Models  •  16.1 MB total</Text>
          {[
            { name: 'BlazeFace-Lite (detector)',       size: '1.4 MB' },
            { name: 'FeatherNetB (anti-spoof)',         size: '0.06 MB' },
            { name: 'MobileFaceNet-S (recognition)',    size: '14.6 MB' },
          ].map(m => (
            <View key={m.name} style={styles.modelRow}>
              <Text style={styles.modelName}>{m.name}</Text>
              <Text style={styles.modelSize}>{m.size}</Text>
            </View>
          ))}
        </View>

        {/* Actions */}
        <View style={styles.actions}>
          <TouchableOpacity
            style={[styles.actionBtn, styles.primaryAction]}
            onPress={() => navigation.navigate('Verify', { workerId: 'EMP_001' })}
          >
            <Text style={styles.actionIcon}>🔍</Text>
            <Text style={styles.primaryActionText}>VERIFY WORKER</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.actionBtn}
            onPress={() => navigation.navigate('Enrol')}
          >
            <Text style={styles.actionIcon}>➕</Text>
            <Text style={styles.actionText}>ENROL NEW</Text>
          </TouchableOpacity>
        </View>

        {/* Footer */}
        <Text style={styles.footer}>
          Zero network calls • SQLite AES-256 • HMAC-SHA256 audit log
        </Text>
      </ScrollView>
    </SafeAreaView>
  );
}

function StatCard({ label, value, icon, color }) {
  return (
    <View style={styles.statCard}>
      <Text style={styles.statIcon}>{icon}</Text>
      <Text style={[styles.statValue, color && { color }]}>{value}</Text>
      <Text style={styles.statLabel}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  root:   { flex: 1, backgroundColor: '#0A0A0A' },
  scroll: { padding: 20, paddingBottom: 40 },

  header: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 },
  headerTitle: { color: '#fff', fontSize: 22, fontWeight: '800' },
  headerSub:   { color: '#00D97E', fontSize: 12, marginTop: 2 },
  nhaiLogo:    { backgroundColor: '#FF6B00', borderRadius: 8, padding: 8 },
  nhaiLogoText:{ color: '#fff', fontWeight: '900', fontSize: 13 },

  grid: { flexDirection: 'row', flexWrap: 'wrap', gap: 10, marginBottom: 16 },
  statCard:  { flex: 1, minWidth: '45%', backgroundColor: '#161616', borderRadius: 16,
               padding: 16, borderWidth: 0.5, borderColor: '#2A2A2A' },
  statIcon:  { fontSize: 20, marginBottom: 6 },
  statValue: { color: '#fff', fontSize: 24, fontWeight: '800' },
  statLabel: { color: '#555', fontSize: 11, marginTop: 3 },

  syncCard: { backgroundColor: '#111', borderRadius: 16, padding: 16, marginBottom: 16,
              borderWidth: 0.5, borderColor: '#2A2A2A' },
  syncRow:  { flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 6 },
  syncDot:  { width: 8, height: 8, borderRadius: 4 },
  syncLabel:{ color: '#ccc', fontSize: 13 },
  pendingText: { color: '#FF9500', fontSize: 12, marginBottom: 10 },
  syncBtn:  { backgroundColor: 'rgba(0,217,126,0.1)', borderRadius: 10, padding: 10,
              borderWidth: 1, borderColor: 'rgba(0,217,126,0.3)', alignItems: 'center' },
  syncBtnText: { color: '#00D97E', fontSize: 13, fontWeight: '600' },

  modelCard:  { backgroundColor: '#111', borderRadius: 16, padding: 16, marginBottom: 20,
                borderWidth: 0.5, borderColor: '#2A2A2A' },
  modelTitle: { color: '#fff', fontSize: 13, fontWeight: '700', marginBottom: 12 },
  modelRow:   { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 6 },
  modelName:  { color: '#888', fontSize: 12, flex: 1 },
  modelSize:  { color: '#00D97E', fontSize: 12, fontWeight: '600' },

  actions: { gap: 10, marginBottom: 24 },
  actionBtn: { backgroundColor: '#161616', borderRadius: 16, padding: 18,
               flexDirection: 'row', alignItems: 'center', gap: 12,
               borderWidth: 0.5, borderColor: '#2A2A2A' },
  primaryAction: { backgroundColor: '#00D97E' },
  actionIcon: { fontSize: 20 },
  actionText: { color: '#888', fontSize: 15, fontWeight: '700' },
  primaryActionText: { color: '#000', fontSize: 15, fontWeight: '800' },

  footer: { color: '#333', fontSize: 11, textAlign: 'center', lineHeight: 18 },
});
