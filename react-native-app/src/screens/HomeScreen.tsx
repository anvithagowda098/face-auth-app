/**
 * HomeScreen — operations dashboard: live gallery + verification stats, sync
 * state, the real on-device model card, and the primary actions.
 */

import React, { useCallback, useEffect, useState } from 'react';
import { View, StyleSheet, RefreshControl, Pressable } from 'react-native';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';

import { Screen, Text, Card, Button, StatTile, Badge, Icon } from '../ui';
import { palette, spacing, radius } from '../theme';
import { OfflineDB, type DashboardStats } from '../db/OfflineDB';
import { syncManager, type SyncStatus } from '../sync/SyncManager';
import type { RootStackParamList } from '../navigation';

type Props = NativeStackScreenProps<RootStackParamList, 'Home'>;

export default function HomeScreen({ navigation }: Props) {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [sync, setSync] = useState<SyncStatus>('idle');
  const [refreshing, setRefreshing] = useState(false);

  const load = useCallback(async () => {
    try {
      setStats(await OfflineDB.getStats());
    } catch {
      /* db not ready yet */
    }
  }, []);

  useEffect(() => {
    load();
    syncManager.start();
    const unsub = syncManager.onStatusChange(({ status }) => {
      setSync(status);
      if (status === 'synced') load();
    });
    const focus = navigation.addListener('focus', load);
    return () => {
      unsub();
      focus();
      syncManager.stop();
    };
  }, [load, navigation]);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await load();
    setRefreshing(false);
  }, [load]);

  const grantRate = stats ? `${Math.round(stats.grantRate * 100)}` : '—';

  return (
    <Screen
      scroll
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={palette.accent} />
      }
    >
      {/* Header */}
      <View style={styles.header}>
        <View>
          <Text variant="overline" color={palette.accent}>
            EDGE FACE AUTH
          </Text>
          <Text variant="display">Sentinel</Text>
        </View>
        <Badge label="100% Offline" tone="accent" dot />
      </View>

      {/* Stat grid */}
      <View style={styles.grid}>
        <StatTile label="ENROLLED" value={stats?.enrolledWorkers ?? '—'} icon="users" />
        <StatTile label="VERIFICATIONS" value={stats?.totalVerifications ?? '—'} icon="face-scan" />
        <StatTile label="GRANT RATE" value={grantRate} unit="%" icon="shield-check" valueColor={palette.success} />
        <StatTile label="AVG LATENCY" value={stats?.avgLatencyMs ?? '—'} unit="ms" icon="bolt" />
      </View>

      {/* Sync */}
      <SyncCard status={sync} pending={stats?.pendingSync ?? 0} />

      {/* Model card — real artifact, real numbers */}
      <Card style={styles.modelCard}>
        <View style={styles.modelHead}>
          <Icon name="cpu" size={18} color={palette.accent} />
          <Text variant="h2">On-device model</Text>
        </View>
        <ModelRow k="Recognition" v="MobileFaceNet · ArcFace" tag="INT8 · 3.68 MB" />
        <ModelRow k="Detection + align" v="ML Kit · 5-point" tag="on-device" />
        <ModelRow k="Embedding" v="512-d · cosine" tag="≥ 0.28" />
        <ModelRow k="Benchmark (LFW)" v="99.7% · EER 0.006" tag="ORT" last />
      </Card>

      {/* Actions */}
      <View style={styles.actions}>
        <Button
          label="Verify worker"
          icon="face-scan"
          onPress={() => navigation.navigate('Verify', {})}
        />
        <Button
          label="Enrol new worker"
          icon="user-plus"
          variant="secondary"
          onPress={() => navigation.navigate('Enrol')}
        />
      </View>

      <View style={styles.footer}>
        <Icon name="lock" size={13} color={palette.textMuted} />
        <Text variant="caption" color={palette.textMuted}>
          On-device inference · HMAC-SHA256 signed audit log · zero network at auth time
        </Text>
      </View>
    </Screen>
  );
}

function ModelRow({ k, v, tag, last }: { k: string; v: string; tag: string; last?: boolean }) {
  return (
    <View style={[styles.modelRow, !last && styles.modelRowBorder]}>
      <Text variant="label" color={palette.textSecondary}>
        {k}
      </Text>
      <View style={styles.modelRight}>
        <Text variant="label" color={palette.text}>
          {v}
        </Text>
        <Badge label={tag} tone="neutral" />
      </View>
    </View>
  );
}

function SyncCard({ status, pending }: { status: SyncStatus; pending: number }) {
  const map: Record<SyncStatus, { tone: 'neutral' | 'accent' | 'success' | 'danger' | 'warning'; label: string; icon: 'cloud-sync' | 'wifi-off' | 'refresh' | 'alert' }> = {
    idle: { tone: 'neutral', label: 'Waiting for connectivity', icon: 'cloud-sync' },
    offline: { tone: 'warning', label: 'Offline — stored locally', icon: 'wifi-off' },
    syncing: { tone: 'accent', label: 'Syncing audit log…', icon: 'refresh' },
    synced: { tone: 'success', label: 'All records synced', icon: 'cloud-sync' },
    error: { tone: 'danger', label: 'Sync failed — will retry', icon: 'alert' },
  };
  const m = map[status];
  return (
    <Card style={styles.syncCard}>
      <View style={styles.syncRow}>
        <Icon name={m.icon} size={18} color={palette.textSecondary} />
        <Text variant="body" color={palette.text} style={{ flex: 1 }}>
          {m.label}
        </Text>
        {pending > 0 && <Badge label={`${pending} pending`} tone={m.tone} />}
      </View>
      <Pressable onPress={() => syncManager.forceSync()} style={styles.syncBtn} hitSlop={6}>
        <Text variant="label" color={palette.accent}>
          Force sync now
        </Text>
        <Icon name="chevron-right" size={16} color={palette.accent} />
      </Pressable>
    </Card>
  );
}

const styles = StyleSheet.create({
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: spacing.xl,
  },
  grid: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.md, marginBottom: spacing.lg },
  syncCard: { gap: spacing.md, marginBottom: spacing.lg },
  syncRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.md },
  syncBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.xs,
    backgroundColor: palette.accentSoft,
    borderRadius: radius.md,
    paddingVertical: spacing.md,
  },
  modelCard: { gap: spacing.xs, marginBottom: spacing.xl },
  modelHead: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm, marginBottom: spacing.sm },
  modelRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: spacing.md,
  },
  modelRowBorder: { borderBottomWidth: StyleSheet.hairlineWidth, borderColor: palette.border },
  modelRight: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm },
  actions: { gap: spacing.md, marginBottom: spacing.xl },
  footer: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: spacing.xs, paddingHorizontal: spacing.lg },
});
