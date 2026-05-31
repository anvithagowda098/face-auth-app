/**
 * SyncManager.ts — connectivity-aware upload of the local audit log.
 *
 * Watches NetInfo; when the device is online it batches pending access_log rows
 * to the configured endpoint, marks them synced, then purges them (data-
 * minimisation requirement). Auth itself never touches the network — this only
 * ships the tamper-evident log when a link is available.
 */

import NetInfo from '@react-native-community/netinfo';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { OfflineDB } from '../db/OfflineDB';

export type SyncStatus = 'idle' | 'offline' | 'syncing' | 'synced' | 'error';

const ENDPOINT_KEY = 'sync_endpoint';
const DEVICE_KEY = 'device_id';
const DEFAULT_ENDPOINT = 'https://example.invalid/v1/sync'; // configure per deployment
const BATCH = 50;
const MAX_RETRIES = 3;
const RETRY_MS = 5000;

type Listener = (s: { status: SyncStatus; count: number }) => void;

class SyncManager {
  private unsub: (() => void) | null = null;
  private syncing = false;
  private listeners: Listener[] = [];
  private last: SyncStatus = 'idle';

  start() {
    if (this.unsub) return;
    this.unsub = NetInfo.addEventListener(state => {
      if (state.isConnected && state.isInternetReachable) this.onConnected();
      else this.emit('offline', 0);
    });
  }

  stop() {
    this.unsub?.();
    this.unsub = null;
  }

  onStatusChange(cb: Listener): () => void {
    this.listeners.push(cb);
    return () => {
      this.listeners = this.listeners.filter(l => l !== cb);
    };
  }

  forceSync() {
    return this.doSync();
  }

  async setEndpoint(url: string) {
    await AsyncStorage.setItem(ENDPOINT_KEY, url);
  }

  private async getEndpoint() {
    return (await AsyncStorage.getItem(ENDPOINT_KEY)) || DEFAULT_ENDPOINT;
  }

  private async onConnected() {
    if (this.syncing) return;
    const pending = await OfflineDB.getPendingLogs(1);
    if (pending.length > 0) this.doSync();
  }

  private async doSync(retries = 0): Promise<{ synced: number }> {
    if (this.syncing) return { synced: 0 };
    this.syncing = true;
    this.emit('syncing', 0);

    let total = 0;
    try {
      const endpoint = await this.getEndpoint();
      const deviceId = await this.deviceId();
      let batch: Array<Record<string, unknown>>;
      do {
        batch = await OfflineDB.getPendingLogs(BATCH);
        if (batch.length === 0) break;
        const res = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ device_id: deviceId, records: batch }),
          signal: AbortSignal.timeout(15000),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const ids = batch.map(r => r.id as string);
        await OfflineDB.markSynced(ids);
        total += ids.length;
      } while (batch.length === BATCH);

      const purged = await OfflineDB.purgeSynced();
      this.emit('synced', total);
      if (__DEV__) console.log(`[sync] synced=${total} purged=${purged}`);
    } catch (e) {
      if (retries < MAX_RETRIES) {
        setTimeout(() => {
          this.syncing = false;
          this.doSync(retries + 1);
        }, RETRY_MS);
        return { synced: total };
      }
      this.emit('error', total);
    } finally {
      this.syncing = false;
    }
    return { synced: total };
  }

  private emit(status: SyncStatus, count: number) {
    this.last = status;
    this.listeners.forEach(l => l({ status, count }));
  }

  private async deviceId() {
    let id = await AsyncStorage.getItem(DEVICE_KEY);
    if (!id) {
      const bytes = new Uint8Array(6);
      (globalThis.crypto ?? require('react-native-get-random-values')).getRandomValues?.(bytes);
      id = 'DEV-' + Array.from(bytes, b => b.toString(16).padStart(2, '0')).join('').toUpperCase();
      await AsyncStorage.setItem(DEVICE_KEY, id);
    }
    return id;
  }

  get status() {
    return this.last;
  }
}

export const syncManager = new SyncManager();
export default syncManager;
