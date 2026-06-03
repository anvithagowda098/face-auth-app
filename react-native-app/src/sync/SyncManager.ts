/**
 * SyncManager.ts — connectivity-aware upload of the local audit log.
 *
 * Watches expo-network; when the device is online it batches pending access_log
 * rows to the configured endpoint, marks them synced, then purges them (data-
 * minimisation requirement). Auth itself never touches the network — this only
 * ships the tamper-evident log when a link is available. Settings (endpoint,
 * device id) live in the OfflineDB `meta` table.
 */

import * as Network from 'expo-network';
import * as Crypto from 'expo-crypto';
import { OfflineDB } from '../db/OfflineDB';

export type SyncStatus = 'idle' | 'offline' | 'syncing' | 'synced' | 'error';

const ENDPOINT_KEY = 'sync_endpoint';
const DEVICE_KEY = 'device_id';
const DEFAULT_ENDPOINT = 'https://example.invalid/v1/sync'; // configure per deployment
const BATCH = 50;
const MAX_RETRIES = 3;
const RETRY_MS = 5000;

type Listener = (s: { status: SyncStatus; count: number }) => void;
type Subscription = { remove: () => void };

class SyncManager {
  private sub: Subscription | null = null;
  private syncing = false;
  private listeners: Listener[] = [];
  private last: SyncStatus = 'idle';

  start() {
    if (this.sub) return;
    this.sub = Network.addNetworkStateListener(state => {
      // isInternetReachable can be undefined on some platforms — treat as online.
      if (state.isConnected && state.isInternetReachable !== false) this.onConnected();
      else this.emit('offline', 0);
    });
  }

  stop() {
    this.sub?.remove();
    this.sub = null;
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
    await OfflineDB.setMeta(ENDPOINT_KEY, url);
  }

  private async getEndpoint() {
    return (await OfflineDB.getMeta(ENDPOINT_KEY)) || DEFAULT_ENDPOINT;
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
    } catch {
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
    let id = await OfflineDB.getMeta(DEVICE_KEY);
    if (!id) {
      id = 'DEV-' + Crypto.randomUUID().replace(/-/g, '').slice(0, 12).toUpperCase();
      await OfflineDB.setMeta(DEVICE_KEY, id);
    }
    return id;
  }

  get status() {
    return this.last;
  }
}

export const syncManager = new SyncManager();
export default syncManager;
