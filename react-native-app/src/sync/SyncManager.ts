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

/** Result of one sync attempt — surfaced to the UI by forceSync(). */
export interface SyncResult {
  ok: boolean;
  synced: number;
  purged: number;
  error?: string;
}

const ENDPOINT_KEY = 'sync_endpoint';
const DEVICE_KEY = 'device_id';
// Unset by default — the operator configures the AWS endpoint at runtime
// (Home → Sync settings). A blank/placeholder endpoint fails fast with a clear
// message instead of throwing a confusing DNS error.
const PLACEHOLDER_ENDPOINT = 'https://example.invalid/v1/sync';
const BATCH = 50;
const MAX_RETRIES = 3;
const RETRY_MS = 5000;

type Listener = (s: { status: SyncStatus; count: number }) => void;
type Subscription = { remove: () => void };

function isConfigured(url: string | null): url is string {
  return !!url && url !== PLACEHOLDER_ENDPOINT && /^https?:\/\//i.test(url);
}

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

  /** Manual sync (from the UI). Resolves with a visible, structured result. */
  forceSync(): Promise<SyncResult> {
    return this.runSync();
  }

  async setEndpoint(url: string) {
    await OfflineDB.setMeta(ENDPOINT_KEY, url.trim());
  }

  /** The configured endpoint, or null if none/placeholder is set. */
  async getEndpoint(): Promise<string | null> {
    const url = await OfflineDB.getMeta(ENDPOINT_KEY);
    return isConfigured(url) ? url : null;
  }

  private async onConnected() {
    if (this.syncing) return;
    if (!(await this.getEndpoint())) return; // nothing to do until configured
    const pending = await OfflineDB.getPendingLogs(1);
    if (pending.length > 0) this.runWithRetry();
  }

  /** Auto path: retry transient failures a few times in the background. */
  private async runWithRetry(n = 0) {
    const r = await this.runSync();
    if (!r.ok && n < MAX_RETRIES && r.error !== 'no-endpoint') {
      setTimeout(() => this.runWithRetry(n + 1), RETRY_MS);
    }
  }

  private async runSync(): Promise<SyncResult> {
    if (this.syncing) return { ok: false, synced: 0, purged: 0, error: 'already syncing' };

    const endpoint = await this.getEndpoint();
    if (!endpoint) {
      this.emit('error', 0);
      return {
        ok: false,
        synced: 0,
        purged: 0,
        error: 'no-endpoint',
      };
    }

    this.syncing = true;
    this.emit('syncing', 0);

    let total = 0;
    try {
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
      return { ok: true, synced: total, purged };
    } catch (e) {
      this.emit('error', total);
      return { ok: false, synced: total, purged: 0, error: (e as Error).message };
    } finally {
      this.syncing = false;
    }
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
