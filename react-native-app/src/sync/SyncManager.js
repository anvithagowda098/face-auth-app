/**
 * SyncManager.js
 * Monitors connectivity via @react-native-community/netinfo.
 * On connectivity restored → uploads pending access_log rows to AWS,
 * then purges local records (as per NHAI requirement).
 *
 * AWS target: API Gateway → Lambda → DynamoDB / S3
 * The endpoint URL is configured via SYNC_ENDPOINT env / AsyncStorage.
 */

import NetInfo from '@react-native-community/netinfo';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { OfflineDB } from '../db/OfflineDB';

const ENDPOINT_KEY   = 'nhai_sync_endpoint';
const DEFAULT_ENDPOINT = 'https://api.nhai-sentinel.gov.in/v1/sync';  // configure per deployment
const BATCH_SIZE     = 50;
const MAX_RETRIES    = 3;
const RETRY_DELAY_MS = 5000;

class SyncManager {
  constructor() {
    this._unsubscribe   = null;
    this._syncing       = false;
    this._listeners     = [];  // ({status, count}) callbacks
    this._lastStatus    = 'idle';
  }

  // ── Public API ───────────────────────────────────────────────────────────

  /** Start watching network. Call once on app start. */
  start() {
    this._unsubscribe = NetInfo.addEventListener(state => {
      if (state.isConnected && state.isInternetReachable) {
        this._onConnected();
      } else {
        this._emit('offline', 0);
      }
    });
    console.log('[SyncManager] started');
  }

  stop() {
    if (this._unsubscribe) { this._unsubscribe(); this._unsubscribe = null; }
  }

  onStatusChange(cb) {
    this._listeners.push(cb);
    return () => { this._listeners = this._listeners.filter(l => l !== cb); };
  }

  /** Manually trigger sync (e.g. from settings screen). */
  async forcSync() {
    return this._doSync();
  }

  async setEndpoint(url) {
    await AsyncStorage.setItem(ENDPOINT_KEY, url);
  }

  async getEndpoint() {
    return (await AsyncStorage.getItem(ENDPOINT_KEY)) || DEFAULT_ENDPOINT;
  }

  // ── Internal ─────────────────────────────────────────────────────────────

  async _onConnected() {
    if (this._syncing) return;
    const pending = await OfflineDB.getPendingLogs(1);
    if (pending.length === 0) return;
    this._doSync();
  }

  async _doSync(retries = 0) {
    if (this._syncing) return { synced: 0 };
    this._syncing = true;
    this._emit('syncing', 0);

    let totalSynced = 0;
    try {
      const endpoint = await this.getEndpoint();
      let batch;
      do {
        batch = await OfflineDB.getPendingLogs(BATCH_SIZE);
        if (!batch.length) break;

        const payload = {
          device_id:  await this._deviceId(),
          records:    batch,
          batch_size: batch.length,
          synced_at:  new Date().toISOString(),
        };

        const response = await fetch(endpoint, {
          method:  'POST',
          headers: { 'Content-Type': 'application/json',
                     'X-NHAI-Auth': await this._authToken() },
          body:    JSON.stringify(payload),
          signal:  AbortSignal.timeout(15000),
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const ids = batch.map(r => r.id);
        await OfflineDB.markSynced(ids);
        totalSynced += ids.length;

      } while (batch.length === BATCH_SIZE); // continue if more rows exist

      // Purge synced records (NHAI requirement)
      const purged = await OfflineDB.purgeSynced();
      console.log(`[SyncManager] synced=${totalSynced} purged=${purged}`);
      this._emit('synced', totalSynced);

    } catch (err) {
      console.warn('[SyncManager] error:', err.message);
      if (retries < MAX_RETRIES) {
        setTimeout(() => { this._syncing = false; this._doSync(retries + 1); }, RETRY_DELAY_MS);
        return { synced: totalSynced, retrying: true };
      }
      this._emit('error', totalSynced);
    } finally {
      this._syncing = false;
    }

    return { synced: totalSynced };
  }

  _emit(status, count) {
    this._lastStatus = status;
    this._listeners.forEach(l => l({ status, count }));
  }

  async _deviceId() {
    let id = await AsyncStorage.getItem('nhai_device_id');
    if (!id) {
      id = 'DEV-' + Math.random().toString(36).slice(2, 10).toUpperCase();
      await AsyncStorage.setItem('nhai_device_id', id);
    }
    return id;
  }

  async _authToken() {
    // In production: JWT signed with device cert
    return 'NHAI-DEV-TOKEN';
  }

  get status()  { return this._lastStatus; }
  get syncing() { return this._syncing; }
}

export const syncManager = new SyncManager();
export default syncManager;
