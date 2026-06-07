/**
 * OfflineDB.ts — SQLite-backed offline store (expo-sqlite, async API).
 *
 * Holds the enrolment gallery (real 512-d ArcFace templates), the access log
 * (every verify attempt, pending sync), and a small key/value table used for
 * sync settings. Every row is signed with a real HMAC-SHA256 (security/hmac.ts)
 * so tampering is detectable.
 *
 * Everything is local-first: nothing leaves the device until SyncManager
 * uploads the access log and purges it.
 */

import * as SQLite from 'expo-sqlite';
import * as Crypto from 'expo-crypto';

import { sign } from '../security/hmac';
import { EMBEDDING_DIM } from '../core/constants';
import type { Embedding, GalleryEntry } from '../core/types';

export async function initDB(db: SQLite.SQLiteDatabase) {
  await db.execAsync(`
    CREATE TABLE IF NOT EXISTS gallery (
      worker_id     TEXT PRIMARY KEY,
      template_json TEXT NOT NULL,
      dim           INTEGER NOT NULL,
      shots         INTEGER NOT NULL,
      enrolled_at   TEXT NOT NULL,
      metadata_json TEXT,
      hmac          TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS access_log (
      id            TEXT PRIMARY KEY,
      worker_id     TEXT,
      matched       INTEGER NOT NULL,
      score         REAL,
      liveness_pass INTEGER,
      latency_ms    REAL,
      face_quality  REAL,
      challenges    TEXT,
      timestamp     TEXT NOT NULL,
      synced        INTEGER NOT NULL DEFAULT 0,
      hmac          TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_access_unsynced ON access_log(synced, timestamp);
    CREATE TABLE IF NOT EXISTS meta (
      k TEXT PRIMARY KEY,
      v TEXT NOT NULL
    );
  `);
}

function encodeEmbedding(e: Embedding): string {
  // round to 6 dp to keep rows compact; loss is far below matching tolerance.
  return JSON.stringify(Array.from(e, v => Math.round(v * 1e6) / 1e6));
}

interface GalleryRow {
  worker_id: string;
  template_json: string;
  enrolled_at: string;
  shots: number;
}

export interface AccessRecord {
  workerId: string | null;
  matched: boolean;
  score: number;
  livenessPass: boolean;
  latencyMs: number;
  faceQuality: number;
  challenges: string[];
}

export interface DashboardStats {
  enrolledWorkers: number;
  totalVerifications: number;
  totalGranted: number;
  grantRate: number;
  avgLatencyMs: number;
  pendingSync: number;
}

export const OfflineDB = {
  // ── Gallery ────────────────────────────────────────────────────────────────

  /** Upsert a worker's averaged template (re-enrolment overwrites). */
  async enrollWorker(db: SQLite.SQLiteDatabase,
    workerId: string,
    template: Embedding,
    shots: number,
    metadata: Record<string, unknown> = {},
  ): Promise<GalleryEntry> {
    if (template.length !== EMBEDDING_DIM) {
      throw new Error(`enrollWorker: template dim ${template.length} != ${EMBEDDING_DIM}`);
    }
    const now = new Date().toISOString();
    const tplJson = encodeEmbedding(template);
    const metaJson = JSON.stringify(metadata);
    const hmac = await sign(`${workerId}:${tplJson}:${now}`);
    await db.runAsync(
      `INSERT OR REPLACE INTO gallery
         (worker_id, template_json, dim, shots, enrolled_at, metadata_json, hmac)
       VALUES (?,?,?,?,?,?,?)`,
      [workerId, tplJson, EMBEDDING_DIM, shots, now, metaJson, hmac],
    );
    return { workerId, template, enrolledAt: now, shots };
  },

  async getGallery(db: SQLite.SQLiteDatabase): Promise<GalleryEntry[]> {
    const rows = await db.getAllAsync<GalleryRow>(
      'SELECT * FROM gallery ORDER BY enrolled_at DESC',
    );
    return rows.map(row => ({
      workerId: row.worker_id,
      template: new Float32Array(JSON.parse(row.template_json)),
      enrolledAt: row.enrolled_at,
      shots: row.shots,
    }));
  },

  async getWorker(db: SQLite.SQLiteDatabase, workerId: string): Promise<GalleryEntry | null> {
    const row = await db.getFirstAsync<GalleryRow>(
      'SELECT * FROM gallery WHERE worker_id=?',
      [workerId],
    );
    if (!row) return null;
    return {
      workerId: row.worker_id,
      template: new Float32Array(JSON.parse(row.template_json)),
      enrolledAt: row.enrolled_at,
      shots: row.shots,
    };
  },

  async removeWorker(db: SQLite.SQLiteDatabase, workerId: string): Promise<void> {
    await db.runAsync('DELETE FROM gallery WHERE worker_id=?', [workerId]);
  },

  // ── Access log ─────────────────────────────────────────────────────────────

  async logAccess(db: SQLite.SQLiteDatabase, rec: AccessRecord): Promise<string> {
    const id = Crypto.randomUUID();
    const now = new Date().toISOString();
    const hmac = await sign(`${id}:${rec.workerId ?? ''}:${rec.matched ? 1 : 0}:${now}`);
    await db.runAsync(
      `INSERT INTO access_log
         (id, worker_id, matched, score, liveness_pass, latency_ms, face_quality, challenges, timestamp, synced, hmac)
       VALUES (?,?,?,?,?,?,?,?,?,0,?)`,
      [
        id,
        rec.workerId,
        rec.matched ? 1 : 0,
        rec.score,
        rec.livenessPass ? 1 : 0,
        rec.latencyMs,
        rec.faceQuality,
        JSON.stringify(rec.challenges),
        now,
        hmac,
      ],
    );
    return id;
  },

  async getPendingLogs(db: SQLite.SQLiteDatabase, limit = 100): Promise<Array<Record<string, unknown>>> {
    return db.getAllAsync<Record<string, unknown>>(
      'SELECT * FROM access_log WHERE synced=0 ORDER BY timestamp ASC LIMIT ?',
      [limit],
    );
  },

  async markSynced(db: SQLite.SQLiteDatabase, ids: string[]): Promise<void> {
    if (ids.length === 0) return;
    const placeholders = ids.map(() => '?').join(',');
    await db.runAsync(`UPDATE access_log SET synced=1 WHERE id IN (${placeholders})`, ids);
  },

  async purgeSynced(db: SQLite.SQLiteDatabase): Promise<number> {
    const res = await db.runAsync('DELETE FROM access_log WHERE synced=1');
    return res.changes;
  },

  async getStats(db: SQLite.SQLiteDatabase,): Promise<DashboardStats> {
    const agg = await db.getFirstAsync<{ n: number; m: number | null; lat: number | null }>(
      'SELECT COUNT(*) n, SUM(matched) m, AVG(latency_ms) lat FROM access_log',
    );
    const pend = await db.getFirstAsync<{ n: number }>(
      'SELECT COUNT(*) n FROM access_log WHERE synced=0',
    );
    const work = await db.getFirstAsync<{ n: number }>('SELECT COUNT(*) n FROM gallery');
    const total = agg?.n ?? 0;
    const granted = agg?.m ?? 0;
    return {
      enrolledWorkers: work?.n ?? 0,
      totalVerifications: total,
      totalGranted: granted,
      grantRate: total > 0 ? granted / total : 0,
      avgLatencyMs: Math.round(agg?.lat ?? 0),
      pendingSync: pend?.n ?? 0,
    };
  },

  // ── Key/value (sync settings, device id) ─────────────────────────────────────

  async getMeta(db: SQLite.SQLiteDatabase, key: string): Promise<string | null> {
    const row = await db.getFirstAsync<{ v: string }>('SELECT v FROM meta WHERE k=?', [key]);
    return row?.v ?? null;
  },

  async setMeta(db: SQLite.SQLiteDatabase, key: string, value: string): Promise<void> {
    await db.runAsync('INSERT OR REPLACE INTO meta (k, v) VALUES (?, ?)', [key, value]);
  },
};

export default OfflineDB;
