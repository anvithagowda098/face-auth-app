/**
 * OfflineDB.ts — SQLite-backed offline store for the face-auth app.
 *
 * Holds the enrolment gallery (real 512-d ArcFace templates), the access log
 * (every verify attempt, pending sync), and a sync cursor. Every row is signed
 * with a real HMAC-SHA256 (see security/hmac.ts) so tampering is detectable.
 *
 * Everything is local-first: nothing leaves the device until SyncManager
 * uploads the access log and purges it.
 */

import SQLite, { type SQLiteDatabase } from 'react-native-sqlite-storage';
import { v4 as uuidv4 } from 'uuid';

import { sign } from '../security/hmac';
import { EMBEDDING_DIM } from '../core/constants';
import type { Embedding, GalleryEntry } from '../core/types';

SQLite.enablePromise(true);

let _db: SQLiteDatabase | null = null;

async function getDB(): Promise<SQLiteDatabase> {
  if (_db) return _db;
  const db = await SQLite.openDatabase({ name: 'faceauth.db', location: 'default' });
  await db.executeSql(`
    CREATE TABLE IF NOT EXISTS gallery (
      worker_id     TEXT PRIMARY KEY,
      template_json TEXT NOT NULL,
      dim           INTEGER NOT NULL,
      shots         INTEGER NOT NULL,
      enrolled_at   TEXT NOT NULL,
      metadata_json TEXT,
      hmac          TEXT NOT NULL
    );`);
  await db.executeSql(`
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
    );`);
  await db.executeSql(
    'CREATE INDEX IF NOT EXISTS idx_access_unsynced ON access_log(synced, timestamp);',
  );
  _db = db;
  return db;
}

function encodeEmbedding(e: Embedding): string {
  // round to 6 dp to keep rows compact; loss is far below matching tolerance.
  return JSON.stringify(Array.from(e, v => Math.round(v * 1e6) / 1e6));
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
  async enrollWorker(
    workerId: string,
    template: Embedding,
    shots: number,
    metadata: Record<string, unknown> = {},
  ): Promise<GalleryEntry> {
    if (template.length !== EMBEDDING_DIM) {
      throw new Error(`enrollWorker: template dim ${template.length} != ${EMBEDDING_DIM}`);
    }
    const db = await getDB();
    const now = new Date().toISOString();
    const tplJson = encodeEmbedding(template);
    const metaJson = JSON.stringify(metadata);
    const hmac = await sign(`${workerId}:${tplJson}:${now}`);
    await db.executeSql(
      `INSERT OR REPLACE INTO gallery
         (worker_id, template_json, dim, shots, enrolled_at, metadata_json, hmac)
       VALUES (?,?,?,?,?,?,?)`,
      [workerId, tplJson, EMBEDDING_DIM, shots, now, metaJson, hmac],
    );
    return { workerId, template, enrolledAt: now, shots };
  },

  async getGallery(): Promise<GalleryEntry[]> {
    const db = await getDB();
    const [res] = await db.executeSql('SELECT * FROM gallery ORDER BY enrolled_at DESC');
    const out: GalleryEntry[] = [];
    for (let i = 0; i < res.rows.length; i++) {
      const row = res.rows.item(i);
      out.push({
        workerId: row.worker_id,
        template: new Float32Array(JSON.parse(row.template_json)),
        enrolledAt: row.enrolled_at,
        shots: row.shots,
      });
    }
    return out;
  },

  async getWorker(workerId: string): Promise<GalleryEntry | null> {
    const db = await getDB();
    const [res] = await db.executeSql('SELECT * FROM gallery WHERE worker_id=?', [workerId]);
    if (res.rows.length === 0) return null;
    const row = res.rows.item(0);
    return {
      workerId: row.worker_id,
      template: new Float32Array(JSON.parse(row.template_json)),
      enrolledAt: row.enrolled_at,
      shots: row.shots,
    };
  },

  async removeWorker(workerId: string): Promise<void> {
    const db = await getDB();
    await db.executeSql('DELETE FROM gallery WHERE worker_id=?', [workerId]);
  },

  // ── Access log ─────────────────────────────────────────────────────────────

  async logAccess(rec: AccessRecord): Promise<string> {
    const db = await getDB();
    const id = uuidv4();
    const now = new Date().toISOString();
    const hmac = await sign(`${id}:${rec.workerId ?? ''}:${rec.matched ? 1 : 0}:${now}`);
    await db.executeSql(
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

  async getPendingLogs(limit = 100): Promise<Array<Record<string, unknown>>> {
    const db = await getDB();
    const [res] = await db.executeSql(
      'SELECT * FROM access_log WHERE synced=0 ORDER BY timestamp ASC LIMIT ?',
      [limit],
    );
    const out: Array<Record<string, unknown>> = [];
    for (let i = 0; i < res.rows.length; i++) out.push(res.rows.item(i));
    return out;
  },

  async markSynced(ids: string[]): Promise<void> {
    if (ids.length === 0) return;
    const db = await getDB();
    const placeholders = ids.map(() => '?').join(',');
    await db.executeSql(`UPDATE access_log SET synced=1 WHERE id IN (${placeholders})`, ids);
  },

  async purgeSynced(): Promise<number> {
    const db = await getDB();
    const [res] = await db.executeSql('DELETE FROM access_log WHERE synced=1');
    return res.rowsAffected;
  },

  async getStats(): Promise<DashboardStats> {
    const db = await getDB();
    const [agg] = await db.executeSql(
      'SELECT COUNT(*) n, SUM(matched) m, AVG(latency_ms) lat FROM access_log',
    );
    const [pend] = await db.executeSql('SELECT COUNT(*) n FROM access_log WHERE synced=0');
    const [work] = await db.executeSql('SELECT COUNT(*) n FROM gallery');
    const total = agg.rows.item(0).n || 0;
    const granted = agg.rows.item(0).m || 0;
    return {
      enrolledWorkers: work.rows.item(0).n || 0,
      totalVerifications: total,
      totalGranted: granted,
      grantRate: total > 0 ? granted / total : 0,
      avgLatencyMs: Math.round(agg.rows.item(0).lat || 0),
      pendingSync: pend.rows.item(0).n || 0,
    };
  },
};

export default OfflineDB;
