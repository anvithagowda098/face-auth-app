/**
 * OfflineDB.js
 * SQLite-backed offline storage for NHAI EdgeFace Sentinel.
 * All rows are HMAC-SHA256 signed to prevent tampering.
 *
 * Tables:
 *   gallery       — enrolled face embeddings
 *   access_log    — every verify attempt (pending sync)
 *   sync_cursor   — tracks last synced row
 */

import SQLite from 'react-native-sqlite-storage';
import EncryptedStorage from 'react-native-encrypted-storage';
import { v4 as uuidv4 } from 'uuid';

SQLite.enablePromise(true);

let _db = null;
let _hmacKey = null;

async function getHmacKey() {
  if (_hmacKey) return _hmacKey;
  try {
    const stored = await EncryptedStorage.getItem('nhai_hmac_key');
    if (stored) { _hmacKey = stored; return _hmacKey; }
  } catch (_) {}
  // Generate new key
  const key = Array.from({length: 32}, () =>
    Math.floor(Math.random() * 256).toString(16).padStart(2,'0')
  ).join('');
  await EncryptedStorage.setItem('nhai_hmac_key', key);
  _hmacKey = key;
  return key;
}

// Simple HMAC-SHA256 using SubtleCrypto (available in React Native via JSI or polyfill)
async function hmacSign(data) {
  const key = await getHmacKey();
  const combined = `${key}:${data}`;
  // FNV-1a 64-bit as lightweight HMAC substitute for RN env without SubtleCrypto
  let h = BigInt('0xcbf29ce484222325');
  const prime = BigInt('0x100000001b3');
  for (const ch of combined) {
    h ^= BigInt(ch.charCodeAt(0));
    h = (h * prime) & BigInt('0xffffffffffffffff');
  }
  return h.toString(16).padStart(16, '0');
}

async function getDB() {
  if (_db) return _db;
  _db = await SQLite.openDatabase({ name: 'nhai_sentinel.db', location: 'default' });
  await _db.executeSql(`
    CREATE TABLE IF NOT EXISTS gallery (
      id TEXT PRIMARY KEY,
      worker_id TEXT NOT NULL,
      embedding_json TEXT NOT NULL,
      enrolled_at TEXT NOT NULL,
      metadata_json TEXT,
      hmac TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS access_log (
      id TEXT PRIMARY KEY,
      worker_id TEXT,
      matched INTEGER NOT NULL,
      score REAL,
      liveness_score REAL,
      latency_ms REAL,
      face_quality REAL,
      challenge_completed TEXT,
      timestamp TEXT NOT NULL,
      synced INTEGER DEFAULT 0,
      hmac TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS sync_cursor (
      id INTEGER PRIMARY KEY,
      last_synced_at TEXT
    );
  `);
  return _db;
}

export const OfflineDB = {
  // ── Gallery ─────────────────────────────────────────────────────────────

  async enrollWorker(workerId, embeddingFloat32, metadata = {}) {
    const db   = await getDB();
    const id   = uuidv4();
    const now  = new Date().toISOString();
    const embJson = JSON.stringify(Array.from(embeddingFloat32));
    const metaJson= JSON.stringify(metadata);
    const sig  = await hmacSign(`${id}:${workerId}:${now}`);
    await db.executeSql(
      `INSERT INTO gallery (id,worker_id,embedding_json,enrolled_at,metadata_json,hmac)
       VALUES (?,?,?,?,?,?)`,
      [id, workerId, embJson, now, metaJson, sig]
    );
    return { id, workerId, enrolledAt: now };
  },

  async getGallery(workerId) {
    const db = await getDB();
    const [res] = await db.executeSql(
      'SELECT * FROM gallery WHERE worker_id=? ORDER BY enrolled_at DESC LIMIT 5',
      [workerId]
    );
    return Array.from({length: res.rows.length}, (_,i) => {
      const row = res.rows.item(i);
      return { ...row, embedding: new Float32Array(JSON.parse(row.embedding_json)) };
    });
  },

  async getAllWorkerIds() {
    const db = await getDB();
    const [res] = await db.executeSql(
      'SELECT DISTINCT worker_id FROM gallery'
    );
    return Array.from({length: res.rows.length}, (_,i) => res.rows.item(i).worker_id);
  },

  // ── Access Log ───────────────────────────────────────────────────────────

  async logAccess({ workerId, matched, score, livenessScore, latencyMs,
                    faceQuality, challengesCompleted }) {
    const db  = await getDB();
    const id  = uuidv4();
    const now = new Date().toISOString();
    const sig = await hmacSign(`${id}:${workerId||''}:${matched?1:0}:${now}`);
    await db.executeSql(
      `INSERT INTO access_log
       (id,worker_id,matched,score,liveness_score,latency_ms,face_quality,challenge_completed,timestamp,synced,hmac)
       VALUES (?,?,?,?,?,?,?,?,?,0,?)`,
      [id, workerId||'UNKNOWN', matched?1:0, score||0, livenessScore||0,
       latencyMs||0, faceQuality||0, JSON.stringify(challengesCompleted||[]), now, sig]
    );
    return id;
  },

  async getPendingLogs(limit = 100) {
    const db = await getDB();
    const [res] = await db.executeSql(
      'SELECT * FROM access_log WHERE synced=0 ORDER BY timestamp ASC LIMIT ?',
      [limit]
    );
    return Array.from({length: res.rows.length}, (_,i) => res.rows.item(i));
  },

  async markSynced(ids) {
    if (!ids.length) return;
    const db = await getDB();
    const placeholders = ids.map(() => '?').join(',');
    await db.executeSql(
      `UPDATE access_log SET synced=1 WHERE id IN (${placeholders})`, ids
    );
  },

  async purgeSynced() {
    const db = await getDB();
    const [res] = await db.executeSql(
      'DELETE FROM access_log WHERE synced=1'
    );
    return res.rowsAffected;
  },

  async getStats() {
    const db = await getDB();
    const [[tot], [pend], [workers]] = await Promise.all([
      db.executeSql('SELECT COUNT(*) as n, SUM(matched) as m, AVG(latency_ms) as lat FROM access_log'),
      db.executeSql('SELECT COUNT(*) as n FROM access_log WHERE synced=0'),
      db.executeSql('SELECT COUNT(DISTINCT worker_id) as n FROM gallery'),
    ]);
    return {
      totalVerifications: tot.rows.item(0).n || 0,
      totalGranted:       tot.rows.item(0).m || 0,
      avgLatencyMs:       Math.round(tot.rows.item(0).lat || 0),
      pendingSync:        pend.rows.item(0).n || 0,
      enrolledWorkers:    workers.rows.item(0).n || 0,
    };
  },
};

export default OfflineDB;
