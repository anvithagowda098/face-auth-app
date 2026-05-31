/**
 * hmac.ts — tamper-evident signing for local records.
 *
 * The previous implementation labelled an FNV-1a hash as "HMAC-SHA256" — it was
 * neither keyed-secure nor SHA. This uses a real HMAC-SHA256 (js-sha256, pure
 * JS, no native module) over a per-device key held in the OS keystore
 * (react-native-encrypted-storage). Pure JS keeps it verifiable and avoids a
 * native crypto dependency on the bleeding-edge RN version.
 */

import { sha256 } from 'js-sha256';
import EncryptedStorage from 'react-native-encrypted-storage';

const KEY_ID = 'nhai_hmac_key_v2';
let _key: string | null = null;

/** Lazily load (or mint) the device HMAC key from the secure keystore. */
export async function getKey(): Promise<string> {
  if (_key) return _key;
  try {
    const stored = await EncryptedStorage.getItem(KEY_ID);
    if (stored) {
      _key = stored;
      return stored;
    }
  } catch {
    // keystore not ready (e.g. first launch) — fall through to mint
  }
  // 256-bit random key, hex. crypto.getRandomValues is provided by RN's runtime.
  const bytes = new Uint8Array(32);
  (globalThis.crypto ?? require('react-native-get-random-values') ?? globalThis.crypto).getRandomValues(
    bytes,
  );
  const key = Array.from(bytes, b => b.toString(16).padStart(2, '0')).join('');
  await EncryptedStorage.setItem(KEY_ID, key);
  _key = key;
  return key;
}

/** HMAC-SHA256 hex digest of `message` under the device key. */
export async function sign(message: string): Promise<string> {
  const key = await getKey();
  return sha256.hmac(key, message);
}

/** Constant-time-ish verification of a signature. */
export async function verify(message: string, signature: string): Promise<boolean> {
  const expected = await sign(message);
  if (expected.length !== signature.length) return false;
  let diff = 0;
  for (let i = 0; i < expected.length; i++) {
    diff |= expected.charCodeAt(i) ^ signature.charCodeAt(i);
  }
  return diff === 0;
}
