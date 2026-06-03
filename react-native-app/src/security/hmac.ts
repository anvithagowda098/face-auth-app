/**
 * hmac.ts — tamper-evident signing for local records.
 *
 * Real HMAC-SHA256 (js-sha256, pure JS) over a per-device 256-bit key. The key
 * is minted with expo-crypto's CSPRNG and held in the OS keystore via
 * expo-secure-store (Keychain on iOS, Keystore-backed EncryptedSharedPreferences
 * on Android). No native crypto dependency of our own.
 */

import { sha256 } from 'js-sha256';
import * as SecureStore from 'expo-secure-store';
import * as Crypto from 'expo-crypto';

const KEY_ID = 'nhai_hmac_key_v2';
let _key: string | null = null;

/** Lazily load (or mint) the device HMAC key from the secure keystore. */
export async function getKey(): Promise<string> {
  if (_key) return _key;
  try {
    const stored = await SecureStore.getItemAsync(KEY_ID);
    if (stored) {
      _key = stored;
      return stored;
    }
  } catch {
    // keystore not ready (e.g. first launch) — fall through to mint
  }
  // 256-bit random key, hex.
  const bytes = Crypto.getRandomBytes(32);
  const key = Array.from(bytes, b => b.toString(16).padStart(2, '0')).join('');
  await SecureStore.setItemAsync(KEY_ID, key);
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
