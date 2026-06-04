/**
 * FaceCamera.tsx — wraps react-native-vision-camera + the native `faceProcessor`
 * frame-processor plugin behind a small, screen-friendly API.
 *
 *   - streams a live `onStatus(FaceStatus)` every analysed frame (overlay,
 *     liveness, quality gating)
 *   - exposes an imperative `capture()` that resolves the next aligned face as a
 *     DetectedFace (112x112 RGB, decoded from the plugin's base64)
 *
 * All native-only surface lives here. Screens never import vision-camera.
 */

import React, {
  forwardRef,
  useCallback,
  useImperativeHandle,
  useMemo,
  useRef,
} from 'react';
import { StyleSheet, View } from 'react-native';
import {
  Camera,
  useCameraDevice,
  useFrameProcessor,
  VisionCameraProxy,
  runAtTargetFps,
  type Frame,
} from 'react-native-vision-camera';
import { useSharedValue, Worklets } from 'react-native-worklets-core';

import type { FaceStatus } from './types';
import type { DetectedFace } from '../core/types';
import { INPUT_SIZE } from '../core/constants';

// Resolve the native plugin once. Null in environments where it isn't linked.
const plugin = VisionCameraProxy.initFrameProcessorPlugin('faceProcessor', {});

export interface FaceCameraHandle {
  /** Resolve the next good aligned face, or reject after `timeoutMs`. */
  capture: (timeoutMs?: number) => Promise<DetectedFace>;
}

interface Props {
  isActive: boolean;
  onStatus?: (s: FaceStatus) => void;
  /** Analysed frames per second — 8–12 is plenty for auth and saves battery. */
  fps?: number;
}

function decodeBase64ToBytes(b64: string): Uint8Array {
  // RN has global atob via the URL polyfill; fall back to a tiny decoder.
  const bin = typeof atob === 'function' ? atob(b64) : polyfillAtob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

const FaceCamera = forwardRef<FaceCameraHandle, Props>(function FaceCamera(
  { isActive, onStatus, fps = 10 },
  ref,
) {
  const device = useCameraDevice('front');
  const captureWanted = useSharedValue(false);

  // pending capture resolver, set when capture() is called
  const pending = useRef<{
    resolve: (f: DetectedFace) => void;
    reject: (e: Error) => void;
    timer: ReturnType<typeof setTimeout>;
  } | null>(null);

  const deliverStatus = useCallback(
    (s: FaceStatus) => {
      onStatus?.(s);
      if (s.alignedRgbB64 && pending.current) {
        const rgb = decodeBase64ToBytes(s.alignedRgbB64);
        if (rgb.length === INPUT_SIZE * INPUT_SIZE * 3 && s.landmarks) {
          const face: DetectedFace = {
            rgb,
            landmarks: s.landmarks,
            confidence: s.confidence,
            faceRatio: s.faceRatio,
            yaw: s.yaw,
            pitch: s.pitch,
            roll: s.roll,
          };
          clearTimeout(pending.current.timer);
          pending.current.resolve(face);
          pending.current = null;
          captureWanted.value = false;
        }
      }
    },
    [onStatus, captureWanted],
  );

  // Worklet-safe bridge to JS.
  const onStatusJS = useMemo(
    () => Worklets.createRunOnJS(deliverStatus),
    [deliverStatus],
  );

  const frameProcessor = useFrameProcessor(
    (frame: Frame) => {
      'worklet';
      runAtTargetFps(fps, () => {
        'worklet';
        if (!plugin) return;
        const res = plugin.call(frame, { capture: captureWanted.value }) as unknown as
          | FaceStatus
          | null;
        if (res) onStatusJS(res);
      });
    },
    [fps, onStatusJS, captureWanted],
  );

  useImperativeHandle(
    ref,
    () => ({
      capture: (timeoutMs = 6000) =>
        new Promise<DetectedFace>((resolve, reject) => {
          if (!plugin) {
            reject(new Error('faceProcessor plugin not available (native build required)'));
            return;
          }
          if (pending.current) clearTimeout(pending.current.timer);
          const timer = setTimeout(() => {
            pending.current = null;
            captureWanted.value = false;
            reject(new Error('capture timed out — no clear face'));
          }, timeoutMs);
          pending.current = { resolve, reject, timer };
          captureWanted.value = true;
        }),
    }),
    [captureWanted],
  );

  if (!device) return <View style={styles.fill} />;

  return (
    <Camera
      style={StyleSheet.absoluteFill}
      device={device}
      isActive={isActive}
      frameProcessor={frameProcessor}
      pixelFormat="yuv"
    />
  );
});

function polyfillAtob(b64: string): string {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
  let str = b64.replace(/=+$/, '');
  let output = '';
  let bc = 0;
  let bs = 0;
  for (let i = 0; i < str.length; i++) {
    const buffer = chars.indexOf(str[i]);
    if (buffer < 0) continue;
    bs = bc % 4 ? bs * 64 + buffer : buffer;
    if (bc++ % 4) output += String.fromCharCode(255 & (bs >> ((-2 * bc) & 6)));
  }
  return output;
}

const styles = StyleSheet.create({ fill: { flex: 1, backgroundColor: '#000' } });

export default FaceCamera;
