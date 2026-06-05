/**
 * FaceCamera.tsx — wraps react-native-vision-camera + the native `faceProcessor`
 * frame-processor plugin behind a small, screen-friendly API.
 *
 *   - streams a live `onLivenessProgress(LivenessProgress)` every accepted
 *     frame
 *
 * All native-only surface lives here. Screens never import vision-camera.
 */

import {
  useMemo,
  useRef,
  useEffect,
} from 'react';
import { StyleSheet, View, Text } from 'react-native';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  useFrameOutput,
  type Frame,
} from 'react-native-vision-camera';
import { 
  Face,
  useFaceDetector,
} from 'react-native-vision-camera-face-detector';

import {
  warpAndExtractRGB
} from '../core/align';


import { useResizer } from 'react-native-vision-camera-resizer';
import { useTensorflowModel } from 'react-native-fast-tflite';
import { shuffleSecure, CHALLENGE_PROMPT, type LivenessProgress, type ChallengeId } from '../engine/LivenessEngine';
import { type Embedding } from '../core/types';
import { qualityScore } from '../engine/FaceAuthService';
import { createSynchronizable, scheduleOnRN } from 'react-native-worklets';

export interface FaceCameraHandle {
}

interface Props {
  isActive: boolean;
  onStatus?: (s: Face) => void;
  onLivenessProgress?: (progress: LivenessProgress, embeddings?: Embedding[], faceQuality?: number) => void;
}

const EYE_CLOSED = 0.35;
const EYE_OPEN = 0.7;
const SMILE_ON = 0.7;
const YAW_DEG = 18;

const ALL: ChallengeId[] = ['blink_left_eye', 'blink_right_eye', 'smile', 'turn_left', 'turn_right'];

const detectors_armed = createSynchronizable([false, false, false, false, false]);
const detectors_feed = [
  (s: Face, armed: boolean): boolean[] => {
    'worklet'
    const eye = s.leftEyeOpenProbability;
    if (eye === undefined) return [false, false];
    if (eye < EYE_CLOSED) armed = true;
    return [armed, armed && eye > EYE_OPEN]; // closed then re-opened
  },
  (s: Face, armed: boolean): boolean[] => {
    'worklet'
    const eye = s.rightEyeOpenProbability
    if (eye === undefined) return [false, false];
    if (eye < EYE_CLOSED) armed = true;
    return [armed, armed && eye > EYE_OPEN]; // closed then re-opened
  },
  (s: Face): boolean[] => {
    'worklet'
    if (s.smilingProbability === undefined) return [false, false];
    return [true, s.smilingProbability >= SMILE_ON];
  },
  (s: Face): boolean[] => {
    'worklet'
    return [true, s.yawAngle > YAW_DEG];
  },
  (s: Face): boolean[] => {
    'worklet'
    return [true, s.yawAngle < -YAW_DEG];
  }
];

const FaceCamera = (
  { isActive, onLivenessProgress } : Props
) => {

  const device = useCameraDevice('front');

  const faceDetector = useFaceDetector({
    // detection options
    performanceMode: "accurate",
    runLandmarks: true,
    runClassifications: true,
  });

  const faceRecognition = useTensorflowModel(require('../../assets/models/model.tflite'), [])
  const model = faceRecognition.state === 'loaded' ? faceRecognition.model : undefined

  const { resizer } = useResizer({
    width: 0,
    height: 0,
    scaleMode: 'contain',
    pixelLayout: 'planar',
    channelOrder: 'rgb',
    dataType: 'uint8',
  });

  const { hasPermission, requestPermission } = useCameraPermission();

  useEffect(() => {
    if (!hasPermission) requestPermission()
  }, [hasPermission, requestPermission])

  const idxRef = useRef(createSynchronizable(0));
  const embeddingsRef = useRef(createSynchronizable<Embedding[]>([]));
  const facesSharedRef = useRef(createSynchronizable<Face[]>([]));
  const permRef = useRef(createSynchronizable(shuffleSecure(Array.from({ length: ALL.length }, (_, i) => i))));
  const idx = idxRef.current;
  const embeddings = embeddingsRef.current;
  const facesShared = facesSharedRef.current;
  const perm = permRef.current;
  const ALLsync = createSynchronizable(ALL);
  const CPsync = createSynchronizable(CHALLENGE_PROMPT);
  const livenessProgress = () => {
    'worklet'
    const done = idx.getBlocking() >= perm.getBlocking().length;
    const current = done ? null : ALLsync.getBlocking()[perm.getBlocking()[idx.getBlocking()]];
    return {
      current,
      index: idx.getBlocking(),
      challenges: perm.getBlocking().map(i => ALLsync.getBlocking()[i]),
      done,
      prompt: current ? CPsync.getBlocking()[current] : 'Liveness confirmed',
    };
  }
  onLivenessProgress?.(livenessProgress());

  const frameOutput = useFrameOutput({
    pixelFormat: 'yuv',
    dropFramesWhileBusy: true,
    onFrame(frame: Frame) {
      'worklet'
      if (model === undefined) {
        frame.dispose();
        return;
      }
      const faces = faceDetector.detectFaces(frame);
      if (faces.length == 1) {
        const s = faces[0];
        if (s.landmarks === undefined) {
          frame.dispose();
          return;
        }

        if (idx.getBlocking() < perm.getBlocking().length) {
          console.log(idx.getBlocking());
          console.log(perm.getBlocking());
          const perm_idx = perm.getBlocking()[idx.getBlocking()]
          const feed_func = detectors_feed[perm_idx];
          const out = feed_func(s, detectors_armed.getBlocking()[perm_idx]);
          if (out[0]) {
            detectors_armed.setBlocking(prev => {
              prev[perm_idx] = out[0];
              return prev;
            });
            console.log(out);
          }
          if (out[1] != true) {
            frame.dispose();
            return;
          }
          idx.setBlocking(prev => prev+1);
        }
        facesShared.setBlocking(prev => [...prev, s]);
        if (onLivenessProgress === undefined) {
          frame.dispose();
          return;
        }
        scheduleOnRN(onLivenessProgress, livenessProgress(), embeddings.getBlocking(), qualityScore(facesShared.getBlocking()[0]));

        if (resizer === undefined) {
          frame.dispose();
          return;
        }

        const resized = resizer.resize(frame);
        const inputRGB = warpAndExtractRGB(new Uint8Array(resized.getPixelBuffer()), frame.width, frame.height, s.landmarks);
        resized.dispose()
        const inputBuffer = inputRGB.buffer.slice(inputRGB.byteOffset, inputRGB.byteOffset + inputRGB.byteLength);
        const outputs = model.runSync([inputBuffer]);
        const embedding = new Float32Array(outputs[0]);
        embeddings.setBlocking(prev => [...prev, embedding]);
        frame.dispose();
      }
      // ... chain some asynchronous frame processor
      // ... do something asynchronously with frame
      // handleDetectedFaces(faces)
      frame.dispose();
    }
  });

  if (!device) return <View style={styles.fill} />;

  return (
    <Camera
      style={StyleSheet.absoluteFill}
      isActive={isActive}
      device={device}
      outputs={[frameOutput]}
      mirrorMode="on"
    />
  );

};

const styles = StyleSheet.create({ fill: { flex: 1, backgroundColor: '#000' } });

export default FaceCamera;
/* vi: set et sw=2: */
