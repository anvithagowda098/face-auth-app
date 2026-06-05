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
  useFrameOutput,
  useAsyncRunner,
  type Frame,
  type CameraRef,
} from 'react-native-vision-camera';
import { 
  Face,
  useFaceDetector,
  FaceDetectorOptions
} from 'react-native-vision-camera-face-detector';

import {
  warpAndExtractRGB
} from '../core/align';


import { useResizer } from 'react-native-vision-camera-resizer';
import { NitroModules } from 'react-native-nitro-modules';
import { useTensorflowModel } from 'react-native-fast-tflite';
import { shuffleSecure, CHALLENGE_PROMPT, type LivenessProgress, type ChallengeId } from '../engine/LivenessEngine';
import { type Embedding } from '../core/types';
import { qualityScore } from '../engine/FaceAuthService';

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

const detectors_armed = [false, false, false, false, false];
const detectors_feed = [
  (s: Face, armed: boolean): boolean[] => {
    'worklet'
    const eye = s.leftEyeOpenProbability;
    if (eye < EYE_CLOSED) armed = true;
    return [armed, armed && eye > EYE_OPEN]; // closed then re-opened
  },
  (s: Face, armed: boolean): boolean[] => {
    'worklet'
    const eye = s.rightEyeOpenProbability
    if (eye < EYE_CLOSED) armed = true;
    return [armed, armed && eye > EYE_OPEN]; // closed then re-opened
  },
  (s: Face): boolean[] => {
    'worklet'
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
  { isActive, onStatus, onLivenessProgress } : Props
) => {
  const camera = useRef<CameraRef>(null)
  const faceDetectionOptions = useRef<FaceDetectorOptions>( {
    // detection options
    performanceMode: "accurate",
    runLandmarks: true,
    runClassifications: true,
  } ).current;

  const device = useCameraDevice('front');

  const { 
    detectFaces,
  } = useFaceDetector(faceDetectionOptions);

  const faceRecognition = useTensorflowModel(require('../../assets/models/model.tflite'), [])
  const model = faceRecognition.state === 'loaded' ? faceRecognition.model : undefined

  // TfliteModel is a Nitro HybridObject (jsi::NativeState). VisionCamera v4's worklet
  // runtime cannot access jsi::NativeState directly — box it into a jsi::HostObject
  // before capture, then unbox() inside the worklet to run inference.
  // This will not be necessary in VisionCamera v5, which is itself a Nitro Module
  // and can access HybridObjects directly.
  const boxedModel = useMemo(
    () => (model != null ? NitroModules.box(model) : undefined),
    [model]
  )

  const { resizer } = useResizer({
    width: 0,
    height: 0,
    scaleMode: 'contain',
    pixelLayout: 'planar',
    channelOrder: 'rgb',
    dataType: 'uint8',
  });

  useEffect(() => {
    (async () => {
      const status = await Camera.requestCameraPermission();
      // console.log({ status });
    })()
  }, [device]);

  const asyncRunner = useAsyncRunner();
  const embeddings = useSharedValue<Embedding[]>([]);
  const facesShared = useSharedValue<Face[]>([]);
  const perm = useRef(shuffleSecure(Array.from({ length: ALL.length }, (_, i) => i)));
  const idx = useRef(0);
  const livenessProgress = () => {
    'worklet'
    const done = idx.current >= perm.current.length;
    const current = done ? null : ALL[perm.current[idx.current]];
    return {
      current,
      index: idx.current,
      challenges: perm.current.map(i => ALL[i]),
      done,
      prompt: current ? CHALLENGE_PROMPT[current] : 'Liveness confirmed',
    };
  }
  onLivenessProgress?.(livenessProgress());

  const frameOutput = useFrameOutput({
    pixelFormat: 'yuv',
    onFrame(frame: Frame) {
      'worklet'
      const wasHandled = asyncRunner.runAsync(() => {
        'worklet'
        if (boxedModel == null) return;
        const tflite = boxedModel.unbox();
        const faces = detectFaces(frame);
        if (faces.length == 1) {
          const s = faces[0];
          // console.log(s);
          if (s.landmarks === undefined) return;

          if (idx.current < perm.current.length) {
            const out = detectors_feed[perm.current[idx.current]](s, detectors_armed[perm.current[idx.current]]);
            if (out[0]) {
              detectors_armed[perm.current[idx.current]] = out[0];
              console.log(out);
            }
            if (out[1] != true) return;
            idx.current++;
          }
          facesShared.value.push(s);
          console.log(facesShared);
          if (onLivenessProgress !== undefined) {
            // TODO: this only checks first face
            console.log("here");
            console.log("here2");
            onLivenessProgress(livenessProgress(), embeddings.value)
            console.log("here2");
          }

          if (resizer === undefined) return;

          const resized = resizer.resize(frame);
          const inputRGB = warpAndExtractRGB(new Uint8Array(resized.getPixelBuffer()), frame.width, frame.height, s.landmarks);
          resized.dispose()
          const inputBuffer = inputRGB.buffer.slice(inputRGB.byteOffset, inputRGB.byteOffset + inputRGB.byteLength);
          const outputs = tflite.runSync([inputBuffer]);
          const embedding = new Float32Array(outputs[0]);
          embeddings.value.push(embedding);
        }
        // ... chain some asynchronous frame processor
        // ... do something asynchronously with frame
        // handleDetectedFaces(faces)
      });
      if (!wasHandled) {
        frame.dispose();
      }
    }
  });

  if (!device) return <View style={styles.fill} />;

  return (
    <Camera
      style={StyleSheet.absoluteFill}
      isActive={isActive}
      device={device}
      outputs={[frameOutput]}
    />
  );

};

const styles = StyleSheet.create({ fill: { flex: 1, backgroundColor: '#000' } });

export default FaceCamera;
/* vi: set et sw=2: */
