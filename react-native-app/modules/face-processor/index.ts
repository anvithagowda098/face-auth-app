/**
 * face-processor — local Expo module.
 *
 * It exposes NO JavaScript API. Its only job is to register the `faceProcessor`
 * VisionCamera frame-processor plugin (MLKit detect + ArcFace 112x112 warp) at
 * native startup. src/camera/FaceCamera.tsx reaches it via
 * `VisionCameraProxy.initFrameProcessorPlugin('faceProcessor')`.
 *
 * Because it's a local module under modules/, Expo autolinks it during
 * `expo prebuild` — no MainApplication or Podfile edits needed.
 */
export {};
