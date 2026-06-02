/**
 * Babel config for the classic-V4 stack (RN 0.76 + vision-camera 4 + worklets-core).
 *
 * The `react-native-worklets-core/plugin` entry is REQUIRED: without it the
 * frame-processor worklet in src/camera/FaceCamera.tsx builds fine but throws
 * "Regular JS function cannot be shared" the moment a frame is processed.
 *
 * NOTE: there is intentionally no react-native-reanimated plugin here — nothing
 * in the app uses reanimated, so it was removed from package.json. If you ever
 * re-add reanimated (v3.16.x, NOT v4), its plugin MUST be the LAST entry in this
 * list, after the worklets-core plugin.
 */
module.exports = {
  presets: ['module:@react-native/babel-preset'],
  plugins: [['react-native-worklets-core/plugin']],
};
