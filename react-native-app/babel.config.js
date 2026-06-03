/**
 * Babel config — Expo dev build.
 *
 *  - `babel-preset-expo` is the Expo SDK preset (replaces @react-native/babel-preset).
 *  - `react-native-worklets-core/plugin` is REQUIRED for the vision-camera frame
 *    processor in src/camera/FaceCamera.tsx. Without it the worklet builds but
 *    throws "Regular JS function cannot be shared" on the first frame.
 *  - `babel-plugin-transform-import-meta` is needed by onnxruntime-react-native.
 *
 * No reanimated plugin: nothing in the app uses reanimated.
 */
module.exports = function (api) {
  api.cache(true);
  return {
    presets: ['babel-preset-expo'],
    plugins: [['react-native-worklets-core/plugin'], 'babel-plugin-transform-import-meta'],
  };
};
