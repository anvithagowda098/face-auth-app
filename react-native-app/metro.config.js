// Metro config — Expo defaults plus the model file extension so the .onnx ships
// as a bundled asset (resolved at runtime via expo-asset in src/core/embedder.ts).
const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);
config.resolver.assetExts.push('onnx', 'ort');

module.exports = config;
