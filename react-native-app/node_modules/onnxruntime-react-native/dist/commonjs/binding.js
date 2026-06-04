"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.OrtApi = exports.Module = void 0;
var _reactNative = require("react-native");
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

const Module = exports.Module = _reactNative.NativeModules.Onnxruntime;
if (typeof globalThis.OrtApi === 'undefined') {
  Module.install();
}
const OrtApi = exports.OrtApi = globalThis.OrtApi ?? new Proxy({}, {
  get: () => {
    throw new Error('OrtApi is not initialized. Please make sure Onnxruntime installation is successful.');
  }
});
//# sourceMappingURL=binding.js.map