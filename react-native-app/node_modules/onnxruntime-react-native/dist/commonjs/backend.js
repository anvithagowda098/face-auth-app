"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.onnxruntimeBackend = exports.listSupportedBackends = void 0;
var _onnxruntimeCommon = require("onnxruntime-common");
var _binding = require("./binding");
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

const dataTypeStrings = [undefined,
// 0
'float32', 'uint8', 'int8', 'uint16', 'int16', 'int32', 'int64', 'string', 'bool', 'float16', 'float64', 'uint32', 'uint64', undefined,
// 14
undefined,
// 15
undefined,
// 16
undefined,
// 17
undefined,
// 18
undefined,
// 19
undefined,
// 20
'uint4', 'int4'];
const fillNamesAndMetadata = rawMetadata => {
  const names = [];
  const metadata = [];
  for (const m of rawMetadata) {
    names.push(m.name);
    if (!m.isTensor) {
      metadata.push({
        name: m.name,
        isTensor: false
      });
    } else {
      const type = dataTypeStrings[m.type];
      if (type === undefined) {
        throw new Error(`Unsupported data type: ${m.type}`);
      }
      const shape = [];
      for (let i = 0; i < m.shape.length; ++i) {
        const dim = m.shape[i];
        if (dim === -1) {
          shape.push(m.symbolicDimensions[i]);
        } else if (dim >= 0) {
          shape.push(dim);
        } else {
          throw new Error(`Invalid dimension: ${dim}`);
        }
      }
      metadata.push({
        name: m.name,
        isTensor: m.isTensor,
        type,
        shape
      });
    }
  }
  return [names, metadata];
};
class OnnxruntimeSessionHandler {
  #inferenceSession;
  static #initialized = false;
  constructor(session, info) {
    this.#inferenceSession = session;
    this.inputNames = info.inputNames;
    this.outputNames = info.outputNames;
    this.inputMetadata = info.inputMetadata;
    this.outputMetadata = info.outputMetadata;
  }
  static async create(pathOrBuffer, options) {
    if (!OnnxruntimeSessionHandler.#initialized) {
      OnnxruntimeSessionHandler.#initialized = true;
      let logLevel = 2;
      if (_onnxruntimeCommon.env.logLevel) {
        switch (_onnxruntimeCommon.env.logLevel) {
          case 'verbose':
            logLevel = 0;
            break;
          case 'info':
            logLevel = 1;
            break;
          case 'warning':
            logLevel = 2;
            break;
          case 'error':
            logLevel = 3;
            break;
          case 'fatal':
            logLevel = 4;
            break;
          default:
            throw new Error(`Unsupported log level: ${_onnxruntimeCommon.env.logLevel}`);
        }
      }
      _binding.OrtApi.initOrtOnce(logLevel, _onnxruntimeCommon.Tensor);
    }
    const session = _binding.OrtApi.createInferenceSession();
    if (typeof pathOrBuffer === 'string') {
      await session.loadModel(pathOrBuffer, options);
    } else {
      await session.loadModel(pathOrBuffer.buffer, pathOrBuffer.byteOffset, pathOrBuffer.byteLength, options);
    }
    const [inputNames, inputMetadata] = fillNamesAndMetadata(session.inputMetadata);
    const [outputNames, outputMetadata] = fillNamesAndMetadata(session.outputMetadata);
    return new OnnxruntimeSessionHandler(session, {
      inputNames,
      outputNames,
      inputMetadata,
      outputMetadata
    });
  }
  async dispose() {
    this.#inferenceSession.dispose();
  }
  startProfiling() {
    // startProfiling is a no-op.
    // if sessionOptions.enableProfiling is true, profiling will be enabled when the model is loaded.
  }
  endProfiling() {
    this.#inferenceSession.endProfiling();
  }
  async run(feeds, fetches, options) {
    return this.#inferenceSession.run(feeds, fetches, options);
  }
}
class OnnxruntimeBackend {
  async init() {
    return Promise.resolve();
  }
  async createInferenceSessionHandler(pathOrBuffer, options) {
    return OnnxruntimeSessionHandler.create(pathOrBuffer, {
      ...options,
      ortExtLibPath: _binding.Module.ORT_EXTENSIONS_PATH
    });
  }
}
const onnxruntimeBackend = exports.onnxruntimeBackend = new OnnxruntimeBackend();
const listSupportedBackends = exports.listSupportedBackends = _binding.OrtApi.listSupportedBackends;
//# sourceMappingURL=backend.js.map