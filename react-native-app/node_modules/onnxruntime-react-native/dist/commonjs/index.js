"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
var _exportNames = {
  listSupportedBackends: true
};
Object.defineProperty(exports, "listSupportedBackends", {
  enumerable: true,
  get: function () {
    return _backend.listSupportedBackends;
  }
});
var _onnxruntimeCommon = require("onnxruntime-common");
Object.keys(_onnxruntimeCommon).forEach(function (key) {
  if (key === "default" || key === "__esModule") return;
  if (Object.prototype.hasOwnProperty.call(_exportNames, key)) return;
  if (key in exports && exports[key] === _onnxruntimeCommon[key]) return;
  Object.defineProperty(exports, key, {
    enumerable: true,
    get: function () {
      return _onnxruntimeCommon[key];
    }
  });
});
var _backend = require("./backend");
var _version = require("./version");
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

const backends = (0, _backend.listSupportedBackends)();
for (const backend of backends) {
  (0, _onnxruntimeCommon.registerBackend)(backend.name, _backend.onnxruntimeBackend, 1);
}
Object.defineProperty(_onnxruntimeCommon.env.versions, 'react-native', {
  value: _version.version,
  enumerable: true
});
//# sourceMappingURL=index.js.map