import type { Backend, InferenceSessionHandler } from 'onnxruntime-common';
import type { SessionOptions } from './api';
declare class OnnxruntimeBackend implements Backend {
    init(): Promise<void>;
    createInferenceSessionHandler(pathOrBuffer: string | Uint8Array, options?: SessionOptions): Promise<InferenceSessionHandler>;
}
export declare const onnxruntimeBackend: OnnxruntimeBackend;
export declare const listSupportedBackends: () => import("./api").SupportedBackend[];
export {};
//# sourceMappingURL=backend.d.ts.map