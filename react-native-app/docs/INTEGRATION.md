# Sentinel — on-device face auth: integration guide

Real, on-device face authentication wired to the int8 ArcFace recognizer chosen
in `model-pipeline` (`w600k_mbf_int8_static.onnx` — 3.68 MB, 99.7% / EER 0.006).
No simulation: every score is a real ONNX Runtime inference.

## Architecture

```
 Camera frame ─▶ faceProcessor (native, MLKit)
                  ├─ detect + 5 landmarks + pose + eye/smile signals   ──▶ FaceStatus (per frame)
                  └─ on capture: Umeyama align → 112×112 RGB (base64)   ──▶ DetectedFace
                                                       │
 src/core (pure, unit-tested) ◀────────────────────────┘
   preprocess → NCHW float32           embedder (onnxruntime-react-native)
   align/geometry (== skimage)    ──▶  int8 ArcFace → 512-d ──▶ match (cosine ≥ 0.28, margin gate)
                                                       │
 FaceAuthService ── enroll() avg template / verify() 1:1 or 1:N ──▶ OfflineDB (SQLite + HMAC-SHA256)
                                                       └────────────────────▶ SyncManager (upload + purge)
```

The **only** device-dependent surface is `src/camera/FaceCamera.tsx` + the native
`faceProcessor` plugin. Everything else is plain TypeScript.

## What's verified vs what needs a device

| Layer | Status |
| --- | --- |
| Alignment math (`geometry.ts`/`align.ts`) | ✅ unit-tested **and** matched to `skimage.SimilarityTransform` to ~1e-15 |
| preprocess / cosine / matching / averaging | ✅ unit-tested (`npm run test:core`, 14 checks) |
| ORT embedder, native plugins, camera, screens | ⚠️ require a native build on your machine (no Android SDK/device in CI) |

> RN 0.85 / React 19.2 is bleeding-edge. The native lib versions in
> `package.json` are best-known-good ranges — **pin them** against your installed
> RN once `npm install` resolves, and check `onnxruntime-react-native` /
> `react-native-vision-camera` / `react-native-worklets-core` compatibility.

## Build steps

```bash
# 1. Generate the native projects (this is the android/ ios/ scaffold)
npx @react-native-community/cli@latest init EdgeFaceSentinel \
    --version 0.85.3 --directory _native_tmp --skip-install
#    then move _native_tmp/android and _native_tmp/ios next to this package
#    (or run init in place on a fresh checkout). Keep this src/, App.tsx, index.js.

# 2. Install JS deps
npm install

# 3. Bundle the model into the native asset dirs
npm run bundle:model        # copies w600k_mbf_int8_static.onnx

# 4. iOS pods
cd ios && pod install && cd ..

# 5. Run
npm run android   # or: npm run ios
```

## Native plugin placement

Copy the staged sources in `native/` into the generated projects:

**Android**
- `native/android/FaceProcessorPlugin.kt` → `android/app/src/main/java/com/edgefacesentinel/`
- `native/android/FaceProcessorRegistration.kt` → same package; call
  `FaceProcessorRegistration.register()` in `MainApplication.onCreate()`.
- `android/app/build.gradle` → add `implementation 'com.google.mlkit:face-detection:16.1.6'`
- Model asset lives at `android/app/src/main/assets/w600k_mbf_int8_static.onnx`
  (written by `bundle:model`).

**iOS**
- `native/ios/FaceProcessorPlugin.swift` + `FaceProcessorPlugin.m` → add to the app target.
- `Podfile` → `pod 'GoogleMLKit/FaceDetection'`
- Add the `.onnx` under **Copy Bundle Resources**.
- Fix the bridging header name in `FaceProcessorPlugin.m` (`EdgeFaceSentinel-Swift.h`)
  if your module name differs.

## Permissions

- Android `AndroidManifest.xml`: `<uses-permission android:name="android.permission.CAMERA"/>`
- iOS `Info.plist`: `NSCameraUsageDescription`.

## Tuning knobs (`src/core/constants.ts`)

- `COSINE_THRESHOLD` (0.28) — raise for stricter false-accept, lower for convenience.
  Calibrated reference: EER threshold ≈ 0.225, FAR=1e-3 ≈ 0.218 on aligned LFW.
- `MIN_MATCH_MARGIN` (0.06) — ambiguity gate for 1:N identify.
- `ENROLL_SHOTS`, pose/quality gates — enrolment strictness.

## Notes

- **Liveness** uses ML Kit eye-open/smile/yaw signals (`LivenessEngine.ts`), not a
  468-point mesh. It's an anti-replay convenience gate, not a certified PAD; a
  dedicated anti-spoof model can be slotted in behind the same interface.
- **Sync endpoint** is a placeholder (`SyncManager.ts`) — set via
  `syncManager.setEndpoint(url)` per deployment. Auth never uses the network.
