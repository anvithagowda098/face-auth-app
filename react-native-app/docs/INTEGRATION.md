# Sentinel — on-device face auth: build & integration guide

Real, on-device face authentication wired to the int8 ArcFace recognizer chosen
in `model-pipeline` (`w600k_mbf_int8_static.onnx` — 3.68 MB, 99.7% / EER 0.006).
No simulation: every score is a real ONNX Runtime inference.

> **For the person building this:** the `src/`, `App.tsx`, `index.js`, `native/`,
> `package.json`, `babel.config.js`, and `tsconfig.json` are ready. There is **no
> `android/` or `ios/` folder yet** — you generate those once (Step 1), drop in the
> staged native plugin files (Step 6), and build. Follow the steps in order.

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

## Version lock — why these exact versions (read before bumping anything)

This project is pinned to the **proven "classic V4" stack**. Do **not** upgrade
VisionCamera to v5 or React Native to 0.85 — that path pulls in Nitro Modules
(`NitroModules`, `NitroImage`), a different worklet runtime, and a rewritten
frame-processor API, and is what caused the earlier `pod install` / codegen
cascade. The whole point of this lock is to avoid Nitro entirely.

| Package | Pinned | Why |
| --- | --- | --- |
| `react-native` | `0.76.9` | Last RN line VisionCamera **v4** is proven against; New Architecture is on by default and these libs support it. |
| `react` | `18.3.1` | The React version RN 0.76 ships — **not** React 19. |
| `react-native-vision-camera` | `4.6.3` | Pre-Nitro. Uses the classic `FrameProcessorPluginRegistry` + `VisionCameraProxy.initFrameProcessorPlugin` API that `src/camera/FaceCamera.tsx` and the `native/` plugins are written against. |
| `react-native-worklets-core` | `1.5.0` | The worklet runtime v4 frame processors use. **This is the only worklet system in the app** — there is no `react-native-worklets` (SWM) and no `worklets-core` + Nitro overlap. |
| `react-native-reanimated` | *(removed)* | Nothing imports it. It's an *optional* VisionCamera peer, so omitting it is clean. **If you ever add it back, use `^3.16.x` (NOT v4), and its Babel plugin must be the LAST entry in `babel.config.js`.** |
| `onnxruntime-react-native` | `^1.20.1` | Matches the RN 0.76 era. |
| `react-native-screens` / `safe-area-context` | `^4.4.0` / `^4.14.0` | v4/v4 lines that target RN 0.76 (not the 5.x lines, which target newer RN). |
| `@react-native-community/netinfo` | `^11.4.1` | 11.x targets RN 0.76; 12.x targets newer. |
| `@react-native-async-storage/async-storage` | `^2.1.0` | 2.x line for RN 0.76. |
| `uuid` | `^9.0.1` | CJS build that's safe under Hermes/Metro (v11+ is ESM-only and can break Metro). |
| `@react-native/*` presets + `@react-native-community/cli` | `0.76.9` / `15.0.1` | Must match the RN minor exactly. |

`react-native-worklets-core` and `react-native-reanimated` are both **optional**
peers of vision-camera 4.6.3, so npm will not error about the missing reanimated.

## What's verified vs what needs the build

| Layer | Status |
| --- | --- |
| Alignment math (`geometry.ts`/`align.ts`) | ✅ unit-tested **and** matched to `skimage.SimilarityTransform` to ~1e-15 |
| preprocess / cosine / matching / averaging | ✅ unit-tested (`npm run test:core`, 14 checks) |
| ORT embedder, native plugins, camera, screens | ⚠️ require this native build on a real device — verify on-device |

## Build steps (run in order)

```bash
# ──────────────────────────────────────────────────────────────────────────
# 0. From a clean state. If you previously tried a V5/Nitro build, nuke caches
#    first (stale Nitro pods/codegen are sticky):
#      rm -rf node_modules ios/Pods ios/Podfile.lock
#      rm -rf ~/Library/Developer/Xcode/DerivedData/*   # macOS, iOS
#      watchman watch-del-all 2>/dev/null || true
# ──────────────────────────────────────────────────────────────────────────

# 1. Generate the native projects at the MATCHING RN version (creates android/ ios/).
#    Generate into a temp dir, then move android/ and ios/ next to this package.
npx @react-native-community/cli@15.0.1 init EdgeFaceSentinel \
    --version 0.76.9 --directory _native_tmp --skip-install
mv _native_tmp/android ./android
mv _native_tmp/ios ./ios
#    Keep THIS repo's src/, App.tsx, index.js, package.json, babel.config.js,
#    tsconfig.json, app.json — do NOT let the generated ones overwrite them.
#    (app name / module name is "EdgeFaceSentinel", matching app.json.)

# 2. Install JS deps (uses the pinned package.json in this folder)
npm install

# 3. Bundle the model into the native asset dirs (now that android/ ios/ exist)
npm run bundle:model        # copies w600k_mbf_int8_static.onnx

# 4. iOS pods — autolinking pulls vision-camera + worklets-core. Do NOT hand-add
#    any pods (see "Podfile" below).
cd ios && pod install && cd ..

# 5. Sanity check the JS/TS before native build
npm run typecheck && npm run test:core

# 6. Drop in the native plugin sources (see "Native plugin placement")

# 7. Run
npm run android   # or: npm run ios
```

## babel.config.js (already in this repo — don't drop it)

```js
module.exports = {
  presets: ['module:@react-native/babel-preset'],
  plugins: [['react-native-worklets-core/plugin']],
};
```

The `react-native-worklets-core/plugin` line is **required**. Without it the frame
processor in `FaceCamera.tsx` compiles but throws *"Regular JS function cannot be
shared. Try decorating the function with 'worklet'"* the instant a frame arrives.
The scaffold's generated `babel.config.js` does **not** include it — this repo's
version does, so make sure Step 1 didn't overwrite it.

## Native plugin placement

Copy the staged sources in `native/` into the generated projects.

**Android**
- `native/android/FaceProcessorPlugin.kt` → `android/app/src/main/java/com/edgefacesentinel/`
- `native/android/FaceProcessorRegistration.kt` → same package; call
  `FaceProcessorRegistration.register()` in `MainApplication.onCreate()`.
- `android/app/build.gradle` → add
  `implementation 'com.google.mlkit:face-detection:16.1.6'`
- Model asset lives at `android/app/src/main/assets/w600k_mbf_int8_static.onnx`
  (written by `bundle:model`).
- New Architecture is **on** by default in RN 0.76 — leave it on; vision-camera 4,
  worklets-core, and onnxruntime-react-native all support it.

**iOS**
- `native/ios/FaceProcessorPlugin.swift` + `FaceProcessorPlugin.m` → add both to the
  app target in Xcode (drag into the project, "Copy items if needed", app target checked).
- `ios/Podfile` → add the MLKit pod inside the app target:
  `pod 'GoogleMLKit/FaceDetection'`
- Add the `.onnx` under the app target's **Build Phases ▸ Copy Bundle Resources**.
- The first Swift file added triggers Xcode to create a bridging header. Make sure
  the auto-generated umbrella header referenced in `FaceProcessorPlugin.m`
  (`EdgeFaceSentinel-Swift.h`) matches your product module name; rename if it differs.

### Podfile — what NOT to do

- **Do NOT** add `pod 'NitroModules'` or `pod 'NitroImage'`. Those belong to
  VisionCamera **v5** only; with v4 they don't exist on CocoaPods trunk and cause
  *"Unable to find a specification for NitroModules"*. v4 needs no Nitro pods.
- **Do NOT** hand-add a `pod 'react-native-vision-camera'` / `pod 'VisionCamera'`
  or a `pod 'react-native-worklets-core'` line — React Native **autolinking** adds
  them from `node_modules`. Manual entries fight autolinking and re-introduce the
  "can't find specification" / duplicate-symbol errors.
- The only manual pod you add is `pod 'GoogleMLKit/FaceDetection'`.

## Permissions

- Android `android/app/src/main/AndroidManifest.xml`:
  `<uses-permission android:name="android.permission.CAMERA"/>`
- iOS `ios/.../Info.plist`: `NSCameraUsageDescription` (e.g. "Used to authenticate
  toll-plaza workers by face.").

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
