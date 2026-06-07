# Sentinel — on-device face auth: build & integration guide (Expo)

Real, on-device face authentication wired to the int8 ArcFace recognizer chosen
in `model-pipeline` (`w600k_mbf_int8_static.onnx` — 3.68 MB, 99.7% / EER 0.006).
No simulation: every score is a real ONNX Runtime inference.

This is an **Expo dev-build** app (SDK 53 / RN 0.79 / React 19). It is **not**
an Expo Go app — camera, MLKit and ONNX are native, so it must run in a custom
dev client / release build produced by `expo prebuild` (or EAS Build).

> **For the person building this:** the JS/TS app and the native face-warp
> module are ready. There is no `android/` or `ios/` folder — `expo prebuild`
> generates them from `app.json` + the config plugins. Follow the steps in order.

## Why Expo, and why these versions

The previous bare-RN setup fought endless native version mismatches. Expo fixes
the root cause: `expo install` resolves mutually-compatible native versions for
the SDK, and **config plugins generate the native glue** (permissions, pods,
ORT wiring) during prebuild — so the whole class of pod/codegen errors is gone.

Minimal dependency surface (Expo built-ins replace third-party native modules):

| Need | Package |
| --- | --- |
| Camera + frame processor | `react-native-vision-camera` **4.6.4** (pre-Nitro, classic frame-processor API) |
| Worklet runtime | `react-native-worklets-core` **1.6.0** (the only worklet system; no Nitro/`react-native-worklets`) |
| On-device inference | `onnxruntime-react-native` (official Expo config plugin) |
| Face detect + ArcFace warp | **local Expo module** `modules/face-processor` (our validated native code) |
| SQLite gallery + audit log | `expo-sqlite` |
| HMAC key in keystore | `expo-secure-store` (+ `js-sha256` for the HMAC, pure JS) |
| Connectivity for sync | `expo-network` |
| Random ids / key bytes | `expo-crypto` |
| Model asset load | `expo-asset` + `expo-file-system` |
| Icons / overlays | `react-native-svg` |
| Navigation | **none** — a ~50-line `useState` router in `src/navigation.tsx` |

> **Do not** upgrade to VisionCamera v5 — it is a Nitro rewrite that needs
> `react-native-nitro-modules`/`-image` and a different frame-processor API, and
> was the source of the earlier `NitroModules`/codegen build failures.

## What's verified vs what needs the device build

| Layer | Status |
| --- | --- |
| Alignment math (`geometry.ts`/`align.ts`) | ✅ unit-tested **and** matched to `skimage.SimilarityTransform` to ~1e-15 |
| preprocess / cosine / matching / averaging | ✅ unit-tested (`npm run test:core`) |
| ORT embedder, native module, camera, screens | ⚠️ require this dev build on a real device |

## Build steps (run in order)

```bash
# 0. If a previous (bare-RN or V5) attempt left artifacts, clear them:
rm -rf node_modules android ios
rm -rf ~/Library/Developer/Xcode/DerivedData/*    # macOS / iOS only
watchman watch-del-all 2>/dev/null || true

# 1. Install JS deps, then let Expo normalize native versions to the SDK
npm install
npx expo install --fix        # aligns expo-* / vision-camera / svg patch versions

# 2. Copy the model into assets/models/ so Metro bundles it
npm run bundle:model

# 3. Sanity-check the pure TS before any native build
npm run typecheck && npm run test:core

# 4. Generate android/ ios/ from app.json + config plugins (+ autolink the
#    local face-processor module)
npx expo prebuild --clean

# 5. Build & run a dev client on a device (camera needs real hardware)
npx expo run:android         # or: npx expo run:ios   (then `cd ios && pod install` runs automatically)
```

For a cloud build instead of a local toolchain: `eas build --profile development
--platform android` (or `ios`).

## The native module (modules/face-processor)

Our MLKit-detect + ArcFace-warp plugin lives in `modules/face-processor` as a
**local Expo module**, so `expo prebuild` autolinks it — no `MainApplication`
or `Podfile` edits. See `modules/face-processor/README.md`. The one line worth
checking on first iOS build is the VisionCamera Swift registration call in
`FaceProcessorModule.swift` (the compiler will flag it if your VisionCamera
version's signature differs).

## Permissions (handled by config — no manual edits)

- Camera permission text is set by the `react-native-vision-camera` plugin in
  `app.json`; microphone is disabled there.
- `NSCameraUsageDescription` (iOS) and the `CAMERA` permission (Android) are
  injected during prebuild from `app.json`.

## Model bundling

`npm run bundle:model` copies `w600k_mbf_int8_static.onnx` into `assets/models/`.
`metro.config.js` registers the `.onnx` extension, and `src/core/embedder.ts`
loads it via `expo-asset` → reads the bytes → hands ORT a `Uint8Array` (no
platform-specific file-path quirks). Re-run `bundle:model` whenever the model is
re-exported.

## Tuning knobs (`src/core/constants.ts`)

- `COSINE_THRESHOLD` (0.28) — raise for stricter false-accept, lower for convenience.
  Calibrated: EER threshold ≈ 0.225, FAR=1e-3 ≈ 0.218 on aligned LFW.
- `MIN_MATCH_MARGIN` (0.06) — ambiguity gate for 1:N identify.
- `ENROLL_SHOTS`, pose/quality gates — enrolment strictness.

## Notes

- **Liveness** uses ML Kit eye-open/smile/yaw signals (`LivenessEngine.ts`), not a
  468-point mesh. Anti-replay convenience, not certified PAD; a dedicated
  anti-spoof model can slot in behind the same interface.
- **Sync endpoint** is a placeholder — set via `syncManager.setEndpoint(url)`
  (stored in the OfflineDB `meta` table). Auth never uses the network.
- **Safe-area** insets in `src/ui/Screen.tsx` are platform constants (we dropped
  `react-native-safe-area-context`); swap in `useSafeAreaInsets` if you re-add it.
