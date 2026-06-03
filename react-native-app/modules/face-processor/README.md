# face-processor — local Expo module

The `faceProcessor` VisionCamera frame-processor plugin: ML Kit face detection +
the closed-form ArcFace 112×112 similarity warp (identical to `src/core/geometry.ts`,
validated to ~1e-15 against `skimage`). It's a **local Expo module** so
`expo prebuild` autolinks it — no `MainApplication` or `Podfile` edits.

```
modules/face-processor/
  expo-module.config.json     # registers the native modules with Expo autolinking
  index.ts                    # no JS API — registration only
  android/
    build.gradle             # adds mlkit face-detection + vision-camera deps
    src/main/java/com/edgefacesentinel/faceprocessor/
      FaceProcessorModule.kt  # Expo module: registers the plugin in OnCreate
      FaceProcessorPlugin.kt  # the detect + warp logic
  ios/
    FaceProcessor.podspec     # adds GoogleMLKit/FaceDetection + VisionCamera
    FaceProcessorModule.swift # Expo module: registers the plugin in OnCreate
    FaceProcessorPlugin.swift # the detect + warp logic
```

## How registration works

`FaceProcessorModule` (both platforms) registers the plugin under the name
`"faceProcessor"` in its `OnCreate` lifecycle, which runs once at native startup
— before the camera mounts. `src/camera/FaceCamera.tsx` then resolves it via
`VisionCameraProxy.initFrameProcessorPlugin('faceProcessor')`. Registering from
the Expo module (instead of an Android `MainApplication` call / an iOS ObjC
`+load`) is what lets prebuild stay clean.

## If the build can't find the module boilerplate

The `build.gradle` / `podspec` here follow the standard Expo local-module
templates. If your Expo SDK's template differs and the module won't configure,
regenerate the scaffold and drop these sources in:

```bash
npx create-expo-module@latest --local face-processor
# then replace the generated android/ios sources with the four files above,
# keep expo-module.config.json, and ensure these deps are present:
#   android/build.gradle : implementation 'com.google.mlkit:face-detection:16.1.6'
#                          implementation project(':react-native-vision-camera')
#   ios/*.podspec        : s.dependency 'GoogleMLKit/FaceDetection'
#                          s.dependency 'VisionCamera'
```

## The one thing to verify on a device

The iOS registration call in `FaceProcessorModule.swift`
(`FrameProcessorPluginRegistry.addFrameProcessorPlugin("faceProcessor") { … }`)
uses VisionCamera's Swift registry API. If your VisionCamera version exposes a
different signature, the compiler will point at that line — adjust the closure
to match. Everything else (the detect + warp logic) is unchanged from the
version validated against skimage.
