# EdgeFace Sentinel — How to Run

## Prerequisites
- Node.js >= 18, Python >= 3.9
- Android Studio or Xcode
- Physical Android/iOS device recommended

## Step 1 — Install Python deps + run benchmark
```bash
pip install numpy tensorflow-cpu opencv-contrib-python torch
python3 benchmark.py
# Expect: ALL 8 CHECKS PASS ✅
```

## Step 2 — Install React Native deps
```bash
cd react-native-app
npm install
cd ios && pod install && cd ..    # iOS only
```

## Step 3 — Download models (one-time, needs internet)
```bash
bash download_models.sh
# BlazeFace 1.4 MB + MobileFaceNet-S 14.6 MB + FeatherNetB 0.06 MB = 16.1 MB total
```

## Step 4 — Run on device
```bash
npx react-native run-android   # Android
npx react-native run-ios       # iOS
```

## App Flow
1. Home: stats dashboard + sync status
2. ENROL NEW: pick worker ID → 5 face captures
3. VERIFY: liveness challenge (blink/smile/turn) → face match → result card
4. All logged to SQLite → auto-syncs to AWS when Wi-Fi returns → local purge
