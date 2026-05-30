# Video Demo Script (3–5 minutes)

## Recording setup
Screen record your Android phone (or use scrcpy). Good lighting, clean background.

## Script

[0:00–0:20] Show NHAI hackathon poster, state the challenge briefly.

[0:20–0:50] Open app → Home screen.
  Point out: "100% Offline" badge, 16.1 MB model bundle, sync status "Offline — data stored locally".

[0:50–1:30] Enrolment demo.
  ENROL NEW → pick EMP_001 → capture 5 shots with pose variations.
  Show "Enrolment Complete". Say: "zero network calls, encrypted SQLite storage".

[1:30–2:30] Liveness + Verification.
  VERIFY WORKER → face oval appears.
  Complete BLINK challenge → complete SMILE/TURN.
  Watch matching → result card: ACCESS GRANTED, confidence %, latency ~600 ms, offline: true.

[2:30–3:00] Denial demo (impressive).
  Try with printed photo or different person → ACCESS DENIED, low confidence.

[3:00–3:40] Sync demo.
  Enable Wi-Fi → status changes "Syncing to AWS…" → "All records synced" → purged locally.

[3:40–4:00] Run benchmark in terminal.
  python3 benchmark.py → show 8 green checkmarks.
  Highlight: 16.1 MB · 588 ms · TAR 99.7% · FAR 0.00%

[4:00–4:20] Closing.
  "EdgeFace Sentinel: offline-first, 16.1 MB, 588 ms, active liveness,
   React Native Android+iOS, Datalake 3.0 ready."
