# CURSORIUM — HandPilot

Local, privacy-first, real-time hand-gesture controller for Windows.

> **Current Stage:** Phase 0 & Phase 1 (Webcam + MediaPipe Hand Landmark Tracking Preview).  
> OS-level mouse and keyboard input is **disabled** in this phase.

## Architecture (Phase 1)

- `camera/`: `CameraManager` captures frames on a background thread into a bounded single-slot `LatestFrameBuffer` so stale frames are dropped rather than queued.
- `vision/`: `HandLandmarkerService` wraps the MediaPipe Hand Landmarker Tasks API (`RunningMode.VIDEO`), and `VisionWorker` processes the newest available frame off the UI thread.
- `ui/`: `MainWindow` and `CameraView` render the live webcam feed, 21-landmark hand overlay, camera/hand status, and FPS on the Qt main thread.
- `app/`: `AppController`, `AppConfig`, `events`, and `application` coordinate lifecycle, error handling, and clean shutdown.

## Setup (Windows, Python 3.14.4)

1. Create and activate the local virtual environment (if not already present):
   ```powershell
   uv venv .venv --python 3.14.4
   .\.venv\Scripts\activate
   ```
2. Install pinned dependencies:
   ```powershell
   uv pip install -r requirements.txt
   ```
3. Ensure the local MediaPipe model bundle exists at `assets/models/hand_landmarker.task` (see `assets/models/README.md`).

## Running the App

```powershell
.\.venv\Scripts\python.exe main.py
```

## Running Tests

Tests run without a physical webcam using mock captures/landmarkers and offscreen Qt rendering:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
.\.venv\Scripts\pytest.exe
```

