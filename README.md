# CURSORIUM — HandPilot

Local, privacy-first, real-time hand-gesture controller for Windows.

> **Current Stage:** Phase 3 (Mouse Gesture Controls: Click, Double-Click, Right-Click, Drag).  
> OS-level mouse control is **disabled by default** (`os_input_enabled = False`) and requires explicit activation in the UI. Scrolling and advanced gestures are not implemented yet.

## Architecture (Phase 1–3)

- `camera/`: `CameraManager` captures frames on a background thread into a bounded single-slot `LatestFrameBuffer` so stale frames are dropped rather than queued.
- `vision/`: `HandLandmarkerService` wraps MediaPipe Hand Landmarker Tasks API (`RunningMode.VIDEO`), `VisionWorker` processes the newest frame off the UI thread, `landmark_features` computes index extension (`5, 6, 7, 8`) and palm-scaled pinch ratios (`4–8`, `4–12`), and `coordinate_mapper` maps the calibrated active region to clamped screen pixels.
- `gestures/`: `gesture_rules` (hysteresis thresholds & conflict filtering), `GestureRecognizer` (temporal debounce), and `GestureStateManager` (finite-state machine for single click, deferred double-click, right-click, drag, and cooldown).
- `input_control/`: `InputBackend` (`MockInputBackend`, `DesktopInputBackend` via PyAutoGUI), `InputDispatcher` (serialized safety gate & guaranteed `release_all()`), and `ActionMapper`.
- `utils/`: `ExponentialSmoother` for cursor smoothing and `logging_config` for local logging.
- `ui/`: `MainWindow` and `CameraView` render the live webcam feed, active cursor region, pinch indicators, 21-landmark overlay, calibration/gesture sliders, explicit OS control toggle, and `Esc` emergency stop.
- `app/`: `AppController`, `AppConfig`, `events`, and `application` coordinate lifecycle, `Esc` emergency stop, error handling, and clean shutdown.

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

