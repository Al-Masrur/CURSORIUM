# CURSORIUM — HandPilot

Local, privacy-first, real-time hand-gesture controller for Windows.

> **Current Stage:** Phase 4 (Scroll & Pause Controls).  
> OS-level mouse control is **disabled by default** (`os_input_enabled = False`) and requires explicit activation in the UI. Two-finger vertical displacement drives scrolling, and an open palm held steadily toggles Pause/Resume.

## Architecture (Phase 1–4)

- `camera/`: `CameraManager` captures frames on a background thread into a bounded single-slot `LatestFrameBuffer` so stale frames are dropped rather than queued.
- `vision/`: `HandLandmarkerService` wraps MediaPipe Hand Landmarker Tasks API (`RunningMode.VIDEO`), `VisionWorker` processes the newest frame off the UI thread, `landmark_features` computes multi-finger extension (index, middle, ring, pinky, thumb), palm-scaled pinch ratios (`4–8`, `4–12`), two-finger scroll pose detection, and open-palm pause detection, while `coordinate_mapper` maps the calibrated active region to clamped screen pixels.
- `gestures/`: `gesture_rules` (hysteresis thresholds, scroll dead zone, pause cooldown), `GestureRecognizer` (temporal debounce), and `GestureStateManager` (finite-state machine for single click, deferred double-click, right-click, drag, two-finger vertical scroll, open-palm pause/resume, and conflict priorities).
- `input_control/`: `InputBackend` (`MockInputBackend`, `DesktopInputBackend` via PyAutoGUI with `scroll`), `InputDispatcher` (serialized safety gate & guaranteed `release_all()`), and `ActionMapper`.
- `utils/`: `ExponentialSmoother` for cursor smoothing and `logging_config` for local logging.
- `ui/`: `MainWindow` and `CameraView` render the live webcam feed, active cursor region, pinch indicators, scroll indicator badge, paused overlay, 21-landmark overlay, calibration/gesture/scroll sliders, pause/resume button & shortcut, explicit OS control toggle, and `Esc` emergency stop.
- `app/`: `AppController`, `AppConfig`, `events`, and `application` coordinate lifecycle, `Esc` emergency stop, pause/resume toggles, error handling, and clean shutdown.

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

