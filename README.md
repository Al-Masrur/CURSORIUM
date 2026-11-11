# CURSORIUM — HandPilot

Local, privacy-first, real-time hand-gesture controller for Windows.

> **Current Stage:** Phase 2 (Virtual Mouse Movement via Index Fingertip).  
> OS-level mouse control is **disabled by default** (`os_input_enabled = False`) and requires explicit activation in the UI. Clicks, dragging, and scrolling are not implemented yet.

## Architecture (Phase 1 & 2)

- `camera/`: `CameraManager` captures frames on a background thread into a bounded single-slot `LatestFrameBuffer` so stale frames are dropped rather than queued.
- `vision/`: `HandLandmarkerService` wraps the MediaPipe Hand Landmarker Tasks API (`RunningMode.VIDEO`), `VisionWorker` processes the newest frame off the UI thread, `landmark_features` extracts landmark `8` and validates index extension (`5, 6, 7, 8`), and `coordinate_mapper` maps the calibrated active region to clamped screen pixels.
- `input_control/`: `InputBackend` interface with `MockInputBackend` (safe testing) and `DesktopInputBackend` (PyAutoGUI), gated by `InputDispatcher`.
- `utils/`: `ExponentialSmoother` for cursor smoothing and `logging_config` for local logging.
- `ui/`: `MainWindow` and `CameraView` render the live webcam feed, active cursor region, 21-landmark overlay, calibration/sensitivity/smoothing sliders, explicit OS control toggle, and `Esc` emergency stop.
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

