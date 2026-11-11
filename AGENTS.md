# HandPilot — Antigravity Workspace Rules

- Before changing code, read `HANDPILOT_SPEC.md`; it is the product and architecture source of truth.
- Inspect the existing repository first. Preserve unrelated work and existing conventions.
- Implement one small, testable phase at a time. Do not build the whole roadmap in one pass.
- MVP target: Windows desktop, Python, OpenCV, MediaPipe Hand Landmarker Tasks API, PySide6, NumPy, and a replaceable input-backend interface (PyAutoGUI initially).
- Keep camera/vision, feature extraction, gesture recognition, state management, coordinate mapping, input execution, and UI responsibilities separate.
- Keep processing local. Do not record, upload, or log camera frames.
- Use a bounded latest-frame buffer; never let old frames accumulate. Keep Qt widget updates on the UI thread.
- Never execute OS input from raw/noisy gesture detection. Validate gestures over time and route actions through the state manager and serialized input dispatcher.
- Real OS input must be disabled by default during development; test with a mock backend first. Require explicit user enablement.
- Always release held mouse buttons on hand loss, stop, exception, and shutdown. `Esc` is the emergency stop.
- Verify current package/API compatibility before choosing versions. Pin tested dependencies; do not mix MediaPipe legacy Solutions and Tasks API examples.
- Use type hints and tests. Never claim tests passed unless they were actually run.
- Avoid unnecessary dependencies, cloud services, databases, abstractions, or features outside the MVP.
- Do not run destructive commands, install globally, modify system settings, or request administrator access without explicit approval.
- If a detail is unclear, choose the safest reversible default, document it, and continue unless it blocks safe implementation.
- At the end of each task, report changed files, commands/tests run and results, known issues, and the next recommended step.
