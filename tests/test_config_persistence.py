"""Unit tests for Phase 5: config persistence, validation recovery, logging, and hardening."""

from __future__ import annotations

import json
from pathlib import Path

from app.config import (
    AppConfig,
    get_default_config_dir,
    get_default_config_path,
    load_config,
    reset_config_to_defaults,
    save_config,
)
from utils.logging_config import configure_logging, install_exception_hook


def test_default_config_creation_and_paths() -> None:
    cfg = AppConfig()
    assert cfg.os_input_enabled is False
    assert cfg.target_fps == 30
    assert cfg.num_hands == 1

    cfg_dir = get_default_config_dir()
    cfg_path = get_default_config_path()
    assert "HandPilot" in str(cfg_dir)
    assert cfg_path.name == "config.json"


def test_config_save_and_load_roundtrip(tmp_path: Path) -> None:
    custom_path = tmp_path / "test_config.json"
    cfg = AppConfig(
        camera_index=2,
        cursor_sensitivity=1.8,
        cursor_smoothing=0.50,
        drag_hold_ms=450,
        scroll_sensitivity=2.5,
    )

    saved = save_config(cfg, custom_path)
    assert saved is True
    assert custom_path.is_file()

    loaded, notice = load_config(custom_path)
    assert notice is None
    assert loaded.camera_index == 2
    assert abs(loaded.cursor_sensitivity - 1.8) < 1e-4
    assert abs(loaded.cursor_smoothing - 0.50) < 1e-4
    assert loaded.drag_hold_ms == 450
    assert abs(loaded.scroll_sensitivity - 2.5) < 1e-4
    # Safety gate: os_input_enabled must always remain False
    assert loaded.os_input_enabled is False


def test_config_os_input_always_starts_disabled_even_if_tampered(tmp_path: Path) -> None:
    custom_path = tmp_path / "tampered.json"
    cfg = AppConfig()
    data = cfg.to_dict()
    data["os_input_enabled"] = True  # Attempt to force armed input
    custom_path.write_text(json.dumps(data), encoding="utf-8")

    loaded, _ = load_config(custom_path)
    assert loaded.os_input_enabled is False


def test_load_config_handles_corrupt_json_gracefully(tmp_path: Path) -> None:
    custom_path = tmp_path / "corrupt.json"
    custom_path.write_text("{ incomplete json ...", encoding="utf-8")

    loaded, notice = load_config(custom_path)
    assert notice is not None
    assert "Invalid config file" in notice
    assert isinstance(loaded, AppConfig)
    assert loaded.target_fps == 30


def test_load_config_handles_invalid_value_ranges_gracefully(tmp_path: Path) -> None:
    custom_path = tmp_path / "invalid_range.json"
    custom_path.write_text(
        json.dumps({"cursor_sensitivity": 99.0}),  # exceeds max 3.0
        encoding="utf-8",
    )

    loaded, notice = load_config(custom_path)
    assert notice is not None
    assert abs(loaded.cursor_sensitivity - 1.0) < 1e-4


def test_reset_config_to_defaults(tmp_path: Path) -> None:
    custom_path = tmp_path / "reset.json"
    cfg = AppConfig(cursor_sensitivity=2.5)
    save_config(cfg, custom_path)

    reset_cfg = reset_config_to_defaults(custom_path)
    assert abs(reset_cfg.cursor_sensitivity - 1.0) < 1e-4

    loaded, _ = load_config(custom_path)
    assert abs(loaded.cursor_sensitivity - 1.0) < 1e-4


def test_logging_file_creation_and_rotation(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    configure_logging(log_to_file=True, log_dir=log_dir)
    log_file = log_dir / "handpilot.log"
    assert log_file.is_file()


def test_crash_hardening_excepthook_invokes_cleanup() -> None:
    cleaned_up = False

    def on_cleanup() -> None:
        nonlocal cleaned_up
        cleaned_up = True

    install_exception_hook(on_crash_callback=on_cleanup)
    import sys

    try:
        raise RuntimeError("Simulated crash")
    except RuntimeError:
        exc_type, exc_val, tb = sys.exc_info()
        sys.excepthook(exc_type, exc_val, tb)

    assert cleaned_up is True
