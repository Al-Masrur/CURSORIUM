"""Desktop application bootstrap and shutdown wiring with hardening."""

from __future__ import annotations

import atexit
import sys

from PySide6.QtWidgets import QApplication

from app.app_controller import AppController
from app.config import load_config
from ui.main_window import MainWindow
from utils.logging_config import configure_logging, install_exception_hook


def main() -> int:
    """Launch the HandPilot desktop application."""
    configure_logging(log_to_file=True)
    qt_app = QApplication(sys.argv)
    qt_app.setApplicationName("HandPilot")

    config, notice = load_config()
    controller = AppController(config=config, config_notice=notice)

    # Hardening guarantees: release inputs and cleanup on any exit path
    install_exception_hook(on_crash_callback=controller.shutdown)
    atexit.register(controller.shutdown)
    qt_app.aboutToQuit.connect(controller.shutdown)

    window = MainWindow(controller=controller)
    window.show()
    return qt_app.exec()

