"""Desktop application bootstrap and shutdown wiring."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from app.app_controller import AppController
from app.config import AppConfig
from ui.main_window import MainWindow
from utils.logging_config import configure_logging


def main() -> int:
    """Launch the HandPilot Phase 1 desktop application."""
    configure_logging()
    qt_app = QApplication(sys.argv)
    qt_app.setApplicationName("HandPilot")

    config = AppConfig()
    controller = AppController(config=config)
    qt_app.aboutToQuit.connect(controller.shutdown)

    window = MainWindow(controller=controller)
    window.show()
    return qt_app.exec()

