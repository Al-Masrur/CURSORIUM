"""Unit tests for InputBackend, MockInputBackend, DesktopInputBackend, and InputDispatcher."""

from __future__ import annotations

from dataclasses import dataclass

from app.config import AppConfig
from input_control.desktop_backend import DesktopInputBackend
from input_control.dispatcher import InputDispatcher
from input_control.mock_backend import MockInputBackend


@dataclass
class _FakeScreenSize:
    width: int
    height: int


class _FakePyAutoGUI:
    def __init__(self) -> None:
        self.PAUSE = 0.1
        self.FAILSAFE = True
        self.moves: list[tuple[int, int, bool]] = []
        self.ups: list[str] = []

    def size(self) -> _FakeScreenSize:
        return _FakeScreenSize(width=1920, height=1080)

    def moveTo(self, x: int, y: int, _pause: bool = True) -> None:
        self.moves.append((x, y, _pause))

    def mouseUp(self, button: str = "left", _pause: bool = True) -> None:
        self.ups.append(button)


def test_os_input_disabled_by_default_in_config_and_dispatcher() -> None:
    cfg = AppConfig()
    assert cfg.os_input_enabled is False

    backend = MockInputBackend()
    dispatcher = InputDispatcher(backend=backend)
    assert dispatcher.is_enabled is False

    executed = dispatcher.move_cursor(500, 400)
    assert executed is False
    assert backend.move_history == []


def test_dispatcher_moves_when_enabled_and_releases_on_emergency_stop() -> None:
    backend = MockInputBackend()
    dispatcher = InputDispatcher(backend=backend, os_input_enabled=False)

    dispatcher.set_enabled(True)
    assert dispatcher.move_cursor(100, 200) is True
    assert dispatcher.move_cursor(100, 200) is False  # deduplicates identical position
    assert dispatcher.move_cursor(300, 400) is True
    assert backend.move_history == [(100, 200), (300, 400)]

    dispatcher.emergency_stop()
    assert dispatcher.is_enabled is False
    assert backend.release_all_count == 1
    assert dispatcher.move_cursor(500, 500) is False

    dispatcher.close()
    assert backend.closed is True
    assert backend.release_all_count == 2


def test_dispatcher_disables_and_releases_on_backend_error() -> None:
    backend = MockInputBackend(fail_on_move=True)
    errors: list[str] = []
    dispatcher = InputDispatcher(
        backend=backend,
        os_input_enabled=True,
        on_backend_error=errors.append,
    )

    assert dispatcher.move_cursor(250, 250) is False
    assert dispatcher.is_enabled is False
    assert backend.release_all_count == 1
    assert len(errors) == 1
    assert "disabled safely" in errors[0].lower()


def test_desktop_input_backend_clamps_and_configures_pyautogui() -> None:
    fake_api = _FakePyAutoGUI()
    backend = DesktopInputBackend(pyautogui_module=fake_api)

    assert backend.screen_size() == (1920, 1080)
    assert fake_api.PAUSE == 0.0
    assert fake_api.FAILSAFE is False

    backend.move_cursor(-50, 2500)
    backend.move_cursor(640, 480)
    assert fake_api.moves == [(0, 1079, False), (640, 480, False)]

    backend.release_all()
    backend.close()

