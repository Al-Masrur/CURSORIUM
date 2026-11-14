"""Unit tests for InputBackend, MockInputBackend, DesktopInputBackend, InputDispatcher, and ActionMapper."""

from __future__ import annotations

from dataclasses import dataclass

from app.config import AppConfig
from gestures.gesture_rules import GestureEvent, GestureType
from input_control.action_mapper import ActionMapper
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
        self.size_calls = 0
        self.moves: list[tuple[int, int, bool]] = []
        self.clicks: list[tuple[str, int, bool]] = []
        self.downs: list[tuple[str, bool]] = []
        self.ups: list[tuple[str, bool]] = []
        self.scrolls: list[tuple[int, bool]] = []

    def size(self) -> _FakeScreenSize:
        self.size_calls += 1
        return _FakeScreenSize(width=1920, height=1080)

    def moveTo(self, x: int, y: int, _pause: bool = True) -> None:
        self.moves.append((x, y, _pause))

    def click(self, button: str = "left", clicks: int = 1, _pause: bool = True) -> None:
        self.clicks.append((button, clicks, _pause))

    def mouseDown(self, button: str = "left", _pause: bool = True) -> None:
        self.downs.append((button, _pause))

    def mouseUp(self, button: str = "left", _pause: bool = True) -> None:
        self.ups.append((button, _pause))

    def scroll(self, clicks: int, _pause: bool = True) -> None:
        self.scrolls.append((clicks, _pause))


def test_os_input_disabled_by_default_blocks_moves_clicks_and_drag() -> None:
    cfg = AppConfig()
    assert cfg.os_input_enabled is False

    backend = MockInputBackend()
    dispatcher = InputDispatcher(backend=backend)
    assert dispatcher.is_enabled is False

    assert dispatcher.move_cursor(500, 400) is False
    assert dispatcher.click("left", 1) is False
    assert dispatcher.button_down("left") is False
    assert backend.move_history == []
    assert backend.click_history == []
    assert backend.held_buttons == set()


def test_dispatcher_executes_clicks_and_releases_held_buttons_on_emergency_stop() -> None:
    backend = MockInputBackend()
    dispatcher = InputDispatcher(backend=backend, os_input_enabled=False)

    dispatcher.set_enabled(True)
    assert dispatcher.move_cursor(100, 200) is True
    assert dispatcher.move_cursor(100, 200) is False
    assert dispatcher.click("left", 1) is True
    assert dispatcher.click("left", 2) is True
    assert dispatcher.click("right", 1) is True
    assert dispatcher.button_down("left") is True
    assert backend.held_buttons == {"left"}

    dispatcher.emergency_stop()
    assert dispatcher.is_enabled is False
    assert backend.held_buttons == set()
    assert backend.release_all_count == 1
    assert ("up", "left") in backend.button_events

    dispatcher.close()
    assert backend.closed is True
    assert backend.release_all_count == 2


def test_dispatcher_disables_and_releases_on_backend_error() -> None:
    backend = MockInputBackend(fail_on_click=True)
    errors: list[str] = []
    dispatcher = InputDispatcher(
        backend=backend,
        os_input_enabled=True,
        on_backend_error=errors.append,
    )

    assert dispatcher.button_down("left") is True
    assert dispatcher.click("left", 1) is False
    assert dispatcher.is_enabled is False
    assert backend.held_buttons == set()
    assert backend.release_all_count == 1
    assert len(errors) == 1
    assert "disabled safely" in errors[0].lower()


def test_action_mapper_translates_gestures_and_simulates_safely_when_disabled() -> None:
    backend = MockInputBackend()
    dispatcher = InputDispatcher(backend=backend, os_input_enabled=False)
    mapper = ActionMapper(dispatcher)

    sim = mapper.dispatch((GestureEvent(GestureType.LEFT_CLICK, 10),), (100, 100))
    assert sim.executed_actions == ()
    assert sim.action_label == "Left Click (sim)"
    assert backend.click_history == []

    dispatcher.set_enabled(True)
    res = mapper.dispatch(
        (
            GestureEvent(GestureType.DRAG_START, 20),
            GestureEvent(GestureType.DRAG_MOVE, 20),
        ),
        (300, 400),
    )
    assert res.executed_actions == ("drag_start", "move")
    assert backend.held_buttons == {"left"}
    assert backend.last_position == (300, 400)

    end = mapper.dispatch((GestureEvent(GestureType.DRAG_END, 50),), None)
    assert end.executed_actions == ("drag_end",)
    assert backend.held_buttons == set()


def test_desktop_input_backend_clamps_clicks_and_releases_held_buttons() -> None:
    fake_api = _FakePyAutoGUI()
    backend = DesktopInputBackend(pyautogui_module=fake_api)

    assert backend.screen_size() == (1920, 1080)
    assert fake_api.PAUSE == 0.0
    assert fake_api.FAILSAFE is False

    backend.move_cursor(-50, 2500)
    backend.click("left", 2)
    backend.click("right", 1)
    backend.button_down("left")
    backend.button_down("left")  # duplicate down suppressed
    assert fake_api.moves == [(0, 1079, False)]
    assert fake_api.clicks == [("left", 2, False), ("right", 1, False)]
    assert fake_api.downs == [("left", False)]

    backend.release_all()
    assert fake_api.ups == [("left", False)]
    backend.close()


def test_desktop_backend_caches_screen_size_across_cursor_moves() -> None:
    fake_api = _FakePyAutoGUI()
    backend = DesktopInputBackend(pyautogui_module=fake_api)

    for step in range(5):
        backend.move_cursor(100 + step, 200 + step)

    assert fake_api.size_calls == 1
    assert backend.screen_size() == (1920, 1080)
    assert fake_api.size_calls == 2


def test_dispatcher_and_action_mapper_scroll() -> None:
    backend = MockInputBackend()
    dispatcher = InputDispatcher(backend=backend, os_input_enabled=False)
    mapper = ActionMapper(dispatcher)

    # When OS input disabled, scroll is simulated
    sim = mapper.dispatch((GestureEvent(GestureType.SCROLL_MOVE, 10, scroll_delta=3),), None)
    assert sim.executed_actions == ()
    assert "sim" in (sim.action_label or "").lower()
    assert backend.scroll_history == []

    # When enabled, scroll executes on backend
    dispatcher.set_enabled(True)
    res = mapper.dispatch((GestureEvent(GestureType.SCROLL_MOVE, 20, scroll_delta=-2),), None)
    assert res.executed_actions == ("scroll",)
    assert backend.scroll_history == [-2]


def test_desktop_backend_scroll() -> None:
    fake_api = _FakePyAutoGUI()
    backend = DesktopInputBackend(pyautogui_module=fake_api)

    backend.scroll(5)
    backend.scroll(-3)
    backend.scroll(0)  # zero is no-op
    assert fake_api.scrolls == [
        (5 * backend.wheel_multiplier, False),
        (-3 * backend.wheel_multiplier, False),
    ]

    # Explicit multiplier override
    custom_backend = DesktopInputBackend(pyautogui_module=fake_api, wheel_multiplier=1)
    custom_backend.scroll(4)
    assert fake_api.scrolls[-1] == (4, False)
