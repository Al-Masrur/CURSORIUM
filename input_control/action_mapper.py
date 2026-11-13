"""Translates validated gesture events into serialized InputDispatcher commands."""

from __future__ import annotations

from dataclasses import dataclass

from gestures.gesture_rules import GestureEvent, GestureType
from input_control.dispatcher import InputDispatcher


@dataclass(frozen=True, slots=True)
class ActionDispatchResult:
    """Summary of OS or simulated actions produced for a frame."""

    executed_actions: tuple[str, ...]
    action_label: str | None


class ActionMapper:
    """Maps abstract GestureEvents to InputDispatcher calls without direct OS APIs."""

    def __init__(self, dispatcher: InputDispatcher) -> None:
        self._dispatcher = dispatcher

    def dispatch(
        self,
        events: tuple[GestureEvent, ...],
        mapped_position: tuple[int, int] | None,
    ) -> ActionDispatchResult:
        """Execute or simulate commands for the given frame's gesture events."""
        executed: list[str] = []
        last_label: str | None = None
        os_enabled = self._dispatcher.is_enabled

        for event in events:
            gesture = event.gesture

            if gesture in (GestureType.MOVE_CURSOR, GestureType.DRAG_MOVE):
                if mapped_position is not None:
                    moved = self._dispatcher.move_cursor(mapped_position[0], mapped_position[1])
                    if moved:
                        executed.append("move")
                if gesture == GestureType.DRAG_MOVE:
                    last_label = "Dragging" if os_enabled else "Dragging (sim)"

            elif gesture == GestureType.LEFT_CLICK:
                if self._dispatcher.click(button="left", clicks=1):
                    executed.append("left_click")
                last_label = "Left Click" if os_enabled else "Left Click (sim)"

            elif gesture == GestureType.DOUBLE_CLICK:
                if self._dispatcher.click(button="left", clicks=2):
                    executed.append("double_click")
                last_label = "Double-Click" if os_enabled else "Double-Click (sim)"

            elif gesture == GestureType.RIGHT_CLICK:
                if self._dispatcher.click(button="right", clicks=1):
                    executed.append("right_click")
                last_label = "Right Click" if os_enabled else "Right Click (sim)"

            elif gesture == GestureType.DRAG_START:
                if self._dispatcher.button_down(button="left"):
                    executed.append("drag_start")
                last_label = "Drag Start (Left Down)" if os_enabled else "Drag Start (sim)"

            elif gesture == GestureType.DRAG_END:
                if self._dispatcher.button_up(button="left"):
                    executed.append("drag_end")
                last_label = "Drag Released" if os_enabled else "Drag Released (sim)"

            elif gesture == GestureType.SCROLL_START:
                last_label = "Scroll Mode" if os_enabled else "Scroll Mode (sim)"

            elif gesture == GestureType.SCROLL_MOVE:
                if event.scroll_delta != 0:
                    if self._dispatcher.scroll(event.scroll_delta):
                        executed.append("scroll")
                    direction = "Up" if event.scroll_delta > 0 else "Down"
                    last_label = (
                        f"Scroll {direction} ({event.scroll_delta:+d})"
                        if os_enabled
                        else f"Scroll {direction} (sim {event.scroll_delta:+d})"
                    )

            elif gesture == GestureType.SCROLL_END:
                last_label = "Scroll Released" if os_enabled else "Scroll Released (sim)"

            elif gesture == GestureType.PAUSE_TOGGLED:
                last_label = f"Tracking {event.reason.capitalize()}"

        return ActionDispatchResult(
            executed_actions=tuple(executed),
            action_label=last_label,
        )

