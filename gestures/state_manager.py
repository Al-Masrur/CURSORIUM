"""Finite-state machine for click, double-click, right-click, and drag."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from gestures.gesture_rules import GestureEvent, GestureRules, GestureType, PinchType
from gestures.recognizer import GestureRecognizer
from vision.landmark_features import HandFeatures


class GestureFSMState(str, Enum):
    """States owned by GestureStateManager."""

    IDLE = "Idle"
    LEFT_PINCH_HOLD = "Left Pinch Hold"
    DRAGGING = "Dragging"
    WAIT_DOUBLE_CLICK = "Wait Double-Click"
    SECOND_LEFT_PINCH_HOLD = "Second Pinch Hold"
    RIGHT_PINCH_HOLD = "Right Pinch Hold"
    COOLDOWN = "Cooldown"


@dataclass(frozen=True, slots=True)
class StateUpdate:
    """Result of advancing the gesture FSM for one frame."""

    state: GestureFSMState
    active_pinch: PinchType
    events: tuple[GestureEvent, ...]
    cursor_should_move: bool
    dragging: bool


class GestureStateManager:
    """Resolves click, double-click, right-click, and drag transitions without OS calls.

    Double-click policy:
        A short thumb-index pinch release enters ``WAIT_DOUBLE_CLICK`` for
        ``rules.double_click_interval_ms``. If a second quick thumb-index pinch
        completes inside that window, ``DOUBLE_CLICK`` is emitted without firing
        independent single clicks. If the window expires first, one ``LEFT_CLICK``
        is emitted.
    """

    def __init__(
        self,
        rules: GestureRules | None = None,
        recognizer: GestureRecognizer | None = None,
    ) -> None:
        self._rules = rules or GestureRules()
        self._recognizer = recognizer or GestureRecognizer(self._rules)
        self._state: GestureFSMState = GestureFSMState.IDLE
        self._pinch_start_ms: int = 0
        self._first_click_release_ms: int = 0
        self._cooldown_until_ms: int = 0
        self._tracking_lost_since_ms: int | None = None

    @property
    def rules(self) -> GestureRules:
        return self._rules

    @property
    def state(self) -> GestureFSMState:
        return self._state

    @property
    def is_dragging(self) -> bool:
        return self._state == GestureFSMState.DRAGGING

    def update_rules(self, rules: GestureRules) -> None:
        self._rules = rules
        self._recognizer.update_rules(rules)

    def reset(self, timestamp_ms: int = 0, reason: str = "reset") -> tuple[GestureEvent, ...]:
        """Reset gesture state safely and emit DRAG_END if a drag is active."""
        events: list[GestureEvent] = []
        if self._state == GestureFSMState.DRAGGING:
            events.append(
                GestureEvent(
                    gesture=GestureType.DRAG_END,
                    timestamp_ms=timestamp_ms,
                    reason=reason,
                )
            )
        self._state = GestureFSMState.IDLE
        self._pinch_start_ms = 0
        self._first_click_release_ms = 0
        self._cooldown_until_ms = 0
        self._tracking_lost_since_ms = None
        self._recognizer.reset()
        return tuple(events)

    def update(self, features: HandFeatures, timestamp_ms: int) -> StateUpdate:
        """Advance the gesture state machine by one frame."""
        if not features.hand_detected or features.palm_scale is None:
            if self._state == GestureFSMState.DRAGGING and self._rules.tracking_loss_grace_ms > 0:
                if self._tracking_lost_since_ms is None:
                    self._tracking_lost_since_ms = timestamp_ms
                if timestamp_ms - self._tracking_lost_since_ms < self._rules.tracking_loss_grace_ms:
                    return StateUpdate(
                        state=self._state,
                        active_pinch=PinchType.NONE,
                        events=(),
                        cursor_should_move=False,
                        dragging=True,
                    )
            events = self.reset(timestamp_ms=timestamp_ms, reason="tracking_lost")
            return StateUpdate(
                state=self._state,
                active_pinch=PinchType.NONE,
                events=events,
                cursor_should_move=False,
                dragging=False,
            )

        self._tracking_lost_since_ms = None
        pinch = self._recognizer.update(features, timestamp_ms)
        events: list[GestureEvent] = []

        if self._state == GestureFSMState.WAIT_DOUBLE_CLICK:
            if timestamp_ms - self._first_click_release_ms >= self._rules.double_click_interval_ms:
                events.append(GestureEvent(GestureType.LEFT_CLICK, timestamp_ms, "single_timeout"))
                self._state = GestureFSMState.IDLE

        if self._state == GestureFSMState.COOLDOWN:
            if timestamp_ms >= self._cooldown_until_ms and pinch == PinchType.NONE:
                self._state = GestureFSMState.IDLE
            else:
                can_move = pinch == PinchType.NONE and features.index_extended
                if can_move:
                    events.append(GestureEvent(GestureType.MOVE_CURSOR, timestamp_ms))
                return StateUpdate(
                    state=self._state,
                    active_pinch=pinch,
                    events=tuple(events),
                    cursor_should_move=can_move,
                    dragging=False,
                )

        if self._state == GestureFSMState.IDLE:
            return self._handle_idle(features, pinch, timestamp_ms, events)

        if self._state == GestureFSMState.LEFT_PINCH_HOLD:
            return self._handle_left_pinch_hold(pinch, timestamp_ms, events)

        if self._state == GestureFSMState.DRAGGING:
            return self._handle_dragging(pinch, timestamp_ms, events)

        if self._state == GestureFSMState.WAIT_DOUBLE_CLICK:
            return self._handle_wait_double_click(pinch, timestamp_ms, events)

        if self._state == GestureFSMState.SECOND_LEFT_PINCH_HOLD:
            return self._handle_second_pinch_hold(pinch, timestamp_ms, events)

        if self._state == GestureFSMState.RIGHT_PINCH_HOLD:
            return self._handle_right_pinch_hold(pinch, timestamp_ms, events)

        return StateUpdate(
            state=self._state,
            active_pinch=pinch,
            events=tuple(events),
            cursor_should_move=False,
            dragging=False,
        )

    def _handle_idle(
        self,
        features: HandFeatures,
        pinch: PinchType,
        timestamp_ms: int,
        events: list[GestureEvent],
    ) -> StateUpdate:
        if pinch == PinchType.THUMB_INDEX:
            self._state = GestureFSMState.LEFT_PINCH_HOLD
            self._pinch_start_ms = timestamp_ms
            events.append(GestureEvent(GestureType.PINCH_HOLD, timestamp_ms))
            return StateUpdate(self._state, pinch, tuple(events), False, False)

        if pinch == PinchType.THUMB_MIDDLE:
            self._state = GestureFSMState.RIGHT_PINCH_HOLD
            self._pinch_start_ms = timestamp_ms
            events.append(GestureEvent(GestureType.PINCH_HOLD, timestamp_ms))
            return StateUpdate(self._state, pinch, tuple(events), False, False)

        can_move = features.index_extended
        if can_move:
            events.append(GestureEvent(GestureType.MOVE_CURSOR, timestamp_ms))
        return StateUpdate(self._state, pinch, tuple(events), can_move, False)

    def _handle_left_pinch_hold(
        self,
        pinch: PinchType,
        timestamp_ms: int,
        events: list[GestureEvent],
    ) -> StateUpdate:
        if pinch == PinchType.THUMB_INDEX:
            if timestamp_ms - self._pinch_start_ms >= self._rules.drag_hold_ms:
                self._state = GestureFSMState.DRAGGING
                events.append(GestureEvent(GestureType.DRAG_START, timestamp_ms))
                events.append(GestureEvent(GestureType.DRAG_MOVE, timestamp_ms))
                return StateUpdate(self._state, pinch, tuple(events), True, True)
            events.append(GestureEvent(GestureType.PINCH_HOLD, timestamp_ms))
            return StateUpdate(self._state, pinch, tuple(events), False, False)

        # Quick pinch released before drag threshold
        if pinch == PinchType.THUMB_MIDDLE:
            events.append(GestureEvent(GestureType.LEFT_CLICK, timestamp_ms, "quick_switch"))
            self._state = GestureFSMState.RIGHT_PINCH_HOLD
            self._pinch_start_ms = timestamp_ms
            events.append(GestureEvent(GestureType.PINCH_HOLD, timestamp_ms))
            return StateUpdate(self._state, pinch, tuple(events), False, False)

        self._state = GestureFSMState.WAIT_DOUBLE_CLICK
        self._first_click_release_ms = timestamp_ms
        return StateUpdate(self._state, pinch, tuple(events), False, False)

    def _handle_dragging(
        self,
        pinch: PinchType,
        timestamp_ms: int,
        events: list[GestureEvent],
    ) -> StateUpdate:
        if pinch == PinchType.THUMB_INDEX:
            events.append(GestureEvent(GestureType.DRAG_MOVE, timestamp_ms))
            return StateUpdate(self._state, pinch, tuple(events), True, True)

        events.append(GestureEvent(GestureType.DRAG_END, timestamp_ms, "pinch_released"))
        self._enter_cooldown(timestamp_ms)
        return StateUpdate(self._state, pinch, tuple(events), False, False)

    def _handle_wait_double_click(
        self,
        pinch: PinchType,
        timestamp_ms: int,
        events: list[GestureEvent],
    ) -> StateUpdate:
        if pinch == PinchType.THUMB_INDEX:
            self._state = GestureFSMState.SECOND_LEFT_PINCH_HOLD
            self._pinch_start_ms = timestamp_ms
            events.append(GestureEvent(GestureType.PINCH_HOLD, timestamp_ms))
            return StateUpdate(self._state, pinch, tuple(events), False, False)

        if pinch == PinchType.THUMB_MIDDLE:
            events.append(GestureEvent(GestureType.LEFT_CLICK, timestamp_ms, "before_right_pinch"))
            self._state = GestureFSMState.RIGHT_PINCH_HOLD
            self._pinch_start_ms = timestamp_ms
            events.append(GestureEvent(GestureType.PINCH_HOLD, timestamp_ms))
            return StateUpdate(self._state, pinch, tuple(events), False, False)

        return StateUpdate(self._state, pinch, tuple(events), False, False)

    def _handle_second_pinch_hold(
        self,
        pinch: PinchType,
        timestamp_ms: int,
        events: list[GestureEvent],
    ) -> StateUpdate:
        if pinch == PinchType.THUMB_INDEX:
            if timestamp_ms - self._pinch_start_ms >= self._rules.drag_hold_ms:
                events.append(GestureEvent(GestureType.LEFT_CLICK, timestamp_ms, "before_drag"))
                self._state = GestureFSMState.DRAGGING
                events.append(GestureEvent(GestureType.DRAG_START, timestamp_ms))
                events.append(GestureEvent(GestureType.DRAG_MOVE, timestamp_ms))
                return StateUpdate(self._state, pinch, tuple(events), True, True)
            events.append(GestureEvent(GestureType.PINCH_HOLD, timestamp_ms))
            return StateUpdate(self._state, pinch, tuple(events), False, False)

        events.append(GestureEvent(GestureType.DOUBLE_CLICK, timestamp_ms))
        self._enter_cooldown(timestamp_ms)
        return StateUpdate(self._state, pinch, tuple(events), False, False)

    def _handle_right_pinch_hold(
        self,
        pinch: PinchType,
        timestamp_ms: int,
        events: list[GestureEvent],
    ) -> StateUpdate:
        if pinch == PinchType.THUMB_MIDDLE:
            events.append(GestureEvent(GestureType.PINCH_HOLD, timestamp_ms))
            return StateUpdate(self._state, pinch, tuple(events), False, False)

        events.append(GestureEvent(GestureType.RIGHT_CLICK, timestamp_ms))
        self._enter_cooldown(timestamp_ms)
        return StateUpdate(self._state, pinch, tuple(events), False, False)

    def _enter_cooldown(self, timestamp_ms: int) -> None:
        self._state = GestureFSMState.COOLDOWN
        self._cooldown_until_ms = timestamp_ms + self._rules.click_cooldown_ms

