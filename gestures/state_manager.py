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
    SCROLLING = "Scrolling"
    PAUSED = "Paused"


@dataclass(frozen=True, slots=True)
class StateUpdate:
    """Result of advancing the gesture FSM for one frame."""

    state: GestureFSMState
    active_pinch: PinchType
    events: tuple[GestureEvent, ...]
    cursor_should_move: bool
    dragging: bool
    scrolling: bool = False
    scroll_delta: int = 0


class GestureStateManager:
    """Resolves click, double-click, right-click, drag, scroll, and pause transitions without OS calls.

    Double-click policy:
        A short thumb-index pinch release enters ``WAIT_DOUBLE_CLICK`` for
        ``rules.double_click_interval_ms``. If a second quick thumb-index pinch
        completes inside that window, ``DOUBLE_CLICK`` is emitted without firing
        independent single clicks. If the window expires first, one ``LEFT_CLICK``
        is emitted.

    Scroll & Pause policy:
        - Two fingers extended (index + middle) enters ``SCROLLING`` after ``rules.scroll_stable_ms``.
          Vertical hand displacement is converted into directional scroll steps with dead zone.
          Scroll suppresses clicks; drag suppresses scroll.
        - Open palm held steadily for ``rules.pause_hold_ms`` toggles ``PAUSED``.
          When paused, all cursor movements and gesture inputs are suspended.
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
        self._scroll_candidate_since_ms: int | None = None
        self._scroll_anchor_y: float = 0.5
        self._scroll_accumulator: float = 0.0
        self._open_palm_since_ms: int | None = None
        self._pause_cooldown_until_ms: int = 0

    @property
    def rules(self) -> GestureRules:
        return self._rules

    @property
    def state(self) -> GestureFSMState:
        return self._state

    @property
    def is_dragging(self) -> bool:
        return self._state == GestureFSMState.DRAGGING

    @property
    def is_scrolling(self) -> bool:
        return self._state == GestureFSMState.SCROLLING

    @property
    def is_paused(self) -> bool:
        return self._state == GestureFSMState.PAUSED

    def update_rules(self, rules: GestureRules) -> None:
        self._rules = rules
        self._recognizer.update_rules(rules)

    def set_paused(self, paused: bool, timestamp_ms: int = 0) -> tuple[GestureEvent, ...]:
        """Explicitly set or clear the PAUSED state."""
        events: list[GestureEvent] = []
        if paused and self._state != GestureFSMState.PAUSED:
            if self._state == GestureFSMState.DRAGGING:
                events.append(
                    GestureEvent(
                        gesture=GestureType.DRAG_END,
                        timestamp_ms=timestamp_ms,
                        reason="paused",
                    )
                )
            elif self._state == GestureFSMState.SCROLLING:
                events.append(
                    GestureEvent(
                        gesture=GestureType.SCROLL_END,
                        timestamp_ms=timestamp_ms,
                        reason="paused",
                    )
                )
            self._state = GestureFSMState.PAUSED
            self._open_palm_since_ms = None
            self._pause_cooldown_until_ms = timestamp_ms + self._rules.pause_cooldown_ms
            events.append(GestureEvent(GestureType.PAUSE_TOGGLED, timestamp_ms, reason="paused"))
        elif not paused and self._state == GestureFSMState.PAUSED:
            self._state = GestureFSMState.IDLE
            self._open_palm_since_ms = None
            self._pause_cooldown_until_ms = timestamp_ms + self._rules.pause_cooldown_ms
            events.append(GestureEvent(GestureType.PAUSE_TOGGLED, timestamp_ms, reason="resumed"))
        return tuple(events)

    def reset(self, timestamp_ms: int = 0, reason: str = "reset") -> tuple[GestureEvent, ...]:
        """Reset gesture state safely and emit DRAG_END or SCROLL_END if active."""
        events: list[GestureEvent] = []
        if self._state == GestureFSMState.DRAGGING:
            events.append(
                GestureEvent(
                    gesture=GestureType.DRAG_END,
                    timestamp_ms=timestamp_ms,
                    reason=reason,
                )
            )
        elif self._state == GestureFSMState.SCROLLING:
            events.append(
                GestureEvent(
                    gesture=GestureType.SCROLL_END,
                    timestamp_ms=timestamp_ms,
                    reason=reason,
                )
            )
        self._state = GestureFSMState.IDLE
        self._pinch_start_ms = 0
        self._first_click_release_ms = 0
        self._cooldown_until_ms = 0
        self._tracking_lost_since_ms = None
        self._scroll_candidate_since_ms = None
        self._scroll_anchor_y = 0.5
        self._scroll_accumulator = 0.0
        self._open_palm_since_ms = None
        self._recognizer.reset()
        return tuple(events)

    def update(self, features: HandFeatures, timestamp_ms: int) -> StateUpdate:
        """Advance the gesture state machine by one frame."""
        if not features.hand_detected or features.palm_scale is None:
            self._scroll_candidate_since_ms = None
            self._open_palm_since_ms = None
            if self._state == GestureFSMState.PAUSED:
                return StateUpdate(
                    state=self._state,
                    active_pinch=PinchType.NONE,
                    events=(),
                    cursor_should_move=False,
                    dragging=False,
                    scrolling=False,
                )
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
                        scrolling=False,
                    )
            events = self.reset(timestamp_ms=timestamp_ms, reason="tracking_lost")
            return StateUpdate(
                state=self._state,
                active_pinch=PinchType.NONE,
                events=events,
                cursor_should_move=False,
                dragging=False,
                scrolling=False,
            )

        self._tracking_lost_since_ms = None
        events: list[GestureEvent] = []

        # Check open-palm hold to pause / resume
        if features.open_palm:
            if self._open_palm_since_ms is None:
                self._open_palm_since_ms = timestamp_ms
            elif timestamp_ms - self._open_palm_since_ms >= self._rules.pause_hold_ms:
                if timestamp_ms >= self._pause_cooldown_until_ms:
                    if self._state == GestureFSMState.PAUSED:
                        self._state = GestureFSMState.IDLE
                        self._pause_cooldown_until_ms = (
                            timestamp_ms + self._rules.pause_cooldown_ms
                        )
                        self._open_palm_since_ms = None
                        events.append(
                            GestureEvent(
                                GestureType.PAUSE_TOGGLED,
                                timestamp_ms,
                                reason="resumed",
                            )
                        )
                        return StateUpdate(
                            state=self._state,
                            active_pinch=PinchType.NONE,
                            events=tuple(events),
                            cursor_should_move=False,
                            dragging=False,
                            scrolling=False,
                        )
                    else:
                        if self._state == GestureFSMState.DRAGGING:
                            events.append(
                                GestureEvent(
                                    GestureType.DRAG_END,
                                    timestamp_ms,
                                    reason="paused",
                                )
                            )
                        elif self._state == GestureFSMState.SCROLLING:
                            events.append(
                                GestureEvent(
                                    GestureType.SCROLL_END,
                                    timestamp_ms,
                                    reason="paused",
                                )
                            )
                        self._state = GestureFSMState.PAUSED
                        self._pause_cooldown_until_ms = (
                            timestamp_ms + self._rules.pause_cooldown_ms
                        )
                        self._open_palm_since_ms = None
                        events.append(
                            GestureEvent(
                                GestureType.PAUSE_TOGGLED,
                                timestamp_ms,
                                reason="paused",
                            )
                        )
                        return StateUpdate(
                            state=self._state,
                            active_pinch=PinchType.NONE,
                            events=tuple(events),
                            cursor_should_move=False,
                            dragging=False,
                            scrolling=False,
                        )
        else:
            self._open_palm_since_ms = None

        if self._state == GestureFSMState.PAUSED:
            return StateUpdate(
                state=self._state,
                active_pinch=PinchType.NONE,
                events=tuple(events),
                cursor_should_move=False,
                dragging=False,
                scrolling=False,
            )

        if self._state == GestureFSMState.SCROLLING:
            return self._handle_scrolling(features, timestamp_ms, events)

        pinch = self._recognizer.update(features, timestamp_ms)

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
                    scrolling=False,
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
            scrolling=False,
        )

    def _handle_idle(
        self,
        features: HandFeatures,
        pinch: PinchType,
        timestamp_ms: int,
        events: list[GestureEvent],
    ) -> StateUpdate:
        if pinch == PinchType.THUMB_INDEX:
            self._scroll_candidate_since_ms = None
            self._state = GestureFSMState.LEFT_PINCH_HOLD
            self._pinch_start_ms = timestamp_ms
            events.append(GestureEvent(GestureType.PINCH_HOLD, timestamp_ms))
            return StateUpdate(self._state, pinch, tuple(events), False, False)

        if pinch == PinchType.THUMB_MIDDLE:
            self._scroll_candidate_since_ms = None
            self._state = GestureFSMState.RIGHT_PINCH_HOLD
            self._pinch_start_ms = timestamp_ms
            events.append(GestureEvent(GestureType.PINCH_HOLD, timestamp_ms))
            return StateUpdate(self._state, pinch, tuple(events), False, False)

        # Check two-finger scroll pose
        if features.scroll_pose and features.scroll_point is not None:
            if self._scroll_candidate_since_ms is None:
                self._scroll_candidate_since_ms = timestamp_ms
            elif timestamp_ms - self._scroll_candidate_since_ms >= self._rules.scroll_stable_ms:
                self._state = GestureFSMState.SCROLLING
                self._scroll_anchor_y = features.scroll_point.y
                self._scroll_accumulator = 0.0
                self._scroll_candidate_since_ms = None
                events.append(GestureEvent(GestureType.SCROLL_START, timestamp_ms))
                return StateUpdate(
                    self._state,
                    pinch,
                    tuple(events),
                    cursor_should_move=False,
                    dragging=False,
                    scrolling=True,
                )
        else:
            self._scroll_candidate_since_ms = None

        can_move = features.index_extended
        if can_move:
            events.append(GestureEvent(GestureType.MOVE_CURSOR, timestamp_ms))
        return StateUpdate(self._state, pinch, tuple(events), can_move, False)

    def _handle_scrolling(
        self,
        features: HandFeatures,
        timestamp_ms: int,
        events: list[GestureEvent],
    ) -> StateUpdate:
        if not features.scroll_pose or features.scroll_point is None:
            self._state = GestureFSMState.IDLE
            self._scroll_candidate_since_ms = None
            events.append(
                GestureEvent(
                    GestureType.SCROLL_END,
                    timestamp_ms,
                    reason="pose_released",
                )
            )
            self._enter_cooldown(timestamp_ms)
            return StateUpdate(
                state=self._state,
                active_pinch=PinchType.NONE,
                events=tuple(events),
                cursor_should_move=False,
                dragging=False,
                scrolling=False,
            )

        current_y = features.scroll_point.y
        dy = current_y - self._scroll_anchor_y
        delta_steps = 0
        if abs(dy) > self._rules.scroll_deadzone:
            excess = dy - (self._rules.scroll_deadzone if dy > 0 else -self._rules.scroll_deadzone)
            self._scroll_accumulator += -excess * self._rules.scroll_sensitivity * 30.0
            self._scroll_anchor_y = current_y
            delta_steps = int(self._scroll_accumulator)
            if delta_steps != 0:
                self._scroll_accumulator -= delta_steps
                events.append(
                    GestureEvent(
                        GestureType.SCROLL_MOVE,
                        timestamp_ms,
                        scroll_delta=delta_steps,
                    )
                )

        return StateUpdate(
            state=self._state,
            active_pinch=PinchType.NONE,
            events=tuple(events),
            cursor_should_move=False,
            dragging=False,
            scrolling=True,
            scroll_delta=delta_steps,
        )

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

