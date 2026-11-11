"""Unit tests for GestureStateManager finite-state machine (click, double-click, right-click, drag)."""

from __future__ import annotations

from app.events import NormalizedPoint
from gestures.gesture_rules import GestureRules, GestureType
from gestures.state_manager import GestureFSMState, GestureStateManager
from vision.landmark_features import HandFeatures


def _features(
    *,
    thumb_index_ratio: float = 0.80,
    thumb_middle_ratio: float = 0.80,
    hand_detected: bool = True,
    index_extended: bool = True,
) -> HandFeatures:
    return HandFeatures(
        hand_detected=hand_detected,
        index_tip=NormalizedPoint(0.5, 0.5) if hand_detected else None,
        index_extended=index_extended,
        confidence=0.95 if hand_detected else 0.0,
        palm_scale=0.2 if hand_detected else None,
        thumb_index_ratio=thumb_index_ratio if hand_detected else None,
        thumb_middle_ratio=thumb_middle_ratio if hand_detected else None,
    )


def _rules() -> GestureRules:
    return GestureRules(
        pinch_activate_ratio=0.32,
        pinch_release_ratio=0.45,
        pinch_stable_ms=30,
        drag_hold_ms=300,
        double_click_interval_ms=250,
        click_cooldown_ms=150,
    )


def test_single_left_click_on_quick_pinch_release_after_window() -> None:
    fsm = GestureStateManager(rules=_rules())

    # Pinch and hold briefly (< 300ms), then release
    fsm.update(_features(thumb_index_ratio=0.20), timestamp_ms=0)
    fsm.update(_features(thumb_index_ratio=0.20), timestamp_ms=40)
    assert fsm.state == GestureFSMState.LEFT_PINCH_HOLD

    rel = fsm.update(_features(thumb_index_ratio=0.60), timestamp_ms=100)
    assert fsm.state == GestureFSMState.WAIT_DOUBLE_CLICK
    assert [e.gesture for e in rel.events] == []

    # Before double_click_interval_ms expires -> still waiting, no click yet
    wait = fsm.update(_features(thumb_index_ratio=0.70), timestamp_ms=250)
    assert [e.gesture for e in wait.events] == []

    # After 250ms window expires -> single LEFT_CLICK emitted once
    fired = fsm.update(_features(thumb_index_ratio=0.70), timestamp_ms=360)
    assert [e.gesture for e in fired.events if e.gesture != GestureType.MOVE_CURSOR] == [
        GestureType.LEFT_CLICK
    ]
    assert fsm.state == GestureFSMState.IDLE


def test_double_click_emits_no_extra_single_clicks() -> None:
    fsm = GestureStateManager(rules=_rules())
    emitted: list[GestureType] = []

    # First pinch
    for ts, r in ((0, 0.20), (35, 0.20), (90, 0.60)):
        upd = fsm.update(_features(thumb_index_ratio=r), timestamp_ms=ts)
        emitted.extend(e.gesture for e in upd.events)

    assert fsm.state == GestureFSMState.WAIT_DOUBLE_CLICK

    # Second quick pinch within 250ms
    for ts, r in ((140, 0.20), (175, 0.20), (220, 0.60)):
        upd = fsm.update(_features(thumb_index_ratio=r), timestamp_ms=ts)
        emitted.extend(e.gesture for e in upd.events)

    assert GestureType.DOUBLE_CLICK in emitted
    assert GestureType.LEFT_CLICK not in emitted
    assert emitted.count(GestureType.DOUBLE_CLICK) == 1


def test_two_slow_pinches_produce_two_single_clicks() -> None:
    fsm = GestureStateManager(rules=_rules())
    clicks: list[GestureType] = []

    # Pinch 1
    for ts, r in ((0, 0.20), (40, 0.20), (90, 0.60), (400, 0.70)):
        upd = fsm.update(_features(thumb_index_ratio=r), timestamp_ms=ts)
        clicks.extend(
            e.gesture for e in upd.events if e.gesture in (GestureType.LEFT_CLICK, GestureType.DOUBLE_CLICK)
        )

    # Pinch 2 well after window
    for ts, r in ((450, 0.20), (490, 0.20), (540, 0.60), (850, 0.70)):
        upd = fsm.update(_features(thumb_index_ratio=r), timestamp_ms=ts)
        clicks.extend(
            e.gesture for e in upd.events if e.gesture in (GestureType.LEFT_CLICK, GestureType.DOUBLE_CLICK)
        )

    assert clicks == [GestureType.LEFT_CLICK, GestureType.LEFT_CLICK]


def test_thumb_middle_pinch_emits_right_click_once_and_respects_cooldown() -> None:
    fsm = GestureStateManager(rules=_rules())

    fsm.update(_features(thumb_middle_ratio=0.20), timestamp_ms=0)
    fsm.update(_features(thumb_middle_ratio=0.20), timestamp_ms=40)
    assert fsm.state == GestureFSMState.RIGHT_PINCH_HOLD

    rel = fsm.update(_features(thumb_middle_ratio=0.65), timestamp_ms=90)
    assert [e.gesture for e in rel.events] == [GestureType.RIGHT_CLICK]
    assert fsm.state == GestureFSMState.COOLDOWN

    # Trying to re-pinch immediately inside cooldown (90 + 150 = 240ms) is ignored
    fsm.update(_features(thumb_middle_ratio=0.20), timestamp_ms=120)
    fsm.update(_features(thumb_middle_ratio=0.20), timestamp_ms=160)
    rel2 = fsm.update(_features(thumb_middle_ratio=0.65), timestamp_ms=200)
    assert [e.gesture for e in rel2.events if e.gesture == GestureType.RIGHT_CLICK] == []


def test_drag_start_move_and_release_plus_tracking_loss_safety() -> None:
    fsm = GestureStateManager(rules=_rules())

    # Start pinch at 0 -> confirmed at 40 -> hold past 300ms (40 + 300 = 340)
    fsm.update(_features(thumb_index_ratio=0.20), timestamp_ms=0)
    fsm.update(_features(thumb_index_ratio=0.20), timestamp_ms=40)
    drag_start = fsm.update(_features(thumb_index_ratio=0.20, index_extended=False), timestamp_ms=350)
    assert fsm.is_dragging is True
    assert drag_start.cursor_should_move is True
    assert [e.gesture for e in drag_start.events] == [
        GestureType.DRAG_START,
        GestureType.DRAG_MOVE,
    ]

    # Moving while pinched emits DRAG_MOVE
    drag_move = fsm.update(_features(thumb_index_ratio=0.25, index_extended=False), timestamp_ms=380)
    assert [e.gesture for e in drag_move.events] == [GestureType.DRAG_MOVE]

    # Releasing pinch emits DRAG_END
    drag_end = fsm.update(_features(thumb_index_ratio=0.60), timestamp_ms=420)
    assert fsm.is_dragging is False
    assert [e.gesture for e in drag_end.events] == [GestureType.DRAG_END]

    # Now test tracking loss while dragging
    fsm.update(_features(thumb_index_ratio=0.70), timestamp_ms=700)  # clear cooldown
    fsm.update(_features(thumb_index_ratio=0.20), timestamp_ms=720)
    fsm.update(_features(thumb_index_ratio=0.20), timestamp_ms=760)
    fsm.update(_features(thumb_index_ratio=0.20), timestamp_ms=1100)
    assert fsm.is_dragging is True

    lost = fsm.update(_features(hand_detected=False), timestamp_ms=1130)
    assert fsm.is_dragging is False
    assert [e.gesture for e in lost.events] == [GestureType.DRAG_END]


def test_drag_survives_brief_dropout_within_grace_window_and_releases_after() -> None:
    rules = GestureRules(pinch_stable_ms=0, drag_hold_ms=200, tracking_loss_grace_ms=90)
    fsm = GestureStateManager(rules=rules)

    fsm.update(_features(thumb_index_ratio=0.20), timestamp_ms=100)
    fsm.update(_features(thumb_index_ratio=0.20), timestamp_ms=320)
    assert fsm.is_dragging is True

    # Brief 1-frame occlusion within grace freezes movement without dropping drag
    brief_loss = fsm.update(_features(hand_detected=False), timestamp_ms=360)
    assert fsm.is_dragging is True
    assert brief_loss.dragging is True
    assert brief_loss.cursor_should_move is False
    assert brief_loss.events == ()

    # Reacquiring pinched hand resumes drag smoothly
    resumed = fsm.update(_features(thumb_index_ratio=0.20), timestamp_ms=390)
    assert fsm.is_dragging is True
    assert resumed.cursor_should_move is True
    assert [e.gesture for e in resumed.events] == [GestureType.DRAG_MOVE]

    # Prolonged loss past tracking_loss_grace_ms emits DRAG_END safely
    fsm.update(_features(hand_detected=False), timestamp_ms=420)
    expired = fsm.update(_features(hand_detected=False), timestamp_ms=520)
    assert fsm.is_dragging is False
    assert [e.gesture for e in expired.events] == [GestureType.DRAG_END]

