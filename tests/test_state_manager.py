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
    middle_extended: bool = False,
    scroll_pose: bool = False,
    scroll_point: NormalizedPoint | None = None,
    open_palm: bool = False,
) -> HandFeatures:
    return HandFeatures(
        hand_detected=hand_detected,
        index_tip=NormalizedPoint(0.5, 0.5) if hand_detected else None,
        index_extended=index_extended,
        middle_extended=middle_extended,
        confidence=0.95 if hand_detected else 0.0,
        palm_scale=0.2 if hand_detected else None,
        thumb_index_ratio=thumb_index_ratio if hand_detected else None,
        thumb_middle_ratio=thumb_middle_ratio if hand_detected else None,
        scroll_pose=scroll_pose,
        scroll_point=scroll_point,
        open_palm=open_palm,
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


def test_scroll_mode_detection_displacement_and_release() -> None:
    rules = GestureRules(scroll_stable_ms=50, scroll_deadzone=0.02, scroll_sensitivity=1.0)
    fsm = GestureStateManager(rules=rules)

    # 1-frame scroll pose does not immediately enter scroll (needs 50ms)
    fsm.update(
        _features(scroll_pose=True, scroll_point=NormalizedPoint(0.5, 0.5)),
        timestamp_ms=0,
    )
    assert fsm.is_scrolling is False

    # After 50ms stable pose -> SCROLLING active, emits SCROLL_START
    res = fsm.update(
        _features(scroll_pose=True, scroll_point=NormalizedPoint(0.5, 0.5)),
        timestamp_ms=60,
    )
    assert fsm.is_scrolling is True
    assert [e.gesture for e in res.events] == [GestureType.SCROLL_START]

    # Small displacement within dead zone (< 0.02) does not emit scroll
    deadzone_move = fsm.update(
        _features(scroll_pose=True, scroll_point=NormalizedPoint(0.5, 0.51)),
        timestamp_ms=90,
    )
    assert [e.gesture for e in deadzone_move.events] == []

    # Significant displacement upward (y = 0.5 -> 0.35, dy = -0.15) emits positive SCROLL_MOVE (scroll up)
    up_move = fsm.update(
        _features(scroll_pose=True, scroll_point=NormalizedPoint(0.5, 0.35)),
        timestamp_ms=120,
    )
    scroll_events = [e for e in up_move.events if e.gesture == GestureType.SCROLL_MOVE]
    assert len(scroll_events) >= 1
    assert scroll_events[0].scroll_delta > 0

    # Significant displacement downward emits negative SCROLL_MOVE (scroll down)
    down_move = fsm.update(
        _features(scroll_pose=True, scroll_point=NormalizedPoint(0.5, 0.60)),
        timestamp_ms=150,
    )
    down_events = [e for e in down_move.events if e.gesture == GestureType.SCROLL_MOVE]
    assert len(down_events) >= 1
    assert down_events[0].scroll_delta < 0

    # Releasing scroll pose exits scrolling with SCROLL_END
    release = fsm.update(
        _features(scroll_pose=False, scroll_point=None),
        timestamp_ms=180,
    )
    assert fsm.is_scrolling is False
    assert [e.gesture for e in release.events] == [GestureType.SCROLL_END]


def test_scroll_mode_suppresses_clicks_and_drag_suppresses_scroll() -> None:
    rules = GestureRules(scroll_stable_ms=30, drag_hold_ms=100)
    fsm = GestureStateManager(rules=rules)

    # 1. While scrolling, accidental pinch does not trigger click
    fsm.update(
        _features(scroll_pose=True, scroll_point=NormalizedPoint(0.5, 0.5)),
        timestamp_ms=0,
    )
    fsm.update(
        _features(scroll_pose=True, scroll_point=NormalizedPoint(0.5, 0.5)),
        timestamp_ms=40,
    )
    assert fsm.is_scrolling is True

    # Accidental pinch while scrolling pose remains active
    pinch_during_scroll = fsm.update(
        _features(scroll_pose=True, scroll_point=NormalizedPoint(0.5, 0.5), thumb_index_ratio=0.20),
        timestamp_ms=80,
    )
    assert fsm.is_scrolling is True
    assert GestureType.LEFT_CLICK not in [e.gesture for e in pinch_during_scroll.events]
    assert GestureType.PINCH_HOLD not in [e.gesture for e in pinch_during_scroll.events]

    # Reset
    fsm.reset(timestamp_ms=200)

    # 2. While dragging, scroll pose is suppressed
    fsm.update(_features(thumb_index_ratio=0.20), timestamp_ms=300)
    fsm.update(_features(thumb_index_ratio=0.20), timestamp_ms=350)
    fsm.update(_features(thumb_index_ratio=0.20), timestamp_ms=460)
    assert fsm.is_dragging is True

    # Candidate scroll pose while actively dragging
    drag_move = fsm.update(
        _features(thumb_index_ratio=0.20, scroll_pose=True, scroll_point=NormalizedPoint(0.5, 0.5)),
        timestamp_ms=500,
    )
    assert fsm.is_dragging is True
    assert fsm.is_scrolling is False
    assert GestureType.SCROLL_START not in [e.gesture for e in drag_move.events]


def test_open_palm_pause_and_resume_toggle_with_cooldown() -> None:
    rules = GestureRules(pause_hold_ms=200, pause_cooldown_ms=400)
    fsm = GestureStateManager(rules=rules)

    # Open palm held for less than 200ms -> not paused yet
    fsm.update(_features(open_palm=True), timestamp_ms=0)
    res = fsm.update(_features(open_palm=True), timestamp_ms=100)
    assert fsm.is_paused is False
    assert GestureType.PAUSE_TOGGLED not in [e.gesture for e in res.events]

    # Hold past 200ms -> pauses!
    paused = fsm.update(_features(open_palm=True), timestamp_ms=220)
    assert fsm.is_paused is True
    assert [e.gesture for e in paused.events] == [GestureType.PAUSE_TOGGLED]
    assert paused.events[0].reason == "paused"

    # Still holding palm inside cooldown window (< 220 + 400 = 620ms) -> remains paused, does not rapidly oscillate
    still_paused = fsm.update(_features(open_palm=True), timestamp_ms=350)
    assert fsm.is_paused is True
    assert [e.gesture for e in still_paused.events] == []

    # After cooldown expires, continuing open palm toggles resume
    resumed = fsm.update(_features(open_palm=True), timestamp_ms=650)
    assert fsm.is_paused is False
    assert [e.gesture for e in resumed.events] == [GestureType.PAUSE_TOGGLED]
    assert resumed.events[0].reason == "resumed"


def test_pause_suppresses_all_gestures_and_releases_held_buttons() -> None:
    rules = GestureRules(drag_hold_ms=100)
    fsm = GestureStateManager(rules=rules)

    # Start drag (pinch confirmed at 50ms, drag confirmed at 50+100=150ms)
    fsm.update(_features(thumb_index_ratio=0.20), timestamp_ms=0)
    fsm.update(_features(thumb_index_ratio=0.20), timestamp_ms=50)
    fsm.update(_features(thumb_index_ratio=0.20), timestamp_ms=160)
    assert fsm.is_dragging is True

    # Programmatic set_paused releases drag safely
    events = fsm.set_paused(True, timestamp_ms=200)
    assert fsm.is_paused is True
    assert fsm.is_dragging is False
    assert GestureType.DRAG_END in [e.gesture for e in events]
    assert GestureType.PAUSE_TOGGLED in [e.gesture for e in events]

    # While paused, pinch or movement updates produce no cursor movement or clicks
    move_attempt = fsm.update(_features(thumb_index_ratio=0.20, index_extended=True), timestamp_ms=250)
    assert move_attempt.cursor_should_move is False
    assert move_attempt.events == ()
    assert fsm.is_dragging is False

