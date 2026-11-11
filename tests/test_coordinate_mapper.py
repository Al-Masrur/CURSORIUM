"""Unit tests for calibration region, coordinate mapping, clamping, mirroring, and EMA smoothing."""

from __future__ import annotations

import pytest

from app.events import NormalizedPoint
from utils.smoothing import ExponentialSmoother, smoothing_to_alpha
from vision.coordinate_mapper import ActiveRegion, CoordinateMapper


def test_active_region_margins_and_centered_calibration() -> None:
    region = ActiveRegion.from_margins(0.2, 0.1)
    assert region.left == pytest.approx(0.2)
    assert region.right == pytest.approx(0.8)
    assert region.top == pytest.approx(0.1)
    assert region.bottom == pytest.approx(0.9)

    centered = ActiveRegion.centered_around(NormalizedPoint(0.05, 0.95), width=0.5, height=0.5)
    assert centered.left == pytest.approx(0.0)
    assert centered.right == pytest.approx(0.5)
    assert centered.top == pytest.approx(0.5)
    assert centered.bottom == pytest.approx(1.0)

    with pytest.raises(ValueError):
        ActiveRegion(left=0.8, top=0.1, right=0.2, bottom=0.9)


def test_coordinate_mapper_corners_center_and_boundary_clamping() -> None:
    mapper = CoordinateMapper(
        screen_width=1920,
        screen_height=1080,
        active_region=ActiveRegion(0.2, 0.2, 0.8, 0.8),
        sensitivity=1.0,
        smoothing=0.0,
        mirror_x=False,
    )

    assert mapper.map_point(NormalizedPoint(0.2, 0.2)) == (0, 0)
    assert mapper.map_point(NormalizedPoint(0.8, 0.8)) == (1919, 1079)
    # 0.3 / 0.6000000000000001 * 1919 = 959.49999... -> 959 on 1920x1080
    assert mapper.map_point(NormalizedPoint(0.5, 0.5)) == (959, 539)

    # Outside active region clamps cleanly to screen bounds
    assert mapper.map_point(NormalizedPoint(0.0, 0.0)) == (0, 0)
    assert mapper.map_point(NormalizedPoint(1.0, 1.0)) == (1919, 1079)


def test_coordinate_mapper_mirror_and_sensitivity() -> None:
    mirrored = CoordinateMapper(
        screen_width=1001,
        screen_height=1001,
        active_region=ActiveRegion(0.0, 0.0, 1.0, 1.0),
        sensitivity=1.0,
        smoothing=0.0,
        mirror_x=True,
    )
    assert mirrored.map_point(NormalizedPoint(0.25, 0.1)) == (750, 100)

    sensitive = CoordinateMapper(
        screen_width=1001,
        screen_height=1001,
        active_region=ActiveRegion(0.0, 0.0, 1.0, 1.0),
        sensitivity=2.0,
        smoothing=0.0,
    )
    # 0.6 is +0.1 from center 0.5 -> * 2.0 = +0.2 -> 0.7 -> 700
    assert sensitive.map_point(NormalizedPoint(0.6, 0.5)) == (700, 500)


def test_exponential_smoothing_and_reset() -> None:
    assert smoothing_to_alpha(0.0) == pytest.approx(1.0)
    assert smoothing_to_alpha(0.75) == pytest.approx(0.25)

    smoother = ExponentialSmoother(alpha=0.5)
    assert smoother.update(100.0, 200.0) == (100.0, 200.0)
    assert smoother.update(200.0, 400.0) == (150.0, 300.0)

    smoother.reset()
    assert smoother.has_state is False
    # Next update after reset starts cleanly at the new point with no jump from old history
    assert smoother.update(10.0, 20.0) == (10.0, 20.0)


def test_mapper_resets_smoothing_on_calibration_change() -> None:
    mapper = CoordinateMapper(
        screen_width=1001,
        screen_height=1001,
        active_region=ActiveRegion(0.0, 0.0, 1.0, 1.0),
        smoothing=0.5,
    )
    assert mapper.map_point(NormalizedPoint(0.0, 0.0)) == (0, 0)
    # Calibration change must reset smoothing so next point isn't pulled toward (0, 0)
    mapper.set_active_region(ActiveRegion(0.0, 0.0, 1.0, 1.0))
    assert mapper.map_point(NormalizedPoint(0.8, 0.8)) == (800, 800)
