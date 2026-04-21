"""Toy-problem tests for :mod:`rod_sim3d.cylinder_intersection`.

Each test isolates a closed-form case so that any regression in the slice-integral
implementation is diagnosable from a single failing line.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rod_sim3d.cylinder_intersection import (
    Cylinder,
    iou,
    lens_area,
    overlap_volume,
    rectangle_intersection_area,
)


def _cyl(
    center: tuple[float, float, float],
    axis: tuple[float, float, float],
    half_length: float,
    radius: float,
) -> Cylinder:
    axis_arr = np.asarray(axis, dtype=float)
    axis_arr /= np.linalg.norm(axis_arr)
    return Cylinder(
        center=np.asarray(center, dtype=float),
        axis=axis_arr,
        half_length=half_length,
        radius=radius,
    )


# --- 2D rectangle intersection ---------------------------------------------------------


def test_rectangle_intersection_axis_aligned_full_overlap() -> None:
    area = rectangle_intersection_area((1.0, 0.5), (0.0, 0.0), (1.0, 0.0), (1.0, 0.5))
    assert area == pytest.approx(1.0 * 2.0, rel=1e-12)


def test_rectangle_intersection_disjoint_returns_zero() -> None:
    area = rectangle_intersection_area((1.0, 0.5), (10.0, 0.0), (1.0, 0.0), (1.0, 0.5))
    assert area == 0.0


def test_rectangle_intersection_rotated_known() -> None:
    # Unit square overlapping a square rotated by 45 degrees, same center.
    s = math.sqrt(2.0) / 2.0
    area = rectangle_intersection_area((0.5, 0.5), (0.0, 0.0), (s, s), (0.5, 0.5))
    # Overlap is a regular octagon inscribed in the unit square; area = 2*(sqrt(2)-1).
    assert area == pytest.approx(2.0 * (math.sqrt(2.0) - 1.0), rel=1e-9)


# --- Lens area (closed form) -----------------------------------------------------------


def test_lens_area_equals_full_circle_when_coincident() -> None:
    assert lens_area(0.0, 1.0, 1.0) == pytest.approx(math.pi)


def test_lens_area_zero_when_far() -> None:
    assert lens_area(3.0, 1.0, 1.0) == 0.0


def test_lens_area_small_inside_large() -> None:
    # Small circle of radius 0.2 fully inside a circle of radius 1.0 at distance 0.5.
    assert lens_area(0.5, 0.2, 1.0) == pytest.approx(math.pi * 0.04)


# --- Parallel cylinders ----------------------------------------------------------------


def test_parallel_coaxial_gives_shorter_volume() -> None:
    c1 = _cyl((0, 0, 0), (0, 0, 1), half_length=2.0, radius=1.0)
    c2 = _cyl((0, 0, 1.0), (0, 0, 1), half_length=3.0, radius=1.0)
    # Overlap z-range is [-1, 3], length 3 (capped by c1's top at z=2 => length = 3).
    # Recompute: c1 z-range [-2, 2], c2 [-2, 4]. Overlap [-2, 2] length 4.
    overlap_len = 4.0
    expected = overlap_len * math.pi * 1.0**2
    assert overlap_volume(c1, c2) == pytest.approx(expected, rel=1e-12)


def test_parallel_separated_gives_zero() -> None:
    c1 = _cyl((0, 0, 0), (0, 0, 1), half_length=1.0, radius=1.0)
    c2 = _cyl((3.0, 0, 0), (0, 0, 1), half_length=1.0, radius=1.0)
    assert overlap_volume(c1, c2) == 0.0


def test_parallel_partially_overlapping_matches_lens_times_length() -> None:
    c1 = _cyl((0, 0, 0), (0, 0, 1), half_length=5.0, radius=1.0)
    c2 = _cyl((0.5, 0, 2.0), (0, 0, 1), half_length=5.0, radius=1.0)
    # z-overlap: [-5, 5] ∩ [-3, 7] = [-3, 5], length 8.
    expected = 8.0 * lens_area(0.5, 1.0, 1.0)
    assert overlap_volume(c1, c2) == pytest.approx(expected, rel=1e-12)


# --- Steinmetz solid (canonical non-parallel test) -------------------------------------


def test_steinmetz_perpendicular_equal_radius() -> None:
    r = 1.0
    long = 40.0
    c1 = _cyl((0, 0, 0), (1, 0, 0), half_length=long, radius=r)
    c2 = _cyl((0, 0, 0), (0, 1, 0), half_length=long, radius=r)
    expected = 16.0 / 3.0 * r**3
    got = overlap_volume(c1, c2, quadrature_points=64)
    assert got == pytest.approx(expected, rel=1e-4)


def test_steinmetz_oblique_equal_radius_matches_hubbell() -> None:
    r = 1.2
    long = 60.0
    beta = math.radians(40.0)
    c1 = _cyl((0, 0, 0), (1, 0, 0), half_length=long, radius=r)
    c2 = _cyl(
        (0, 0, 0),
        (math.cos(beta), math.sin(beta), 0),
        half_length=long,
        radius=r,
    )
    expected = 16.0 * r**3 / (3.0 * math.sin(beta))
    got = overlap_volume(c1, c2, quadrature_points=64)
    assert got == pytest.approx(expected, rel=1e-3)


def test_far_apart_non_parallel_gives_zero() -> None:
    c1 = _cyl((0, 0, 0), (1, 0, 0), half_length=1.0, radius=0.5)
    c2 = _cyl((10.0, 0, 0), (0, 1, 0), half_length=1.0, radius=0.5)
    assert overlap_volume(c1, c2) == 0.0


def test_iou_is_bounded_by_zero_and_one() -> None:
    c1 = _cyl((0, 0, 0), (1, 0, 0), half_length=1.0, radius=0.5)
    c2 = _cyl((0.2, 0.3, 0), (0, 1, 0), half_length=1.0, radius=0.5)
    value = iou(c1, c2)
    assert 0.0 <= value <= 1.0


def test_identical_cylinders_iou_is_one() -> None:
    c1 = _cyl((0, 0, 0), (1, 0, 0), half_length=2.0, radius=0.6)
    c2 = _cyl((0, 0, 0), (1, 0, 0), half_length=2.0, radius=0.6)
    assert iou(c1, c2) == pytest.approx(1.0, rel=1e-12)


def test_disjoint_cylinders_iou_is_zero() -> None:
    c1 = _cyl((0, 0, 0), (1, 0, 0), half_length=1.0, radius=0.5)
    c2 = _cyl((5.0, 0, 0), (0, 1, 0), half_length=1.0, radius=0.5)
    assert iou(c1, c2) == 0.0


def test_symmetry_of_overlap_volume() -> None:
    c1 = _cyl((0.1, 0.0, 0.0), (1, 0, 0), half_length=1.5, radius=0.4)
    c2 = _cyl((0.0, 0.2, 0.1), (1, 2, 3), half_length=1.2, radius=0.3)
    assert overlap_volume(c1, c2) == pytest.approx(overlap_volume(c2, c1), rel=1e-10)
