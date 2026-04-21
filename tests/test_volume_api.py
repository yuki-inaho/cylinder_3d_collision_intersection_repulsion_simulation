"""Tests for the unified intersection-volume API.

Covers:
 - ``Method`` enum dispatch
 - ``SliceParams`` / ``PolytopeParams`` wiring
 - Divergence-theorem volume matches scipy ConvexHull.volume
 - Results are consistent between methods for canonical configurations
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from rod_sim3d.cylinder_intersection import Cylinder
from rod_sim3d.cylinder_polytope import intersection_polytope
from rod_sim3d.volume import (
    Method,
    PolytopeParams,
    SliceParams,
    intersection_volume,
    intersection_volume_and_shape,
    polytope_volume_from_mesh,
)


def _cyl(c, a, L, r):
    ax = np.asarray(a, dtype=float)
    ax /= np.linalg.norm(ax)
    return Cylinder(np.asarray(c, dtype=float), ax, L / 2, r)


# --- Enum API --------------------------------------------------------------------------


def test_slice_method_matches_default() -> None:
    c1 = _cyl([0, 0, 0], [1, 0, 0], 40, 1.0)
    c2 = _cyl([0, 0, 0], [0, 1, 0], 40, 1.0)
    v_default = intersection_volume(c1, c2)  # default SLICE
    v_explicit = intersection_volume(c1, c2, Method.SLICE, SliceParams(16))
    assert v_default == pytest.approx(v_explicit, rel=1e-12)


def test_slice_quadrature_accuracy_reaches_machine_epsilon() -> None:
    """Higher Q should reach machine precision for the Steinmetz configuration.

    The slice integrand is C∞ after the sin-substitution, so Gauss-Legendre
    converges super-algebraically; both Q=16 and Q=64 hit float precision.
    """

    c1 = _cyl([0, 0, 0], [1, 0, 0], 40, 1.0)
    c2 = _cyl([0, 0, 0], [0, 1, 0], 40, 1.0)
    expected = 16.0 / 3.0
    for q in (8, 16, 32, 64):
        err = abs(intersection_volume(c1, c2, Method.SLICE, SliceParams(q)) - expected)
        # Q=8 needs some slack; Q>=16 is at machine precision.
        if q >= 16:
            assert err < 1e-10, f"Q={q}: err={err:.3e}"
        else:
            assert err < 1e-3, f"Q={q}: err={err:.3e}"


def test_polytope_method_volume_matches_polytope_object() -> None:
    c1 = _cyl([0, 0, 0], [1, 0, 0], 40, 1.0)
    c2 = _cyl([0, 0, 0], [0, 1, 0], 40, 1.0)
    v_dispatch = intersection_volume(c1, c2, Method.POLYTOPE, PolytopeParams(16))
    poly = intersection_polytope(c1, c2, n_sides=16)
    assert poly is not None
    assert v_dispatch == pytest.approx(poly.volume, rel=1e-12)


def test_polytope_n_sides_converges_to_steinmetz() -> None:
    c1 = _cyl([0, 0, 0], [1, 0, 0], 40, 1.0)
    c2 = _cyl([0, 0, 0], [0, 1, 0], 40, 1.0)
    expected = 16.0 / 3.0
    errors = [
        abs(intersection_volume(c1, c2, Method.POLYTOPE, PolytopeParams(n)) - expected)
        for n in (8, 16, 32, 64)
    ]
    assert errors[0] > errors[-1]
    assert errors[-1] / expected < 0.005


def test_params_type_mismatch_raises() -> None:
    c1 = _cyl([0, 0, 0], [1, 0, 0], 40, 1.0)
    c2 = _cyl([0, 0, 0], [0, 1, 0], 40, 1.0)
    with pytest.raises(TypeError):
        intersection_volume(c1, c2, Method.SLICE, PolytopeParams(16))
    with pytest.raises(TypeError):
        intersection_volume(c1, c2, Method.POLYTOPE, SliceParams(16))


def test_disjoint_returns_zero_for_both_methods() -> None:
    c1 = _cyl([0, 0, 0], [1, 0, 0], 1, 0.3)
    c2 = _cyl([10, 0, 0], [0, 1, 0], 1, 0.3)
    assert intersection_volume(c1, c2, Method.SLICE) == 0.0
    assert intersection_volume(c1, c2, Method.POLYTOPE) == 0.0


def test_intersection_volume_and_shape_returns_both() -> None:
    c1 = _cyl([0, 0, 0], [1, 0, 0], 40, 1.0)
    c2 = _cyl([0, 0, 0], [0, 1, 0], 40, 1.0)
    v, poly = intersection_volume_and_shape(c1, c2, PolytopeParams(16))
    assert v > 0
    assert poly is not None
    assert poly.volume == pytest.approx(v, rel=1e-12)


# --- Divergence theorem ---------------------------------------------------------------


def test_divergence_volume_matches_scipy_for_cube() -> None:
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=float,
    )
    hull = ConvexHull(vertices)
    v_scipy = float(hull.volume)
    v_div_heuristic = polytope_volume_from_mesh(vertices, hull.simplices)
    v_div_normals = polytope_volume_from_mesh(
        vertices, hull.simplices, outward_normals=hull.equations[:, :3]
    )
    assert v_scipy == pytest.approx(1.0)
    assert v_div_heuristic == pytest.approx(v_scipy, rel=1e-12)
    assert v_div_normals == pytest.approx(v_scipy, rel=1e-12)


def test_divergence_volume_matches_scipy_for_cylinder_polytope() -> None:
    c1 = _cyl([0, 0, 0], [1, 0, 0], 40, 1.0)
    c2 = _cyl([0, 0, 0], [0, 1, 0], 40, 1.0)
    for n in (8, 16, 32, 64):
        poly = intersection_polytope(c1, c2, n)
        assert poly is not None
        v = polytope_volume_from_mesh(poly.vertices, poly.simplices)
        assert v == pytest.approx(poly.volume, rel=1e-10)


def test_divergence_volume_translation_invariant() -> None:
    """Translating the mesh should not change its volume (up to float noise)."""

    c = _cyl([0, 0, 0], [1, 0, 0], 2, 0.5)
    poly = intersection_polytope(c, c, n_sides=16)
    assert poly is not None
    v0 = polytope_volume_from_mesh(poly.vertices, poly.simplices)
    shifted = poly.vertices + np.array([10.0, -5.0, 2.0])
    v1 = polytope_volume_from_mesh(shifted, poly.simplices)
    assert v1 == pytest.approx(v0, rel=1e-9)


def test_tetrahedron_volume_is_one_sixth() -> None:
    """Canonical tetrahedron at origin with unit edges."""

    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    hull = ConvexHull(vertices)
    v = polytope_volume_from_mesh(vertices, hull.simplices)
    assert v == pytest.approx(1.0 / 6.0, rel=1e-12)


def test_slice_and_polytope_agree_within_polytope_tolerance() -> None:
    """For a sufficiently refined N-gon, the two methods should agree."""

    c1 = _cyl([0, 0, 0], [1, 0, 0.1], 2, 0.6)
    c2 = _cyl([0.3, 0.2, 0], [0.1, 1, 0.3], 2, 0.5)
    v_slice = intersection_volume(c1, c2, Method.SLICE, SliceParams(64))
    v_poly = intersection_volume(c1, c2, Method.POLYTOPE, PolytopeParams(128))
    # Inscribed N-gon undershoots by factor (N sin(π/N) cos(π/N)) / π.
    undershoot = 128.0 * math.sin(math.pi / 128) * math.cos(math.pi / 128) / math.pi
    assert v_poly == pytest.approx(v_slice * undershoot, rel=2e-2)
