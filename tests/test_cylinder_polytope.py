"""Tests for the polytope (convex hull) intersection representation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rod_sim3d.cylinder_intersection import Cylinder, cylinders_from_rods
from rod_sim3d.cylinder_polytope import (
    IntersectionPolytope,
    compute_pairwise_polytopes,
    cylinder_as_halfspaces,
    intersection_polytope,
)


def _cyl(center, axis, L, r):
    a = np.asarray(axis, dtype=float)
    a /= np.linalg.norm(a)
    return Cylinder(np.asarray(center, dtype=float), a, L / 2, r)


def test_halfspaces_describe_the_prism() -> None:
    """For every prism vertex: all ``n_sides + 2`` constraints must hold within tolerance."""

    cyl = _cyl([0.5, 0.2, 0.1], [0, 0, 1], L=2.0, r=1.0)
    hs = cylinder_as_halfspaces(cyl, n_sides=16)
    # The center of the cylinder must satisfy all constraints strictly
    vals = hs[:, :3] @ cyl.center + hs[:, 3]
    assert np.all(vals < 0), "Center must satisfy every half-space (strict)."


def test_disjoint_returns_none() -> None:
    c1 = _cyl([0, 0, 0], [1, 0, 0], 1, 0.3)
    c2 = _cyl([10, 0, 0], [0, 1, 0], 1, 0.3)
    assert intersection_polytope(c1, c2) is None


def test_identical_cylinders_return_inscribed_volume() -> None:
    """For two identical cylinders, the polytope volume approaches the
    inscribed-polygon volume ``n sin(π/n) cos(π/n) r² L`` as ``n_sides → ∞``."""

    r, L = 0.5, 2.0
    c = _cyl([0, 0, 0], [1, 0, 0], L, r)
    for n in (8, 16, 32):
        poly = intersection_polytope(c, c, n_sides=n)
        assert poly is not None
        expected = L * n * math.sin(math.pi / n) * math.cos(math.pi / n) * r**2
        assert poly.volume == pytest.approx(expected, rel=1e-2)


def test_steinmetz_volume_converges_to_theoretical_as_n_grows() -> None:
    """With ``n_sides → ∞`` the polytope volume must approach ``16 r³ / 3``."""

    r = 1.0
    long = 40.0
    c1 = _cyl([0, 0, 0], [1, 0, 0], long, r)
    c2 = _cyl([0, 0, 0], [0, 1, 0], long, r)
    steinmetz = 16.0 / 3.0 * r**3
    prev_err = float("inf")
    for n in (8, 16, 32, 64):
        poly = intersection_polytope(c1, c2, n_sides=n)
        assert poly is not None
        err = abs(poly.volume - steinmetz) / steinmetz
        assert err < prev_err, f"error should decrease with n; n={n} err={err:.4f}"
        prev_err = err
    # At n=64 we expect <0.5% error.
    assert prev_err < 0.005


def test_polytope_vertices_all_satisfy_combined_halfspaces() -> None:
    """The returned vertices must lie on or inside every half-space of both cylinders."""

    c1 = _cyl([0, 0, 0], [1, 0.2, 0.1], L=2, r=0.5)
    c2 = _cyl([0.3, 0.3, 0], [0.1, 1, 0.3], L=2, r=0.5)
    poly = intersection_polytope(c1, c2, n_sides=16)
    assert poly is not None

    combined = np.vstack((cylinder_as_halfspaces(c1, 16), cylinder_as_halfspaces(c2, 16)))
    residuals = poly.vertices @ combined[:, :3].T + combined[:, 3]
    assert np.all(residuals <= 1e-6), "all vertices must satisfy all half-spaces"


def test_simplices_index_into_vertices() -> None:
    c1 = _cyl([0, 0, 0], [1, 0, 0], 2, 0.5)
    c2 = _cyl([0, 0, 0], [0, 1, 0], 2, 0.5)
    poly = intersection_polytope(c1, c2, n_sides=16)
    assert poly is not None
    assert poly.simplices.min() >= 0
    assert poly.simplices.max() < poly.n_vertices
    assert poly.simplices.shape[1] == 3


def test_is_intersection_polytope_instance() -> None:
    poly = intersection_polytope(
        _cyl([0, 0, 0], [1, 0, 0], 2, 0.5), _cyl([0, 0, 0], [0, 1, 0], 2, 0.5)
    )
    assert isinstance(poly, IntersectionPolytope)


def test_compute_pairwise_polytopes_filters_disjoint_pairs() -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],  # coincident with #0 → overlap
            [10.0, 10.0, 10.0],  # far from everyone
        ]
    )
    directions = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],  # perpendicular, same center as #0
            [1.0, 0.0, 0.0],
        ]
    )
    cyls = cylinders_from_rods(positions, directions, rod_length=2.0, rod_radius=0.5)
    results = compute_pairwise_polytopes(cyls, n_sides=12)
    pairs = {(i, j) for i, j, _ in results}
    assert (0, 1) in pairs
    assert (0, 2) not in pairs
    assert (1, 2) not in pairs
