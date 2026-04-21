"""Polytope (convex hull) approximation of the intersection of two cylinders.

This complements :mod:`rod_sim3d.cylinder_intersection`, which returns only the
scalar volume. When we need the *shape* of ``C_1 ∩ C_2`` — for mesh rendering,
CSG, or downstream geometric queries — an explicit convex polytope is the
right representation.

Strategy
--------
Approximate each cylinder as an ``N_sides``-prism (two axial caps + ``N_sides``
side faces), giving ``N_sides + 2`` half-spaces per cylinder. The intersection of
the two half-space sets is itself a convex polytope, and we recover its vertices
via :class:`scipy.spatial.HalfspaceIntersection` (Qhull). A subsequent
:class:`scipy.spatial.ConvexHull` pass gives us triangular faces.

The polytope approximation is an *inscribed* one: the polygonal cross-section
has radius ``r cos(π / N_sides)``, smaller than the true cylinder. As
``N_sides → ∞`` the volume converges to the exact ``V_cap``.

Pure functions, numpy + scipy only — ready for a future Rust port.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection

from rod_sim3d._array import FloatArray, IntArray
from rod_sim3d.cylinder_intersection import (
    Cylinder,
    _aabb_broad_phase,
    _aabb_overlaps,
    _min_axis_distance,
)


@dataclass(slots=True, frozen=True)
class IntersectionPolytope:
    """Convex polytope approximating ``C_1 ∩ C_2``.

    Attributes
    ----------
    vertices : shape ``(V, 3)``
        Polytope vertices in 3D.
    simplices : shape ``(F, 3)``
        Triangular face indices into ``vertices``.
    volume : float
        Volume of the polytope (for validation against the slice integral).
    """

    vertices: FloatArray
    simplices: IntArray
    volume: float

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.simplices.shape[0])


def intersection_polytope(
    c1: Cylinder, c2: Cylinder, n_sides: int = 16
) -> IntersectionPolytope | None:
    """Return a polytope approximating ``C_1 ∩ C_2``, or ``None`` if empty.

    ``n_sides`` controls the fidelity of the side faces; larger values give tighter
    approximations at quadratic cost (``O(n_sides^2)`` from Qhull). Typical values:

    - 8: coarse preview (~10% volume error)
    - 16: good default (~3% volume error)
    - 32: near-exact for rendering

    Short-circuits:
      - AABB rejection is ~20 µs, vs ~2 ms for the LP that would otherwise detect it.
      - Axis-segment distance >= r1 + r2 proves the cylinders don't touch and is a
        sufficient (not necessary) rejection test.
    """

    if not _aabb_overlaps(c1, c2) or _min_axis_distance(c1, c2) >= c1.radius + c2.radius:
        return None

    halfspaces = _combined_halfspaces(c1, c2, n_sides)
    interior = _interior_point(c1, c2, halfspaces)
    if interior is None:
        return None

    vertices = _safe_halfspace_vertices(halfspaces, interior)
    if vertices is None or vertices.shape[0] < 4:
        return None

    hull = _safe_convex_hull(vertices)
    if hull is None:
        return None

    return IntersectionPolytope(
        vertices=vertices[hull.vertices],
        simplices=_reindex_simplices(hull.simplices, hull.vertices),
        volume=float(hull.volume),
    )


def _safe_halfspace_vertices(halfspaces: FloatArray, interior: FloatArray) -> FloatArray | None:
    """Run Qhull's half-space intersection and dedup, swallowing degenerate errors."""

    try:
        hsi = HalfspaceIntersection(halfspaces, interior)
    except Exception:
        return None
    return _deduplicate(np.asarray(hsi.intersections, dtype=float))


def _safe_convex_hull(vertices: FloatArray) -> ConvexHull | None:
    try:
        return ConvexHull(vertices)
    except Exception:
        return None


def compute_pairwise_polytopes(
    cylinders: list[Cylinder], n_sides: int = 16
) -> list[tuple[int, int, IntersectionPolytope]]:
    """Return ``(i, j, polytope)`` for every overlapping pair.

    Uses the vectorized AABB broad-phase from :mod:`rod_sim3d.cylinder_intersection`
    so that disjoint pairs never reach the ``HalfspaceIntersection`` call.
    """

    candidates = _aabb_broad_phase(cylinders)
    out: list[tuple[int, int, IntersectionPolytope]] = []
    for i, j in candidates:
        poly = intersection_polytope(cylinders[i], cylinders[j], n_sides)
        if poly is not None:
            out.append((i, j, poly))
    return out


def cylinder_as_halfspaces(cyl: Cylinder, n_sides: int) -> FloatArray:
    """Return ``(n_sides + 2, 4)`` half-spaces ``[a, b]`` with ``a·x + b ≤ 0`` for the prism."""

    axis = cyl.axis / np.linalg.norm(cyl.axis)
    e1, e2 = _perpendicular_frame(axis)
    inscribed_radius = cyl.radius * float(np.cos(np.pi / n_sides))

    halfspaces = np.empty((n_sides + 2, 4), dtype=float)
    # Side faces: each outward normal m_k, offset r_inscribed.
    angles = 2.0 * np.pi * np.arange(n_sides) / n_sides
    normals = np.cos(angles)[:, None] * e1 + np.sin(angles)[:, None] * e2
    halfspaces[:n_sides, :3] = normals
    halfspaces[:n_sides, 3] = -(normals @ cyl.center + inscribed_radius)
    # Top cap:  axis · x  ≤ axis · center + half_length
    halfspaces[n_sides, :3] = axis
    halfspaces[n_sides, 3] = -(axis @ cyl.center + cyl.half_length)
    # Bottom cap: -axis · x ≤ -(axis · center - half_length)
    halfspaces[n_sides + 1, :3] = -axis
    halfspaces[n_sides + 1, 3] = axis @ cyl.center - cyl.half_length
    return halfspaces


def _combined_halfspaces(c1: Cylinder, c2: Cylinder, n_sides: int) -> FloatArray:
    return np.vstack((cylinder_as_halfspaces(c1, n_sides), cylinder_as_halfspaces(c2, n_sides)))


def _perpendicular_frame(axis: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Return two unit vectors orthogonal to ``axis`` and to each other."""

    # Pick the axis with smallest magnitude to avoid parallelism.
    seed = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(axis, seed)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    return e1, e2


def _interior_point(c1: Cylinder, c2: Cylinder, halfspaces: FloatArray) -> FloatArray | None:
    """Find any strictly interior point of the combined half-space set.

    We try the cheap geometric guess first (midpoint of the axis-segment closest points)
    which succeeds whenever the two cylinders actually intersect, and only fall back
    to the Chebyshev-center LP on the rare near-tangent cases. On a typical frame
    this skips a ~2 ms linprog call per pair.
    """

    guess = _axis_midpoint(c1, c2)
    if guess is not None and _is_strictly_inside(guess, halfspaces):
        return guess
    return _chebyshev_center(halfspaces)


def _axis_midpoint(c1: Cylinder, c2: Cylinder) -> FloatArray | None:
    """Midpoint of the closest-approach points on the two axis segments, if it exists."""

    p, q = _closest_points_on_segments(
        c1.center - c1.half_length * c1.axis,
        c1.center + c1.half_length * c1.axis,
        c2.center - c2.half_length * c2.axis,
        c2.center + c2.half_length * c2.axis,
    )
    if p is None or q is None:
        return None
    return 0.5 * (p + q)


def _closest_points_on_segments(
    a0: FloatArray, a1: FloatArray, b0: FloatArray, b1: FloatArray
) -> tuple[FloatArray | None, FloatArray | None]:
    """Return the pair of closest points on two 3D line segments."""

    u = a1 - a0
    v = b1 - b0
    w = a0 - b0
    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w))
    e = float(np.dot(v, w))
    denom = a * c - b * b
    if a < 1e-20 or c < 1e-20:
        return None, None
    if denom < 1e-20:
        s = 0.0
        t = min(1.0, max(0.0, e / c))
    else:
        s = min(1.0, max(0.0, (b * e - c * d) / denom))
        t = min(1.0, max(0.0, (a * e - b * d) / denom))
    return a0 + s * u, b0 + t * v


def _is_strictly_inside(point: FloatArray, halfspaces: FloatArray, margin: float = 1e-9) -> bool:
    residuals = halfspaces[:, :3] @ point + halfspaces[:, 3]
    return bool(np.all(residuals < -margin))


def _chebyshev_center(halfspaces: FloatArray) -> FloatArray | None:
    """Chebyshev center via ``scipy.optimize.linprog`` — fallback when the cheap
    interior-point guess fails (near-tangent, edge cases)."""

    a = halfspaces[:, :3]
    b = halfspaces[:, 3]
    norms = np.linalg.norm(a, axis=1)
    c_obj = np.array([0.0, 0.0, 0.0, -1.0])
    a_ub = np.hstack([a, norms[:, None]])
    b_ub = -b
    bounds = [(None, None)] * 3 + [(0.0, None)]
    res = linprog(c_obj, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success or res.x[3] <= 1e-9:
        return None
    return np.asarray(res.x[:3], dtype=float)


def _deduplicate(points: FloatArray, tol: float = 1e-9) -> FloatArray:
    """Vectorized dedup via lex-sort + diff test.

    Replaces a Python O(n²) loop that dominated the polytope runtime on dense
    Qhull output. For ~30 vertices the savings are ~1 ms per pair.
    """

    if points.shape[0] == 0:
        return points
    quantum = 1.0 / max(tol, 1e-15)
    keys = np.round(points * quantum).astype(np.int64)
    _, unique_idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(unique_idx)]


def _reindex_simplices(simplices: np.ndarray, hull_vertex_indices: np.ndarray) -> IntArray:
    """Remap ``ConvexHull.simplices`` to indices into the reduced ``vertices``."""

    index_map = {int(orig): int(new) for new, orig in enumerate(hull_vertex_indices)}
    reindexed = np.empty_like(simplices)
    for i, tri in enumerate(simplices):
        reindexed[i] = [index_map[int(v)] for v in tri]
    return reindexed.astype(np.int64)
