"""Exact intersection volume of two finite-length 3D cylinders.

The non-parallel case reduces to the 1D integral

    V = q * integral_{-pi/2..pi/2} A(t(theta)) cos(theta) d(theta)

with the sin-substitution ``t = m + q sin(theta)`` that annihilates the square-root
singularity of ``h_i(t) = sqrt(r_i^2 - (t - alpha_i)^2)`` at the endpoints. Two
closed-form regimes are handled directly:

1. **Parallel axes** (``|u_1 x u_2| = 0``):
   ``V_cap = L_overlap * lens_area(d; r1, r2)``.

2. **Non-parallel axes** reduce to a 1D integral along the common-normal direction
   ``n = (u_1 x u_2) / |u_1 x u_2|``::

       V_cap = integral_{t0..t1} Area( R1(t) & R2(t) ) dt

   where ``R1(t), R2(t)`` are the two rectangular cross-sections in the slice plane
   ``{x : n.x = t}``. We apply the sin-substitution ``t = m + q sin(theta)`` to
   kill the sqrt singularity at the endpoints and evaluate the θ integral with
   Gauss-Legendre quadrature. The 2D rectangle intersection is done with
   Sutherland-Hodgman clipping, which is exact for the convex polygon case and
   costs O(16) per call.

Every function here is pure and operates on plain numpy arrays, so a future Rust
port can translate these to ``nalgebra``/``ndarray`` without design change.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from rod_sim3d._array import FloatArray
from rod_sim3d.geometry import segment_distance

_PARALLEL_SIN_THRESHOLD: float = 1e-9


@dataclass(slots=True, frozen=True)
class Cylinder:
    """Finite cylinder defined by center, unit axis, half-length, and radius."""

    center: FloatArray  # shape (3,)
    axis: FloatArray  # shape (3,), unit length
    half_length: float
    radius: float

    @property
    def length(self) -> float:
        return 2.0 * self.half_length

    @property
    def volume(self) -> float:
        return float(np.pi * self.radius**2 * self.length)

    def aabb(self) -> tuple[FloatArray, FloatArray]:
        """Return axis-aligned bounding box ``(lo, hi)`` each of shape ``(3,)``.

        The bound accounts for the on-axis half-length and a generous per-axis radius
        budget (``r * sqrt(1 - u_i^2)`` per axis), which is tighter than ``+r``.
        """

        axial = self.half_length * np.abs(self.axis)
        radial = self.radius * np.sqrt(np.maximum(1.0 - self.axis**2, 0.0))
        extent = axial + radial
        return self.center - extent, self.center + extent


def overlap_volume(c1: Cylinder, c2: Cylinder, quadrature_points: int = 16) -> float:
    """Return the exact intersection volume of two cylinders (numerical in 1D only).

    Applies AABB and axis-distance broad-phase checks before dispatching to the
    parallel or non-parallel kernel.
    """

    if not _aabb_overlaps(c1, c2):
        return 0.0
    if _min_axis_distance(c1, c2) >= c1.radius + c2.radius:
        return 0.0

    cross = np.cross(c1.axis, c2.axis)
    sin_angle = float(np.linalg.norm(cross))
    if sin_angle < _PARALLEL_SIN_THRESHOLD:
        return _parallel_overlap_volume(c1, c2)
    return _non_parallel_overlap_volume(c1, c2, quadrature_points, cross, sin_angle)


def iou(c1: Cylinder, c2: Cylinder, quadrature_points: int = 16) -> float:
    """Return Jaccard (IoU) of the two cylinders' volumes."""

    inter = overlap_volume(c1, c2, quadrature_points)
    union = c1.volume + c2.volume - inter
    return 0.0 if union <= 0.0 else inter / union


def cylinders_from_rods(
    positions: FloatArray,
    directions: FloatArray,
    rod_length: float,
    rod_radius: float,
) -> list[Cylinder]:
    """Bulk constructor turning the state arrays of a simulation into Cylinders."""

    half_length = 0.5 * rod_length
    return [
        Cylinder(
            center=np.asarray(positions[i], dtype=float),
            axis=np.asarray(directions[i], dtype=float),
            half_length=half_length,
            radius=rod_radius,
        )
        for i in range(positions.shape[0])
    ]


def compute_pairwise_overlaps(
    cylinders: list[Cylinder],
    quadrature_points: int = 16,
    min_volume: float = 0.0,
) -> list[tuple[int, int, float]]:
    """Return ``(i, j, volume)`` for every pair with a non-trivial overlap.

    A vectorized broad-phase first eliminates non-candidate pairs by AABB overlap,
    turning the worst-case ``O(N^2)`` slice-integral cost into ``O(k)`` narrow-phase
    evaluations for the ``k`` pairs that actually pass. Beyond ~50 rods this matters
    a lot because the narrow-phase integrand is expensive compared to a handful of
    array comparisons.
    """

    candidates = _aabb_broad_phase(cylinders)
    results: list[tuple[int, int, float]] = []
    for i, j in candidates:
        volume = overlap_volume(cylinders[i], cylinders[j], quadrature_points)
        if volume > min_volume:
            results.append((i, j, volume))
    return results


def _aabb_broad_phase(cylinders: list[Cylinder]) -> list[tuple[int, int]]:
    """Return index pairs ``(i, j)`` with ``i < j`` whose AABBs overlap.

    This implementation precomputes every cylinder's AABB once and then compares all
    pairs in numpy. For N in the low hundreds the memory cost (``6 * N`` floats) is
    negligible and the 6-vector-per-pair comparison is dominated by cache bandwidth.
    """

    n = len(cylinders)
    if n < 2:
        return []

    los = np.empty((n, 3), dtype=float)
    his = np.empty((n, 3), dtype=float)
    for i, cyl in enumerate(cylinders):
        lo, hi = cyl.aabb()
        los[i] = lo
        his[i] = hi

    idx_i, idx_j = np.triu_indices(n, k=1)
    overlap = np.all(los[idx_i] <= his[idx_j], axis=1) & np.all(los[idx_j] <= his[idx_i], axis=1)
    keep_i = idx_i[overlap]
    keep_j = idx_j[overlap]
    return [(int(i), int(j)) for i, j in zip(keep_i, keep_j, strict=True)]


def lens_area(d: float, r1: float, r2: float) -> float:
    """Area of the lens formed by two overlapping circles.

    Closed-form expression used for the parallel-axis cylinder case and as a
    fast sanity check in tests.
    """

    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return float(np.pi * min(r1, r2) ** 2)
    a = r1 * r1 * np.arccos((d * d + r1 * r1 - r2 * r2) / (2.0 * d * r1))
    b = r2 * r2 * np.arccos((d * d + r2 * r2 - r1 * r1) / (2.0 * d * r2))
    # Bramagupta-style area of the kite spanned by the two intersection points.
    sqrt_term = (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)
    return float(a + b - 0.5 * np.sqrt(max(sqrt_term, 0.0)))


def rectangle_intersection_area(
    half_extents_1: tuple[float, float],
    rect2_center_xy: tuple[float, float],
    rect2_axis_xy: tuple[float, float],
    half_extents_2: tuple[float, float],
) -> float:
    """2D intersection area of two rectangles.

    Rectangle 1 is axis-aligned with half-extents ``half_extents_1``, centered at the
    origin. Rectangle 2 is centered at ``rect2_center_xy`` with its first axis in
    direction ``rect2_axis_xy`` (unit vector) and half-extents ``half_extents_2``.
    """

    if half_extents_1[0] <= 0.0 or half_extents_1[1] <= 0.0:
        return 0.0
    if half_extents_2[0] <= 0.0 or half_extents_2[1] <= 0.0:
        return 0.0

    rect1 = _rect_vertices_axis_aligned(half_extents_1)
    rect2 = _rect_vertices_rotated(rect2_center_xy, rect2_axis_xy, half_extents_2)
    clipped = _sutherland_hodgman(rect1, rect2)
    return _polygon_area(clipped)


def _parallel_overlap_volume(c1: Cylinder, c2: Cylinder) -> float:
    u = c1.axis
    proj1 = float(np.dot(c1.center, u))
    proj2 = float(np.dot(c2.center, u))
    lo = max(proj1 - c1.half_length, proj2 - c2.half_length)
    hi = min(proj1 + c1.half_length, proj2 + c2.half_length)
    overlap_len = max(0.0, hi - lo)
    if overlap_len == 0.0:
        return 0.0
    delta = c2.center - c1.center
    perp = delta - np.dot(delta, u) * u
    d = float(np.linalg.norm(perp))
    return overlap_len * lens_area(d, c1.radius, c2.radius)


def _non_parallel_overlap_volume(
    c1: Cylinder,
    c2: Cylinder,
    quadrature_points: int,
    cross: FloatArray,
    sin_angle: float,
) -> float:
    """Compute the slice integral for non-parallel cylinders.

    The per-pair scalars are computed in Python, then a single Numba-jitted kernel
    runs the Gauss-Legendre loop (~5-10x faster than the pure-Python path). The
    kernel is byte-compatible with the pure-Python fallback if Numba is absent.
    """

    from rod_sim3d._volume_kernels import slice_integrand_loop  # noqa: PLC0415

    n = cross / sin_angle
    alpha1 = float(np.dot(n, c1.center))
    alpha2 = float(np.dot(n, c2.center))

    t_low = max(alpha1 - c1.radius, alpha2 - c2.radius)
    t_high = min(alpha1 + c1.radius, alpha2 + c2.radius)
    if t_high <= t_low:
        return 0.0

    # 2D local frame aligned with cylinder 1 (g1 = u1, g2 = n x u1).
    g2 = np.cross(n, c1.axis)
    delta = c2.center - c1.center
    d_3d = delta - np.dot(delta, n) * n
    gamma = float(np.dot(c1.axis, c2.axis))

    m = 0.5 * (t_low + t_high)
    q = 0.5 * (t_high - t_low)

    nodes, weights = _gauss_legendre(quadrature_points)
    theta_nodes = 0.5 * np.pi * np.asarray(nodes)
    weight_arr = np.asarray(weights)

    total = slice_integrand_loop(
        theta_nodes,
        weight_arr,
        m,
        q,
        alpha1,
        alpha2,
        c1.radius,
        c2.radius,
        c1.half_length,
        c2.half_length,
        float(np.dot(d_3d, c1.axis)),
        float(np.dot(d_3d, g2)),
        gamma,
        sin_angle,
    )
    return q * 0.5 * np.pi * total


def _safe_half_width(radius: float, delta: float) -> float:
    inside = radius * radius - delta * delta
    return float(np.sqrt(inside)) if inside > 0.0 else 0.0


def _aabb_overlaps(c1: Cylinder, c2: Cylinder) -> bool:
    lo1, hi1 = c1.aabb()
    lo2, hi2 = c2.aabb()
    return bool(np.all(lo1 <= hi2) and np.all(lo2 <= hi1))


def _min_axis_distance(c1: Cylinder, c2: Cylinder) -> float:
    a0 = c1.center - c1.half_length * c1.axis
    a1 = c1.center + c1.half_length * c1.axis
    b0 = c2.center - c2.half_length * c2.axis
    b1 = c2.center + c2.half_length * c2.axis
    return segment_distance(a0, a1, b0, b1)


# --- 2D polygon helpers --------------------------------------------------------------


def _rect_vertices_axis_aligned(half_extents: tuple[float, float]) -> list[tuple[float, float]]:
    hx, hy = half_extents
    return [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]


def _rect_vertices_rotated(
    center: tuple[float, float],
    axis: tuple[float, float],
    half_extents: tuple[float, float],
) -> list[tuple[float, float]]:
    cx, cy = center
    ax, ay = axis
    px, py = -ay, ax
    hx, hy = half_extents
    corners = [
        (cx - hx * ax - hy * px, cy - hx * ay - hy * py),
        (cx + hx * ax - hy * px, cy + hx * ay - hy * py),
        (cx + hx * ax + hy * px, cy + hx * ay + hy * py),
        (cx - hx * ax + hy * px, cy - hx * ay + hy * py),
    ]
    return corners


def _sutherland_hodgman(
    subject: list[tuple[float, float]], clipper: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Intersect ``subject`` with convex polygon ``clipper`` using Sutherland-Hodgman.

    Both polygons must be convex and CCW. Result is a convex polygon (possibly empty).
    """

    output = subject
    for i in range(len(clipper)):
        if not output:
            return output
        input_poly = output
        output = []
        edge_start = clipper[i]
        edge_end = clipper[(i + 1) % len(clipper)]
        prev = input_poly[-1]
        prev_inside = _is_left(edge_start, edge_end, prev)
        for current in input_poly:
            current_inside = _is_left(edge_start, edge_end, current)
            if current_inside:
                if not prev_inside:
                    output.append(_line_intersection(prev, current, edge_start, edge_end))
                output.append(current)
            elif prev_inside:
                output.append(_line_intersection(prev, current, edge_start, edge_end))
            prev = current
            prev_inside = current_inside
    return output


def _is_left(
    edge_start: tuple[float, float],
    edge_end: tuple[float, float],
    point: tuple[float, float],
) -> bool:
    ex, ey = edge_end[0] - edge_start[0], edge_end[1] - edge_start[1]
    px, py = point[0] - edge_start[0], point[1] - edge_start[1]
    return ex * py - ey * px >= 0.0


def _line_intersection(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    p4: tuple[float, float],
) -> tuple[float, float]:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-16:
        return p2
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def _polygon_area(polygon: list[tuple[float, float]]) -> float:
    if len(polygon) < 3:
        return 0.0
    total = 0.0
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return 0.5 * abs(total)


@lru_cache(maxsize=32)
def _gauss_legendre(n: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Cache Gauss-Legendre nodes/weights keyed by ``n``."""

    nodes, weights = np.polynomial.legendre.leggauss(n)
    return tuple(map(float, nodes)), tuple(map(float, weights))
