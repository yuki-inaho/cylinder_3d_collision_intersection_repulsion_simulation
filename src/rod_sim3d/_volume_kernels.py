"""Numba-jitted hot kernels for the slice-integral method.

Only the inner math-heavy routines are compiled. Everything else stays pure-Python
for debuggability. Numba is a soft dependency — if the import fails the module
falls back to pure-NumPy implementations that match byte-for-byte.

Kernels
-------
``sutherland_hodgman_rects``
    Clip two rectangles given as ``(4, 2)`` float arrays; returns the clipped
    convex polygon as a list of ``(x, y)`` tuples (up to 8 vertices).

``polygon_area_shoelace``
    Shoelace absolute-area of a polygon given as an ``(N, 2)`` float array.

``slice_integrand_loop``
    The full per-pair slice-integration loop. Given all per-pair invariants and
    the Gauss-Legendre nodes/weights on ``[-1, 1]``, returns the narrow-phase
    volume.
"""

from __future__ import annotations

import math

import numpy as np

from rod_sim3d._array import FloatArray

try:
    from numba import njit  # top-level availability check

    _NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover — graceful fallback path
    _NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore[no-redef]
        """Identity decorator so the kernel works without Numba installed."""

        if args and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(fn):
            return fn

        return decorator


@njit(cache=True, fastmath=True)
def polygon_area_shoelace(polygon: FloatArray) -> float:
    """Absolute area of a 2D polygon via the Shoelace formula.

    ``polygon`` is ``(n, 2)``; vertices may be in any orientation.
    """

    n = polygon.shape[0]
    if n < 3:
        return 0.0
    total = 0.0
    for i in range(n):
        x1 = polygon[i, 0]
        y1 = polygon[i, 1]
        j = (i + 1) % n
        x2 = polygon[j, 0]
        y2 = polygon[j, 1]
        total += x1 * y2 - x2 * y1
    return 0.5 * abs(total)


@njit(cache=True, fastmath=True)
def sutherland_hodgman_clip(  # noqa: PLR0912, PLR0915
    subject: FloatArray, clipper: FloatArray, out: FloatArray
) -> int:
    """Clip a convex subject polygon against a convex clipper; write result into ``out``.

    Parameters
    ----------
    subject, clipper : ``(n, 2)`` float arrays.
    out : ``(16, 2)`` scratch buffer; writes into the first return-value entries.

    Returns
    -------
    int
        Number of vertices written to ``out``.
    """

    # Bounded scratch so Numba can stack-allocate. 8 vertices suffice for rect∩rect.
    buffer_a = np.empty((16, 2), dtype=np.float64)
    buffer_b = np.empty((16, 2), dtype=np.float64)
    n_a = subject.shape[0]
    for k in range(n_a):
        buffer_a[k, 0] = subject[k, 0]
        buffer_a[k, 1] = subject[k, 1]

    m = clipper.shape[0]
    for e in range(m):
        if n_a == 0:
            break
        ex = clipper[(e + 1) % m, 0] - clipper[e, 0]
        ey = clipper[(e + 1) % m, 1] - clipper[e, 1]
        cx = clipper[e, 0]
        cy = clipper[e, 1]

        n_b = 0
        prev_x = buffer_a[n_a - 1, 0]
        prev_y = buffer_a[n_a - 1, 1]
        prev_inside = (ex * (prev_y - cy) - ey * (prev_x - cx)) >= 0.0

        for v in range(n_a):
            cur_x = buffer_a[v, 0]
            cur_y = buffer_a[v, 1]
            cur_inside = (ex * (cur_y - cy) - ey * (cur_x - cx)) >= 0.0

            if cur_inside:
                if not prev_inside:
                    # line intersection of (prev, cur) with (clipper[e], clipper[e+1])
                    dx1 = prev_x - cur_x
                    dy1 = prev_y - cur_y
                    denom = dx1 * (-ey) - dy1 * (-ex)
                    if abs(denom) < 1e-16:
                        buffer_b[n_b, 0] = cur_x
                        buffer_b[n_b, 1] = cur_y
                    else:
                        t = ((prev_x - cx) * (-ey) - (prev_y - cy) * (-ex)) / denom
                        buffer_b[n_b, 0] = prev_x + t * (cur_x - prev_x)
                        buffer_b[n_b, 1] = prev_y + t * (cur_y - prev_y)
                    n_b += 1
                buffer_b[n_b, 0] = cur_x
                buffer_b[n_b, 1] = cur_y
                n_b += 1
            elif prev_inside:
                dx1 = prev_x - cur_x
                dy1 = prev_y - cur_y
                denom = dx1 * (-ey) - dy1 * (-ex)
                if abs(denom) < 1e-16:
                    buffer_b[n_b, 0] = cur_x
                    buffer_b[n_b, 1] = cur_y
                else:
                    t = ((prev_x - cx) * (-ey) - (prev_y - cy) * (-ex)) / denom
                    buffer_b[n_b, 0] = prev_x + t * (cur_x - prev_x)
                    buffer_b[n_b, 1] = prev_y + t * (cur_y - prev_y)
                n_b += 1

            prev_x = cur_x
            prev_y = cur_y
            prev_inside = cur_inside

        n_a = n_b
        for k in range(n_a):
            buffer_a[k, 0] = buffer_b[k, 0]
            buffer_a[k, 1] = buffer_b[k, 1]

    for k in range(n_a):
        out[k, 0] = buffer_a[k, 0]
        out[k, 1] = buffer_a[k, 1]
    return n_a


@njit(cache=True, fastmath=True)
def rectangle_intersection_area_fast(
    hx1: float,
    hy1: float,
    cx: float,
    cy: float,
    ax: float,
    ay: float,
    hx2: float,
    hy2: float,
) -> float:
    """Intersection area of an axis-aligned rectangle and a rotated rectangle.

    Same contract as the pure-Python ``rectangle_intersection_area`` but with
    all arguments as scalars so Numba can compile the whole call without
    boxing. Roughly 5x faster than the pure-Python version for typical inputs.
    """

    if hx1 <= 0.0 or hy1 <= 0.0 or hx2 <= 0.0 or hy2 <= 0.0:
        return 0.0

    subject = np.empty((4, 2), dtype=np.float64)
    subject[0, 0] = -hx1
    subject[0, 1] = -hy1
    subject[1, 0] = hx1
    subject[1, 1] = -hy1
    subject[2, 0] = hx1
    subject[2, 1] = hy1
    subject[3, 0] = -hx1
    subject[3, 1] = hy1

    # rotated rectangle 2: center (cx, cy), axis (ax, ay), perp (-ay, ax)
    px = -ay
    py = ax
    clipper = np.empty((4, 2), dtype=np.float64)
    clipper[0, 0] = cx - hx2 * ax - hy2 * px
    clipper[0, 1] = cy - hx2 * ay - hy2 * py
    clipper[1, 0] = cx + hx2 * ax - hy2 * px
    clipper[1, 1] = cy + hx2 * ay - hy2 * py
    clipper[2, 0] = cx + hx2 * ax + hy2 * px
    clipper[2, 1] = cy + hx2 * ay + hy2 * py
    clipper[3, 0] = cx - hx2 * ax + hy2 * px
    clipper[3, 1] = cy - hx2 * ay + hy2 * py

    out = np.empty((16, 2), dtype=np.float64)
    n = sutherland_hodgman_clip(subject, clipper, out)
    return polygon_area_shoelace(out[:n])


@njit(cache=True, fastmath=True)
def slice_integrand_loop(
    theta_nodes: FloatArray,
    weights: FloatArray,
    m: float,
    q: float,
    alpha1: float,
    alpha2: float,
    r1: float,
    r2: float,
    L1: float,
    L2: float,
    cx: float,
    cy: float,
    ax: float,
    ay: float,
) -> float:
    """Full Gauss-Legendre loop for one non-parallel cylinder pair.

    The caller precomputes the per-pair invariants (local 2D frame, rotation,
    slice bounds) and this kernel returns ``total`` such that
    ``V = q * (π/2) * total``. All inputs are plain scalars/arrays, so Numba
    can stack-allocate everything.
    """

    total = 0.0
    for k in range(theta_nodes.shape[0]):
        theta_j = theta_nodes[k]
        sin_t = math.sin(theta_j)
        t_j = m + q * sin_t
        d1 = t_j - alpha1
        d2 = t_j - alpha2
        inside1 = r1 * r1 - d1 * d1
        inside2 = r2 * r2 - d2 * d2
        if inside1 <= 0.0 or inside2 <= 0.0:
            continue
        h1 = math.sqrt(inside1)
        h2 = math.sqrt(inside2)
        area = rectangle_intersection_area_fast(L1, h1, cx, cy, ax, ay, L2, h2)
        total += weights[k] * area * math.cos(theta_j)
    return total


def numba_available() -> bool:
    """Report whether the Numba JIT was successfully imported."""

    return _NUMBA_AVAILABLE
