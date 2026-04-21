"""Pure geometric utilities.

All functions are stateless and operate on plain numpy arrays. Target shapes are stated
in the docstrings so a future Rust port (on top of ``ndarray`` / ``nalgebra``) is a
near-mechanical translation.
"""

from __future__ import annotations

import numpy as np

from rod_sim3d._array import FloatArray


def normalize(vectors: FloatArray, axis: int = -1, eps: float = 1e-14) -> FloatArray:
    """Return ``vectors / ||vectors||`` with a lower-bounded denominator.

    ``eps`` only guards against division by zero. Near-zero vectors remain near zero.
    """

    norms = np.linalg.norm(vectors, axis=axis, keepdims=True)
    return vectors / np.maximum(norms, eps)


def random_unit_vectors(rng: np.random.Generator, n: int) -> FloatArray:
    """Sample ``n`` unit vectors uniformly on the 2-sphere (shape ``(n, 3)``)."""

    return normalize(rng.normal(size=(n, 3)))


def project_perpendicular(vectors: FloatArray, axes: FloatArray) -> FloatArray:
    """Remove the component of each vector parallel to the corresponding unit axis.

    Shapes: ``vectors, axes : (N, 3)`` with ``axes`` assumed row-wise unit vectors.
    """

    return vectors - np.sum(vectors * axes, axis=-1, keepdims=True) * axes


def rod_endpoints(positions: FloatArray, directions: FloatArray, length: float) -> FloatArray:
    """Return endpoints with shape ``(N, 2, 3)``.

    ``endpoints[i, 0] = x_i - (L/2) u_i`` and ``endpoints[i, 1] = x_i + (L/2) u_i``.
    """

    half = 0.5 * length * directions
    return np.stack((positions - half, positions + half), axis=1)


def rotate_vectors(vectors: FloatArray, rotation_vectors: FloatArray) -> FloatArray:
    """Rotate ``vectors`` by axis-angle ``rotation_vectors`` (Rodrigues' formula).

    Each row of ``rotation_vectors`` is an axis scaled by the rotation angle in radians.
    Stable for small angles; falls through to normalization when all rotations vanish.
    """

    theta = np.linalg.norm(rotation_vectors, axis=1)
    active = theta > 1e-14
    result = vectors.copy()
    if not np.any(active):
        return normalize(result)

    k = rotation_vectors[active] / theta[active, None]
    v = vectors[active]
    cos_t = np.cos(theta[active])[:, None]
    sin_t = np.sin(theta[active])[:, None]
    dot_kv = np.sum(k * v, axis=1)[:, None]
    result[active] = v * cos_t + np.cross(k, v) * sin_t + k * dot_kv * (1.0 - cos_t)
    return normalize(result)


def segment_distance(a0: FloatArray, a1: FloatArray, b0: FloatArray, b1: FloatArray) -> float:
    """Shortest distance between two finite 3D line segments.

    Uses the standard closest-points formulation with a safeguard for the near-parallel
    case. Returns a single Python float.
    """

    small = 1e-12
    u = a1 - a0
    v = b1 - b0
    w = a0 - b0
    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w))
    e = float(np.dot(v, w))
    denom = a * c - b * b

    s_num, s_den, t_num, t_den = _closest_points_on_segments(a, b, c, d, e, denom, small)

    sc = 0.0 if abs(s_num) < small else s_num / s_den
    tc = 0.0 if abs(t_num) < small else t_num / t_den
    return float(np.linalg.norm(w + sc * u - tc * v))


def _closest_points_on_segments(
    a: float, b: float, c: float, d: float, e: float, denom: float, small: float
) -> tuple[float, float, float, float]:
    """Return ``(s_num, s_den, t_num, t_den)`` for the closest-points parameters."""

    if denom < small:
        return 0.0, 1.0, e, c

    s_num = b * e - c * d
    t_num = a * e - b * d
    s_den = denom
    t_den = denom
    if s_num < 0.0:
        s_num = 0.0
        t_num, t_den = e, c
    elif s_num > s_den:
        s_num = s_den
        t_num, t_den = e + b, c

    if t_num < 0.0:
        t_num = 0.0
        s_num, s_den = _clip_s_bounds(-d, a, s_den)
    elif t_num > t_den:
        t_num = t_den
        s_num, s_den = _clip_s_bounds(-d + b, a, s_den)
    return s_num, s_den, t_num, t_den


def _clip_s_bounds(target: float, a: float, s_den: float) -> tuple[float, float]:
    if target < 0.0:
        return 0.0, s_den
    if target > a:
        return s_den, s_den
    return target, a


def gauss_legendre_segment(n_points: int, length: float) -> tuple[FloatArray, FloatArray]:
    """Gauss-Legendre nodes and weights for ``[-length/2, length/2]``.

    Returns ``(nodes, weights)`` each of shape ``(n_points,)``.
    """

    nodes, weights = np.polynomial.legendre.leggauss(n_points)
    return 0.5 * length * nodes, 0.5 * length * weights
