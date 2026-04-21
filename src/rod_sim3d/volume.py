"""Unified intersection-volume API for two cylinders.

Two methods are supported:

``Method.SLICE``
    Exact slice-integral method (see :mod:`rod_sim3d.cylinder_intersection`). The
    cylinder is *not* approximated; the 3D volume is reduced to a 1D integral via
    the common-normal slicing, with each slice giving an exact 2D rectangle
    intersection. Numerical error is only in the 1D Gauss-Legendre quadrature.

``Method.POLYTOPE``
    N-gon prism approximation (see :mod:`rod_sim3d.cylinder_polytope`). Each
    cylinder becomes a convex polytope; their intersection is another convex
    polytope. Volume is computed via the **divergence theorem**
    (``V = |1/6 Sum_i a_i . (b_i x c_i)|`` over triangular faces), which is a
    signed-tetrahedra decomposition from the origin — ``O(F)`` in the number of
    faces, with no branching. Accuracy is limited by ``n_sides``.

Each method has its own parameter record (``SliceParams`` / ``PolytopeParams``),
selected by ``method``. The dispatcher :func:`intersection_volume` returns the
scalar volume regardless of method; use :func:`intersection_volume_and_shape`
when the explicit polytope is also required.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from rod_sim3d._array import FloatArray
from rod_sim3d.cylinder_intersection import Cylinder, overlap_volume
from rod_sim3d.cylinder_polytope import IntersectionPolytope, intersection_polytope


class Method(StrEnum):
    """Enumeration of available intersection-volume methods."""

    SLICE = "slice"
    POLYTOPE = "polytope"


@dataclass(slots=True, frozen=True)
class SliceParams:
    """Parameters for :data:`Method.SLICE`.

    Attributes
    ----------
    quadrature_points : int
        Number of Gauss-Legendre nodes on ``[-π/2, π/2]`` after the sin-substitution.
        Q=16 gives relative error ~1e-5 for typical configurations; Q=64 ~1e-8.
    """

    quadrature_points: int = 16


@dataclass(slots=True, frozen=True)
class PolytopeParams:
    """Parameters for :data:`Method.POLYTOPE`.

    Attributes
    ----------
    n_sides : int
        Number of side faces per cylinder (inscribed ``N``-gon). Typical values
        8 / 16 / 32 / 64 give volume errors ~10% / 3% / 1% / 0.25%.
    """

    n_sides: int = 16


VolumeParams = SliceParams | PolytopeParams


def intersection_volume(
    c1: Cylinder,
    c2: Cylinder,
    method: Method = Method.SLICE,
    params: VolumeParams | None = None,
) -> float:
    """Return the scalar intersection volume via the requested ``method``.

    If ``params`` is ``None`` a default instance of the matching parameter class
    is used. Unknown combinations (e.g. ``PolytopeParams`` with ``Method.SLICE``)
    raise ``TypeError`` — this keeps the contract unambiguous.
    """

    if method is Method.SLICE:
        slice_params = _as_slice_params(params)
        return overlap_volume(c1, c2, slice_params.quadrature_points)
    if method is Method.POLYTOPE:
        poly_params = _as_polytope_params(params)
        poly = intersection_polytope(c1, c2, poly_params.n_sides)
        return 0.0 if poly is None else poly.volume
    raise ValueError(f"Unknown intersection method: {method!r}")


def intersection_volume_and_shape(
    c1: Cylinder,
    c2: Cylinder,
    params: PolytopeParams | None = None,
) -> tuple[float, IntersectionPolytope | None]:
    """Return both volume and the explicit polytope for :data:`Method.POLYTOPE`.

    The slice method cannot produce a shape, so this function is always
    polytope-based.
    """

    poly_params = _as_polytope_params(params)
    poly = intersection_polytope(c1, c2, poly_params.n_sides)
    volume = 0.0 if poly is None else poly.volume
    return volume, poly


def polytope_volume_from_mesh(
    vertices: FloatArray,
    simplices: npt.NDArray[np.integer],
    outward_normals: FloatArray | None = None,
) -> float:
    """Volume of a convex polytope by the divergence theorem.

    For each outward-oriented triangular face with vertices ``(a, b, c)``, the
    signed volume of the tetrahedron at the origin is ``det(a, b, c) / 6``. The
    sum of these signed volumes — over all faces — equals the total volume.

    ``scipy.spatial.ConvexHull.simplices`` does **not** guarantee per-simplex
    orientation, so we check each triangle's orientation against the outward
    normal and flip when needed. Pass ``ConvexHull.equations[:, :3]`` in
    ``outward_normals`` to skip the centroid-based fallback.

    Complexity ``O(F)`` in the face count. No branching after orientation,
    SIMD-friendly, and translates literally to Rust / C++ / CUDA.

    Parameters
    ----------
    vertices : ``(V, 3)`` float array.
    simplices : ``(F, 3)`` int array indexing ``vertices``.
    outward_normals : optional ``(F, 3)`` outward normals per face (from
        :attr:`scipy.spatial.ConvexHull.equations`).
    """

    # Accept any integer dtype (scipy's ConvexHull.simplices is int32 on some platforms).
    simplices_i = np.asarray(simplices, dtype=np.int64)
    a = vertices[simplices_i[:, 0]]
    b = vertices[simplices_i[:, 1]]
    c = vertices[simplices_i[:, 2]]

    triangle_normals = np.cross(b - a, c - a)
    reference_normals = (
        outward_normals if outward_normals is not None else _centroid_outward_normals(a, b, c)
    )
    flip = np.einsum("ij,ij->i", triangle_normals, reference_normals) < 0.0

    # Flip by swapping b and c where needed — vectorized.
    b_oriented = np.where(flip[:, None], c, b)
    c_oriented = np.where(flip[:, None], b, c)

    signed = np.einsum("ij,ij->i", a, np.cross(b_oriented, c_oriented))
    return float(abs(signed.sum()) / 6.0)


def _centroid_outward_normals(a: FloatArray, b: FloatArray, c: FloatArray) -> FloatArray:
    """Heuristic outward normals: from face centroid away from polytope centroid."""

    face_centroids = (a + b + c) / 3.0
    polytope_centroid = np.concatenate([a, b, c]).mean(axis=0)
    return face_centroids - polytope_centroid


def _as_slice_params(params: VolumeParams | None) -> SliceParams:
    if params is None:
        return SliceParams()
    if not isinstance(params, SliceParams):
        raise TypeError(f"params must be SliceParams for Method.SLICE, got {type(params).__name__}")
    return params


def _as_polytope_params(params: VolumeParams | None) -> PolytopeParams:
    if params is None:
        return PolytopeParams()
    if not isinstance(params, PolytopeParams):
        raise TypeError(
            f"params must be PolytopeParams for Method.POLYTOPE, got {type(params).__name__}"
        )
    return params
