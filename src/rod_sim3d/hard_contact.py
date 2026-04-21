"""Rigid-body capsule contact resolution via impulse.

This module is what makes the ``InteractionModel.HARD_CONTACT`` mode honest.
It does *not* emit forces that the standard integrator adds to ``v`` — instead,
after the drift step, it detects capsules in contact and adjusts velocities by
discrete impulses so the non-penetration constraint ``g_ij >= 0`` and the
restitution relation ``v_n^+ = -e v_n^-`` are satisfied.

Theoretical model
-----------------
Each rod is a capsule: a line segment of length ``L`` inflated by a contact
radius ``a``. Two capsules are in contact when the minimum distance between
their axis segments satisfies ``d_ij <= 2 a``. The contact normal is

    n_ij = (c_i - c_j) / |c_i - c_j|

where ``c_i, c_j`` are the closest points on the two axis segments. The contact
point offsets from the two centers of mass are ``r_i = s*_ij u_i`` and
``r_j = t*_ij u_j`` (the axial offsets; the radial offset is neglected because
``a << L``).

For an approaching contact (``v_n^- = n . (V_i - V_j) < 0``) the normal impulse
magnitude is

    P = -(1 + e) v_n^- / K

with effective inverse mass

    K = 1/m_i + 1/m_j
      + (r_i x n)^T I_i^{-1} (r_i x n)
      + (r_j x n)^T I_j^{-1} (r_j x n)

The capsule's spin around its own axis is physically unobservable, so we use
``I^{-1} = (1 / I_perp) (I_3 - u u^T)``.

Wall contacts are analogous with the far side treated as infinite mass: only
the rod's ``(v, omega)`` update, and the effective inverse mass drops the
``1/m_wall`` and ``I_wall^{-1}`` terms.

References:
  "Interaction models" doc (user spec) — sections 4 / 7 / 8.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rod_sim3d._array import FloatArray
from rod_sim3d.state import RodState

# Penetration tolerance: treat contacts with ``g < -GAP_TOL`` as active.
# Positive tolerance avoids spurious contacts when pairs are exactly tangent.
GAP_TOL: float = 1e-9


@dataclass(slots=True, frozen=True)
class ContactParams:
    """Scalars the impulse solver needs per simulation."""

    mass: float
    inertia: float
    contact_radius: float
    restitution: float
    wall_restitution: float


# --- segment-segment closest points --------------------------------------------------


def segment_closest_points(
    a0: FloatArray, a1: FloatArray, b0: FloatArray, b1: FloatArray
) -> tuple[float, FloatArray, FloatArray]:
    """Closest points on two 3D segments.

    Returns ``(distance, point_on_A, point_on_B)``. The standard clamped-linear
    formulation (Lumelsky/Eberly) with safeguards for the near-parallel case.
    """

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
        # Degenerate: one (or both) segments is a point. Clamp trivially.
        s = 0.0
        t = min(1.0, max(0.0, e / c)) if c > 0 else 0.0
    elif denom < 1e-20 * a * c:
        # Near-parallel: pin s = 0 and solve for t.
        s = 0.0
        t = min(1.0, max(0.0, e / c))
    else:
        s = min(1.0, max(0.0, (b * e - c * d) / denom))
        t = min(1.0, max(0.0, (a * e - b * d) / denom))

    point_a = a0 + s * u
    point_b = b0 + t * v
    dist = float(np.linalg.norm(point_a - point_b))
    return dist, point_a, point_b


# --- pair contact resolution ---------------------------------------------------------


def resolve_pair_contacts(state: RodState, rod_length: float, params: ContactParams) -> int:
    """Apply impulsive responses for every capsule pair in penetration.

    Mutates ``state.velocities`` and ``state.omegas`` in place. Returns the
    number of contacts resolved (useful for logging/stats).
    """

    n = state.n_rods
    if n < 2:
        return 0

    half_l = 0.5 * rod_length
    positions = state.positions
    directions = state.directions
    velocities = state.velocities
    omegas = state.omegas

    count = 0
    for i in range(n - 1):
        a0 = positions[i] - half_l * directions[i]
        a1 = positions[i] + half_l * directions[i]
        for j in range(i + 1, n):
            b0 = positions[j] - half_l * directions[j]
            b1 = positions[j] + half_l * directions[j]
            dist, ci, cj = segment_closest_points(a0, a1, b0, b1)
            gap = dist - 2.0 * params.contact_radius
            if gap >= -GAP_TOL:
                continue
            if dist < 1e-12:
                # Concentric degenerate: nudge along a fallback direction.
                n_ij = (
                    directions[i]
                    if np.linalg.norm(directions[i]) > 0
                    else np.array([1.0, 0.0, 0.0])
                )
            else:
                n_ij = (ci - cj) / dist

            ri = ci - positions[i]
            rj = cj - positions[j]
            v_rel = (
                velocities[i] + np.cross(omegas[i], ri) - velocities[j] - np.cross(omegas[j], rj)
            )
            v_n = float(np.dot(n_ij, v_rel))
            if v_n >= 0.0:
                # Already separating: no impulse required.
                continue

            impulse_mag = _pair_impulse_magnitude(
                ri, rj, n_ij, directions[i], directions[j], params, v_n
            )
            impulse_vec = impulse_mag * n_ij
            _apply_pair_impulse(velocities, omegas, directions, i, j, ri, rj, impulse_vec, params)
            count += 1
    return count


def _pair_impulse_magnitude(
    ri: FloatArray,
    rj: FloatArray,
    n_ij: FloatArray,
    u_i: FloatArray,
    u_j: FloatArray,
    params: ContactParams,
    v_n: float,
) -> float:
    """Return ``P = -(1+e) v_n / K`` with axis-perpendicular inertia."""

    ri_x_n = np.cross(ri, n_ij)
    rj_x_n = np.cross(rj, n_ij)
    # I^{-1} = (1/I_perp) (I_3 - u u^T), so w^T I^{-1} w = (1/I_perp)(|w|^2 - (u.w)^2).
    k_angular_i = (_sq_norm(ri_x_n) - float(np.dot(u_i, ri_x_n)) ** 2) / params.inertia
    k_angular_j = (_sq_norm(rj_x_n) - float(np.dot(u_j, rj_x_n)) ** 2) / params.inertia
    k_total = 2.0 / params.mass + k_angular_i + k_angular_j
    return -(1.0 + params.restitution) * v_n / k_total


def _apply_pair_impulse(
    velocities: FloatArray,
    omegas: FloatArray,
    directions: FloatArray,
    i: int,
    j: int,
    ri: FloatArray,
    rj: FloatArray,
    impulse_vec: FloatArray,
    params: ContactParams,
) -> None:
    velocities[i] += impulse_vec / params.mass
    velocities[j] -= impulse_vec / params.mass
    omegas[i] += _inverse_inertia_apply(directions[i], np.cross(ri, impulse_vec), params.inertia)
    omegas[j] += _inverse_inertia_apply(directions[j], np.cross(rj, -impulse_vec), params.inertia)


# --- wall contact resolution ---------------------------------------------------------


def resolve_wall_contacts(
    state: RodState, box: FloatArray, rod_length: float, params: ContactParams
) -> int:
    """Apply impulsive responses for every rod in contact with a box wall.

    The box is ``[0, box[0]] x [0, box[1]] x [0, box[2]]`` with inward-pointing
    normals. Six half-spaces are checked per rod per step.
    """

    n = state.n_rods
    if n == 0:
        return 0
    half_l = 0.5 * rod_length
    count = 0

    for axis in range(3):
        for sign, wall_value in ((+1.0, 0.0), (-1.0, float(box[axis]))):
            # Inward normal has ``normal[axis] = +sign``, others zero.
            for i in range(n):
                n_w = np.zeros(3)
                n_w[axis] = sign
                # Extreme point of the axis segment most into the wall:
                #   p(s) = x + s u;  minimize n_w . p(s) over s in [-L/2, L/2].
                u_i = state.directions[i]
                if sign * u_i[axis] > 0.0:
                    s_star = -half_l
                elif sign * u_i[axis] < 0.0:
                    s_star = +half_l
                else:
                    s_star = 0.0
                contact_point = state.positions[i] + s_star * u_i
                # Signed distance of the point to the wall (positive when inside box).
                # For sign=+1 (wall at axis=0):        signed = p[axis] - 0
                # For sign=-1 (wall at axis=box[axis]): signed = box[axis] - p[axis]
                # Unified: signed = sign * (p[axis] - wall_value).
                signed = sign * (contact_point[axis] - wall_value)
                gap = signed - params.contact_radius
                if gap >= -GAP_TOL:
                    continue

                ri = s_star * u_i
                v_contact = state.velocities[i] + np.cross(state.omegas[i], ri)
                v_n = float(np.dot(n_w, v_contact))
                if v_n >= 0.0:
                    continue

                ri_x_n = np.cross(ri, n_w)
                k_angular = (_sq_norm(ri_x_n) - float(np.dot(u_i, ri_x_n)) ** 2) / params.inertia
                k_total = 1.0 / params.mass + k_angular
                impulse_mag = -(1.0 + params.wall_restitution) * v_n / k_total
                impulse_vec = impulse_mag * n_w
                state.velocities[i] += impulse_vec / params.mass
                state.omegas[i] += _inverse_inertia_apply(
                    u_i, np.cross(ri, impulse_vec), params.inertia
                )
                count += 1
    return count


# --- Inertia helpers ---------------------------------------------------------------


def _inverse_inertia_apply(u: FloatArray, torque_vec: FloatArray, inertia: float) -> FloatArray:
    """Return ``I^{-1} tau`` for a thin rod: project out the axial component."""

    projected = torque_vec - float(np.dot(u, torque_vec)) * u
    return projected / inertia


def _sq_norm(v: FloatArray) -> float:
    return float(np.dot(v, v))
