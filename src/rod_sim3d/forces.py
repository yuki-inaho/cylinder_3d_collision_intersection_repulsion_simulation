"""Force and torque accumulators (pure functions).

Everything here is a pure function: arrays go in, arrays come out. The orchestrating
:class:`~rod_sim3d.simulation.Simulation` simply composes these, which keeps the
dependency graph acyclic and makes each piece independently testable.

Rust migration
--------------
Each function corresponds 1:1 to a function in a future Rust module. The
quadrature-based pair force scales as ``O(N^2 M^2)``; the outer rod loops
parallelize trivially in Rayon.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from rod_sim3d._array import FloatArray
from rod_sim3d.config import PairPotentialConfig, WallPotentialConfig
from rod_sim3d.geometry import project_perpendicular
from rod_sim3d.potentials import (
    regularized_distance,
    shifted_force_vectors,
    shifted_potential,
    wall_force_magnitude,
)


class ForceTorque(NamedTuple):
    """Named tuple of forces ``(N, 3)`` and torques ``(N, 3)``."""

    forces: FloatArray
    torques: FloatArray


def zero_force_torque(n_rods: int) -> ForceTorque:
    return ForceTorque(
        forces=np.zeros((n_rods, 3), dtype=float),
        torques=np.zeros((n_rods, 3), dtype=float),
    )


def accumulate_pair_forces(
    positions: FloatArray,
    directions: FloatArray,
    nodes: FloatArray,
    weights: FloatArray,
    cfg: PairPotentialConfig,
    out: ForceTorque,
) -> None:
    """Accumulate pair-force contributions into ``out`` in place.

    Uses Gauss-Legendre quadrature on every rod pair. Forces obey Newton's third law
    exactly (the same ``weighted_g`` is added to rod ``i`` and subtracted from ``j``).
    """

    if cfg.strength == 0.0:
        return

    weight_matrix = weights[:, None] * weights[None, :]
    n = positions.shape[0]
    for i in range(n - 1):
        points_i = positions[i][None, :] + nodes[:, None] * directions[i][None, :]
        lever_i = nodes[:, None] * directions[i][None, :]
        for j in range(i + 1, n):
            points_j = positions[j][None, :] + nodes[:, None] * directions[j][None, :]
            lever_j = nodes[:, None] * directions[j][None, :]
            r_vec = points_i[:, None, :] - points_j[None, :, :]
            weighted_g = weight_matrix[:, :, None] * shifted_force_vectors(r_vec, cfg)

            fij = np.sum(weighted_g, axis=(0, 1))
            out.forces[i] += fij
            out.forces[j] -= fij

            out.torques[i] += np.sum(np.cross(lever_i[:, None, :], weighted_g), axis=(0, 1))
            out.torques[j] += np.sum(np.cross(lever_j[None, :, :], -weighted_g), axis=(0, 1))


def accumulate_wall_forces(
    positions: FloatArray,
    directions: FloatArray,
    nodes: FloatArray,
    weights: FloatArray,
    box: FloatArray,
    cfg: WallPotentialConfig,
    out: ForceTorque,
    rod_radius: float = 0.0,
) -> None:
    """Accumulate wall-repulsion contributions into ``out`` in place.

    Passing ``rod_radius > 0`` shifts the effective wall distance inward by ``r``
    so that the rod bounces before its cylindrical surface crosses the wall.
    """

    if cfg.strength == 0.0:
        return

    unit_axes = np.eye(3)
    n = positions.shape[0]
    for i in range(n):
        points = positions[i][None, :] + nodes[:, None] * directions[i][None, :]
        levers = nodes[:, None] * directions[i][None, :]
        point_forces = np.zeros_like(points)

        for axis in range(3):
            lower_mag = wall_force_magnitude(points[:, axis] - rod_radius, cfg)
            point_forces += lower_mag[:, None] * unit_axes[axis]

            upper_mag = wall_force_magnitude((box[axis] - points[:, axis]) - rod_radius, cfg)
            point_forces -= upper_mag[:, None] * unit_axes[axis]

        weighted_forces = weights[:, None] * point_forces
        out.forces[i] += np.sum(weighted_forces, axis=0)
        out.torques[i] += np.sum(np.cross(levers, weighted_forces), axis=0)


def compute_total_force_torque(
    positions: FloatArray,
    directions: FloatArray,
    nodes: FloatArray,
    weights: FloatArray,
    box: FloatArray,
    pair: PairPotentialConfig,
    wall: WallPotentialConfig,
    rod_radius: float = 0.0,
) -> ForceTorque:
    """Return the combined pair + wall forces and torques.

    Torques are projected into the plane perpendicular to each rod axis, matching
    the ``omega_i . u_i = 0`` constraint. ``rod_radius`` is forwarded to the wall
    kernel so the finite cylinder surface bounces off the walls while pair
    interactions, if any, remain on the center line.
    """

    result = zero_force_torque(positions.shape[0])
    accumulate_pair_forces(positions, directions, nodes, weights, pair, result)
    accumulate_wall_forces(
        positions, directions, nodes, weights, box, wall, result, rod_radius=rod_radius
    )
    return ForceTorque(
        forces=result.forces,
        torques=project_perpendicular(result.torques, directions),
    )


def pair_potential_energy(
    positions: FloatArray,
    directions: FloatArray,
    nodes: FloatArray,
    weights: FloatArray,
    cfg: PairPotentialConfig,
) -> float:
    """Total pair potential energy by Gauss-Legendre quadrature."""

    if cfg.strength == 0.0:
        return 0.0

    weight_matrix = weights[:, None] * weights[None, :]
    energy = 0.0
    n = positions.shape[0]
    for i in range(n - 1):
        points_i = positions[i][None, :] + nodes[:, None] * directions[i][None, :]
        for j in range(i + 1, n):
            points_j = positions[j][None, :] + nodes[:, None] * directions[j][None, :]
            r_vec = points_i[:, None, :] - points_j[None, :, :]
            rho = regularized_distance(r_vec, cfg.softening)
            energy += float(np.sum(weight_matrix * shifted_potential(rho, cfg)))
    return energy


def wall_potential_energy(
    positions: FloatArray,
    directions: FloatArray,
    nodes: FloatArray,
    weights: FloatArray,
    box: FloatArray,
    cfg: WallPotentialConfig,
    rod_radius: float = 0.0,
) -> float:
    """Total wall potential energy by Gauss-Legendre quadrature.

    The effective wall distance is shifted inward by ``rod_radius`` so that this
    function and :func:`accumulate_wall_forces` evaluate the *same* potential —
    without the shift the energy reported at ``rod_radius > 0`` would be the
    unintegrated force, and ``F = -∇U`` would be violated.
    """

    if cfg.strength == 0.0:
        return 0.0

    energy = 0.0
    n = positions.shape[0]
    for i in range(n):
        points = positions[i][None, :] + nodes[:, None] * directions[i][None, :]
        point_energy = np.zeros(points.shape[0], dtype=float)
        for axis in range(3):
            lower_distance = np.maximum(points[:, axis] - rod_radius, cfg.softening)
            upper_distance = np.maximum((box[axis] - points[:, axis]) - rod_radius, cfg.softening)
            point_energy += shifted_potential(lower_distance, cfg)
            point_energy += shifted_potential(upper_distance, cfg)
        energy += float(np.sum(weights * point_energy))
    return energy


def kinetic_energy(
    velocities: FloatArray, omegas: FloatArray, mass: float, inertia: float
) -> float:
    """Total translational + rotational kinetic energy."""

    return 0.5 * mass * float(np.sum(velocities**2)) + 0.5 * inertia * float(np.sum(omegas**2))
