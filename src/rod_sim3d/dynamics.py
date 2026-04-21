"""Time-integration primitives.

The integrator is the semi-implicit Euler variant with exact exponential damping
factors: applied to an isolated damped rod, it reproduces ``v(t) = v0 * exp(-gamma t)``
exactly instead of the first-order ``v0 * (1 - gamma dt)`` approximation. This makes
the step robust even when ``dt`` is not tiny relative to ``1 / gamma``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rod_sim3d._array import FloatArray
from rod_sim3d.forces import ForceTorque
from rod_sim3d.geometry import project_perpendicular, rotate_vectors
from rod_sim3d.state import RodState


@dataclass(slots=True, frozen=True)
class IntegratorParams:
    """Constants shared across integration steps."""

    dt: float
    mass: float
    inertia: float
    linear_damping: float
    angular_damping: float
    max_linear_speed: float | None
    max_angular_speed: float | None


def step(state: RodState, forces_torques: ForceTorque, params: IntegratorParams) -> None:
    """Advance ``state`` in place by one step ``dt``.

    Order of operations:

    1. Kick: update velocities and angular velocities by ``F/m`` and ``tau/I``.
    2. Damp: apply ``exp(-gamma dt)`` factors (exact for the isolated-rod case).
    3. Clip: optionally cap translational / rotational speeds.
    4. Project: re-enforce ``omega . u = 0``.
    5. Drift: advance positions and rotate unit directions by Rodrigues' formula.
    6. Constraints: re-normalize and re-project.
    """

    state.velocities += (params.dt / params.mass) * forces_torques.forces
    state.omegas += (params.dt / params.inertia) * forces_torques.torques

    _apply_exponential_damping(state, params)
    _clip_speeds(state, params)
    state.omegas[:] = project_perpendicular(state.omegas, state.directions)

    state.positions += params.dt * state.velocities
    state.directions[:] = rotate_vectors(state.directions, params.dt * state.omegas)
    state.enforce_constraints()


def _apply_exponential_damping(state: RodState, params: IntegratorParams) -> None:
    if params.linear_damping > 0.0:
        state.velocities *= np.exp(-params.linear_damping * params.dt)
    if params.angular_damping > 0.0:
        state.omegas *= np.exp(-params.angular_damping * params.dt)


def _clip_speeds(state: RodState, params: IntegratorParams) -> None:
    if params.max_linear_speed is not None:
        state.velocities[:] = _clip_vector_norms(state.velocities, params.max_linear_speed)
    if params.max_angular_speed is not None:
        state.omegas[:] = _clip_vector_norms(state.omegas, params.max_angular_speed)


def _clip_vector_norms(vectors: FloatArray, max_norm: float) -> FloatArray:
    norms = np.linalg.norm(vectors, axis=1)
    scale = np.ones_like(norms)
    mask = norms > max_norm
    scale[mask] = max_norm / np.maximum(norms[mask], 1e-14)
    return vectors * scale[:, None]
