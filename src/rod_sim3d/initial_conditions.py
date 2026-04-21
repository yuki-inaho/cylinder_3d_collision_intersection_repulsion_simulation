"""Random initial condition sampling.

The two responsibilities split cleanly:

- :func:`sample_rod_placement` decides *where* each rod is, respecting a minimum
  pairwise segment distance. Only geometric.
- :func:`build_initial_state` builds a :class:`RodState` from the placement and converts
  random forces/torques into initial velocities/angular velocities via an impulse.

Splitting them this way makes each independently testable and lets a Rust port treat
placement as a pure geometric sampler.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rod_sim3d._array import FloatArray
from rod_sim3d.config import Config
from rod_sim3d.geometry import (
    normalize,
    project_perpendicular,
    random_unit_vectors,
    rod_endpoints,
    segment_distance,
)
from rod_sim3d.state import RodState


@dataclass(slots=True)
class InitialKick:
    """Raw random forces and torques sampled at t = 0.

    Persisting them separately from the state is useful for reproducibility:
    storing ``forces`` and ``torques`` alongside the seed lets a reader recover
    the *why* of the initial velocities (an impulse of magnitude
    ``forces * kick_duration``).
    """

    forces: FloatArray
    torques: FloatArray
    kick_duration: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "definition": (
                "v0 = forces * kick_duration / mass; "
                "omega0 = torques_perpendicular * kick_duration / inertia"
            ),
            "kick_duration": self.kick_duration,
            "forces": self.forces.tolist(),
            "torques": self.torques.tolist(),
        }

    def save_json(self, path: str | Path, mass: float, inertia: float) -> None:
        payload = self.to_dict()
        payload["mass"] = mass
        payload["inertia"] = inertia
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def sample_rod_placement(config: Config, rng: np.random.Generator) -> tuple[FloatArray, FloatArray]:
    """Rejection-sample ``N`` non-overlapping rods inside the box.

    Returns
    -------
    positions, directions : each of shape ``(N, 3)``.

    Raises
    ------
    RuntimeError
        If the sampler cannot place all rods within ``max_sampling_attempts``.
    """

    n = config.system.n_rods
    length = config.system.rod_length
    box = np.asarray(config.system.box, dtype=float)
    clearance = config.initial.clearance

    positions: list[FloatArray] = []
    directions: list[FloatArray] = []
    endpoints: list[FloatArray] = []

    for _ in range(config.initial.max_sampling_attempts):
        if len(positions) == n:
            break
        direction = random_unit_vectors(rng, 1)[0]
        lower = clearance + 0.5 * length * np.abs(direction)
        upper = box - clearance - 0.5 * length * np.abs(direction)
        if np.any(lower >= upper):
            continue
        position = rng.uniform(lower, upper)
        candidate = rod_endpoints(position[None, :], direction[None, :], length)[0]
        if _accepted_by_distance(candidate, endpoints, config.initial.min_segment_distance):
            positions.append(position)
            directions.append(direction)
            endpoints.append(candidate)

    if len(positions) != n:
        raise RuntimeError(
            "Could not place all rods without violating min_segment_distance. "
            "Lower initial.min_segment_distance, reduce system.n_rods, or enlarge system.box."
        )

    return (
        np.asarray(positions, dtype=float),
        normalize(np.asarray(directions, dtype=float)),
    )


def build_initial_state(config: Config, inertia: float) -> tuple[RodState, InitialKick]:
    """Sample a placement and convert random kicks into initial velocities."""

    rng = np.random.default_rng(config.initial.seed)
    positions, directions = sample_rod_placement(config, rng)

    n = config.system.n_rods
    raw_forces = rng.normal(scale=config.initial.initial_force_scale, size=(n, 3))
    raw_torques = rng.normal(scale=config.initial.initial_torque_scale, size=(n, 3))
    torques = project_perpendicular(raw_torques, directions)

    kick_duration = config.initial.kick_duration
    velocities = raw_forces * (kick_duration / config.system.mass)
    omegas = project_perpendicular(torques * (kick_duration / inertia), directions)

    state = RodState(
        positions=positions,
        directions=directions,
        velocities=velocities,
        omegas=omegas,
    )
    state.enforce_constraints()
    return state, InitialKick(forces=raw_forces, torques=torques, kick_duration=kick_duration)


def _accepted_by_distance(
    candidate: FloatArray, existing: list[FloatArray], min_distance: float
) -> bool:
    if min_distance <= 0.0:
        return True
    return all(
        segment_distance(candidate[0], candidate[1], other[0], other[1]) >= min_distance
        for other in existing
    )
