"""Thin orchestrator that composes state, forces, and integrator.

Everything physics-related is delegated to the pure functions in ``potentials``,
``forces``, ``dynamics``, and ``initial_conditions``. ``Simulation`` itself only
holds precomputed quadrature weights and the integrator parameters derived from the
``Config`` — the single responsibility of this class is *wiring*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from rod_sim3d._array import FloatArray
from rod_sim3d.config import Config, InteractionModel
from rod_sim3d.dynamics import IntegratorParams
from rod_sim3d.dynamics import step as integrate_one_step
from rod_sim3d.forces import (
    ForceTorque,
    compute_total_force_torque,
    kinetic_energy,
    pair_potential_energy,
    wall_potential_energy,
    zero_force_torque,
)
from rod_sim3d.geometry import gauss_legendre_segment, rod_endpoints
from rod_sim3d.hard_contact import (
    ContactParams,
    resolve_pair_contacts,
    resolve_wall_contacts,
)
from rod_sim3d.initial_conditions import InitialKick, build_initial_state
from rod_sim3d.state import RodState


class EnergyBreakdown(NamedTuple):
    kinetic: float
    potential_pair: float
    potential_wall: float

    @property
    def total(self) -> float:
        return self.kinetic + self.potential_pair + self.potential_wall


@dataclass(slots=True)
class Simulation:
    """Composes :class:`RodState`, quadrature, and integrator parameters.

    The class is intentionally thin; it only wires together pure computations so
    that any single piece can be reused or replaced (e.g. swapping the integrator
    for Velocity-Verlet) without touching the rest.
    """

    config: Config
    state: RodState
    quadrature_nodes: FloatArray
    quadrature_weights: FloatArray
    inertia: float
    initial_kick: InitialKick
    step_index: int = 0

    @classmethod
    def from_config(cls, config: Config) -> Simulation:
        nodes, weights = gauss_legendre_segment(
            config.system.quadrature_points, config.system.rod_length
        )
        inertia = config.system.mass * config.system.rod_length**2 / 12.0
        state, kick = build_initial_state(config, inertia)
        return cls(
            config=config,
            state=state,
            quadrature_nodes=nodes,
            quadrature_weights=weights,
            inertia=inertia,
            initial_kick=kick,
        )

    @property
    def time(self) -> float:
        return self.step_index * self.config.dynamics.dt

    def endpoints(self) -> FloatArray:
        return rod_endpoints(
            self.state.positions, self.state.directions, self.config.system.rod_length
        )

    def compute_forces(self) -> ForceTorque:
        """Return pair + wall forces/torques according to the interaction model.

        HARD_CONTACT returns zero here; its physics lives in the post-drift
        impulsive solver. NONE also returns zero (pair) plus the soft wall
        kernel only if ``wall.strength > 0``. SOFT_REPULSION delegates to the
        full potential-based kernel.
        """

        model = self.config.pair_interaction.model
        if model is InteractionModel.SOFT_REPULSION:
            return compute_total_force_torque(
                self.state.positions,
                self.state.directions,
                self.quadrature_nodes,
                self.quadrature_weights,
                self._box_array(),
                self.config.pair,
                self.config.wall,
                rod_radius=self.config.system.rod_radius,
            )
        if model is InteractionModel.NONE:
            # Only the soft wall potential (if any) emits forces.
            zero_pair = type(self.config.pair)(strength=0.0)
            return compute_total_force_torque(
                self.state.positions,
                self.state.directions,
                self.quadrature_nodes,
                self.quadrature_weights,
                self._box_array(),
                zero_pair,
                self.config.wall,
                rod_radius=self.config.system.rod_radius,
            )
        # HARD_CONTACT: skip force-based pair and wall kernels entirely.
        return zero_force_torque(self.state.n_rods)

    def step(self, n_steps: int = 1) -> None:
        params = self._integrator_params()
        model = self.config.pair_interaction.model
        for _ in range(n_steps):
            integrate_one_step(self.state, self.compute_forces(), params)
            if model is InteractionModel.HARD_CONTACT:
                self._resolve_hard_contacts()
            self.step_index += 1

    def _resolve_hard_contacts(self) -> None:
        """Apply impulse-based wall and pair contacts in place on ``self.state``."""

        derived = self.config.resolve_interaction()
        params = ContactParams(
            mass=self.config.system.mass,
            inertia=self.inertia,
            contact_radius=derived.contact_radius,
            restitution=derived.restitution,
            wall_restitution=derived.wall_restitution,
        )
        resolve_wall_contacts(self.state, self._box_array(), self.config.system.rod_length, params)
        resolve_pair_contacts(self.state, self.config.system.rod_length, params)
        self.state.enforce_constraints()

    def energy(self) -> EnergyBreakdown:
        return EnergyBreakdown(
            kinetic=kinetic_energy(
                self.state.velocities, self.state.omegas, self.config.system.mass, self.inertia
            ),
            potential_pair=pair_potential_energy(
                self.state.positions,
                self.state.directions,
                self.quadrature_nodes,
                self.quadrature_weights,
                self.config.pair,
            ),
            potential_wall=wall_potential_energy(
                self.state.positions,
                self.state.directions,
                self.quadrature_nodes,
                self.quadrature_weights,
                self._box_array(),
                self.config.wall,
                rod_radius=self.config.system.rod_radius,
            ),
        )

    def _box_array(self) -> FloatArray:
        return np.asarray(self.config.system.box, dtype=float)

    def _integrator_params(self) -> IntegratorParams:
        dyn = self.config.dynamics
        return IntegratorParams(
            dt=dyn.dt,
            mass=self.config.system.mass,
            inertia=self.inertia,
            linear_damping=dyn.linear_damping,
            angular_damping=dyn.angular_damping,
            max_linear_speed=dyn.max_linear_speed,
            max_angular_speed=dyn.max_angular_speed,
        )
