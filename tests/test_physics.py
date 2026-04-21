"""Smoke tests for the physics primitives.

These cover the small, stable set of invariants: Newton's third law on the pair
force, zero net angular change from internal forces, and the direction/omega
constraints after a step.
"""

from __future__ import annotations

import numpy as np

from rod_sim3d.config import (
    Config,
    DynamicsConfig,
    InitialConfig,
    PairPotentialConfig,
    SystemConfig,
    WallPotentialConfig,
)
from rod_sim3d.geometry import gauss_legendre_segment, project_perpendicular
from rod_sim3d.initial_conditions import InitialKick, build_initial_state
from rod_sim3d.simulation import Simulation
from rod_sim3d.state import RodState


def make_pair_only_simulation() -> Simulation:
    cfg = Config(
        system=SystemConfig(
            n_rods=2, rod_length=1.0, mass=1.0, box=(10.0, 10.0, 10.0), quadrature_points=5
        ),
        pair=PairPotentialConfig(
            strength=0.05, length_scale=0.4, cutoff=2.0, exponent=6.0, softening=0.03
        ),
        wall=WallPotentialConfig(strength=0.0),
        dynamics=DynamicsConfig(dt=0.001, linear_damping=0.0, angular_damping=0.0),
        initial=InitialConfig(seed=1),
    )
    nodes, weights = gauss_legendre_segment(cfg.system.quadrature_points, cfg.system.rod_length)
    inertia = cfg.system.mass * cfg.system.rod_length**2 / 12.0
    positions = np.array([[4.5, 5.0, 5.0], [5.2, 5.0, 5.0]], dtype=float)
    directions = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    state = RodState(
        positions=positions,
        directions=directions,
        velocities=np.zeros((2, 3)),
        omegas=np.zeros((2, 3)),
    )
    state.enforce_constraints()
    _, kick = build_initial_state(cfg, inertia)
    return Simulation(cfg, state, nodes, weights, inertia, kick)


def test_pair_forces_satisfy_action_reaction() -> None:
    sim = make_pair_only_simulation()
    result = sim.compute_forces()
    np.testing.assert_allclose(np.sum(result.forces, axis=0), np.zeros(3), atol=1e-10)


def test_internal_pair_forces_conserve_total_angular_momentum_about_origin() -> None:
    sim = make_pair_only_simulation()
    result = sim.compute_forces()
    total = np.sum(np.cross(sim.state.positions, result.forces) + result.torques, axis=0)
    np.testing.assert_allclose(total, np.zeros(3), atol=1e-10)


def test_direction_norm_and_omega_constraint_after_step() -> None:
    cfg = Config(
        system=SystemConfig(
            n_rods=4, rod_length=1.0, mass=1.0, box=(5.0, 5.0, 5.0), quadrature_points=3
        ),
        initial=InitialConfig(seed=2, initial_force_scale=1.0, initial_torque_scale=1.0),
    )
    sim = Simulation.from_config(cfg)
    sim.step(5)
    np.testing.assert_allclose(np.linalg.norm(sim.state.directions, axis=1), np.ones(4), atol=1e-12)
    np.testing.assert_allclose(
        np.sum(sim.state.omegas * sim.state.directions, axis=1), np.zeros(4), atol=1e-12
    )


def test_initial_torques_are_perpendicular_to_rods() -> None:
    cfg = Config(
        system=SystemConfig(n_rods=8, rod_length=1.0, mass=1.0, box=(5.0, 5.0, 5.0)),
        initial=InitialConfig(seed=4),
    )
    inertia = cfg.system.mass * cfg.system.rod_length**2 / 12.0
    state, kick = build_initial_state(cfg, inertia)
    assert isinstance(kick, InitialKick)
    perpendicular = project_perpendicular(kick.torques, state.directions)
    np.testing.assert_allclose(kick.torques, perpendicular, atol=1e-12)
