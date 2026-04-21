"""Tests for the finite-radius wall bounce.

These complement the zero-thickness tests in ``test_toy_problems.py`` by specifically
exercising the ``rod_radius`` path through ``accumulate_wall_forces``.
"""

from __future__ import annotations

import numpy as np

from rod_sim3d.config import (
    Config,
    DynamicsConfig,
    InitialConfig,
    PairPotentialConfig,
    RenderConfig,
    SystemConfig,
    WallPotentialConfig,
)
from rod_sim3d.geometry import gauss_legendre_segment
from rod_sim3d.initial_conditions import InitialKick
from rod_sim3d.simulation import Simulation
from rod_sim3d.state import RodState


def _make_sim(*, rod_radius: float, positions: np.ndarray, velocities: np.ndarray) -> Simulation:
    n = positions.shape[0]
    cfg = Config(
        system=SystemConfig(
            n_rods=n,
            rod_length=1.0,
            rod_radius=rod_radius,
            mass=1.0,
            box=(10.0, 10.0, 10.0),
            quadrature_points=4,
        ),
        pair=PairPotentialConfig(strength=0.0),
        wall=WallPotentialConfig(
            strength=0.25, length_scale=0.4, cutoff=0.9, exponent=8.0, softening=0.02
        ),
        dynamics=DynamicsConfig(dt=0.001, linear_damping=0.0, angular_damping=0.0),
        initial=InitialConfig(seed=0, clearance=0.01, min_segment_distance=0.0),
        render=RenderConfig(backend="none", frames=1, substeps_per_frame=1),
    )
    nodes, weights = gauss_legendre_segment(cfg.system.quadrature_points, cfg.system.rod_length)
    inertia = cfg.system.mass * cfg.system.rod_length**2 / 12.0
    state = RodState(
        positions=positions.astype(float).copy(),
        directions=np.tile(np.array([[0.0, 1.0, 0.0]]), (n, 1)).astype(float),
        velocities=velocities.astype(float).copy(),
        omegas=np.zeros((n, 3)),
    )
    state.enforce_constraints()
    kick = InitialKick(forces=np.zeros((n, 3)), torques=np.zeros((n, 3)), kick_duration=0.0)
    return Simulation(cfg, state, nodes, weights, inertia, kick)


def test_radius_pushes_repulsion_earlier() -> None:
    """Same center position, different radii: the larger radius must feel a stronger
    force because the wall kernel sees the surface, not the center line."""

    pos = np.array([[0.9, 5.0, 5.0]])
    sim_no_r = _make_sim(rod_radius=0.0, positions=pos, velocities=np.zeros((1, 3)))
    sim_with_r = _make_sim(rod_radius=0.3, positions=pos, velocities=np.zeros((1, 3)))
    f_no_r = sim_no_r.compute_forces().forces[0, 0]
    f_with_r = sim_with_r.compute_forces().forces[0, 0]
    assert f_with_r > f_no_r, (
        f"Finite radius should increase wall push (got {f_no_r} vs {f_with_r})"
    )


def test_cylinder_does_not_penetrate_wall() -> None:
    """A rod with radius 0.25 shot toward the x=0 wall must not leave its surface
    crossing the wall."""

    pos = np.array([[0.7, 5.0, 5.0]])
    vel = np.array([[-0.8, 0.0, 0.0]])
    sim = _make_sim(rod_radius=0.25, positions=pos, velocities=vel)
    for _ in range(2000):
        sim.step(1)
    x_center = sim.state.positions[0, 0]
    assert x_center > 0.25, f"Cylinder surface crossed the wall (center at x={x_center:.3f})"


def test_zero_radius_matches_legacy_behavior() -> None:
    """With rod_radius=0 the bounce position should match the line-rod behavior."""

    pos = np.array([[0.6, 5.0, 5.0]])
    vel = np.array([[-0.6, 0.0, 0.0]])
    sim = _make_sim(rod_radius=0.0, positions=pos, velocities=vel)
    for _ in range(1500):
        sim.step(1)
    assert sim.state.positions[0, 0] > 0.05
    assert sim.state.velocities[0, 0] > 0.0


def test_wall_force_equals_minus_gradient_of_wall_energy() -> None:
    """Verify ``F = -∇U`` for the wall potential with a finite rod_radius.

    Regression guard for the bug where ``wall_potential_energy`` ignored the radius
    and only the force subtracted it — producing a huge inconsistency between the
    reported potential energy and the actual force field.
    """

    x0 = 0.7
    h = 1.0e-5
    for rod_radius in (0.0, 0.15, 0.25):
        sim_c = _make_sim(
            rod_radius=rod_radius,
            positions=np.array([[x0, 5.0, 5.0]]),
            velocities=np.zeros((1, 3)),
        )
        sim_c.state.directions[0] = np.array([0.0, 0.0, 1.0])  # perpendicular to x-wall
        sim_c.state.enforce_constraints()

        sim_p = _make_sim(
            rod_radius=rod_radius,
            positions=np.array([[x0 + h, 5.0, 5.0]]),
            velocities=np.zeros((1, 3)),
        )
        sim_p.state.directions[0] = np.array([0.0, 0.0, 1.0])
        sim_p.state.enforce_constraints()

        sim_m = _make_sim(
            rod_radius=rod_radius,
            positions=np.array([[x0 - h, 5.0, 5.0]]),
            velocities=np.zeros((1, 3)),
        )
        sim_m.state.directions[0] = np.array([0.0, 0.0, 1.0])
        sim_m.state.enforce_constraints()

        fx = float(sim_c.compute_forces().forces[0, 0])
        dE_dx = (sim_p.energy().potential_wall - sim_m.energy().potential_wall) / (2 * h)
        assert abs(dE_dx + fx) < 1e-5 * max(1.0, abs(fx)), (
            f"F != -dE/dx at rod_radius={rod_radius}: F_x={fx:.4e}, dE/dx={dE_dx:.4e}"
        )


def test_total_energy_stays_bounded_with_finite_radius_and_walls() -> None:
    """With small damping the total energy is bounded and monotonically non-increasing
    when the rod has a finite radius and interacts only with walls."""

    sim = _make_sim(
        rod_radius=0.2,
        positions=np.array([[0.6, 5.0, 5.0]]),
        velocities=np.array([[-0.2, 0.0, 0.0]]),
    )
    sim.config.dynamics = type(sim.config.dynamics)(
        dt=0.0005, linear_damping=0.1, angular_damping=0.1
    )
    prev = sim.energy().total
    for _ in range(600):
        sim.step(1)
        now = sim.energy().total
        # Loose bound: damping + discretization together keep the trend non-increasing
        # over a window, with small room for first-order integrator noise.
        assert now <= prev + 1e-4, f"Energy grew spuriously: {prev} -> {now}"
        prev = now
