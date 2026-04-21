"""End-to-end toy-problem tests for the rod dynamics simulator.

These tests are intentionally written as "physics sanity checks" rather than unit tests.
Each one isolates a single physical behavior so that a failure points to a specific
conceptual bug (e.g. energy is not conserved, a rod passes through a wall, the length
of a rod drifts, etc.).
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
from rod_sim3d.geometry import gauss_legendre_segment, segment_distance
from rod_sim3d.initial_conditions import InitialKick
from rod_sim3d.simulation import Simulation
from rod_sim3d.state import RodState


def _make_simulation(
    *,
    positions: np.ndarray,
    directions: np.ndarray,
    velocities: np.ndarray | None = None,
    omegas: np.ndarray | None = None,
    rod_length: float = 1.0,
    box: tuple[float, float, float] = (10.0, 10.0, 10.0),
    pair: PairPotentialConfig | None = None,
    wall: WallPotentialConfig | None = None,
    dynamics: DynamicsConfig | None = None,
    quadrature_points: int = 5,
) -> Simulation:
    """Construct a Simulation with explicit state, avoiding random initialization."""

    n = positions.shape[0]
    if velocities is None:
        velocities = np.zeros_like(positions)
    if omegas is None:
        omegas = np.zeros_like(positions)

    cfg = Config(
        system=SystemConfig(
            n_rods=n,
            rod_length=rod_length,
            mass=1.0,
            box=box,
            quadrature_points=quadrature_points,
        ),
        pair=pair or PairPotentialConfig(strength=0.0),
        wall=wall or WallPotentialConfig(strength=0.0),
        dynamics=dynamics or DynamicsConfig(dt=0.002, linear_damping=0.0, angular_damping=0.0),
        initial=InitialConfig(seed=0, clearance=0.01, min_segment_distance=0.0),
        render=RenderConfig(backend="none", frames=1, substeps_per_frame=1),
    )
    nodes, weights = gauss_legendre_segment(cfg.system.quadrature_points, cfg.system.rod_length)
    inertia = cfg.system.mass * cfg.system.rod_length**2 / 12.0
    state = RodState(
        positions=positions.astype(float).copy(),
        directions=directions.astype(float).copy(),
        velocities=velocities.astype(float).copy(),
        omegas=omegas.astype(float).copy(),
    )
    state.enforce_constraints()
    kick = InitialKick(
        forces=np.zeros((n, 3)),
        torques=np.zeros((n, 3)),
        kick_duration=0.0,
    )
    return Simulation(cfg, state, nodes, weights, inertia, kick)


def _rod_lengths(sim: Simulation) -> np.ndarray:
    ends = sim.endpoints()
    return np.linalg.norm(ends[:, 1] - ends[:, 0], axis=1)


def test_single_free_rod_travels_in_a_straight_line_with_constant_speed() -> None:
    """With no interactions and no damping, a rod follows x(t) = x(0) + v(0) * t."""

    sim = _make_simulation(
        positions=np.array([[5.0, 5.0, 5.0]]),
        directions=np.array([[1.0, 0.0, 0.0]]),
        velocities=np.array([[0.3, 0.0, 0.0]]),
        dynamics=DynamicsConfig(dt=0.002, linear_damping=0.0, angular_damping=0.0),
    )
    x0 = sim.state.positions[0].copy()
    v0 = sim.state.velocities[0].copy()
    sim.step(500)
    expected = x0 + v0 * (500 * sim.config.dynamics.dt)
    np.testing.assert_allclose(sim.state.positions[0], expected, atol=1e-9)
    np.testing.assert_allclose(sim.state.velocities[0], v0, atol=1e-12)


def test_rod_length_is_preserved_over_many_steps() -> None:
    sim = _make_simulation(
        positions=np.array([[5.0, 5.0, 5.0], [6.5, 5.0, 5.0]]),
        directions=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        velocities=np.array([[0.05, 0.02, 0.0], [-0.03, 0.0, 0.01]]),
        omegas=np.array([[0.0, 0.0, 0.5], [0.2, -0.1, 0.0]]),
        pair=PairPotentialConfig(
            strength=0.05, length_scale=0.3, cutoff=0.9, exponent=8.0, softening=0.03
        ),
    )
    for _ in range(200):
        sim.step(5)
        lengths = _rod_lengths(sim)
        np.testing.assert_allclose(lengths, np.ones_like(lengths), atol=1e-10)


def test_two_close_rods_accelerate_apart() -> None:
    """When two rods are inside each other's cutoff, the pair force must push them apart."""

    sim = _make_simulation(
        positions=np.array([[5.0, 5.0, 5.0], [5.3, 5.0, 5.0]]),
        directions=np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        pair=PairPotentialConfig(
            strength=0.1, length_scale=0.4, cutoff=1.5, exponent=8.0, softening=0.02
        ),
    )
    result = sim.compute_forces()

    separation = sim.state.positions[1] - sim.state.positions[0]
    f0_along = float(np.dot(result.forces[0], separation))
    f1_along = float(np.dot(result.forces[1], separation))
    assert f0_along < 0.0, "Rod 0 should be pushed along -separation (away from rod 1)."
    assert f1_along > 0.0, "Rod 1 should be pushed along +separation (away from rod 0)."
    np.testing.assert_allclose(result.forces[0] + result.forces[1], np.zeros(3), atol=1e-10)


def test_two_close_rods_separate_over_time() -> None:
    sim = _make_simulation(
        positions=np.array([[5.0, 5.0, 5.0], [5.3, 5.0, 5.0]]),
        directions=np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        pair=PairPotentialConfig(
            strength=0.1, length_scale=0.4, cutoff=1.5, exponent=8.0, softening=0.02
        ),
        dynamics=DynamicsConfig(dt=0.002, linear_damping=0.0, angular_damping=0.0),
    )
    d0 = float(np.linalg.norm(sim.state.positions[1] - sim.state.positions[0]))
    sim.step(400)
    d1 = float(np.linalg.norm(sim.state.positions[1] - sim.state.positions[0]))
    assert d1 > d0 + 0.05, f"Rods should have separated; d0={d0:.3f}, d1={d1:.3f}"


def test_rod_near_wall_is_pushed_inward() -> None:
    sim = _make_simulation(
        positions=np.array([[0.35, 5.0, 5.0]]),
        directions=np.array([[0.0, 1.0, 0.0]]),
        wall=WallPotentialConfig(
            strength=0.2, length_scale=0.4, cutoff=0.9, exponent=8.0, softening=0.02
        ),
    )
    result = sim.compute_forces()
    assert result.forces[0, 0] > 0.0, "A rod near the x=0 wall must feel +x force."


def test_rod_with_inward_velocity_bounces_off_wall() -> None:
    sim = _make_simulation(
        positions=np.array([[0.6, 5.0, 5.0]]),
        directions=np.array([[0.0, 1.0, 0.0]]),
        velocities=np.array([[-0.6, 0.0, 0.0]]),
        wall=WallPotentialConfig(
            strength=0.3, length_scale=0.4, cutoff=0.9, exponent=8.0, softening=0.02
        ),
        dynamics=DynamicsConfig(dt=0.001, linear_damping=0.0, angular_damping=0.0),
    )
    for _ in range(1500):
        sim.step(1)
    assert sim.state.positions[0, 0] > 0.05, "Rod must not cross the wall."
    assert sim.state.velocities[0, 0] > 0.0, "Rod must end up moving away from the wall."


def test_total_linear_momentum_is_conserved_for_internal_forces_only() -> None:
    sim = _make_simulation(
        positions=np.array([[5.0, 5.0, 5.0], [5.3, 5.0, 5.0]]),
        directions=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        velocities=np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]]),
        pair=PairPotentialConfig(
            strength=0.1, length_scale=0.4, cutoff=1.5, exponent=8.0, softening=0.02
        ),
        dynamics=DynamicsConfig(dt=0.001, linear_damping=0.0, angular_damping=0.0),
    )
    p0 = np.sum(sim.state.velocities, axis=0) * sim.config.system.mass
    sim.step(500)
    p1 = np.sum(sim.state.velocities, axis=0) * sim.config.system.mass
    np.testing.assert_allclose(p0, p1, atol=5e-10)


def test_energy_is_monotonically_non_increasing_with_damping() -> None:
    """With only damping (no potentials), energy must never increase and must decay."""

    linear_damping = 2.0
    angular_damping = 2.0
    dt = 0.005
    n_steps = 600
    sim = _make_simulation(
        positions=np.array([[5.0, 5.0, 5.0]]),
        directions=np.array([[1.0, 0.0, 0.0]]),
        velocities=np.array([[0.4, 0.2, -0.1]]),
        omegas=np.array([[0.0, 0.3, 0.0]]),
        dynamics=DynamicsConfig(
            dt=dt, linear_damping=linear_damping, angular_damping=angular_damping
        ),
    )
    e0 = sim.energy().total
    prev = e0
    for _ in range(n_steps):
        sim.step(1)
        current = sim.energy().total
        assert current <= prev + 1e-12, f"Energy increased: {prev} -> {current}"
        prev = current
    # With damping >= min(linear_damping, angular_damping) == 2.0 and t = 3.0 s,
    # velocities are attenuated by exp(-6) ~ 2.5e-3, so the energy ratio is ~ (2.5e-3)^2 = 6e-6.
    assert prev / e0 < 1e-4, f"Energy should have decayed: E0={e0:.3e}, E_final={prev:.3e}"


def test_energy_is_approximately_conserved_when_far_apart_and_undamped() -> None:
    """With no damping and no interactions (far apart), E should stay essentially constant."""

    sim = _make_simulation(
        positions=np.array([[2.0, 5.0, 5.0], [8.0, 5.0, 5.0]]),
        directions=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        velocities=np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]]),
        omegas=np.array([[0.0, 0.0, 0.2], [0.1, 0.0, 0.0]]),
        pair=PairPotentialConfig(
            strength=0.05, length_scale=0.3, cutoff=0.5, exponent=8.0, softening=0.02
        ),
        dynamics=DynamicsConfig(dt=0.002, linear_damping=0.0, angular_damping=0.0),
    )
    e0 = sim.energy().total
    sim.step(1000)
    e1 = sim.energy().total
    rel_err = abs(e1 - e0) / abs(e0)
    assert rel_err < 1e-3, f"Relative energy drift too large: {rel_err:.3e}"


def test_omega_stays_perpendicular_to_rod_axis_over_many_steps() -> None:
    sim = _make_simulation(
        positions=np.array([[5.0, 5.0, 5.0]]),
        directions=np.array([[1.0, 0.0, 0.0]]),
        omegas=np.array([[0.0, 0.5, 0.3]]),
        dynamics=DynamicsConfig(dt=0.002, linear_damping=0.0, angular_damping=0.0),
    )
    for _ in range(500):
        sim.step(1)
        parallel = float(np.dot(sim.state.omegas[0], sim.state.directions[0]))
        assert abs(parallel) < 1e-10


def test_rods_do_not_penetrate_each_other_for_many_steps() -> None:
    """A conservative/weakly damped run of two rods placed near each other should never
    drive their minimum separation below a small fraction of the rod length."""

    sim = _make_simulation(
        positions=np.array([[5.0, 5.0, 5.0], [5.4, 5.0, 5.0]]),
        directions=np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        velocities=np.array([[0.0, 0.0, 0.0], [-0.2, 0.0, 0.0]]),
        pair=PairPotentialConfig(
            strength=0.15, length_scale=0.5, cutoff=1.2, exponent=8.0, softening=0.03
        ),
        dynamics=DynamicsConfig(dt=0.0005, linear_damping=0.05, angular_damping=0.05),
    )
    min_separation = float("inf")
    for _ in range(800):
        sim.step(1)
        ends = sim.endpoints()
        d = segment_distance(ends[0, 0], ends[0, 1], ends[1, 0], ends[1, 1])
        min_separation = min(min_separation, d)
    assert min_separation > 0.05, f"Rods got too close: min_separation={min_separation:.3f}"


def test_random_initial_kick_converts_force_to_velocity_correctly() -> None:
    """random_initial_state must satisfy v(0) = F0 * kick_duration / mass."""

    cfg = Config(
        system=SystemConfig(n_rods=8, rod_length=1.0, mass=2.5, box=(5.0, 5.0, 5.0)),
        initial=InitialConfig(seed=123, initial_force_scale=3.0, kick_duration=0.05),
    )
    sim = Simulation.from_config(cfg)
    expected_v = sim.initial_kick.forces * (cfg.initial.kick_duration / cfg.system.mass)
    np.testing.assert_allclose(sim.state.velocities, expected_v, atol=1e-12)


def test_random_initial_kick_converts_torque_to_omega_correctly() -> None:
    cfg = Config(
        system=SystemConfig(n_rods=8, rod_length=1.2, mass=1.5, box=(5.0, 5.0, 5.0)),
        initial=InitialConfig(seed=321, initial_torque_scale=2.0, kick_duration=0.04),
    )
    sim = Simulation.from_config(cfg)
    expected_omega = sim.initial_kick.torques * (cfg.initial.kick_duration / sim.inertia)
    np.testing.assert_allclose(sim.state.omegas, expected_omega, atol=1e-12)
