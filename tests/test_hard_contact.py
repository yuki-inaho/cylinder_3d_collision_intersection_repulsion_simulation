"""Physics tests for the ``InteractionModel.HARD_CONTACT`` mode.

Covers the three classical rigid-body invariants:

1. Linear momentum is conserved across an isolated capsule-capsule collision.
2. Total kinetic energy is conserved for ``restitution = 1`` (elastic) and
   strictly decreases for ``restitution < 1`` (inelastic).
3. A capsule fired at a box wall bounces with the prescribed normal reversal
   and never penetrates its surface.

A final regression test locks in the contract that the solver must not apply
spurious impulses to already-separating contacts (no "sticky" artifacts).
"""

from __future__ import annotations

import numpy as np
import pytest

from rod_sim3d.config import (
    Config,
    DynamicsConfig,
    InitialConfig,
    InteractionModel,
    PairInteractionConfig,
    PairPotentialConfig,
    RenderConfig,
    SystemConfig,
    WallPotentialConfig,
)
from rod_sim3d.geometry import segment_distance
from rod_sim3d.hard_contact import (
    ContactParams,
    resolve_pair_contacts,
    resolve_wall_contacts,
    segment_closest_points,
)
from rod_sim3d.simulation import Simulation


def _hard_config(
    *,
    n_rods: int = 2,
    rod_length: float = 1.0,
    rod_radius: float = 0.2,
    mass: float = 1.0,
    box: tuple[float, float, float] = (10.0, 10.0, 10.0),
    quadrature_points: int = 4,
    restitution: float = 1.0,
    wall_restitution: float = 1.0,
    seed: int = 0,
) -> Config:
    return Config(
        system=SystemConfig(
            n_rods=n_rods,
            rod_length=rod_length,
            rod_radius=rod_radius,
            mass=mass,
            box=box,
            quadrature_points=quadrature_points,
        ),
        pair=PairPotentialConfig(strength=0.0),
        wall=WallPotentialConfig(strength=0.0),
        dynamics=DynamicsConfig(dt=0.001, linear_damping=0.0, angular_damping=0.0),
        initial=InitialConfig(seed=seed, clearance=0.01, min_segment_distance=0.0),
        render=RenderConfig(backend="none", frames=1, substeps_per_frame=1),
        pair_interaction=PairInteractionConfig(
            model=InteractionModel.HARD_CONTACT,
            contact_radius=rod_radius,
            restitution=restitution,
            wall_restitution=wall_restitution,
        ),
    )


def _place_head_on(sim: Simulation, speed: float = 1.0) -> None:
    """Put two parallel rods on a head-on collision course."""

    sim.state.positions[0] = np.array([4.5, 5.0, 5.0])
    sim.state.positions[1] = np.array([5.5, 5.0, 5.0])
    sim.state.directions[0] = np.array([0.0, 0.0, 1.0])
    sim.state.directions[1] = np.array([0.0, 0.0, 1.0])
    sim.state.velocities[0] = np.array([+speed, 0.0, 0.0])
    sim.state.velocities[1] = np.array([-speed, 0.0, 0.0])
    sim.state.omegas[:] = 0.0
    sim.state.enforce_constraints()


# --- segment_closest_points sanity -----------------------------------------------------


def test_segment_closest_points_perpendicular_gap() -> None:
    a0 = np.array([0.0, 0.0, 0.0])
    a1 = np.array([1.0, 0.0, 0.0])
    b0 = np.array([0.5, 1.0, 0.0])
    b1 = np.array([0.5, -1.0, 0.0])
    dist, pa, pb = segment_closest_points(a0, a1, b0, b1)
    assert dist == pytest.approx(0.0)
    assert pa == pytest.approx(np.array([0.5, 0.0, 0.0]))
    assert pb == pytest.approx(np.array([0.5, 0.0, 0.0]))


def test_segment_closest_points_parallel_offset() -> None:
    a0 = np.array([0.0, 0.0, 0.0])
    a1 = np.array([1.0, 0.0, 0.0])
    b0 = np.array([0.0, 1.0, 0.0])
    b1 = np.array([1.0, 1.0, 0.0])
    dist, _, _ = segment_closest_points(a0, a1, b0, b1)
    assert dist == pytest.approx(1.0)


# --- Pair collisions -------------------------------------------------------------------


def test_elastic_head_on_conserves_linear_momentum() -> None:
    cfg = _hard_config(restitution=1.0)
    sim = Simulation.from_config(cfg)
    _place_head_on(sim, speed=1.0)

    p_before = cfg.system.mass * sim.state.velocities.sum(axis=0)
    # Run long enough to guarantee contact resolution (about 0.3 s covers it).
    for _ in range(400):
        sim.step(1)
    p_after = cfg.system.mass * sim.state.velocities.sum(axis=0)

    np.testing.assert_allclose(p_before, p_after, atol=1e-10)


def test_elastic_head_on_conserves_kinetic_energy() -> None:
    cfg = _hard_config(restitution=1.0)
    sim = Simulation.from_config(cfg)
    _place_head_on(sim, speed=1.0)

    e_before = sim.energy().kinetic
    for _ in range(400):
        sim.step(1)
    e_after = sim.energy().kinetic

    assert e_after == pytest.approx(e_before, rel=1e-9)


def test_inelastic_collision_strictly_loses_energy() -> None:
    cfg = _hard_config(restitution=0.5)
    sim = Simulation.from_config(cfg)
    _place_head_on(sim, speed=1.0)

    e_before = sim.energy().kinetic
    for _ in range(400):
        sim.step(1)
    e_after = sim.energy().kinetic

    assert e_after < e_before * 0.99  # at least 1% of KE gone


def test_perfectly_inelastic_collision_gives_near_zero_normal_relative_velocity() -> None:
    cfg = _hard_config(restitution=0.0)
    sim = Simulation.from_config(cfg)
    _place_head_on(sim, speed=1.0)

    for _ in range(400):
        sim.step(1)

    # After e=0 impact, the two contact-point velocities along the contact normal
    # should be essentially equal; the rods move together in the contact direction.
    v_rel_x = sim.state.velocities[1, 0] - sim.state.velocities[0, 0]
    assert abs(v_rel_x) < 1e-6


def test_pair_stays_non_penetrating() -> None:
    """Even after multiple bounces the capsule pair must never share a point."""

    cfg = _hard_config(restitution=0.9)
    sim = Simulation.from_config(cfg)
    _place_head_on(sim, speed=1.5)

    contact_radius = cfg.pair_interaction.contact_radius
    assert contact_radius is not None
    rod_len = cfg.system.rod_length
    min_axis_dist = float("inf")
    for _ in range(600):
        sim.step(1)
        # Axis-axis closest distance shouldn't drop much below 2 * contact_radius.
        d = segment_distance(
            sim.state.positions[0] - 0.5 * rod_len * sim.state.directions[0],
            sim.state.positions[0] + 0.5 * rod_len * sim.state.directions[0],
            sim.state.positions[1] - 0.5 * rod_len * sim.state.directions[1],
            sim.state.positions[1] + 0.5 * rod_len * sim.state.directions[1],
        )
        min_axis_dist = min(min_axis_dist, d)

    # Tolerance: the impulse solver acts after the drift, so a single step can
    # penetrate by up to |v_rel| * dt. With v_rel ~ 3.0 and dt = 1e-3 this is
    # ~3e-3; allow 10x headroom for repeated bounces.
    assert min_axis_dist > 2 * contact_radius - 3e-2, (
        f"penetration too large: min_axis_dist={min_axis_dist:.4f}"
    )


# --- Wall collisions -------------------------------------------------------------------


def test_wall_bounce_reverses_normal_velocity_with_elastic_restitution() -> None:
    cfg = _hard_config(wall_restitution=1.0, box=(4.0, 4.0, 4.0))
    sim = Simulation.from_config(cfg)
    sim.state.positions[0] = np.array([0.5, 2.0, 2.0])
    sim.state.positions[1] = np.array([3.5, 2.0, 2.0])  # far from walls
    sim.state.directions[:] = np.array([0.0, 0.0, 1.0])
    sim.state.velocities[0] = np.array([-0.8, 0.0, 0.0])
    sim.state.velocities[1] = np.array([0.0, 0.0, 0.0])
    sim.state.omegas[:] = 0.0
    sim.state.enforce_constraints()

    v0_before = sim.state.velocities[0, 0]
    for _ in range(1200):
        sim.step(1)
    v0_after = sim.state.velocities[0, 0]

    # Elastic bounce: sign flips and magnitude is preserved (modulo tiny drift).
    assert v0_after > 0.0
    assert abs(v0_after) == pytest.approx(abs(v0_before), rel=5e-2)
    # Cylinder surface must still be inside the box.
    assert sim.state.positions[0, 0] >= cfg.system.rod_radius - 5e-3


def test_wall_bounce_inelastic_reduces_speed() -> None:
    cfg = _hard_config(wall_restitution=0.5)
    sim = Simulation.from_config(cfg)
    sim.state.positions[0] = np.array([0.5, 5.0, 5.0])
    sim.state.positions[1] = np.array([8.5, 5.0, 5.0])
    sim.state.directions[:] = np.array([0.0, 0.0, 1.0])
    sim.state.velocities[0] = np.array([-1.0, 0.0, 0.0])
    sim.state.velocities[1] = np.array([0.0, 0.0, 0.0])
    sim.state.omegas[:] = 0.0
    sim.state.enforce_constraints()

    for _ in range(1000):
        sim.step(1)

    # Normal component should have been damped by e_w = 0.5
    assert 0.2 < sim.state.velocities[0, 0] < 0.8  # between inelastic and elastic


# --- "No impulse on separating contact" ------------------------------------------------


def test_solver_ignores_separating_contacts() -> None:
    """Two overlapping capsules moving apart should *not* receive an impulse;
    the impulse is only applied to approaching contacts (``v_n < 0``)."""

    cfg = _hard_config()
    sim = Simulation.from_config(cfg)
    # Deliberately overlapping but flying apart
    sim.state.positions[0] = np.array([5.0, 5.0, 5.0])
    sim.state.positions[1] = np.array([5.1, 5.0, 5.0])
    sim.state.directions[:] = np.array([0.0, 0.0, 1.0])
    sim.state.velocities[0] = np.array([-1.0, 0.0, 0.0])
    sim.state.velocities[1] = np.array([+1.0, 0.0, 0.0])
    sim.state.omegas[:] = 0.0
    sim.state.enforce_constraints()

    v_before = sim.state.velocities.copy()
    params = ContactParams(
        mass=1.0,
        inertia=sim.inertia,
        contact_radius=0.2,
        restitution=1.0,
        wall_restitution=1.0,
    )
    resolved = resolve_pair_contacts(sim.state, cfg.system.rod_length, params)
    assert resolved == 0, "Separating pair should not trigger an impulse"
    np.testing.assert_array_equal(sim.state.velocities, v_before)


def test_resolve_wall_contacts_does_nothing_when_far_from_walls() -> None:
    cfg = _hard_config(box=(10.0, 10.0, 10.0))
    sim = Simulation.from_config(cfg)
    sim.state.positions[0] = np.array([5.0, 5.0, 5.0])
    sim.state.positions[1] = np.array([6.0, 5.0, 5.0])
    sim.state.velocities[:] = 0.0
    sim.state.omegas[:] = 0.0
    sim.state.enforce_constraints()

    params = ContactParams(
        mass=1.0,
        inertia=sim.inertia,
        contact_radius=0.2,
        restitution=1.0,
        wall_restitution=1.0,
    )
    assert resolve_wall_contacts(sim.state, np.asarray([10.0, 10.0, 10.0]), 1.0, params) == 0
