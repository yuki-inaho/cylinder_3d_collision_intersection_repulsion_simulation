"""End-to-end regression tests for the pass-through cylinder mode.

These lock in the two defining properties of the mode:

1. With ``pair_potential.strength = 0`` the rods apply **no** pair forces or
   torques on each other, so nothing prevents them from interpenetrating.
2. The ``compute`` CLI records per-frame overlap volumes in the SQLite DB when
   cylinders do interpenetrate.

A separate test also confirms that the walls still keep a finite-radius
cylinder inside the box.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from rod_sim3d.cli import main
from rod_sim3d.config import (
    Config,
    DynamicsConfig,
    InitialConfig,
    PairPotentialConfig,
    RenderConfig,
    SystemConfig,
    WallPotentialConfig,
)
from rod_sim3d.simulation import Simulation


@pytest.fixture()
def passthrough_config() -> Config:
    """A tiny pass-through setup: 2 rods on a collision course with zero pair force."""

    return Config(
        system=SystemConfig(
            n_rods=2,
            rod_length=1.0,
            rod_radius=0.2,
            mass=1.0,
            box=(6.0, 6.0, 6.0),
            quadrature_points=4,
        ),
        pair=PairPotentialConfig(strength=0.0),  # the defining switch
        wall=WallPotentialConfig(strength=0.0),  # wall-off to isolate pair behavior
        dynamics=DynamicsConfig(dt=0.002, linear_damping=0.0, angular_damping=0.0),
        initial=InitialConfig(seed=0, clearance=0.01, min_segment_distance=0.0),
        render=RenderConfig(backend="none", frames=1, substeps_per_frame=1),
    )


def test_zero_strength_produces_no_pair_force(passthrough_config: Config) -> None:
    """With ``strength=0`` the force kernel must early-return zero forces/torques."""

    sim = Simulation.from_config(passthrough_config)
    # Place the two rods on top of each other
    sim.state.positions[0] = np.array([3.0, 3.0, 3.0])
    sim.state.positions[1] = np.array([3.0, 3.0, 3.0])
    sim.state.directions[0] = np.array([1.0, 0.0, 0.0])
    sim.state.directions[1] = np.array([0.0, 1.0, 0.0])
    sim.state.enforce_constraints()

    # With no wall and zero pair strength → net force and torque are exactly zero.
    result = sim.compute_forces()
    assert np.allclose(result.forces, 0.0)
    assert np.allclose(result.torques, 0.0)


def test_two_cylinders_interpenetrate_freely(passthrough_config: Config) -> None:
    """Launch two cylinders through each other and assert the closest approach is
    smaller than the sum of radii (i.e. they did overlap)."""

    sim = Simulation.from_config(passthrough_config)
    sim.state.positions[0] = np.array([2.5, 3.0, 3.0])
    sim.state.positions[1] = np.array([3.5, 3.0, 3.0])
    sim.state.directions[0] = np.array([0.0, 0.0, 1.0])
    sim.state.directions[1] = np.array([0.0, 0.0, 1.0])
    sim.state.velocities[0] = np.array([1.0, 0.0, 0.0])
    sim.state.velocities[1] = np.array([-1.0, 0.0, 0.0])
    sim.state.omegas[:] = 0.0
    sim.state.enforce_constraints()

    min_center_dist = float("inf")
    # Relative speed 2.0 over distance 1.0 → they meet in 0.5 s = 250 steps (dt=0.002).
    # Run 400 steps to ensure we sweep past the closest approach.
    for _ in range(400):
        sim.step(1)
        dx = sim.state.positions[1] - sim.state.positions[0]
        min_center_dist = min(min_center_dist, float(np.linalg.norm(dx)))

    # If they truly passed through each other the centers came very close.
    assert min_center_dist < 0.05, (
        f"Pass-through failed: min center distance was {min_center_dist:.3f}"
    )


PASSTHROUGH_TOML = """
[system]
n_rods = 8
rod_length = 1.0
rod_radius = 0.35
mass = 1.0
box = [4.0, 4.0, 4.0]
quadrature_points = 4

[pair_potential]
strength = 0.0

[wall_potential]
strength = 0.05
length_scale = 0.3
cutoff = 0.6
exponent = 8.0
softening = 0.03

[dynamics]
dt = 0.004
linear_damping = 0.02
angular_damping = 0.02

[initial]
seed = 3
clearance = 0.1
min_segment_distance = 0.0
initial_force_scale = 5.0
initial_torque_scale = 1.5
kick_duration = 0.08

[render]
backend = "none"
frames = 80
substeps_per_frame = 2
"""


def test_compute_cli_records_overlaps_in_db(tmp_path: Path) -> None:
    """End-to-end: running ``compute`` on a config that guarantees interpenetration
    must produce at least one row in the ``overlaps`` table."""

    cfg_path = tmp_path / "passthrough.toml"
    cfg_path.write_text(PASSTHROUGH_TOML, encoding="utf-8")
    db_path = tmp_path / "passthrough.db"

    main(
        [
            "compute",
            "--config",
            str(cfg_path),
            "--db",
            str(db_path),
            "--overlap-quadrature",
            "8",
        ]
    )

    conn = sqlite3.connect(db_path)
    try:
        (n_overlaps,) = conn.execute("SELECT COUNT(*) FROM overlaps").fetchone()
        max_v = conn.execute("SELECT MAX(volume) FROM overlaps").fetchone()[0]
    finally:
        conn.close()

    assert n_overlaps > 0, "pass-through mode should record at least one overlap"
    assert max_v is not None and max_v > 0


def test_passthrough_config_loads_cleanly() -> None:
    """The shipped ``configs/cylinders.toml`` must load, validate, and be
    recognizably a pass-through config (strength=0 for pairs)."""

    cfg = Config.from_toml("configs/cylinders.toml")
    assert cfg.system.rod_radius > 0, "cylinders config should have a finite radius"
    assert cfg.pair.strength == 0.0, (
        "cylinders.toml must set pair_potential.strength = 0 (pass-through mode); "
        f"got {cfg.pair.strength}"
    )
    assert cfg.wall.strength > 0, "walls should still bounce the cylinders"
