"""Toy-problem tests for the shifted-force repulsive potential itself."""

from __future__ import annotations

import numpy as np

from rod_sim3d.config import PairPotentialConfig, WallPotentialConfig
from rod_sim3d.potentials import (
    shifted_force_magnitude,
    shifted_force_vectors,
    shifted_potential,
)


def _pair_cfg() -> PairPotentialConfig:
    return PairPotentialConfig(
        strength=0.05, length_scale=0.4, cutoff=1.5, exponent=8.0, softening=0.02
    )


def test_potential_is_zero_beyond_cutoff() -> None:
    cfg = _pair_cfg()
    rho = np.array([cfg.cutoff, cfg.cutoff + 0.1, 3.0])
    phi = shifted_potential(rho, cfg)
    np.testing.assert_allclose(phi, np.zeros_like(phi))


def test_force_magnitude_is_zero_beyond_cutoff() -> None:
    cfg = _pair_cfg()
    rho = np.array([cfg.cutoff, cfg.cutoff + 0.1, 3.0])
    mag = shifted_force_magnitude(rho, cfg)
    np.testing.assert_allclose(mag, np.zeros_like(mag))


def test_potential_is_continuous_at_cutoff() -> None:
    cfg = _pair_cfg()
    eps = 1e-6
    phi_inside = shifted_potential(np.array([cfg.cutoff - eps]), cfg)[0]
    phi_outside = shifted_potential(np.array([cfg.cutoff + eps]), cfg)[0]
    assert abs(phi_inside - phi_outside) < 1e-6


def test_force_is_continuous_at_cutoff() -> None:
    """The shifted-force construction guarantees d phi/d rho = 0 at r = cutoff."""
    cfg = _pair_cfg()
    eps = 1e-6
    inside = shifted_force_magnitude(np.array([cfg.cutoff - eps]), cfg)[0]
    outside = shifted_force_magnitude(np.array([cfg.cutoff + eps]), cfg)[0]
    assert abs(inside) < 1e-4
    assert outside == 0.0


def test_potential_is_monotonically_decreasing_inside_cutoff() -> None:
    cfg = _pair_cfg()
    rho = np.linspace(0.08, cfg.cutoff - 1e-4, 256)
    phi = shifted_potential(rho, cfg)
    diffs = np.diff(phi)
    assert np.all(diffs <= 1e-12), "Repulsive potential must decrease as rho grows."


def test_force_vector_points_away_from_partner() -> None:
    cfg = _pair_cfg()
    rng = np.random.default_rng(0)
    r_vec = rng.normal(size=(64, 3)) * 0.4
    g = shifted_force_vectors(r_vec, cfg)
    rho = np.linalg.norm(r_vec, axis=1, keepdims=True)
    active = rho[:, 0] < cfg.cutoff
    if np.any(active):
        projection = np.sum(g[active] * r_vec[active], axis=1)
        assert np.all(projection >= -1e-12), "Force on point 1 should push it away from point 2."


def test_wall_potential_matches_pair_potential_functional_form() -> None:
    wall_cfg = WallPotentialConfig(
        strength=0.08, length_scale=0.35, cutoff=0.75, exponent=8.0, softening=0.025
    )
    r = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9])
    mag = shifted_force_magnitude(np.maximum(r, wall_cfg.softening), wall_cfg)
    assert mag[0] > mag[1] > mag[2] > mag[3] > mag[4] > 0.0
    assert mag[-1] == 0.0
