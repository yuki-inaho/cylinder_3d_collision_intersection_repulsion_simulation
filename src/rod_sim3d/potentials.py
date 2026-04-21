"""Pairwise-repulsive shifted-force potential.

The potential is designed so that both the potential value and its derivative vanish at
the cutoff radius ``rc``, eliminating discontinuities in force and in energy at the
cutoff. Every function here is pure and array-polymorphic; it maps trivially to Rust.

Mathematical definition
-----------------------
For ``0 < r < rc`` (all other ``r`` give zero):

    phi(r) = A [ (lam/r)^p - (lam/rc)^p + p lam^p rc^(-p-1) (r - rc) ]

with

    dphi/dr = A p lam^p [ r^(-p-1) - rc^(-p-1) ] * (-1)    <- gradient is -force_magnitude

so the force magnitude along ``R/r`` is

    f(r) = -dphi/dr = A p lam^p [ r^(-p-1) - rc^(-p-1) ]   for r < rc, else 0.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from rod_sim3d._array import FloatArray


class RepulsivePotentialCfg(Protocol):
    """Parameters accepted by :func:`shifted_potential` and :func:`shifted_force_magnitude`.

    A Python Protocol here doubles as specification for the Rust port: any struct
    exposing these scalar fields can participate in the same interaction algebra.
    """

    strength: float
    length_scale: float
    cutoff: float
    exponent: float
    softening: float


def regularized_distance(r_vec: FloatArray, softening: float) -> FloatArray:
    """Return ``sqrt(|R|^2 + softening^2)`` elementwise along the last axis.

    The softening term keeps the force finite when two quadrature points coincide,
    preserving well-posedness for any initial configuration.
    """

    return np.sqrt(np.sum(r_vec * r_vec, axis=-1) + softening * softening)


def shifted_potential(rho: FloatArray, cfg: RepulsivePotentialCfg) -> FloatArray:
    """Shifted-force inverse-power potential evaluated at distance ``rho``.

    ``rho`` may be any-rank. Output has the same shape.
    """

    rho_arr = np.asarray(rho, dtype=float)
    out = np.zeros_like(rho_arr)
    mask = rho_arr < cfg.cutoff
    if not np.any(mask):
        return out

    lam_p, p, rc = _potential_constants(cfg)
    rm = rho_arr[mask]
    out[mask] = cfg.strength * (
        lam_p * rm ** (-p) - lam_p * rc ** (-p) + p * lam_p * rc ** (-(p + 1.0)) * (rm - rc)
    )
    return out


def shifted_force_magnitude(rho: FloatArray, cfg: RepulsivePotentialCfg) -> FloatArray:
    """Return ``-d phi / d rho`` (i.e. the radial repulsive force magnitude)."""

    rho_arr = np.asarray(rho, dtype=float)
    out = np.zeros_like(rho_arr)
    mask = rho_arr < cfg.cutoff
    if not np.any(mask):
        return out

    lam_p, p, rc = _potential_constants(cfg)
    rm = rho_arr[mask]
    out[mask] = cfg.strength * p * lam_p * (rm ** (-(p + 1.0)) - rc ** (-(p + 1.0)))
    return out


def shifted_force_vectors(r_vec: FloatArray, cfg: RepulsivePotentialCfg) -> FloatArray:
    """Force on the source point due to the partner, for displacement ``r_vec``.

    ``r_vec`` has shape ``(..., 3)``; the returned array has the same shape.
    Points away from the partner (along ``R/|R|``) when the force is active.
    """

    rho = regularized_distance(r_vec, cfg.softening)
    magnitude = shifted_force_magnitude(rho, cfg)
    return magnitude[..., None] * r_vec / rho[..., None]


def wall_force_magnitude(distance_to_wall: FloatArray, cfg: RepulsivePotentialCfg) -> FloatArray:
    """One-sided wall repulsion magnitude.

    ``distance_to_wall`` is signed. Negative values mean the point has crossed the
    wall; the magnitude is then saturated at the softening distance and still points
    inward via the sign applied by the caller.
    """

    distance = np.asarray(distance_to_wall, dtype=float)
    active = distance < cfg.cutoff
    effective = np.maximum(distance, cfg.softening)
    magnitude = shifted_force_magnitude(effective, cfg)
    return np.where(active, magnitude, 0.0)


def _potential_constants(cfg: RepulsivePotentialCfg) -> tuple[float, float, float]:
    return cfg.length_scale**cfg.exponent, cfg.exponent, cfg.cutoff
