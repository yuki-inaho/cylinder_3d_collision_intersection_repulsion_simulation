"""Regression tests for the replay tube-mesh helpers.

These run without opening a window — they build the PolyData/Tube in memory and
inspect the geometric bounds directly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from rod_sim3d.replay import _build_tube_mesh, _refresh_tube_mesh  # noqa: E402


def _tube_span_yz(mesh: Any) -> tuple[float, float]:
    b = mesh.bounds
    return float(b.y_max - b.y_min), float(b.z_max - b.z_min)


def test_build_uses_requested_radius() -> None:
    endpoints = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    mesh = _build_tube_mesh(pv, endpoints, radius=0.22)
    y, z = _tube_span_yz(mesh)
    assert y == pytest.approx(2 * 0.22, rel=1e-3)
    assert z == pytest.approx(2 * 0.22, rel=1e-3)


def test_refresh_preserves_requested_radius() -> None:
    """Regression: a previous implementation reconstructed the radius from mesh bounds
    (and returned 0.05 regardless of the configured value), making tubes look
    progressively wrong after the first frame. Pin it down here."""

    endpoints_a = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    endpoints_b = np.array([[[5.0, 5.0, 5.0], [6.0, 5.0, 5.0]]])

    mesh = _build_tube_mesh(pv, endpoints_a, radius=0.22)
    _refresh_tube_mesh(mesh, endpoints_b, radius=0.22)
    y, z = _tube_span_yz(mesh)
    assert y == pytest.approx(2 * 0.22, rel=1e-3)
    assert z == pytest.approx(2 * 0.22, rel=1e-3)


def test_refresh_with_different_radius_changes_span() -> None:
    endpoints = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    mesh = _build_tube_mesh(pv, endpoints, radius=0.2)
    _refresh_tube_mesh(mesh, endpoints, radius=0.5)
    y, z = _tube_span_yz(mesh)
    assert y == pytest.approx(1.0, rel=1e-3)  # 2 * 0.5
    assert z == pytest.approx(1.0, rel=1e-3)
