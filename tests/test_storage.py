"""Round-trip tests for :mod:`rod_sim3d.storage`.

We verify that frames written through :func:`write_frame` are readable via
:func:`load_frame` with identical array values (up to float precision).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from rod_sim3d.storage import (
    iter_frame_ids,
    load_config_from_db,
    load_frame,
    open_database,
    write_frame,
    write_meta,
)


def test_single_frame_roundtrip(tmp_path) -> None:
    db = tmp_path / "test.db"
    positions = np.array([[0.1, 0.2, 0.3], [1.4, 1.5, 1.6]])
    directions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    overlaps = [(0, 1, 0.03)]

    with open_database(db) as conn:
        write_meta(conn, json.dumps({"system": {"n_rods": 2}}))
        write_frame(conn, 0, 0.5, positions, directions, overlaps=overlaps)

    with open_database(db) as conn:
        snap = load_frame(conn, 0)

    assert snap.frame_id == 0
    assert snap.time == pytest.approx(0.5)
    np.testing.assert_allclose(snap.positions, positions)
    np.testing.assert_allclose(snap.directions, directions)
    assert snap.overlaps == [(0, 1, pytest.approx(0.03))]


def test_many_frames_ordered(tmp_path) -> None:
    db = tmp_path / "many.db"
    with open_database(db) as conn:
        for f in range(10):
            positions = np.full((3, 3), float(f))
            directions = np.tile([1.0, 0.0, 0.0], (3, 1))
            write_frame(conn, f, f * 0.01, positions, directions)

    with open_database(db) as conn:
        ids = list(iter_frame_ids(conn))
    assert ids == list(range(10))


def test_meta_roundtrip(tmp_path) -> None:
    db = tmp_path / "meta.db"
    cfg_payload = {"system": {"n_rods": 4, "rod_radius": 0.1}}
    with open_database(db) as conn:
        write_meta(conn, json.dumps(cfg_payload))

    with open_database(db) as conn:
        recovered = load_config_from_db(conn)
    assert recovered == cfg_payload
