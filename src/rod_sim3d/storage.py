"""Offline simulation storage using SQLite.

The goal is a clean split between compute and render: a long simulation is written
to disk once; interactive animation replays the stored snapshots without any physics.

Schema
------
::

    meta     (key TEXT PRIMARY KEY, value TEXT)              -- config JSON and version
    frames   (frame_id INT PRIMARY KEY, t REAL)              -- one row per recorded frame
    states   (frame_id, rod_id, x, y, z, ux, uy, uz, PRIMARY KEY (frame_id, rod_id))
    overlaps (frame_id, i, j, volume, PRIMARY KEY (frame_id, i, j))

Array-valued columns are stored as tightly as possible; SQLite's REAL is double
precision, which is enough for our use case. For very long runs the writer batches
inserts inside a single transaction to keep throughput near the SSD's limit.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rod_sim3d._array import FloatArray

SCHEMA_VERSION: str = "1"


@dataclass(slots=True, frozen=True)
class FrameSnapshot:
    """In-memory snapshot returned by :func:`load_frame` during replay."""

    frame_id: int
    time: float
    positions: FloatArray
    directions: FloatArray
    overlaps: list[tuple[int, int, float]]  # (i, j, volume) with i < j


def initialize_schema(conn: sqlite3.Connection) -> None:
    """Create tables (idempotent) and set pragmas favoring write throughput."""

    conn.executescript(
        """
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;

        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS frames (
            frame_id INTEGER PRIMARY KEY,
            t REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS states (
            frame_id INTEGER NOT NULL,
            rod_id INTEGER NOT NULL,
            x REAL NOT NULL, y REAL NOT NULL, z REAL NOT NULL,
            ux REAL NOT NULL, uy REAL NOT NULL, uz REAL NOT NULL,
            PRIMARY KEY (frame_id, rod_id)
        );
        CREATE TABLE IF NOT EXISTS overlaps (
            frame_id INTEGER NOT NULL,
            i INTEGER NOT NULL,
            j INTEGER NOT NULL,
            volume REAL NOT NULL,
            PRIMARY KEY (frame_id, i, j)
        );
        CREATE INDEX IF NOT EXISTS idx_overlaps_frame ON overlaps (frame_id);
        """
    )


@contextmanager
def open_database(path: str | Path) -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection with the schema applied and a pragma layer tuned for writes."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target)
    try:
        initialize_schema(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


def write_meta(conn: sqlite3.Connection, config_json: str) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
        [("schema_version", SCHEMA_VERSION), ("config_json", config_json)],
    )


def write_frame(
    conn: sqlite3.Connection,
    frame_id: int,
    time: float,
    positions: FloatArray,
    directions: FloatArray,
    overlaps: list[tuple[int, int, float]] | None = None,
) -> None:
    """Write one full frame into the database in a single transaction batch."""

    conn.execute(
        "INSERT OR REPLACE INTO frames(frame_id, t) VALUES (?, ?)", (frame_id, float(time))
    )
    rows = [
        (frame_id, rod_id, *map(float, positions[rod_id]), *map(float, directions[rod_id]))
        for rod_id in range(positions.shape[0])
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO states(frame_id, rod_id, x, y, z, ux, uy, uz) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    if overlaps:
        conn.executemany(
            "INSERT OR REPLACE INTO overlaps(frame_id, i, j, volume) VALUES (?, ?, ?, ?)",
            [(frame_id, int(i), int(j), float(v)) for (i, j, v) in overlaps],
        )


def load_meta(conn: sqlite3.Connection) -> dict[str, str]:
    return {key: value for key, value in conn.execute("SELECT key, value FROM meta")}


def load_config_from_db(conn: sqlite3.Connection) -> dict:
    payload = load_meta(conn).get("config_json")
    if not payload:
        raise ValueError("Database has no 'config_json' meta entry; was it written?")
    return json.loads(payload)


def iter_frame_ids(conn: sqlite3.Connection) -> Iterator[int]:
    for (frame_id,) in conn.execute("SELECT frame_id FROM frames ORDER BY frame_id"):
        yield int(frame_id)


def load_frame(conn: sqlite3.Connection, frame_id: int) -> FrameSnapshot:
    row = conn.execute("SELECT t FROM frames WHERE frame_id = ?", (frame_id,)).fetchone()
    if row is None:
        raise KeyError(f"Frame {frame_id} is not in the database.")
    time = float(row[0])

    states = conn.execute(
        "SELECT rod_id, x, y, z, ux, uy, uz FROM states WHERE frame_id = ? ORDER BY rod_id",
        (frame_id,),
    ).fetchall()
    positions = np.asarray([r[1:4] for r in states], dtype=float)
    directions = np.asarray([r[4:7] for r in states], dtype=float)

    overlaps_rows = conn.execute(
        "SELECT i, j, volume FROM overlaps WHERE frame_id = ? ORDER BY i, j",
        (frame_id,),
    ).fetchall()
    overlaps = [(int(i), int(j), float(v)) for (i, j, v) in overlaps_rows]

    return FrameSnapshot(
        frame_id=frame_id,
        time=time,
        positions=positions,
        directions=directions,
        overlaps=overlaps,
    )
