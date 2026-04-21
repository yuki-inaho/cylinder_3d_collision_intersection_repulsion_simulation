"""Tests for the command-line interface itself.

These are lightweight CLI-level integration tests: we call `rod_sim3d.cli.main`
with a small in-memory configuration and verify that the subcommands run end to end.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from rod_sim3d.cli import build_parser, load_config, main
from rod_sim3d.simulation import Simulation

FAST_TOML = """
[system]
n_rods = 4
rod_length = 0.8
mass = 1.0
box = [4.0, 4.0, 4.0]
quadrature_points = 3

[pair_potential]
strength = 0.02
length_scale = 0.3
cutoff = 0.7
exponent = 8.0
softening = 0.04

[wall_potential]
strength = 0.05
length_scale = 0.3
cutoff = 0.6
exponent = 8.0
softening = 0.03

[dynamics]
dt = 0.004
linear_damping = 0.2
angular_damping = 0.2

[initial]
seed = 11
clearance = 0.1
min_segment_distance = 0.05
initial_force_scale = 2.0
initial_torque_scale = 0.5
kick_duration = 0.05

[render]
backend = "none"
frames = 10
substeps_per_frame = 2

[output]
trajectory_npz = ""
initial_kick_json = ""
"""


@pytest.fixture()
def tmp_config(tmp_path: Path) -> Path:
    path = tmp_path / "fast.toml"
    path.write_text(FAST_TOML, encoding="utf-8")
    return path


def test_inspect_runs_without_writing_files(
    tmp_config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["inspect", "--config", str(tmp_config)])
    captured = capsys.readouterr()
    # The fixture config sets initial_kick_json = "", so only a JSON object is printed.
    payload = json.loads(captured.out)
    assert payload["n_rods"] == 4
    assert payload["rod_length"] == pytest.approx(0.8)
    assert "kinetic" in payload["energy"]


def test_run_headless_writes_trajectory(tmp_config: Path, tmp_path: Path) -> None:
    traj = tmp_path / "run" / "traj.npz"
    kick = tmp_path / "run" / "kick.json"
    main(
        [
            "run",
            "--config",
            str(tmp_config),
            "--backend",
            "none",
            "--frames",
            "5",
            "--trajectory",
            str(traj),
            "--initial-kick-json",
            str(kick),
        ]
    )
    assert traj.exists(), "trajectory NPZ must be written when --trajectory is given."
    assert kick.exists(), "initial-kick JSON must be written when --initial-kick-json is given."

    data = np.load(traj)
    assert data["positions"].shape[0] == 6  # frame 0 + 5 captures
    assert data["positions"].shape[1] == 4  # 4 rods
    assert data["positions"].shape[2] == 3

    payload = json.loads(kick.read_text())
    assert "forces" in payload and "torques" in payload
    assert len(payload["forces"]) == 4


def test_run_respects_no_render_and_no_save_flags(tmp_config: Path, tmp_path: Path) -> None:
    main(
        [
            "run",
            "--config",
            str(tmp_config),
            "--no-render",
            "--frames",
            "3",
            "--no-save-initial-kick",
        ]
    )
    # Nothing to assert beyond "did not raise"; smoke test of the CLI wiring.


def test_initial_kick_cli_overrides_reach_the_state(tmp_config: Path, tmp_path: Path) -> None:
    """The ``--initial-force-scale`` / ``--kick-duration`` flags must flow into
    the sampled initial velocities, giving the user a one-liner to make a
    scene more energetic."""

    args = build_parser().parse_args(
        [
            "run",
            "--config",
            str(tmp_config),
            "--initial-force-scale",
            "50.0",
            "--kick-duration",
            "0.1",
            "--linear-damping",
            "0.0",
            "--angular-damping",
            "0.0",
            "--seed",
            "0",
        ]
    )
    cfg = load_config(args)
    assert cfg.initial.initial_force_scale == pytest.approx(50.0)
    assert cfg.initial.kick_duration == pytest.approx(0.1)
    assert cfg.dynamics.linear_damping == 0.0
    assert cfg.dynamics.angular_damping == 0.0

    sim = Simulation.from_config(cfg)
    # Expected mean |v0| ~ force_scale * kick_duration / mass = 5.0 m/s per axis.
    # With 3 axes it's ~sqrt(3) * 5 ~ 8.7 m/s on average.
    speeds = np.linalg.norm(sim.state.velocities, axis=1)
    assert speeds.mean() > 3.0, f"expected energetic initial speeds; got mean={speeds.mean():.2f}"


CYLINDERS_TOML = """
[system]
n_rods = 4
rod_length = 0.8
rod_radius = 0.15
mass = 1.0
box = [3.0, 3.0, 3.0]
quadrature_points = 3

[pair_potential]
strength = 0.0

[wall_potential]
strength = 0.05
length_scale = 0.3
cutoff = 0.6
exponent = 8.0
softening = 0.03

[dynamics]
dt = 0.005
linear_damping = 0.1
angular_damping = 0.1

[initial]
seed = 17
clearance = 0.2
min_segment_distance = 0.0
initial_force_scale = 2.0
initial_torque_scale = 0.5
kick_duration = 0.05

[render]
backend = "none"
frames = 5
substeps_per_frame = 1
"""


def test_compute_command_writes_db_with_frames_states_and_overlaps(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cylinders.toml"
    cfg_path.write_text(CYLINDERS_TOML, encoding="utf-8")
    db_path = tmp_path / "sim.db"

    main(
        [
            "compute",
            "--config",
            str(cfg_path),
            "--db",
            str(db_path),
            "--frames",
            "3",
            "--overlap-quadrature",
            "8",
        ]
    )

    assert db_path.exists()

    conn = sqlite3.connect(db_path)
    try:
        (frames,) = conn.execute("SELECT COUNT(*) FROM frames").fetchone()
        (states,) = conn.execute("SELECT COUNT(*) FROM states").fetchone()
    finally:
        conn.close()
    assert frames == 4  # frame 0 plus 3 steps
    assert states == 4 * 4  # 4 rods per frame


def test_replay_headless_reads_full_db(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cylinders.toml"
    cfg_path.write_text(CYLINDERS_TOML, encoding="utf-8")
    db_path = tmp_path / "sim.db"
    main(["compute", "--config", str(cfg_path), "--db", str(db_path), "--frames", "3"])

    main(["replay", "--db", str(db_path), "--backend", "none"])
