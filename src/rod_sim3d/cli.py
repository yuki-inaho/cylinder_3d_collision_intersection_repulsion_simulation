"""Command line interface for RodSim3D.

The CLI is a thin translation layer:

- :func:`build_parser` defines the argparse surface.
- :func:`load_config` + :func:`apply_run_overrides` turn ``argparse.Namespace`` into a
  validated :class:`Config`.
- :func:`run_command` and :func:`inspect_command` wire the :class:`Simulation` to the
  renderer and optional recorders.

Every piece is a small, named function so that a future port (to a Rust binary plus
pyo3, or to ``clap``) can walk these functions one by one.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from rod_sim3d._array import FloatArray
from rod_sim3d.config import Config
from rod_sim3d.cylinder_intersection import compute_pairwise_overlaps, cylinders_from_rods
from rod_sim3d.renderer import FrameCallback, animate, options_from_simulation
from rod_sim3d.replay import ReplayOptions, replay
from rod_sim3d.simulation import Simulation
from rod_sim3d.storage import load_config_from_db, open_database, write_frame, write_meta


class TrajectoryRecorder:
    """Collect simulation snapshots and write a compressed NPZ trajectory.

    Each ``capture`` takes a deep copy of the state arrays so the stored timeline is
    decoupled from the live simulation.
    """

    def __init__(self) -> None:
        self.frames: list[int] = []
        self.times: list[float] = []
        self.positions: list[FloatArray] = []
        self.directions: list[FloatArray] = []
        self.velocities: list[FloatArray] = []
        self.omegas: list[FloatArray] = []

    def capture(self, sim: Simulation, frame: int) -> None:
        self.frames.append(frame)
        self.times.append(sim.time)
        self.positions.append(sim.state.positions.copy())
        self.directions.append(sim.state.directions.copy())
        self.velocities.append(sim.state.velocities.copy())
        self.omegas.append(sim.state.omegas.copy())

    def save(self, path: str | Path, sim: Simulation) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            frames=np.asarray(self.frames, dtype=int),
            times=np.asarray(self.times, dtype=float),
            positions=np.asarray(self.positions, dtype=float),
            directions=np.asarray(self.directions, dtype=float),
            velocities=np.asarray(self.velocities, dtype=float),
            omegas=np.asarray(self.omegas, dtype=float),
            initial_forces=sim.initial_kick.forces,
            initial_torques=sim.initial_kick.torques,
            config_json=json.dumps(sim.config.to_dict(), ensure_ascii=False),
        )


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = _COMMANDS.get(args.command or "")
    if command is None:
        parser.print_help()
        return
    command(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rod-sim3d",
        description=(
            "3D rod / capsule dynamics in a box with three selectable interaction "
            "models (NONE / SOFT_REPULSION / HARD_CONTACT). See docs/model.md for "
            "the physics, and configs/ for example TOMLs.\n\n"
            "Typical workflow:\n"
            "  rod-sim3d compute --config configs/cylinders_hard.toml "
            "--db runs/hard.db --frames 1500\n"
            "  rod-sim3d replay  --db runs/hard.db --backend pyvista\n"
            "(or run `just demo` / `just hard` for one-line shortcuts)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(
        dest="command",
        metavar="{run,compute,replay,inspect}",
        help="subcommand; pass -h after a subcommand for its own flags",
    )

    run = sub.add_parser(
        "run",
        help="run a simulation and animate it live (no SQLite, no replay)",
        description=(
            "Run a simulation and render it in real time. Preferred for the "
            "SOFT_REPULSION mode, or for quick inspection of configs. For long "
            "runs use `compute` + `replay` which separates physics from rendering "
            "and records overlaps to disk.\n\n"
            "Example:\n"
            "  rod-sim3d run --config configs/cylinders_hard.toml --frames 800"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_config_args(run)
    run.add_argument(
        "--backend",
        choices=["pyvista", "matplotlib", "none"],
        default=None,
        help="rendering backend (default: value from [render].backend in TOML)",
    )
    run.add_argument(
        "--no-render",
        action="store_true",
        help="alias for --backend none (headless run, advances physics only)",
    )
    run.add_argument(
        "--frames",
        type=int,
        default=None,
        help="total rendered frames (default: [render].frames in TOML)",
    )
    run.add_argument(
        "--substeps-per-frame",
        type=int,
        default=None,
        help="physics sub-steps per rendered frame (integrator dt stays the same)",
    )
    run.add_argument(
        "--trajectory",
        type=str,
        default=None,
        help="optional path to dump positions/velocities as a compressed NPZ",
    )
    run.add_argument(
        "--initial-kick-json",
        type=str,
        default=None,
        help="optional path to dump the sampled random F0 / tau0 vectors as JSON",
    )
    run.add_argument(
        "--no-save-initial-kick",
        action="store_true",
        help="disable the default initial-kick JSON dump",
    )

    inspect_cmd = sub.add_parser(
        "inspect",
        help="sample the initial condition and print a diagnostic JSON",
        description=(
            "Build the initial state from a TOML config, print kinetic / potential "
            "energy and the norms of the sampled initial F0 / tau0. Does not step "
            "the simulation. Useful for tuning [initial] parameters.\n\n"
            "Example:\n"
            "  rod-sim3d inspect --config configs/cylinders_hard.toml --seed 42"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_config_args(inspect_cmd)
    inspect_cmd.add_argument(
        "--initial-kick-json",
        type=str,
        default=None,
        help="optional path to dump the sampled random F0 / tau0 as JSON",
    )

    compute_cmd = sub.add_parser(
        "compute",
        help="headless physics run; state + overlaps written to a SQLite DB",
        description=(
            "Advance the simulation and record every rendered frame to a SQLite "
            "database. No viewer is opened. The DB is the input to `replay`.\n\n"
            "Stored tables: meta (config JSON), frames, states (x/u per rod), "
            "overlaps (intersection volumes of every pair in the NONE mode, or "
            "residual penetration in HARD_CONTACT).\n\n"
            "Example:\n"
            "  rod-sim3d compute --config configs/cylinders.toml --db runs/demo.db \\\n"
            "                    --frames 1200 --overlap-quadrature 16"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_config_args(compute_cmd)
    compute_cmd.add_argument(
        "--db",
        type=str,
        required=True,
        help="output SQLite path (directories are created if needed)",
    )
    compute_cmd.add_argument(
        "--frames",
        type=int,
        default=None,
        help="total rendered frames (default: [render].frames in TOML)",
    )
    compute_cmd.add_argument(
        "--substeps-per-frame",
        type=int,
        default=None,
        help="physics sub-steps per rendered frame",
    )
    compute_cmd.add_argument(
        "--overlap-quadrature",
        type=int,
        default=16,
        help=(
            "Gauss-Legendre points for the slice-integral intersection-volume "
            "computation (default: 16; raise to 32 for tighter precision at ~2x cost)"
        ),
    )
    compute_cmd.add_argument(
        "--overlap-stride",
        type=int,
        default=1,
        help="compute overlaps every N rendered frames (default: 1 = every frame)",
    )

    replay_cmd = sub.add_parser(
        "replay",
        help="replay a previously computed SQLite DB (no physics; just render)",
        description=(
            "Read a SQLite DB produced by `compute` and animate it. The physics "
            "is NOT re-run, so frame rate is governed purely by the render loop.\n\n"
            "Viewer keys:\n"
            "  Esc / Q : quit\n"
            "  R       : reset to frame 0\n"
            "  L       : toggle loop mode\n\n"
            "Example:\n"
            "  rod-sim3d replay --db runs/demo.db --backend pyvista --opacity 0.5"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    replay_cmd.add_argument(
        "--db",
        type=str,
        required=True,
        help="input SQLite path (produced by `rod-sim3d compute`)",
    )
    replay_cmd.add_argument(
        "--backend",
        choices=["pyvista", "matplotlib", "none"],
        default="pyvista",
        help=(
            "rendering backend. 'pyvista' = interactive 3D (default); "
            "'matplotlib' = fallback; 'none' = headless iteration for smoke tests"
        ),
    )
    replay_cmd.add_argument(
        "--pause",
        type=float,
        default=0.01,
        help="inter-frame wall-clock pause in seconds (default: 0.01)",
    )
    replay_cmd.add_argument(
        "--hide-overlaps",
        action="store_true",
        help="skip drawing the magenta intersection polytopes",
    )
    replay_cmd.add_argument(
        "--opacity",
        type=float,
        default=None,
        help=(
            "rod translucency in [0, 1]. 1 = opaque (good for HARD_CONTACT); "
            "0.4-0.6 is nice for the NONE mode so you can see overlapping capsules. "
            "Depth peeling is auto-enabled when < 1. Default: value from "
            "[render].opacity in the DB's embedded config"
        ),
    )
    replay_cmd.add_argument(
        "--export-gif",
        type=str,
        default=None,
        help=(
            "write the replay as an animated GIF (off-screen rendering, no viewer "
            "window). Only valid for --backend pyvista"
        ),
    )
    replay_cmd.add_argument(
        "--export-fps",
        type=int,
        default=20,
        help="GIF frame rate (default: 20)",
    )
    replay_cmd.add_argument(
        "--export-max-frames",
        type=int,
        default=300,
        help=(
            "cap on GIF length; frames are sub-sampled as "
            "ceil(total_frames / this) (default: 300 → ~15s at 20 fps)"
        ),
    )

    return parser


def _add_common_config_args(parser: argparse.ArgumentParser) -> None:
    """CLI flags shared by ``run`` / ``compute`` / ``inspect``.

    These all override the corresponding TOML value when given, otherwise the
    TOML value wins. Handy for sweeping a parameter without editing the file.
    """

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="path to a TOML config (see configs/ for presets)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="override [initial].seed (reproducible random placement + kicks)",
    )
    parser.add_argument(
        "--n-rods",
        type=int,
        default=None,
        help="override [system].n_rods",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help=(
            "override [dynamics].dt. In HARD_CONTACT, ensure max |v| * dt < "
            "contact_radius to avoid tunneling"
        ),
    )
    # --- Initial kick overrides -------------------------------------------
    # These scale the random initial force / torque (and how long that kick is
    # integrated into v(0) / omega(0)). Cheap knobs for making a scene look
    # livelier without touching the TOML:
    #     v(0)     = F0 * kick_duration / mass
    #     omega(0) = P_perp(tau0) * kick_duration / inertia
    parser.add_argument(
        "--initial-force-scale",
        type=float,
        default=None,
        help=(
            "stddev sigma_F of the random initial force (overrides "
            "[initial].initial_force_scale). Larger = higher initial speed"
        ),
    )
    parser.add_argument(
        "--initial-torque-scale",
        type=float,
        default=None,
        help=(
            "stddev sigma_tau of the random initial torque (overrides "
            "[initial].initial_torque_scale). Larger = faster tumbling"
        ),
    )
    parser.add_argument(
        "--kick-duration",
        type=float,
        default=None,
        help=(
            "effective impulse duration T_kick in seconds (overrides "
            "[initial].kick_duration). Scales both v(0) and omega(0) linearly"
        ),
    )
    parser.add_argument(
        "--linear-damping",
        type=float,
        default=None,
        help=(
            "override [dynamics].linear_damping gamma_t. Set to 0 to conserve "
            "kinetic energy (together with angular-damping = 0)"
        ),
    )
    parser.add_argument(
        "--angular-damping",
        type=float,
        default=None,
        help="override [dynamics].angular_damping gamma_r (0 = no rotational damping)",
    )


def load_config(args: argparse.Namespace) -> Config:
    cfg = Config.from_toml(args.config) if args.config else Config()
    if args.seed is not None:
        cfg.initial.seed = args.seed
    if args.n_rods is not None:
        cfg.system.n_rods = args.n_rods
    if args.dt is not None:
        cfg.dynamics.dt = args.dt
    if args.initial_force_scale is not None:
        cfg.initial.initial_force_scale = args.initial_force_scale
    if args.initial_torque_scale is not None:
        cfg.initial.initial_torque_scale = args.initial_torque_scale
    if args.kick_duration is not None:
        cfg.initial.kick_duration = args.kick_duration
    if args.linear_damping is not None:
        cfg.dynamics.linear_damping = args.linear_damping
    if args.angular_damping is not None:
        cfg.dynamics.angular_damping = args.angular_damping
    cfg.validate()
    return cfg


def apply_run_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    if args.no_render:
        cfg.render.backend = "none"
    elif args.backend is not None:
        cfg.render.backend = args.backend
    if args.frames is not None:
        cfg.render.frames = args.frames
    if args.substeps_per_frame is not None:
        cfg.render.substeps_per_frame = args.substeps_per_frame
    if args.trajectory is not None:
        cfg.output.trajectory_npz = args.trajectory
    if args.initial_kick_json is not None:
        cfg.output.initial_kick_json = args.initial_kick_json
    if args.no_save_initial_kick:
        cfg.output.initial_kick_json = None
    cfg.validate()
    return cfg


def run_command(args: argparse.Namespace) -> None:
    cfg = apply_run_overrides(load_config(args), args)
    sim = Simulation.from_config(cfg)

    if cfg.output.initial_kick_json:
        sim.initial_kick.save_json(cfg.output.initial_kick_json, cfg.system.mass, sim.inertia)
        print(f"initial kick written: {cfg.output.initial_kick_json}")

    recorder = TrajectoryRecorder() if cfg.output.trajectory_npz else None
    if recorder is not None:
        recorder.capture(sim, frame=0)

    animate(sim, options=options_from_simulation(sim), on_frame=_make_frame_callback(recorder))

    if recorder is not None and cfg.output.trajectory_npz:
        recorder.save(cfg.output.trajectory_npz, sim)
        print(f"trajectory written: {cfg.output.trajectory_npz}")

    energy = sim.energy()
    print(
        f"done: steps={sim.step_index}, time={sim.time:.6g}, "
        f"energy_total={energy.total:.6g}, kinetic={energy.kinetic:.6g}"
    )


def inspect_command(args: argparse.Namespace) -> None:
    cfg = load_config(args)
    sim = Simulation.from_config(cfg)
    energy = sim.energy()
    payload: dict[str, Any] = {
        "n_rods": sim.state.n_rods,
        "time": sim.time,
        "box": list(cfg.system.box),
        "rod_length": cfg.system.rod_length,
        "inertia": sim.inertia,
        "energy": {
            "kinetic": energy.kinetic,
            "potential_pair": energy.potential_pair,
            "potential_wall": energy.potential_wall,
            "total": energy.total,
        },
        "initial_force_norms": np.linalg.norm(sim.initial_kick.forces, axis=1).tolist(),
        "initial_torque_norms": np.linalg.norm(sim.initial_kick.torques, axis=1).tolist(),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    path = args.initial_kick_json or cfg.output.initial_kick_json
    if path:
        sim.initial_kick.save_json(path, cfg.system.mass, sim.inertia)
        print(f"initial kick written: {path}")


def _make_frame_callback(recorder: TrajectoryRecorder | None) -> FrameCallback | None:
    if recorder is None:
        return None

    def on_frame(sim: Simulation, frame: int) -> None:
        recorder.capture(sim, frame + 1)

    return on_frame


def compute_command(args: argparse.Namespace) -> None:
    cfg = load_config(args)
    if args.frames is not None:
        cfg.render.frames = args.frames
    if args.substeps_per_frame is not None:
        cfg.render.substeps_per_frame = args.substeps_per_frame
    cfg.validate()

    sim = Simulation.from_config(cfg)
    total_frames = cfg.render.frames
    substeps = cfg.render.substeps_per_frame
    stride = max(1, args.overlap_stride)

    with open_database(args.db) as conn:
        write_meta(conn, json.dumps(cfg.to_dict(), ensure_ascii=False))
        _record_frame(
            conn,
            sim,
            cfg,
            frame_id=0,
            quadrature_points=args.overlap_quadrature,
            stride=stride,
        )

        progress = tqdm(
            range(1, total_frames + 1),
            desc="compute",
            unit="frame",
            dynamic_ncols=True,
        )
        for frame_id in progress:
            sim.step(substeps)
            _record_frame(
                conn,
                sim,
                cfg,
                frame_id=frame_id,
                quadrature_points=args.overlap_quadrature,
                stride=stride,
            )

    print(f"compute done: frames={total_frames + 1}, n_rods={sim.state.n_rods}, db={args.db}")


def replay_command(args: argparse.Namespace) -> None:
    conn = sqlite3.connect(args.db)
    try:
        cfg = Config.from_mapping(load_config_from_db(conn))
        opacity = args.opacity if args.opacity is not None else cfg.render.opacity
        options = ReplayOptions(
            backend=args.backend,
            window_size=tuple(cfg.render.window_size),
            pause=args.pause,
            show_box=cfg.render.show_box,
            show_overlaps=not args.hide_overlaps,
            opacity=float(opacity),
            overlap_opacity=float(cfg.render.overlap_opacity),
            export_gif=args.export_gif or "",
            export_fps=int(args.export_fps),
            export_max_frames=int(args.export_max_frames),
        )
        replay(conn, cfg, options)
    finally:
        conn.close()


def _record_frame(
    conn: sqlite3.Connection,
    sim: Simulation,
    cfg: Config,
    *,
    frame_id: int,
    quadrature_points: int,
    stride: int,
) -> None:
    overlaps: list[tuple[int, int, float]] = []
    if cfg.system.rod_radius > 0.0 and frame_id % stride == 0:
        cylinders = cylinders_from_rods(
            sim.state.positions,
            sim.state.directions,
            cfg.system.rod_length,
            cfg.system.rod_radius,
        )
        overlaps = compute_pairwise_overlaps(cylinders, quadrature_points=quadrature_points)
    write_frame(
        conn,
        frame_id=frame_id,
        time=sim.time,
        positions=sim.state.positions,
        directions=sim.state.directions,
        overlaps=overlaps,
    )


_COMMANDS: dict[str, Callable[[argparse.Namespace], None]] = {
    "run": run_command,
    "inspect": inspect_command,
    "compute": compute_command,
    "replay": replay_command,
}


if __name__ == "__main__":
    main()
