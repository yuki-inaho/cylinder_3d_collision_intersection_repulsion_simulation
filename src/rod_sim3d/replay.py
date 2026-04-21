"""Replay a simulation from a SQLite trajectory with finite-radius cylinder rendering.

Decouples compute from render: :mod:`rod_sim3d.cli` stores a simulation via
:mod:`rod_sim3d.storage`, and this module replays the frames using whichever
backend the user picks. The replay itself does no physics; it only reads
snapshots and draws them, which keeps interactive frame rates high.

Overlaps are visualized by rebuilding the exact convex intersection polytope
of every overlapping capsule pair for the current frame and rendering it as a
single translucent magenta body. Visual weight scales with the intersection
volume naturally. See :func:`_upsert_overlap_mesh`.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rod_sim3d._array import FloatArray
from rod_sim3d._playback_controls import (
    help_text,
    install_matplotlib_playback_controls,
    install_pyvista_playback_controls,
    pyvista_window_is_gone,
)
from rod_sim3d.config import Config
from rod_sim3d.cylinder_intersection import cylinders_from_rods
from rod_sim3d.cylinder_polytope import intersection_polytope
from rod_sim3d.storage import FrameSnapshot, iter_frame_ids, load_frame

# Number of prism sides used to approximate each cylinder when building the
# intersection convex polytope for visual highlight. 16 is a visual/speed sweet
# spot — volumes land within ~3% of the slice-integral truth.
_OVERLAP_MESH_N_SIDES: int = 16
# Fixed highlight color; alpha comes from ReplayOptions.overlap_opacity.
_OVERLAP_COLOR: str = "magenta"


@dataclass(slots=True, frozen=True)
class ReplayOptions:
    """Visualization options for replay.

    ``opacity`` controls rod translucency:
      - ``1.0`` : fully opaque (HARD_CONTACT style)
      - ``0.4 - 0.7`` : see-through capsules, useful for the NONE / pass-through
        mode where visible overlaps make the mode intuitive
      - ``< 0.3`` : nearly invisible, overlaps dominate
    Depth peeling is enabled automatically when ``opacity < 1.0`` so stacked
    translucent cylinders render in the right order.

    ``overlap_opacity`` controls the alpha of the magenta intersection-polytope
    mesh. Higher values keep the highlight solid even when ``opacity`` is low.

    ``export_gif`` / ``export_fps`` / ``export_max_frames`` switch the pyvista
    backend from interactive playback into off-screen GIF capture. When
    ``export_gif`` is non-empty the Plotter is created with ``off_screen=True``,
    the file is opened via ``plotter.open_gif``, and one frame is written per
    rendered step. Frame sub-sampling uses ``ceil(total / export_max_frames)``
    so the resulting file stays modest in size.
    """

    backend: str = "pyvista"
    window_size: tuple[int, int] = (1100, 850)
    pause: float = 0.01
    show_box: bool = True
    show_overlaps: bool = True
    opacity: float = 1.0
    overlap_opacity: float = 0.9
    export_gif: str = ""
    export_fps: int = 20
    export_max_frames: int = 300


def replay(conn: sqlite3.Connection, config: Config, options: ReplayOptions) -> None:
    """Iterate stored frames and draw them with the requested backend."""

    frame_ids = list(iter_frame_ids(conn))
    if not frame_ids:
        raise RuntimeError("Database has no frames; run the compute step first.")

    backend = options.backend.lower()
    if backend == "pyvista":
        _replay_pyvista(conn, config, frame_ids, options)
    elif backend == "matplotlib":
        _replay_matplotlib(conn, config, frame_ids, options)
    elif backend == "none":
        _replay_headless(conn, frame_ids)
    else:
        raise ValueError(f"Unknown replay backend: {options.backend!r}")


def _replay_headless(conn: sqlite3.Connection, frame_ids: Iterable[int]) -> None:
    for frame_id in frame_ids:
        load_frame(conn, frame_id)


def _replay_pyvista(
    conn: sqlite3.Connection,
    config: Config,
    frame_ids: list[int],
    options: ReplayOptions,
) -> None:
    import pyvista as pv  # noqa: PLC0415 — deferred to keep import-time cheap

    if options.export_gif:
        _export_pyvista_gif(pv, conn, config, frame_ids, options)
        return

    first = load_frame(conn, frame_ids[0])
    endpoints = _endpoints(first.positions, first.directions, config.system.rod_length)

    rod_radius = config.system.rod_radius
    tube_mesh = _build_tube_mesh(pv, endpoints, rod_radius)
    plotter: Any = pv.Plotter(window_size=list(options.window_size))
    _maybe_enable_depth_peeling(plotter, options.opacity)
    plotter.add_mesh(
        tube_mesh,
        scalars="rod_id",
        cmap="turbo",
        show_scalar_bar=False,
        opacity=options.opacity,
    )
    overlap_actor: Any = None
    if options.show_overlaps:
        overlap_actor = _upsert_overlap_mesh(
            pv, plotter, None, first, config, options.overlap_opacity
        )

    _decorate_scene(pv, plotter, config, options)
    plotter.add_text(
        _status_text(first, len(frame_ids), loop=False),
        position="upper_left",
        font_size=10,
        name="status",
    )
    plotter.add_text(
        help_text(),
        position="lower_right",
        font_size=9,
        color="gray",
        name="help",
    )
    controls = install_pyvista_playback_controls(plotter)
    plotter.show(interactive_update=True, auto_close=False)

    frame_idx = 1  # the next index to render (frame 0 is already shown)
    while True:
        if controls.stop or pyvista_window_is_gone(plotter):
            break
        if controls.reset_pending:
            controls.reset_pending = False
            frame_idx = 0

        if frame_idx >= len(frame_ids):
            if controls.loop:
                frame_idx = 0
            else:
                # Idle at the last frame: keep the window responsive to keys.
                try:
                    plotter.update()
                except RuntimeError:
                    break
                continue

        snap = load_frame(conn, frame_ids[frame_idx])
        endpoints = _endpoints(snap.positions, snap.directions, config.system.rod_length)
        _refresh_tube_mesh(tube_mesh, endpoints, rod_radius)

        if options.show_overlaps:
            overlap_actor = _upsert_overlap_mesh(
                pv, plotter, overlap_actor, snap, config, options.overlap_opacity
            )

        plotter.add_text(
            _status_text(snap, len(frame_ids), loop=controls.loop),
            position="upper_left",
            font_size=10,
            name="status",
        )
        try:
            plotter.update()
        except RuntimeError:
            # VTK raises when the render window is destroyed mid-update (user clicked X).
            break
        frame_idx += 1

    if not pyvista_window_is_gone(plotter):
        plotter.close()


def _export_pyvista_gif(
    pv: Any,
    conn: sqlite3.Connection,
    config: Config,
    frame_ids: list[int],
    options: ReplayOptions,
) -> None:
    """Render the replay off-screen and emit a GIF at ``options.export_gif``.

    The Plotter is created with ``off_screen=True``. Frames are sub-sampled so
    the resulting file has at most ``options.export_max_frames`` frames, which
    keeps file sizes and decode cost reasonable for README embeds.
    """

    from math import ceil  # noqa: PLC0415

    from tqdm import tqdm  # noqa: PLC0415

    Path(options.export_gif).parent.mkdir(parents=True, exist_ok=True)

    stride = max(1, ceil(len(frame_ids) / max(1, options.export_max_frames)))
    sampled_frames = frame_ids[::stride]

    first = load_frame(conn, sampled_frames[0])
    endpoints = _endpoints(first.positions, first.directions, config.system.rod_length)
    rod_radius = config.system.rod_radius
    tube_mesh = _build_tube_mesh(pv, endpoints, rod_radius)

    plotter: Any = pv.Plotter(off_screen=True, window_size=list(options.window_size))
    _maybe_enable_depth_peeling(plotter, options.opacity)
    plotter.add_mesh(
        tube_mesh,
        scalars="rod_id",
        cmap="turbo",
        show_scalar_bar=False,
        opacity=options.opacity,
    )
    overlap_actor: Any = None
    if options.show_overlaps:
        overlap_actor = _upsert_overlap_mesh(
            pv, plotter, None, first, config, options.overlap_opacity
        )
    _decorate_scene(pv, plotter, config, options)
    plotter.add_text(
        _status_text(first, len(frame_ids), loop=False),
        position="upper_left",
        font_size=10,
        name="status",
    )

    plotter.open_gif(options.export_gif, fps=options.export_fps)
    plotter.show(auto_close=False)
    plotter.write_frame()

    progress = tqdm(
        sampled_frames[1:],
        desc=f"gif→{Path(options.export_gif).name}",
        unit="frame",
        dynamic_ncols=True,
    )
    for frame_id in progress:
        snap = load_frame(conn, frame_id)
        endpoints = _endpoints(snap.positions, snap.directions, config.system.rod_length)
        _refresh_tube_mesh(tube_mesh, endpoints, rod_radius)
        if options.show_overlaps:
            overlap_actor = _upsert_overlap_mesh(
                pv, plotter, overlap_actor, snap, config, options.overlap_opacity
            )
        plotter.add_text(
            _status_text(snap, len(frame_ids), loop=False),
            position="upper_left",
            font_size=10,
            name="status",
        )
        plotter.write_frame()

    plotter.close()


def _replay_matplotlib(
    conn: sqlite3.Connection,
    config: Config,
    frame_ids: list[int],
    options: ReplayOptions,
) -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    plt.ion()
    fig: Any = plt.figure(figsize=(options.window_size[0] / 120, options.window_size[1] / 120))
    ax: Any = fig.add_subplot(111, projection="3d")
    box = np.asarray(config.system.box, dtype=float)
    ax.set_xlim(0, box[0])
    ax.set_ylim(0, box[1])
    ax.set_zlim(0, box[2])
    ax.set_box_aspect(tuple(box))

    controls = install_matplotlib_playback_controls(fig)

    frame_idx = 0
    while True:
        if controls.stop or not plt.fignum_exists(fig.number):
            break
        if controls.reset_pending:
            controls.reset_pending = False
            frame_idx = 0

        if frame_idx >= len(frame_ids):
            if controls.loop:
                frame_idx = 0
            else:
                # Idle on last frame: pump events so keys keep working.
                plt.pause(options.pause)
                continue

        ax.cla()
        ax.set_xlim(0, box[0])
        ax.set_ylim(0, box[1])
        ax.set_zlim(0, box[2])
        ax.set_box_aspect(tuple(box))
        snap = load_frame(conn, frame_ids[frame_idx])
        endpoints = _endpoints(snap.positions, snap.directions, config.system.rod_length)
        for i, (a, b) in enumerate(endpoints):
            color = "tab:red" if _rod_in_overlap(i, snap.overlaps) else "tab:blue"
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color=color, linewidth=3.0)
        ax.set_title(f"{_status_text(snap, len(frame_ids), loop=controls.loop)}    {help_text()}")
        plt.pause(options.pause)
        frame_idx += 1

    plt.ioff()
    if plt.fignum_exists(fig.number):
        plt.close(fig)


# --- Geometry and PyVista helpers ------------------------------------------------------


def _endpoints(positions: FloatArray, directions: FloatArray, rod_length: float) -> FloatArray:
    half = 0.5 * rod_length * directions
    return np.stack((positions - half, positions + half), axis=1)


def _build_tube_mesh(pv: Any, endpoints: FloatArray, radius: float) -> Any:
    """Construct a PolyData of rod polylines and convert to a tube mesh with given radius."""

    return _polylines_to_tube(pv, endpoints, radius)


def _refresh_tube_mesh(mesh: Any, endpoints: FloatArray, radius: float) -> None:
    """Rebuild the tube mesh for the new ``endpoints`` and copy it into ``mesh``.

    We always pass the same ``radius`` that was used at construction time; otherwise
    the on-screen thickness would drift frame to frame. ``copy_from`` swaps the VTK
    data blocks behind the PolyData without invalidating the actor reference, so the
    existing render pipeline keeps working.
    """

    import pyvista as pv  # noqa: PLC0415

    fresh = _polylines_to_tube(pv, endpoints, radius)
    mesh.copy_from(fresh)


def _maybe_enable_depth_peeling(plotter: Any, opacity: float) -> None:
    """Turn on order-independent transparency for translucent rods.

    Depth peeling lets overlapping semi-transparent cylinders render in the correct
    front-to-back order. It has a small cost, so we only enable it when actually
    needed (``opacity < 1``). PyVista on some platforms silently fails to enable
    peeling; we swallow the error rather than crash the replay.
    """

    if opacity >= 1.0:
        return
    try:
        plotter.enable_depth_peeling(number_of_peels=6)
    except Exception:
        # Some backends (e.g. OpenGL without depth-peeling extensions) raise here.
        # The viewer still works; it just may have minor blending artefacts.
        return


def _polylines_to_tube(pv: Any, endpoints: FloatArray, radius: float) -> Any:
    """Build a tube mesh for the given rod endpoints at the requested radius."""

    n = endpoints.shape[0]
    points = endpoints.reshape(-1, 3)
    lines = np.empty((n, 3), dtype=np.int64)
    lines[:, 0] = 2
    lines[:, 1] = 2 * np.arange(n)
    lines[:, 2] = 2 * np.arange(n) + 1
    poly = pv.PolyData(points, lines=lines.reshape(-1))
    poly.cell_data["rod_id"] = np.arange(n)
    tube_radius = max(radius, 1e-3)
    return poly.tube(radius=tube_radius, n_sides=16, capping=True)


def _upsert_overlap_mesh(
    pv: Any,
    plotter: Any,
    existing_actor: Any,
    snap: FrameSnapshot,
    config: Config,
    opacity: float,
) -> Any:
    """Draw the convex intersection polytope of every overlapping capsule pair.

    Each ``C_i ∩ C_j`` is approximated by an N-gon-prism intersection (see
    :func:`rod_sim3d.cylinder_polytope.intersection_polytope`) and the resulting
    triangulated convex polytope is merged into a single PolyData. Rendered in
    a single flat magenta colour with ``opacity`` coming from the caller so the
    highlight alpha is tunable from the config / CLI.
    """

    if existing_actor is not None:
        plotter.remove_actor(existing_actor, render=False)
    if not snap.overlaps:
        return None

    cylinders = cylinders_from_rods(
        snap.positions,
        snap.directions,
        config.system.rod_length,
        config.system.rod_radius,
    )
    mesh = _build_intersection_polytope_mesh(pv, cylinders, snap.overlaps)
    if mesh is None:
        return None
    return plotter.add_mesh(
        mesh,
        color=_OVERLAP_COLOR,
        opacity=opacity,
        show_scalar_bar=False,
        specular=0.2,
        smooth_shading=True,
    )


def _build_intersection_polytope_mesh(
    pv: Any,
    cylinders: list,
    overlaps: list[tuple[int, int, float]],
) -> Any:
    """Merge the convex intersection polytopes of all overlap pairs into one PolyData."""

    vertex_blocks: list[FloatArray] = []
    face_blocks: list[FloatArray] = []
    vertex_offset = 0
    for i, j, _vol in overlaps:
        poly = intersection_polytope(cylinders[i], cylinders[j], n_sides=_OVERLAP_MESH_N_SIDES)
        if poly is None:
            continue
        verts = poly.vertices
        simplices = poly.simplices
        # PyVista PolyData `faces` format: [3, v0, v1, v2, 3, v0, v1, v2, ...].
        lead = np.full((simplices.shape[0], 1), 3, dtype=np.int64)
        faces_flat = np.hstack((lead, simplices + vertex_offset)).ravel()
        vertex_blocks.append(verts)
        face_blocks.append(faces_flat)
        vertex_offset += verts.shape[0]

    if not vertex_blocks:
        return None

    points = np.vstack(vertex_blocks)
    faces = np.concatenate(face_blocks)
    return pv.PolyData(points, faces=faces)


def _decorate_scene(pv: Any, plotter: Any, config: Config, options: ReplayOptions) -> None:
    plotter.set_background("white")
    if options.show_box:
        box = np.asarray(config.system.box, dtype=float)
        outline = pv.Box(bounds=(0.0, box[0], 0.0, box[1], 0.0, box[2])).outline()
        plotter.add_mesh(outline, color="black", line_width=1.5)
    center = tuple(0.5 * np.asarray(config.system.box, dtype=float))
    distance = 2.5 * float(np.linalg.norm(config.system.box))
    position = (
        center[0] + distance,
        center[1] - distance,
        center[2] + 0.7 * distance,
    )
    plotter.camera_position = [position, center, (0.0, 0.0, 1.0)]


def _rod_in_overlap(rod_id: int, overlaps: list[tuple[int, int, float]]) -> bool:
    return any(rod_id in (i, j) for (i, j, _v) in overlaps)


def _status_text(snap: FrameSnapshot, total_frames: int, *, loop: bool = False) -> str:
    loop_tag = "  [loop]" if loop else ""
    return (
        f"frame={snap.frame_id}/{total_frames}  t={snap.time:.3f}  "
        f"rods={snap.positions.shape[0]}  overlaps={len(snap.overlaps)}{loop_tag}"
    )
