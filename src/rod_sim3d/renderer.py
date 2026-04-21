"""Animation backends for RodSim3D.

The public :func:`animate` function dispatches to one of the concrete backends via
the :class:`Backend` protocol. Adding a new backend is a pure extension: implement the
protocol, register the factory in :data:`_BACKENDS`. ``Simulation`` itself never needs
to learn about rendering.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from rod_sim3d._array import FloatArray, IntArray
from rod_sim3d._playback_controls import (
    PlaybackControls,
    help_text,
    install_matplotlib_playback_controls,
    install_pyvista_playback_controls,
    pyvista_window_is_gone,
)
from rod_sim3d.initial_conditions import build_initial_state
from rod_sim3d.simulation import Simulation

FrameCallback = Callable[[Simulation, int], None]


@dataclass(slots=True, frozen=True)
class AnimationOptions:
    """Immutable options passed to every backend."""

    backend: str
    frames: int
    substeps_per_frame: int
    line_width: float
    window_size: tuple[int, int]
    show_box: bool
    camera_position: str
    matplotlib_pause: float


def options_from_simulation(sim: Simulation) -> AnimationOptions:
    render = sim.config.render
    return AnimationOptions(
        backend=render.backend,
        frames=render.frames,
        substeps_per_frame=render.substeps_per_frame,
        line_width=render.line_width,
        window_size=render.window_size,
        show_box=render.show_box,
        camera_position=render.camera_position,
        matplotlib_pause=render.matplotlib_pause,
    )


class Backend(Protocol):
    """Backend interface.

    Implementations are free to open windows, spawn threads, or render off-screen; the
    only contract is: advance ``sim`` by ``options.frames * options.substeps_per_frame``
    steps, invoking ``on_frame`` after each rendered frame.
    """

    def run(
        self, sim: Simulation, options: AnimationOptions, on_frame: FrameCallback | None
    ) -> None: ...


def animate(
    sim: Simulation,
    options: AnimationOptions | None = None,
    on_frame: FrameCallback | None = None,
) -> None:
    """Run an animation with the requested backend.

    Raises
    ------
    ValueError
        When ``options.backend`` does not match any registered backend.
    """

    resolved = options or options_from_simulation(sim)
    backend = _resolve_backend(resolved.backend)
    backend.run(sim, resolved, on_frame)


def _resolve_backend(name: str) -> Backend:
    try:
        factory = _BACKENDS[name.lower()]
    except KeyError as exc:
        known = ", ".join(sorted(_BACKENDS))
        raise ValueError(f"Unknown render backend: {name!r} (known: {known})") from exc
    return factory()


class HeadlessBackend:
    """Step the simulation without any visualization. Useful for trajectory generation."""

    def run(
        self, sim: Simulation, options: AnimationOptions, on_frame: FrameCallback | None
    ) -> None:
        for frame in range(options.frames):
            sim.step(options.substeps_per_frame)
            if on_frame is not None:
                on_frame(sim, frame)


class PyVistaBackend:
    """Fast OpenGL backend via PyVista.

    Falls back to Matplotlib if PyVista fails to import so that the CLI keeps working
    on minimal systems.
    """

    def run(
        self, sim: Simulation, options: AnimationOptions, on_frame: FrameCallback | None
    ) -> None:
        try:
            self._run(sim, options, on_frame)
        except ImportError as exc:
            print(
                f"PyVista is not importable; falling back to matplotlib. Original error: {exc}",
                file=sys.stderr,
            )
            MatplotlibBackend().run(sim, options, on_frame)

    @staticmethod
    def _run(sim: Simulation, options: AnimationOptions, on_frame: FrameCallback | None) -> None:
        # pyvista / vtk import is deferred to keep headless users from ever loading VTK.
        import pyvista as pv  # noqa: PLC0415

        n_rods = sim.state.n_rods
        poly: Any = pv.PolyData(_rod_points(sim), lines=_rod_lines(n_rods))
        poly.cell_data["rod_id"] = np.arange(n_rods)

        plotter: Any = pv.Plotter(window_size=list(options.window_size))
        plotter.add_mesh(
            poly,
            scalars="rod_id",
            cmap="turbo",
            line_width=options.line_width,
            render_lines_as_tubes=True,
            show_scalar_bar=False,
        )
        if options.show_box:
            box = np.asarray(sim.config.system.box, dtype=float)
            outline = pv.Box(bounds=(0.0, box[0], 0.0, box[1], 0.0, box[2])).outline()
            plotter.add_mesh(outline, color="black", line_width=1.5)

        plotter.add_text(
            _status_text(sim, 0, loop=False),
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
        plotter.set_background("white")  # ty: ignore[invalid-argument-type]
        _set_pyvista_camera(plotter, sim, options.camera_position)
        controls = install_pyvista_playback_controls(plotter)

        plotter.show(interactive_update=True, auto_close=False)

        frame = 0
        while True:
            if controls.stop or pyvista_window_is_gone(plotter):
                break
            if controls.reset_pending:
                controls.reset_pending = False
                _reset_simulation(sim)
                frame = 0

            if frame >= options.frames:
                if controls.loop:
                    _reset_simulation(sim)
                    frame = 0
                else:
                    try:
                        plotter.update()
                    except RuntimeError:
                        break
                    continue

            sim.step(options.substeps_per_frame)
            poly.points = _rod_points(sim)
            poly.GetPoints().Modified()
            poly.Modified()
            plotter.add_text(
                _status_text(sim, frame + 1, loop=controls.loop),
                position="upper_left",
                font_size=10,
                name="status",
            )
            if on_frame is not None:
                on_frame(sim, frame)
            try:
                plotter.update()
            except RuntimeError:
                break
            frame += 1
        if not pyvista_window_is_gone(plotter):
            plotter.close()


class MatplotlibBackend:
    """Slow but universally available fallback using ``mpl_toolkits.mplot3d``."""

    def run(
        self, sim: Simulation, options: AnimationOptions, on_frame: FrameCallback | None
    ) -> None:
        # Deferred to keep import-time cheap when the user never needs matplotlib.
        import matplotlib.pyplot as plt  # noqa: PLC0415

        fig, lines, title = self._setup(plt, sim, options)
        controls = install_matplotlib_playback_controls(fig)
        self._animate_loop(plt, sim, options, on_frame, lines, title, controls)
        plt.ioff()
        if plt.fignum_exists(fig.number):
            plt.close(fig)

    @staticmethod
    def _setup(plt: Any, sim: Simulation, options: AnimationOptions) -> tuple[Any, list, Any]:
        plt.ion()
        fig: Any = plt.figure(figsize=(options.window_size[0] / 120, options.window_size[1] / 120))
        ax: Any = fig.add_subplot(111, projection="3d")
        box = np.asarray(sim.config.system.box, dtype=float)
        ax.set_xlim(0, box[0])
        ax.set_ylim(0, box[1])
        ax.set_zlim(0, box[2])
        ax.set_box_aspect(tuple(box))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        if options.show_box:
            _draw_matplotlib_box(ax, box)

        endpoints = sim.endpoints()
        lines: list = []
        for i in range(sim.state.n_rods):
            (line,) = ax.plot(
                endpoints[i, :, 0],
                endpoints[i, :, 1],
                endpoints[i, :, 2],
                linewidth=max(1.0, options.line_width / 2.0),
            )
            lines.append(line)
        title = ax.set_title(f"{_status_text(sim, 0, loop=False)}    {help_text()}")
        fig.canvas.draw()
        fig.canvas.flush_events()
        return fig, lines, title

    @staticmethod
    def _animate_loop(
        plt: Any,
        sim: Simulation,
        options: AnimationOptions,
        on_frame: FrameCallback | None,
        lines: list,
        title: Any,
        controls: PlaybackControls,
    ) -> None:
        frame = 0
        while True:
            if controls.stop or not plt.fignum_exists(plt.gcf().number):
                break
            if controls.reset_pending:
                controls.reset_pending = False
                _reset_simulation(sim)
                frame = 0

            if frame >= options.frames:
                if controls.loop:
                    _reset_simulation(sim)
                    frame = 0
                else:
                    plt.pause(options.matplotlib_pause)
                    continue

            sim.step(options.substeps_per_frame)
            endpoints = sim.endpoints()
            for i, line in enumerate(lines):
                line.set_data(endpoints[i, :, 0], endpoints[i, :, 1])
                line.set_3d_properties(endpoints[i, :, 2])
            title.set_text(f"{_status_text(sim, frame + 1, loop=controls.loop)}    {help_text()}")
            if on_frame is not None:
                on_frame(sim, frame)
            plt.pause(options.matplotlib_pause)
            frame += 1


_BACKENDS: dict[str, Callable[[], Backend]] = {
    "none": HeadlessBackend,
    "pyvista": PyVistaBackend,
    "matplotlib": MatplotlibBackend,
}


def _rod_points(sim: Simulation) -> FloatArray:
    return sim.endpoints().reshape((-1, 3))


def _rod_lines(n_rods: int) -> IntArray:
    lines = np.empty((n_rods, 3), dtype=np.int64)
    lines[:, 0] = 2
    lines[:, 1] = 2 * np.arange(n_rods)
    lines[:, 2] = 2 * np.arange(n_rods) + 1
    return lines.reshape(-1)


def _status_text(sim: Simulation, frame: int, *, loop: bool = False) -> str:
    loop_tag = "  [loop]" if loop else ""
    return f"frame={frame}  t={sim.time:.3f}  rods={sim.state.n_rods}{loop_tag}"


def _reset_simulation(sim: Simulation) -> None:
    """Reset the live simulation to a fresh initial state (for the R key and L loop)."""

    state, kick = build_initial_state(sim.config, sim.inertia)
    sim.state = state
    sim.initial_kick = kick
    sim.step_index = 0


def _set_pyvista_camera(plotter: Any, sim: Simulation, camera_position: str) -> None:
    box = np.asarray(sim.config.system.box, dtype=float)
    center = (float(0.5 * box[0]), float(0.5 * box[1]), float(0.5 * box[2]))
    distance = 2.5 * float(np.linalg.norm(box))
    position, view_up = _camera_from_label(camera_position, center, distance)
    plotter.camera_position = [position, center, view_up]


def _camera_from_label(
    label: str, center: tuple[float, float, float], distance: float
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if label == "xy":
        return (center[0], center[1], center[2] + distance), (0.0, 1.0, 0.0)
    if label == "xz":
        return (center[0], center[1] - distance, center[2]), (0.0, 0.0, 1.0)
    if label == "yz":
        return (center[0] + distance, center[1], center[2]), (0.0, 0.0, 1.0)
    return (
        (center[0] + distance, center[1] - distance, center[2] + 0.7 * distance),
        (0.0, 0.0, 1.0),
    )


def _draw_matplotlib_box(ax: Any, box: FloatArray) -> None:
    corners = np.array(
        [
            [0, 0, 0],
            [box[0], 0, 0],
            [box[0], box[1], 0],
            [0, box[1], 0],
            [0, 0, box[2]],
            [box[0], 0, box[2]],
            [box[0], box[1], box[2]],
            [0, box[1], box[2]],
        ],
        dtype=float,
    )
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for a, b in edges:
        ax.plot(
            [corners[a, 0], corners[b, 0]],
            [corners[a, 1], corners[b, 1]],
            [corners[a, 2], corners[b, 2]],
            linewidth=0.8,
        )
