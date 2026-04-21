"""Shared keyboard-driven playback controls for PyVista and matplotlib viewers.

Centralises the three commands wired up by :mod:`rod_sim3d.replay` and
:mod:`rod_sim3d.renderer`:

- ``Esc`` / ``Q``: stop the loop.
- ``R``: reset to the beginning (loop resets the index; live run resets the state).
- ``L``: toggle looping — when the loop reaches its last frame, go back to 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class PlaybackControls:
    """Flags set by keyboard handlers and polled by the animation loop.

    ``reset_pending`` is one-shot — the caller must clear it after acting on it.
    ``loop`` is toggled by the L key.
    """

    stop: bool = False
    reset_pending: bool = False
    loop: bool = False


def install_pyvista_playback_controls(plotter: Any) -> PlaybackControls:
    """Bind ``Esc/Q`` (stop), ``R`` (reset), ``L`` (loop toggle) to a PyVista plotter."""

    controls = PlaybackControls()

    def on_stop() -> None:
        controls.stop = True

    def on_reset() -> None:
        controls.reset_pending = True

    def on_loop() -> None:
        controls.loop = not controls.loop

    for keys, callback in (
        (("q", "Q", "Escape", "escape"), on_stop),
        (("r", "R"), on_reset),
        (("l", "L"), on_loop),
    ):
        for key in keys:
            try:
                plotter.add_key_event(key, callback)
            except Exception:
                # Some backends reject unknown key names silently; iterate past them.
                continue

    return controls


def install_matplotlib_playback_controls(fig: Any) -> PlaybackControls:
    """Bind ``Esc/Q`` / ``R`` / ``L`` and the window close button to a matplotlib figure."""

    controls = PlaybackControls()

    def on_key(event: Any) -> None:
        key = event.key
        if key in ("q", "Q", "escape"):
            controls.stop = True
        elif key in ("r", "R"):
            controls.reset_pending = True
        elif key in ("l", "L"):
            controls.loop = not controls.loop

    def on_close(_event: Any) -> None:
        controls.stop = True

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", on_close)
    return controls


def pyvista_window_is_gone(plotter: Any) -> bool:
    """True if the user already closed the PyVista window (X button or internal close)."""

    render_window = getattr(plotter, "render_window", None)
    if render_window is None:
        return True
    return bool(getattr(plotter, "_closed", False))


def help_text() -> str:
    """Footer text shown in the viewer to remind the user of the keybindings."""

    return "Esc/Q: quit  |  R: reset  |  L: loop"
