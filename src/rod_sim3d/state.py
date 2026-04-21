"""Runtime state of the rod ensemble.

Keeping the state in a tiny dataclass (instead of scattered arrays on a God-object)
makes the data layout explicit. The same struct can be reused verbatim in a Rust port:
four row-major ``[N, 3]`` matrices.
"""

from __future__ import annotations

from dataclasses import dataclass

from rod_sim3d._array import FloatArray
from rod_sim3d.geometry import normalize, project_perpendicular


@dataclass(slots=True)
class RodState:
    """Dynamical state of all rods.

    Attributes
    ----------
    positions : ``(N, 3)`` center of each rod
    directions : ``(N, 3)`` unit axis of each rod
    velocities : ``(N, 3)`` center velocity of each rod
    omegas : ``(N, 3)`` angular velocity, always perpendicular to the rod axis
    """

    positions: FloatArray
    directions: FloatArray
    velocities: FloatArray
    omegas: FloatArray

    def copy(self) -> RodState:
        return RodState(
            positions=self.positions.copy(),
            directions=self.directions.copy(),
            velocities=self.velocities.copy(),
            omegas=self.omegas.copy(),
        )

    @property
    def n_rods(self) -> int:
        return int(self.positions.shape[0])

    def enforce_constraints(self) -> None:
        """Re-normalize directions and project out spin around the rod axis.

        Called at every integration step so numerical drift never accumulates.
        """

        self.directions[:] = normalize(self.directions)
        self.omegas[:] = project_perpendicular(self.omegas, self.directions)
