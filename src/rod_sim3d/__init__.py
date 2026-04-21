"""RodSim3D: soft-repulsive equal-length rod dynamics in a 3D box.

The public surface is deliberately small and stable; richer submodules (``forces``,
``potentials``, ``dynamics``, ``renderer`` etc.) are available for programmatic use.
"""

from rod_sim3d.config import Config
from rod_sim3d.initial_conditions import InitialKick
from rod_sim3d.simulation import EnergyBreakdown, Simulation
from rod_sim3d.state import RodState

__all__ = [
    "Config",
    "EnergyBreakdown",
    "InitialKick",
    "RodState",
    "Simulation",
]
