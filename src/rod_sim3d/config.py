"""Configuration objects and TOML loading for RodSim3D.

The simulation is deliberately configured through small dataclasses instead of a large
framework.  This keeps the project self-contained and makes it easy to inspect or modify
when experimenting with the physical model.
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np


class InteractionModel(StrEnum):
    """Explicit model for rod-rod (and rod-wall) interactions.

    Three physically distinct models are kept separate rather than trying to cast
    them as parameter choices of a single smooth potential. See
    ``docs/physics/interaction_models.md`` for the theoretical derivation.

    Attributes
    ----------
    NONE
        No interaction: ``F_ij = 0``, ``tau_ij = 0``. Rods pass through each
        other; only initial velocities and wall interactions (if any) act.
    SOFT_REPULSION
        Continuous potential-based repulsion. Generates smooth ODE-style
        forces/torques. Allows (small) overlap at finite stiffness; this is
        *not* a rigid-body contact — large ``strength`` only approximates it.
    HARD_CONTACT
        Non-penetration constraint ``g_ij >= 0`` with impulsive collision
        response at ``g_ij = 0``. Governed by a restitution coefficient
        (and optional friction); requires a finite ``contact_radius``.
    """

    NONE = "none"
    SOFT_REPULSION = "soft_repulsion"
    HARD_CONTACT = "hard_contact"


@dataclass(slots=True)
class SystemConfig:
    """Geometry and inertial parameters.

    ``rod_radius`` is the *render / bounding* radius. For hard-contact physics
    the authoritative value is ``PairInteractionConfig.contact_radius`` (which
    defaults to ``rod_radius`` when unset). Keeping them distinct lets the user
    draw a thick cylinder while using a thinner capsule for collision, or vice
    versa.
    """

    n_rods: int = 32
    rod_length: float = 1.0
    rod_radius: float = 0.0
    mass: float = 1.0
    box: tuple[float, float, float] = (6.0, 6.0, 6.0)
    quadrature_points: int = 5


@dataclass(slots=True)
class PairInteractionConfig:
    """Which physical model governs rod-rod contacts.

    ``model = NONE``: rods pass through each other (no pair forces, no contacts).
    ``model = SOFT_REPULSION``: the ``PairPotentialConfig`` block is used to
        generate smooth repulsive forces and torques. Good for visualization;
        not a rigid-body model.
    ``model = HARD_CONTACT``: the capsule non-penetration constraint is
        enforced via impulse at each step. Uses ``contact_radius`` and
        ``restitution``; the ``PairPotentialConfig`` block is ignored.
    """

    model: InteractionModel = InteractionModel.SOFT_REPULSION
    contact_radius: float | None = None
    restitution: float = 0.8
    wall_restitution: float = 0.8


@dataclass(slots=True)
class DerivedInteraction:
    """Resolved parameters ready for physics kernels.

    Populated by :meth:`Config.resolve_interaction` after loading so downstream
    modules do not need to re-interpret the config.
    """

    model: InteractionModel
    contact_radius: float
    restitution: float
    wall_restitution: float


@dataclass(slots=True)
class PairPotentialConfig:
    """Soft shifted-force repulsion between material points on different rods."""

    strength: float = 0.03
    length_scale: float = 0.30
    cutoff: float = 0.85
    exponent: float = 8.0
    softening: float = 0.035


@dataclass(slots=True)
class WallPotentialConfig:
    """Soft one-sided shifted-force repulsion from the six walls."""

    strength: float = 0.08
    length_scale: float = 0.35
    cutoff: float = 0.75
    exponent: float = 8.0
    softening: float = 0.025


@dataclass(slots=True)
class DynamicsConfig:
    """Time integration and damping."""

    dt: float = 0.004
    linear_damping: float = 0.12
    angular_damping: float = 0.12
    max_linear_speed: float | None = 20.0
    max_angular_speed: float | None = 30.0


@dataclass(slots=True)
class InitialConfig:
    """Random initialization and random initial force/torque kick.

    The random initial force and torque are not kept as time-dependent external forces.
    They are converted into an impulse at t=0:

        v(0)     = F0 * kick_duration / mass
        omega(0) = P_perp(tau0) * kick_duration / inertia

    where P_perp removes the component of torque parallel to the rod axis.
    """

    seed: int = 7
    clearance: float = 0.12
    min_segment_distance: float = 0.12
    max_sampling_attempts: int = 20_000
    initial_force_scale: float = 4.0
    initial_torque_scale: float = 1.0
    kick_duration: float = 0.08


@dataclass(slots=True)
class RenderConfig:
    """Animation settings.

    ``opacity`` controls rod translucency in the PyVista replay. Use a value
    below 1 (e.g. 0.5) to see through capsules — particularly useful for the
    pass-through (``InteractionModel.NONE``) mode so overlap regions are
    visible through the outer surfaces.

    ``overlap_opacity`` controls the alpha of the magenta intersection-polytope
    mesh drawn in the NONE replay. Higher = more solid highlight.
    """

    backend: str = "pyvista"
    frames: int = 2500
    substeps_per_frame: int = 3
    line_width: float = 6.0
    window_size: tuple[int, int] = (1100, 850)
    show_box: bool = True
    camera_position: str = "iso"
    matplotlib_pause: float = 0.001
    opacity: float = 1.0
    overlap_opacity: float = 0.9


@dataclass(slots=True)
class OutputConfig:
    """Optional output files."""

    trajectory_npz: str | None = None
    initial_kick_json: str | None = "runs/initial_kick.json"


@dataclass(slots=True)
class Config:
    """Complete simulation configuration."""

    system: SystemConfig = field(default_factory=SystemConfig)
    pair: PairPotentialConfig = field(default_factory=PairPotentialConfig)
    wall: WallPotentialConfig = field(default_factory=WallPotentialConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    initial: InitialConfig = field(default_factory=InitialConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    pair_interaction: PairInteractionConfig = field(default_factory=PairInteractionConfig)

    @classmethod
    def from_toml(cls, path: str | Path) -> Config:
        """Load a configuration from a TOML file.

        Unknown keys are intentionally ignored, allowing users to keep comments or future
        options in local config files without breaking older versions of the code.
        """

        path = Path(path)
        with path.open("rb") as f:
            data = tomllib.load(f)
        return cls.from_mapping(data)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> Config:
        system = _build(SystemConfig, data.get("system", {}))
        pair = _build(PairPotentialConfig, data.get("pair_potential", data.get("pair", {})))
        wall = _build(WallPotentialConfig, data.get("wall_potential", data.get("wall", {})))
        dynamics = _build(DynamicsConfig, data.get("dynamics", {}))
        initial = _build(InitialConfig, data.get("initial", {}))
        render = _build(RenderConfig, data.get("render", {}))
        output = _build(OutputConfig, data.get("output", {}))
        pair_interaction = _build_pair_interaction(
            data.get("pair_interaction", {}), pair_potential=pair
        )
        cfg = cls(
            system=system,
            pair=pair,
            wall=wall,
            dynamics=dynamics,
            initial=initial,
            render=render,
            output=output,
            pair_interaction=pair_interaction,
        )
        cfg.validate()
        return cfg

    def resolve_interaction(self) -> DerivedInteraction:
        """Return the fully-resolved interaction parameters ready for physics."""

        contact_radius = (
            self.pair_interaction.contact_radius
            if self.pair_interaction.contact_radius is not None
            else self.system.rod_radius
        )
        return DerivedInteraction(
            model=self.pair_interaction.model,
            contact_radius=float(contact_radius),
            restitution=float(self.pair_interaction.restitution),
            wall_restitution=float(self.pair_interaction.wall_restitution),
        )

    def validate(self) -> None:
        """Raise ``ValueError`` if the configuration is internally inconsistent.

        Checks are grouped by sub-object so each helper stays short. A single aggregated
        ``validate`` is convenient for the CLI entrypoint.
        """

        _validate_system(self.system)
        _validate_potentials(self.pair, self.wall)
        _validate_dynamics(self.dynamics)
        _validate_initial(self.initial)
        _validate_render(self.render)
        _validate_box_versus_rod(self.system)
        _validate_pair_interaction(self.pair_interaction, self.system)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build(cls: type[Any], raw: dict[str, Any]) -> Any:
    """Construct a dataclass from a mapping, ignoring unknown keys."""

    if not raw:
        return cls()
    allowed = cls.__dataclass_fields__.keys()  # type: ignore[attr-defined]
    kwargs = {k: _coerce_tuple(v) for k, v in raw.items() if k in allowed}
    return cls(**kwargs)


def _coerce_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(value)
    return value


def _validate_system(system: SystemConfig) -> None:
    if system.n_rods <= 0:
        raise ValueError("system.n_rods must be positive.")
    if system.rod_length <= 0:
        raise ValueError("system.rod_length must be positive.")
    if system.rod_radius < 0:
        raise ValueError("system.rod_radius must be non-negative.")
    if system.mass <= 0:
        raise ValueError("system.mass must be positive.")
    if any(v <= 0 for v in system.box):
        raise ValueError("Every box dimension must be positive.")
    if system.quadrature_points < 2:
        raise ValueError("system.quadrature_points must be at least 2.")


def _validate_potentials(pair: PairPotentialConfig, wall: WallPotentialConfig) -> None:
    for label, cfg in (("pair", pair), ("wall", wall)):
        if cfg.cutoff <= 0:
            raise ValueError(f"{label}_potential.cutoff must be positive.")
        if cfg.length_scale <= 0:
            raise ValueError(f"{label}_potential.length_scale must be positive.")
        if cfg.exponent <= 0:
            raise ValueError(f"{label}_potential.exponent must be positive.")
        if cfg.softening <= 0:
            raise ValueError(f"{label}_potential.softening must be positive.")


def _validate_dynamics(dyn: DynamicsConfig) -> None:
    if dyn.dt <= 0:
        raise ValueError("dynamics.dt must be positive.")


def _validate_initial(initial: InitialConfig) -> None:
    if initial.kick_duration < 0:
        raise ValueError("initial.kick_duration must be non-negative.")


def _validate_render(render: RenderConfig) -> None:
    if render.substeps_per_frame <= 0:
        raise ValueError("render.substeps_per_frame must be positive.")
    if render.frames <= 0:
        raise ValueError("render.frames must be positive.")


def _validate_box_versus_rod(system: SystemConfig) -> None:
    box = np.asarray(system.box, dtype=float)
    if np.any(box <= system.rod_length):
        raise ValueError("Each box dimension should exceed rod_length for random initialization.")


def _validate_pair_interaction(pi: PairInteractionConfig, system: SystemConfig) -> None:
    if pi.contact_radius is not None and pi.contact_radius < 0:
        raise ValueError("pair_interaction.contact_radius must be non-negative if given.")
    if not 0.0 <= pi.restitution <= 1.0:
        raise ValueError("pair_interaction.restitution must be in [0, 1].")
    if not 0.0 <= pi.wall_restitution <= 1.0:
        raise ValueError("pair_interaction.wall_restitution must be in [0, 1].")
    if pi.model is InteractionModel.HARD_CONTACT:
        # A capsule with zero radius has measure-zero contact manifold; the
        # impulsive contact solver has nothing to bite on.
        effective_radius = pi.contact_radius if pi.contact_radius is not None else system.rod_radius
        if effective_radius <= 0.0:
            raise ValueError(
                "HARD_CONTACT requires a positive contact_radius or rod_radius "
                "(capsule collisions are ill-defined for zero-thickness lines)."
            )


def _build_pair_interaction(
    raw: dict[str, Any], pair_potential: PairPotentialConfig
) -> PairInteractionConfig:
    """Build the interaction config with a gentle back-compat default.

    If the user did not supply a ``[pair_interaction]`` block, infer the model
    from ``pair_potential.strength``: zero strength → ``NONE``, otherwise
    ``SOFT_REPULSION``. This keeps pre-existing configs working unchanged.
    """

    if raw:
        allowed = PairInteractionConfig.__dataclass_fields__.keys()
        kwargs: dict[str, Any] = {k: _coerce_tuple(v) for k, v in raw.items() if k in allowed}
        if "model" in kwargs and isinstance(kwargs["model"], str):
            kwargs["model"] = InteractionModel(kwargs["model"])
        return PairInteractionConfig(**kwargs)
    inferred_model = (
        InteractionModel.NONE if pair_potential.strength == 0.0 else InteractionModel.SOFT_REPULSION
    )
    return PairInteractionConfig(model=inferred_model)
